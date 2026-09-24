"""Continuity through public generation, without synthesis or encoder downloads."""

import asyncio
import io
import json
from dataclasses import asdict
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
import torch

from rho_tts import CancellationToken, ContinuityConfig
from rho_tts import continuity as module
from rho_tts.exceptions import ValidationError
from rho_tts.isolation.worker import Worker
from tests.test_isolation.test_proxy import TestProviderProxy as ProxyHarness
from tests.test_pipeline import FakeTTS


def candidates(tts, amplitudes):
    values = iter(amplitudes)
    seeds = []

    def generate(text):
        seeds.append(tts.seed)
        return torch.sin(torch.arange(16000) * 0.07) * next(values)

    tts._generate_audio = Mock(side_effect=generate)
    return seeds


def reference(tmp_path):
    path = tmp_path / "previous.wav"
    sf.write(path, np.sin(np.arange(16000) * 0.07) * 0.1, 16000)
    return path


class TestContinuity:
    def setup_method(self):
        self.original_features = module._features
        self.monkeypatch = pytest.MonkeyPatch()
        self.tts = FakeTTS()
        self.tts.max_iterations = 3
        self.tts._validate_accent_drift = Mock(return_value=(0.01, True))
        self.tts._validate_text_match = Mock(return_value=(True, 1.0, "Hello"))
        self.tts._validate_sound_decay = Mock(return_value=(1.0, True))
        # Save real WAVs without loading torchaudio's optional codec stack.
        self.tts._save_wav = lambda path, audio, rate: sf.write(path, audio.cpu().numpy().T, rate)
        # Retain real frame-level measurement; speaker identity is deterministic.
        self.monkeypatch.setattr(
            module, "_features", lambda samples, rate: (np.array([1.0, 0.0]), module.speech_level(samples, rate))
        )

    def teardown_method(self):
        self.monkeypatch.undo()

    def test_accent_failure_reports_text_validation_as_skipped(self):
        tts = self.tts
        candidates(tts, [0.1, 0.1, 0.1])
        tts._validate_accent_drift.return_value = (0.8, False)

        result = tts.generate("Hello", continuity=ContinuityConfig())

        tts._validate_text_match.assert_not_called()
        segment = result.continuity["segments"][0]
        assert segment["fallback"]
        assert all(attempt["text_passed"] is None for attempt in segment["attempts"])
        assert result.text_similarity is None

    def test_retry_reports_only_measured_text_verdicts(self):
        tts = self.tts
        candidates(tts, [0.1, 0.1, 0.1])
        tts._validate_accent_drift.side_effect = [(0.8, False), (0.01, True), (0.01, True)]
        tts._validate_text_match.side_effect = [(False, 0.2, "wrong"), (True, 0.9, "Hello")]

        result = tts.generate("Hello", continuity=ContinuityConfig())

        segment = result.continuity["segments"][0]
        assert [attempt["text_passed"] for attempt in segment["attempts"]] == [None, False, True]
        assert segment["selected_attempt"] == 3
        assert not segment["fallback"]
        assert result.text_similarity == 0.9

    def test_retry_budget_and_seed_selection(self, tmp_path):
        tts = self.tts
        seeds = candidates(tts, [0.8, 0.1])
        result = tts.generate("Hello", continuity=ContinuityConfig(previous_audio=reference(tmp_path)))
        assert len(seeds) == 2
        assert seeds[1] == seeds[0] + 1
        report = result.continuity
        assert report["passed"]
        assert report["segments"][0]["selected_attempt"] == 2
        assert [a["passed"] for a in report["segments"][0]["attempts"]] == [False, True]

    def test_exhaustion_selects_closest_audio(self, tmp_path):
        tts = self.tts
        candidates(tts, [0.8, 0.3, 0.6])
        result = tts.generate("Hello", continuity=ContinuityConfig(previous_audio=reference(tmp_path)))
        assert not result.continuity["passed"]
        assert result.continuity["segments"][0]["selected_attempt"] == 2
        assert result.audio.abs().max().item() == pytest.approx(0.3, abs=0.001)
        assert tts._generate_audio.call_count == 3

    @pytest.mark.parametrize("strict", [False, True])
    def test_strict_exhaustion_does_not_publish(self, tmp_path, strict):
        tts = self.tts
        candidates(tts, [0.8, 0.3, 0.6])
        tts.strict_validation = strict
        target = tmp_path / "output.wav"
        with pytest.raises(ValidationError, match="3 attempts"):
            tts.generate(
                "Hello",
                str(target),
                continuity=ContinuityConfig(previous_audio=reference(tmp_path), allow_fallback=strict),
            )
        assert not target.exists()

    def test_internal_segments_and_list_items_share_only_call_context(self):
        tts = self.tts
        candidates(tts, [0.1, 0.8, 0.1, 0.8, 0.1])
        tts._split_text_into_segments = lambda text, max_chars: ["one", "two"] if text == "split" else [text]
        results = tts.generate(["split", "next"], continuity=ContinuityConfig())
        assert [len(r.continuity["segments"]) for r in results] == [2, 1]
        assert results[0].continuity["segments"][1]["selected_attempt"] == 2
        assert results[1].continuity["segments"][0]["selected_attempt"] == 2
        candidates(tts, [0.8])
        fresh = tts.generate("Unrelated", continuity=ContinuityConfig())
        assert fresh.continuity["segments"][0]["attempts"][0]["reason"].startswith("First segment")

    def test_existing_validation_and_continuity_share_budget(self, tmp_path):
        tts = self.tts
        candidates(tts, [0.1, 0.8, 0.1])
        tts._validate_text_match.side_effect = [(False, 0.2, "wrong"), (True, 1.0, "Hello"), (True, 1.0, "Hello")]
        result = tts.generate("Hello", continuity=ContinuityConfig(previous_audio=reference(tmp_path)))
        assert result.continuity["segments"][0]["selected_attempt"] == 3
        assert tts._generate_audio.call_count == 3
        assert tts._validate_text_match.call_count == 3

    def test_unavailable_encoder_fails_even_with_fallback(self, monkeypatch, tmp_path):
        tts = self.tts
        monkeypatch.setattr(module, "_features", Mock(side_effect=RuntimeError("encoder missing")))
        with pytest.raises(ValidationError, match="encoder missing"):
            tts.generate("Hello", str(tmp_path / "out.wav"), continuity=ContinuityConfig())
        assert not (tmp_path / "out.wav").exists()

    def test_missing_reference_fails_before_synthesis(self, tmp_path):
        tts = self.tts
        candidates(tts, [0.1])
        with pytest.raises(ValidationError, match="reference unavailable"):
            tts.generate("Hello", continuity=ContinuityConfig(previous_audio=tmp_path / "missing.wav"))
        tts._generate_audio.assert_not_called()

    def test_cancelled_comparison_does_not_publish_or_retry(self, monkeypatch, tmp_path):
        tts = self.tts
        token = CancellationToken()
        candidates(tts, [0.1])
        original = module._features

        def cancel(samples, rate):
            token.cancel()
            return original(samples, rate)

        monkeypatch.setattr(module, "_features", cancel)
        result = tts.generate(
            "Hello", str(tmp_path / "out.wav"), cancellation_token=token, continuity=ContinuityConfig()
        )
        assert result is None
        assert tts._generate_audio.call_count == 1
        assert not (tmp_path / "out.wav").exists()

    def test_padding_and_loudness(self):
        tone = np.sin(np.arange(16000) * 0.07) * 0.1
        assert module.speech_level(tone, 16000) == pytest.approx(
            module.speech_level(np.r_[tone, np.zeros(32000)], 16000)
        )
        assert module.speech_level(tone * 10, 16000) - module.speech_level(tone, 16000) == pytest.approx(20)
        with pytest.raises(ValueError, match="speech"):
            module.speech_level(np.zeros(16000), 16000)

    def test_speaker_mismatch(self, monkeypatch):
        features = iter([(np.array([1.0, 0.0]), -20.0), (np.array([0.0, 1.0]), -20.0)])
        monkeypatch.setattr(module, "_features", lambda *a: next(features))
        validator = module.ContinuityValidator(ContinuityConfig())
        _, validator.previous = validator.evaluate(torch.ones(16000), 16000)
        result, _ = validator.evaluate(torch.ones(16000), 16000)
        assert not result["passed"]
        assert result["speaker_similarity"] == 0
        assert result["loudness_difference_db"] == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_similarity": float("nan")},
            {"min_similarity": 0},
            {"max_loudness_db": float("inf")},
            {"max_loudness_db": 0},
        ],
    )
    def test_invalid_config(self, kwargs):
        with pytest.raises(ValueError):
            ContinuityConfig(**kwargs)

    @pytest.mark.parametrize("batch", [False, True])
    def test_worker_round_trip(self, tmp_path, batch):
        tts = self.tts
        output = io.StringIO()
        worker = Worker(protocol_out=output)
        worker._tts = tts
        # The proxy uses actual JSON messages through the worker with fake synthesis.
        proxy, transport = ProxyHarness()._make_proxy([{"type": "ready", "sample_rate": 16000}])

        def send(kind, **kwargs):
            output.seek(0)
            output.truncate()
            worker._handle_generate(json.loads(json.dumps(kwargs)))
            return json.loads(output.getvalue())

        transport.send.side_effect = send
        context = ContinuityConfig(previous_audio=reference(tmp_path))
        candidates(tts, [0.8, 0.1, 0.1] if batch else [0.8, 0.1])
        result = proxy.generate(["Hello", "World"] if batch else "Hello", str(tmp_path / "out.wav"), continuity=context)
        first = result[0] if batch else result
        assert first.continuity["passed"]
        assert first.continuity["segments"][0]["selected_attempt"] == 2
        assert transport.send.call_args.kwargs["continuity"] == asdict(context)
        proxy.close()

    def test_async_context(self, tmp_path):
        tts = self.tts
        candidates(tts, [0.8, 0.1])
        result = asyncio.run(
            tts.async_generate("Hello", continuity=ContinuityConfig(previous_audio=reference(tmp_path)))
        )
        assert result.continuity["segments"][0]["selected_attempt"] == 2

    def test_packaged_worker_installs_matching_version(self, monkeypatch, tmp_path):
        from rho_tts import __version__
        from rho_tts.isolation import venv_manager

        monkeypatch.setattr(venv_manager, "_find_project_root", lambda: None)
        run = Mock(return_value=Mock(returncode=0))
        monkeypatch.setattr(venv_manager.subprocess, "run", run)
        venv_manager.VenvManager("breeze", venvs_root=tmp_path)._install_package()
        assert run.call_args.args[0][-1] == f"rho-tts[breeze,validation]=={__version__}"

    def test_decay_retry_restarts_from_original_context(self):
        tts = self.tts
        candidates(tts, [0.1, 0.18, 0.05, 0.05])
        tts._split_text_into_segments = lambda text, max_chars: ["one", "two"]
        tts._validate_sound_decay.side_effect = [(0.0, False), (1.0, True)]
        result = tts.generate("split", continuity=ContinuityConfig())
        assert result.continuity["segments"][0]["attempts"][0]["reason"].startswith("First segment")
        assert tts._generate_audio.call_count == 4

    def test_encoder_features_reject_invalid_embeddings(self, monkeypatch):
        import sys
        from types import SimpleNamespace

        monkeypatch.setattr(module, "_features", self.original_features)
        monkeypatch.setitem(sys.modules, "resemblyzer", SimpleNamespace(preprocess_wav=lambda wav, **kw: wav))
        monkeypatch.setattr(
            module, "_encoder", lambda: SimpleNamespace(embed_utterance=lambda wav: np.array([np.nan, 0.0]))
        )
        with pytest.raises(ValidationError, match="invalid speaker features"):
            module.ContinuityValidator(ContinuityConfig()).evaluate(torch.ones(16000) * 0.1, 16000)
