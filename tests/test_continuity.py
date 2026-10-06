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


VOICE = dict(pitch_hz=100.0, tilt_db=-18.0, voiced_seconds=1.0)


def harmonic(pitch, seconds=2.0, rate=16000, brightness=1.0):
    """A voiced-sounding tone: harmonics of ``pitch`` whose upper partials scale with ``brightness``."""
    t = np.arange(int(seconds * rate)) / rate
    partials = [(k, 1 / k * (brightness if k * pitch > 2000 else 1)) for k in range(1, int(6000 / pitch))]
    wave = sum(a * np.sin(2 * np.pi * k * pitch * t) for k, a in partials)
    return (0.1 * wave / np.abs(wave).max()).astype(np.float32)


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
        # Pitch tracking is exercised directly below; elsewhere every clip sounds alike.
        self.monkeypatch.setattr(module, "voice_features", lambda samples, rate=16000: VOICE)

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
        selected = result.acceptance["segments"][0]
        assert selected["validators"]["text"]["status"] == "skipped"
        assert selected["fallback_reasons"] == ["accent failed", "retries exhausted"]

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
        attempts = result.acceptance['segments'][0]['attempts']
        assert attempts[1]['seed'] != attempts[0]['seed']
        assert tts.seed == 42
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

    @pytest.mark.parametrize("drifts, selected", [
        # A loud take that passes accent loses to a level take that drifts slightly...
        ([0.05, 0.2, 0.9], 2),
        # ...but a level take with far worse drift loses to a slightly loud one.
        ([0.05, 0.9, 0.9], 1),
    ])
    def test_exhaustion_sums_threshold_misses(self, tmp_path, drifts, selected):
        tts = self.tts
        threshold = tts.accent_drift_threshold
        tts._validate_accent_drift = Mock(side_effect=[(d, d <= threshold) for d in drifts])
        # Previous speech is 0.1; 0.25 is 8 dB louder (limit 6 dB), 0.1 matches.
        candidates(tts, [0.25, 0.1, 0.8])
        result = tts.generate("Hello", continuity=ContinuityConfig(previous_audio=reference(tmp_path)))
        assert result.continuity["segments"][0]["selected_attempt"] == selected
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
        # Positive samples are one speaker, negative samples another.
        monkeypatch.setattr(
            module, "_features", lambda samples, rate: (np.array([1.0, 0.0] if samples.mean() > 0 else [0.0, 1.0]), -20.0)
        )
        validator = module.ContinuityValidator(ContinuityConfig())
        _, speech = validator.evaluate(torch.ones(32000) * 0.1, 16000)
        validator.advance(speech)
        result, _ = validator.evaluate(-torch.ones(32000) * 0.1, 16000)
        assert not result["passed"]
        assert result["speaker_judged"]
        assert result["speaker_similarity"] == 0
        assert result["loudness_difference_db"] == pytest.approx(0, abs=1e-6)

    def test_short_predecessor_is_not_the_whole_anchor(self):
        validator = module.ContinuityValidator(ContinuityConfig())
        clip = lambda seconds: np.ones(int(seconds * 16000), dtype=np.float32)
        for seconds in (0.8, 2.0, 2.0):
            validator.advance(clip(seconds))
        # The 0.8 s clip is dropped only once newer speech covers the window.
        assert [len(s) / 16000 for s in validator.previous] == [2.0, 2.0]
        validator.advance(clip(4.0))
        assert [len(s) / 16000 for s in validator.previous] == [4.0]

    def test_reference_voice_checks_the_first_segment(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            module, "_features", lambda samples, rate: (np.array([1.0, 0.0] if samples.mean() > 0 else [0.0, 1.0]), -20.0)
        )
        voice = tmp_path / "voice.wav"
        sf.write(voice, np.full(24000, 0.1), 24000)
        validator = module.ContinuityValidator(ContinuityConfig(), reference_audio=str(voice))
        result, _ = validator.evaluate(-torch.ones(32000) * 0.1, 16000)
        assert not result["passed"]
        assert result["reference_voice"]
        assert result["anchor_seconds"] == pytest.approx(1.0)
        assert result["loudness_difference_db"] is None

    def test_provider_reference_voice_anchors_generation(self, tmp_path):
        tts = self.tts
        tts.reference_audio_path = str(reference(tmp_path))
        candidates(tts, [0.1])
        result = tts.generate("Hello", continuity=ContinuityConfig())
        attempt = result.continuity["segments"][0]["attempts"][0]
        assert attempt["reference_voice"] and attempt["passed"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_similarity": float("nan")},
            {"min_similarity": 0},
            {"max_loudness_db": float("inf")},
            {"max_loudness_db": 0},
            {"max_pitch_semitones": 0},
            {"max_pitch_semitones": float("nan")},
            {"max_tilt_db": -1},
            {"max_tilt_db": float("inf")},
            {"max_adjacent_pitch_semitones": 0},
            {"max_adjacent_pitch_semitones": float("nan")},
            {"max_adjacent_tilt_db": -1},
            {"max_adjacent_tilt_db": float("inf")},
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
        assert first.acceptance["accepted"]
        assert first.acceptance["segments"][0]["selected_attempt"] == 2
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


class TestVoiceConsistency:
    """Checks resemblyzer cannot make: a short clip, pitch, brightness and voicing."""

    def setup_method(self):
        self.monkeypatch = pytest.MonkeyPatch()
        # Speakers differ by sign; loudness is real.
        self.monkeypatch.setattr(module, "_features", lambda samples, rate: (
            np.array([1.0, 0.0] if samples.mean() >= 0 else [0.0, 1.0]), module.speech_level(samples, rate)))

    def teardown_method(self):
        self.monkeypatch.undo()

    def voices(self, *features):
        """Queue voice_features results: each anchor change measures the anchor, then each candidate."""
        values = iter(features)
        self.monkeypatch.setattr(module, "voice_features", lambda samples, rate=16000: next(values))

    def accepted(self, validator, seconds=2.0):
        _, speech = validator.evaluate(torch.ones(int(seconds * 16000)) * 0.1, 16000)
        validator.advance(speech)

    def test_short_clip_reports_but_does_not_judge_the_speaker(self):
        self.voices(VOICE, VOICE)
        validator = module.ContinuityValidator(ContinuityConfig())
        self.accepted(validator)
        result, _ = validator.evaluate(-torch.ones(int(0.8 * 16000)) * 0.1, 16000)
        assert result["speaker_similarity"] == 0
        assert not result["speaker_judged"]
        assert result["passed"]

    @pytest.mark.parametrize("pitch, passed", [(100 * 2 ** (2.9 / 12), True), (100 * 2 ** (4.6 / 12), False),
                                               (100 * 2 ** (-4.4 / 12), False)])
    def test_pitch_jump_fails(self, pitch, passed):
        self.voices(VOICE, dict(VOICE, pitch_hz=pitch))
        validator = module.ContinuityValidator(ContinuityConfig())
        self.accepted(validator)
        result, _ = validator.evaluate(torch.ones(32000) * 0.1, 16000)
        assert result["pitch_difference_semitones"] == pytest.approx(12 * np.log2(pitch / 100))
        assert result["passed"] is passed

    def test_brightness_jump_fails(self):
        self.voices(VOICE, dict(VOICE, tilt_db=-12.0))
        validator = module.ContinuityValidator(ContinuityConfig())
        self.accepted(validator)
        result, _ = validator.evaluate(torch.ones(32000) * 0.1, 16000)
        assert result["tilt_difference_db"] == pytest.approx(6.0)
        assert not result["passed"]

    def level_voice(self, tilt_per_unit=0.0):
        """Pitch rises 20 st, and tilt by ``tilt_per_unit`` dB, per unit of mean amplitude above 0.1.

        Linear in the mean, so a concatenated window measures as the average of its clips.
        """
        def features(samples, rate=16000):
            shift = (float(np.mean(samples)) - 0.1)
            return dict(VOICE, pitch_hz=100 * 2 ** (shift * 20 / 12), tilt_db=VOICE["tilt_db"] + shift * tilt_per_unit)
        self.monkeypatch.setattr(module, "voice_features", features)

    def sentence(self, validator, level):
        result, speech = validator.evaluate(torch.ones(32000) * level, 16000)
        return result, speech

    @pytest.mark.parametrize("limit, passed", [(None, True), (3.5, True), (2.5, False)])
    def test_jump_from_the_previous_sentence(self, limit, passed):
        # Accepted 0 st then +2 st; the window averages +1 st. A -1 st candidate is
        # 2 st from the window but 3 st below the sentence just before it.
        self.level_voice()
        validator = module.ContinuityValidator(ContinuityConfig(max_loudness_db=30, max_adjacent_pitch_semitones=limit))
        for level in (0.1, 0.2):
            validator.advance(self.sentence(validator, level)[1])
        result, _ = self.sentence(validator, 0.05)
        assert result["pitch_difference_semitones"] == pytest.approx(-2.0)
        if limit is None:
            assert result["adjacent_pitch_difference_semitones"] is None
        else:
            assert result["adjacent_pitch_difference_semitones"] == pytest.approx(-3.0)
        assert result["passed"] is passed

    def test_brightness_jump_from_the_previous_sentence(self):
        self.level_voice(tilt_per_unit=40.0)
        validator = module.ContinuityValidator(ContinuityConfig(
            max_loudness_db=30, max_pitch_semitones=24, max_adjacent_pitch_semitones=24, max_adjacent_tilt_db=3))
        for level in (0.1, 0.2):
            validator.advance(self.sentence(validator, level)[1])
        result, _ = self.sentence(validator, 0.1)
        assert result["tilt_difference_db"] == pytest.approx(-2.0)
        assert result["adjacent_tilt_difference_db"] == pytest.approx(-4.0)
        assert result["maximum_adjacent_tilt_difference_db"] == 3
        assert not result["passed"]

    def test_adjacent_check_follows_a_restored_window(self):
        self.level_voice()
        validator = module.ContinuityValidator(ContinuityConfig(max_loudness_db=30, max_adjacent_pitch_semitones=1.5))
        validator.advance(self.sentence(validator, 0.1)[1])
        snapshot = validator.previous
        validator.advance(self.sentence(validator, 0.2)[1])
        validator.previous = snapshot
        result, _ = self.sentence(validator, 0.15)
        assert result["adjacent_pitch_difference_semitones"] == pytest.approx(1.0)
        assert result["passed"]

    def test_unmeasurable_pitch_is_not_judged_but_silent_voicing_fails(self):
        quiet = dict(pitch_hz=None, tilt_db=None, voiced_seconds=0.1)
        self.voices(VOICE, quiet, dict(quiet, voiced_seconds=0.0))
        validator = module.ContinuityValidator(ContinuityConfig())
        self.accepted(validator)
        result, _ = validator.evaluate(torch.ones(32000) * 0.1, 16000)
        assert result["passed"] and result["pitch_difference_semitones"] is None
        result, _ = validator.evaluate(torch.ones(32000) * 0.1, 16000)
        assert result["unvoiced"] and not result["passed"]

    def test_reference_anchors_pitch_until_speech_fills_the_window(self, tmp_path):
        voice = tmp_path / "voice.wav"
        sf.write(voice, np.full(32000, 0.1), 16000)
        anchors = []
        high = dict(VOICE, pitch_hz=130.0)

        def features(samples, rate=16000):
            anchors.append(len(samples) / 16000)
            return VOICE if len(anchors) == 1 else high
        self.monkeypatch.setattr(module, "voice_features", features)
        validator = module.ContinuityValidator(ContinuityConfig(), reference_audio=str(voice))
        result, speech = validator.evaluate(torch.ones(32000) * 0.1, 16000)
        assert not result["passed"]  # 130 Hz against the 100 Hz reference
        validator.advance(speech)
        validator.evaluate(torch.ones(32000) * 0.1, 16000)
        validator.advance(speech)
        result, _ = validator.evaluate(torch.ones(32000) * 0.1, 16000)
        assert result["passed"]
        # Measured: reference, candidate, reference + 2 s, candidate, 4 s of speech alone, candidate.
        assert anchors == [2.0, 2.0, 4.0, 2.0, 4.0, 2.0]

    def test_voice_features_measure_pitch_and_brightness(self):
        dull = module.voice_features(harmonic(110))
        bright = module.voice_features(harmonic(110, brightness=4.0))
        assert dull["pitch_hz"] == pytest.approx(110, rel=0.03)
        assert dull["voiced_seconds"] > 1.5
        assert bright["tilt_db"] - dull["tilt_db"] == pytest.approx(20 * np.log10(4), abs=1.5)

    def test_voice_features_need_enough_voiced_speech(self):
        result = module.voice_features(harmonic(110, seconds=0.15))
        assert result["pitch_hz"] is None and result["tilt_db"] is None
