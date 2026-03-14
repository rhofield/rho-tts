"""Tests for BaseTTS._run_pipeline() — the core orchestration loop."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from rho_tts.base_tts import BaseTTS
from rho_tts.cancellation import CancellationToken, CancelledException


class FakeTTS(BaseTTS):
    """Minimal concrete subclass for testing _run_pipeline()."""

    def __init__(self, sr=16000):
        self.device = "cpu"
        self.seed = 42
        self.deterministic = False
        self.phonetic_mapping = {}
        self.silence_threshold_db = -50.0
        self.crossfade_duration_sec = 0.05
        self.trim_silence = True
        self.fade_duration_sec = 0.02
        self.force_sentence_split = True
        self.inter_sentence_pause_sec = 0.1
        self._voice_encoder = None
        self.reference_embedding = None
        self._sample_rate = sr
        self.max_chars_per_segment = 800
        self.max_iterations = 1
        self.accent_drift_threshold = 0.17
        self.text_similarity_threshold = 0.85
        self.voice_id = None
        self.drift_model_path = None
        self._max_chars_explicit = True
        self._max_model_chars = 3000

    def _generate_audio(self, text, **kwargs):
        # Return 1 second of non-silent audio (sine wave) to survive trimming
        t = torch.linspace(0, 1, self._sample_rate)
        return torch.sin(2 * 3.14159 * 440 * t) * 0.5

    @property
    def sample_rate(self):
        return self._sample_rate


class TestRunPipelineBasic:
    def test_single_text_returns_one_result(self):
        tts = FakeTTS()
        token = CancellationToken()
        results = tts._run_pipeline(["Hello world"], token)

        assert len(results) == 1
        assert results[0] is not None
        audio, seg_count, metadata = results[0]
        assert isinstance(audio, torch.Tensor)
        assert audio.numel() > 0
        assert seg_count >= 1
        assert isinstance(metadata, dict)

    def test_multiple_texts_returns_multiple_results(self):
        tts = FakeTTS()
        token = CancellationToken()
        results = tts._run_pipeline(["Hello", "World", "Test"], token)

        assert len(results) == 3
        for r in results:
            assert r is not None
            audio, seg_count, metadata = r
            assert audio.numel() > 0

    def test_empty_list_returns_empty(self):
        tts = FakeTTS()
        token = CancellationToken()
        results = tts._run_pipeline([], token)
        assert results == []


class TestRunPipelineCancellation:
    def test_cancel_before_start_raises(self):
        tts = FakeTTS()
        token = CancellationToken()
        token.cancel()

        with pytest.raises(CancelledException):
            tts._run_pipeline(["Hello"], token)

    def test_cancel_between_items_raises(self):
        call_count = 0

        class CancellingTTS(FakeTTS):
            def _generate_audio(self, text, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    token.cancel()
                return super()._generate_audio(text, **kwargs)

        tts = CancellingTTS()
        token = CancellationToken()

        with pytest.raises(CancelledException):
            tts._run_pipeline(["First", "Second", "Third"], token)

    def test_cancel_during_iteration_raises(self):
        call_count = 0

        class CancelOnSecondIter(FakeTTS):
            def __init__(self):
                super().__init__()
                self.max_iterations = 5

            def _generate_audio(self, text, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    token.cancel()
                return super()._generate_audio(text, **kwargs)

        tts = CancelOnSecondIter()
        token = CancellationToken()

        # Mock validation to fail, forcing iteration loop to retry and hit cancel
        with patch.object(tts, '_validate_accent_drift', return_value=(0.5, False)):
            with pytest.raises(CancelledException):
                tts._run_pipeline(["Hello"], token)


class TestRunPipelinePhoneticMapping:
    def test_phonetic_mapping_applied(self):
        generated_texts = []

        class TrackingTTS(FakeTTS):
            def _generate_audio(self, text, **kwargs):
                generated_texts.append(text)
                return super()._generate_audio(text, **kwargs)

        tts = TrackingTTS()
        tts.phonetic_mapping = {"hello": "heh-low"}
        token = CancellationToken()
        tts._run_pipeline(["hello world"], token)

        assert "heh-low world" in generated_texts


class TestRunPipelineSegmentation:
    def test_long_text_split_into_segments(self):
        segments_generated = []

        class TrackingTTS(FakeTTS):
            def _generate_audio(self, text, **kwargs):
                segments_generated.append(text)
                return super()._generate_audio(text, **kwargs)

        tts = TrackingTTS()
        tts.max_chars_per_segment = 50
        # Two sentences that force splitting
        long_text = "This is the first sentence. This is the second sentence."
        token = CancellationToken()
        tts._run_pipeline([long_text], token)

        assert len(segments_generated) >= 2

    def test_segment_count_matches(self):
        tts = FakeTTS()
        tts.max_chars_per_segment = 50
        long_text = "First sentence. Second sentence. Third sentence."
        token = CancellationToken()
        results = tts._run_pipeline([long_text], token)

        assert results[0] is not None
        _, seg_count, _ = results[0]
        assert seg_count >= 2


class TestRunPipelineGenerationErrors:
    def test_runtime_error_oom_retries(self):
        """OOM errors should be caught and retried, not crash the pipeline."""
        call_count = 0

        class OOMThenSuccessTTS(FakeTTS):
            def __init__(self):
                super().__init__()
                self.max_iterations = 3

            def _generate_audio(self, text, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("CUDA out of memory")
                return super()._generate_audio(text, **kwargs)

        tts = OOMThenSuccessTTS()
        token = CancellationToken()
        results = tts._run_pipeline(["Hello"], token)

        assert results[0] is not None
        assert call_count >= 2

    def test_value_error_propagates(self):
        """ValueError (config errors) should propagate, not retry."""

        class BadConfigTTS(FakeTTS):
            def _generate_audio(self, text, **kwargs):
                raise ValueError("Invalid config")

        tts = BadConfigTTS()
        tts.max_iterations = 3
        token = CancellationToken()

        with pytest.raises(ValueError, match="Invalid config"):
            tts._run_pipeline(["Hello"], token)

    def test_generic_exception_retries(self):
        """Other exceptions are caught and retried."""
        call_count = 0

        class FailOnceTTS(FakeTTS):
            def __init__(self):
                super().__init__()
                self.max_iterations = 3

            def _generate_audio(self, text, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("transient error")
                return super()._generate_audio(text, **kwargs)

        tts = FailOnceTTS()
        token = CancellationToken()
        results = tts._run_pipeline(["Hello"], token)
        assert results[0] is not None

    def test_all_segments_fail_returns_none(self):
        """If all generations fail, the item result is None."""

        class AlwaysFailTTS(FakeTTS):
            def __init__(self):
                super().__init__()
                self.max_iterations = 2

            def _generate_audio(self, text, **kwargs):
                raise Exception("always fails")

        tts = AlwaysFailTTS()
        token = CancellationToken()
        results = tts._run_pipeline(["Hello"], token)
        assert results[0] is None


class TestRunPipelineValidation:
    """Test the validation loop (max_iterations > 1)."""

    def test_skips_validation_when_max_iterations_one(self):
        tts = FakeTTS()
        tts.max_iterations = 1
        token = CancellationToken()

        # Should not call _validate_accent_drift or _validate_text_match
        with patch.object(tts, '_validate_accent_drift') as mock_drift, \
             patch.object(tts, '_validate_text_match') as mock_text:
            results = tts._run_pipeline(["Hello"], token)
            mock_drift.assert_not_called()
            mock_text.assert_not_called()
        assert results[0] is not None

    def test_validation_pass_on_first_try(self):
        tts = FakeTTS()
        tts.max_iterations = 3
        token = CancellationToken()

        with patch.object(tts, '_validate_accent_drift', return_value=(0.05, True)), \
             patch.object(tts, '_validate_text_match', return_value=(True, 0.95, "Hello")):
            results = tts._run_pipeline(["Hello"], token)

        assert results[0] is not None

    def test_validation_retry_on_drift_failure(self):
        tts = FakeTTS()
        tts.max_iterations = 3
        drift_call_count = 0
        token = CancellationToken()

        def mock_drift(path):
            nonlocal drift_call_count
            drift_call_count += 1
            if drift_call_count == 1:
                return (0.5, False)  # fail first time
            return (0.05, True)  # pass second time

        with patch.object(tts, '_validate_accent_drift', side_effect=mock_drift), \
             patch.object(tts, '_validate_text_match', return_value=(True, 0.95, "Hello")):
            results = tts._run_pipeline(["Hello"], token)

        assert results[0] is not None
        assert drift_call_count >= 2

    def test_validation_retry_on_text_failure(self):
        tts = FakeTTS()
        tts.max_iterations = 3
        text_call_count = 0
        token = CancellationToken()

        def mock_text(path, text):
            nonlocal text_call_count
            text_call_count += 1
            if text_call_count == 1:
                return (False, 0.3, "wrong text")
            return (True, 0.95, "Hello")

        with patch.object(tts, '_validate_accent_drift', return_value=(0.05, True)), \
             patch.object(tts, '_validate_text_match', side_effect=mock_text):
            results = tts._run_pipeline(["Hello"], token)

        assert results[0] is not None
        assert text_call_count >= 2

    def test_max_iterations_exhausted_uses_best(self):
        """When all iterations fail validation, should use best audio."""
        tts = FakeTTS()
        tts.max_iterations = 2
        token = CancellationToken()

        with patch.object(tts, '_validate_accent_drift', return_value=(0.3, False)), \
             patch.object(tts, '_validate_text_match', return_value=(True, 0.95, "Hello")):
            results = tts._run_pipeline(["Hello"], token)

        # Should still return the "best" result even though validation never passed
        assert results[0] is not None

    def test_metadata_contains_drift_and_similarity(self):
        tts = FakeTTS()
        tts.max_iterations = 2
        token = CancellationToken()

        with patch.object(tts, '_validate_accent_drift', return_value=(0.1, True)), \
             patch.object(tts, '_validate_text_match', return_value=(True, 0.92, "Hello")):
            results = tts._run_pipeline(["Hello"], token)

        assert results[0] is not None
        _, _, metadata = results[0]
        assert "drift_prob" in metadata
        assert "text_similarity" in metadata
        assert metadata["drift_prob"] == pytest.approx(0.1, abs=0.01)
        assert metadata["text_similarity"] == pytest.approx(0.92, abs=0.01)

    def test_validation_error_caught_and_continues(self):
        """If validation itself throws, it should be caught and continue."""
        tts = FakeTTS()
        tts.max_iterations = 2
        token = CancellationToken()

        with patch.object(tts, '_validate_accent_drift', side_effect=Exception("model broken")):
            results = tts._run_pipeline(["Hello"], token)

        # Should still produce a result (falls through to best/last audio)
        assert results[0] is not None


class TestRunPipelinePostProcess:
    def test_post_process_called(self):
        class PostProcessTTS(FakeTTS):
            def _post_process_audio(self, audio):
                return audio * 2.0

        tts = PostProcessTTS()
        token = CancellationToken()
        results = tts._run_pipeline(["Hello"], token)
        assert results[0] is not None


class TestRunPipelineProgressCallback:
    def test_progress_callback_called(self):
        tts = FakeTTS()
        token = CancellationToken()
        messages = []

        tts._run_pipeline(["Hello"], token, progress_callback=messages.append)

        assert len(messages) >= 1
        assert any("segment" in m.lower() for m in messages)

    def test_progress_callback_per_segment(self):
        tts = FakeTTS()
        tts.max_chars_per_segment = 30
        token = CancellationToken()
        messages = []

        tts._run_pipeline(["First sentence. Second sentence."], token,
                          progress_callback=messages.append)

        assert len(messages) >= 2
