"""Tests for context manager and close() on BaseTTS."""

from unittest.mock import patch, MagicMock

import torch

from rho_tts.base_tts import BaseTTS


class FakeTTS(BaseTTS):
    """Minimal concrete subclass for testing."""

    def __init__(self):
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
        self._sample_rate = 16000
        self.max_chars_per_segment = 800
        self.max_iterations = 1
        self.accent_drift_threshold = 0.17
        self.text_similarity_threshold = 0.85
        self.sound_decay_threshold = 0.3
        self.max_decay_retries = 3
        self.voice_id = None
        self.drift_model_path = None
        self._max_chars_explicit = True
        self._max_model_chars = 3000
        self.closed = False

    def _generate_audio(self, text, **kwargs):
        return torch.zeros(self._sample_rate)

    def close(self):
        self.closed = True

    @property
    def sample_rate(self):
        return self._sample_rate


class TestContextManager:
    def test_enter_returns_self(self):
        tts = FakeTTS()
        assert tts.__enter__() is tts

    def test_exit_calls_close(self):
        tts = FakeTTS()
        tts.__exit__(None, None, None)
        assert tts.closed

    def test_with_statement(self):
        with FakeTTS() as tts:
            assert isinstance(tts, FakeTTS)
        assert tts.closed

    def test_close_called_on_exception(self):
        tts = FakeTTS()
        try:
            with tts:
                raise ValueError("test")
        except ValueError:
            pass
        assert tts.closed

    def test_base_close_is_noop(self):
        """BaseTTS.close() should not raise even without override."""
        # Use FakeTTS but call BaseTTS.close directly
        tts = FakeTTS()
        BaseTTS.close(tts)  # should not raise

    def test_exit_returns_false(self):
        """__exit__ should not suppress exceptions."""
        tts = FakeTTS()
        result = tts.__exit__(ValueError, ValueError("test"), None)
        assert result is False
