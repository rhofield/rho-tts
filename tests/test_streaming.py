"""Tests for the stream() generator API."""

import torch

from rho_tts.base_tts import BaseTTS
from rho_tts.cancellation import CancellationToken
from rho_tts.result import GenerationResult


class FakeTTS(BaseTTS):
    """Minimal concrete subclass for testing streaming."""

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

    def _generate_audio(self, text, **kwargs):
        return torch.zeros(self._sample_rate)

    @property
    def sample_rate(self):
        return self._sample_rate


class TestStream:
    def test_yields_generation_results(self):
        tts = FakeTTS()
        results = list(tts.stream("Hello world. This is a test."))
        assert len(results) >= 1
        for r in results:
            assert isinstance(r, GenerationResult)
            assert r.audio is not None
            assert r.sample_rate == 16000
            assert r.segments_count == 1

    def test_single_segment(self):
        tts = FakeTTS()
        tts.force_sentence_split = False
        results = list(tts.stream("Hello"))
        assert len(results) == 1

    def test_stream_cancellation(self):
        tts = FakeTTS()
        token = CancellationToken()
        token.cancel()

        results = list(tts.stream("Hello world. This is a test.", cancellation_token=token))
        assert len(results) == 0

    def test_stream_is_generator(self):
        tts = FakeTTS()
        gen = tts.stream("Hello")
        import types
        assert isinstance(gen, types.GeneratorType)
