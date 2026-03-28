"""Tests for async_generate()."""

import asyncio

import torch

from rho_tts.base_tts import BaseTTS
from rho_tts.result import GenerationResult


class FakeTTS(BaseTTS):
    """Minimal concrete subclass for testing async."""

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


class TestAsyncGenerate:
    def test_async_generate_returns_result(self):
        tts = FakeTTS()
        result = asyncio.run(tts.async_generate("Hello world"))
        assert isinstance(result, GenerationResult)
        assert result.audio is not None
        assert result.sample_rate == 16000

    def test_async_generate_in_memory(self):
        tts = FakeTTS()
        result = asyncio.run(tts.async_generate("Hello"))
        assert result.path is None
        assert result.audio is not None

    def test_async_generate_list(self):
        tts = FakeTTS()
        result = asyncio.run(tts.async_generate(["Hello", "World"]))
        assert isinstance(result, list)
        assert len(result) == 2
