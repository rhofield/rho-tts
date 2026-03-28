"""Tests for the unified generate() method — both file and in-memory modes."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import torch

from rho_tts.base_tts import BaseTTS
from rho_tts.result import GenerationResult
from rho_tts.exceptions import FormatConversionError


class FakeTTS(BaseTTS):
    """Minimal concrete subclass for testing generate()."""

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
        self.sound_decay_threshold = 0.3
        self.max_decay_retries = 3
        self.voice_id = None
        self.drift_model_path = None
        self._max_chars_explicit = True
        self._max_model_chars = 3000

    def _generate_audio(self, text, **kwargs):
        # Return 1 second of silence
        return torch.zeros(self._sample_rate)

    @property
    def sample_rate(self):
        return self._sample_rate


class TestGenerateWithOutputPath:
    def test_single_text_returns_generation_result(self):
        tts = FakeTTS()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            result = tts.generate("Hello world", path)
            assert isinstance(result, GenerationResult)
            assert result.path == path
            assert result.audio is not None
            assert result.sample_rate == 16000
            assert result.duration_sec > 0
            assert result.segments_count >= 1
            assert result.format == "wav"
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_list_texts_returns_list(self):
        tts = FakeTTS()
        with tempfile.TemporaryDirectory() as td:
            base = os.path.join(td, "out")
            result = tts.generate(["Hello", "World"], base)
            assert isinstance(result, list)
            assert len(result) == 2
            for r in result:
                assert isinstance(r, GenerationResult)
                assert r.audio is not None


class TestGenerateInMemory:
    def test_no_output_path_returns_audio_only(self):
        tts = FakeTTS()
        result = tts.generate("Hello world")
        assert isinstance(result, GenerationResult)
        assert result.path is None
        assert result.audio is not None
        assert result.sample_rate == 16000
        assert result.duration_sec > 0

    def test_list_in_memory(self):
        tts = FakeTTS()
        result = tts.generate(["Hello", "World"])
        assert isinstance(result, list)
        assert len(result) == 2
        for r in result:
            assert r.path is None
            assert r.audio is not None


class TestGenerateMetadata:
    def test_duration_is_positive(self):
        tts = FakeTTS(sr=16000)
        result = tts.generate("Hello")
        assert result is not None
        # Audio may be trimmed (silence trimming) but should have positive duration
        assert result.duration_sec > 0
        assert result.sample_rate == 16000

    def test_segments_count(self):
        tts = FakeTTS()
        result = tts.generate("Hello")
        assert result is not None
        assert result.segments_count >= 1


class TestFormatValidation:
    def test_unsupported_format_raises(self):
        tts = FakeTTS()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            from rho_tts.exceptions import FormatConversionError
            with __import__('pytest').raises(FormatConversionError, match="Unsupported format"):
                tts.generate("Hello", path, format="aac")
        finally:
            if os.path.exists(path):
                os.remove(path)
