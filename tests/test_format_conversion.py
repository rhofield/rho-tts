"""Tests for audio format conversion."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
import torch

from rho_tts.base_tts import BaseTTS
from rho_tts.exceptions import FormatConversionError


class FakeTTS(BaseTTS):
    """Minimal concrete subclass for format tests."""

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
        self.voice_id = None
        self.drift_model_path = None
        self._max_chars_explicit = True
        self._max_model_chars = 3000

    def _generate_audio(self, text, **kwargs):
        return torch.zeros(16000)

    @property
    def sample_rate(self):
        return self._sample_rate


class TestConvertFormat:
    def test_convert_wav_to_mp3_mock(self):
        """Test format conversion with mocked pydub."""
        mock_segment = MagicMock()
        mock_pydub = MagicMock()
        mock_pydub.AudioSegment.from_wav.return_value = mock_segment

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
            f.write(b"fake wav data")

        try:
            with patch.dict("sys.modules", {"pydub": mock_pydub}):
                result = BaseTTS._convert_format(wav_path, "mp3")
                expected_mp3 = wav_path.rsplit(".", 1)[0] + ".mp3"
                assert result == expected_mp3
                mock_pydub.AudioSegment.from_wav.assert_called_once_with(wav_path)
                mock_segment.export.assert_called_once_with(expected_mp3, format="mp3")
        finally:
            for ext in [".wav", ".mp3"]:
                p = wav_path.rsplit(".", 1)[0] + ext
                if os.path.exists(p):
                    os.remove(p)

    def test_unsupported_format_raises_at_generate(self):
        """FormatConversionError raised for invalid formats in generate()."""
        tts = FakeTTS()
        with pytest.raises(FormatConversionError, match="Unsupported format"):
            tts.generate("hello", format="aac")

    def test_conversion_error_wraps_exception(self):
        """FormatConversionError wraps pydub failures."""
        mock_pydub = MagicMock()
        mock_pydub.AudioSegment.from_wav.side_effect = Exception("codec error")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
            f.write(b"fake wav data")

        try:
            with patch.dict("sys.modules", {"pydub": mock_pydub}):
                with pytest.raises(FormatConversionError, match="Failed to convert"):
                    BaseTTS._convert_format(wav_path, "mp3")
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
