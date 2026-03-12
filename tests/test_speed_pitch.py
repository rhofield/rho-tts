"""Tests for speed and pitch control."""

import torch

from rho_tts.base_tts import BaseTTS
from rho_tts.result import GenerationResult


class FakeTTS(BaseTTS):
    """Minimal concrete subclass for testing speed/pitch."""

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

    def _generate_audio(self, text, **kwargs):
        # Return 1 second of a sine wave (more interesting than silence for pitch tests)
        t = torch.linspace(0, 1, self._sample_rate)
        return torch.sin(2 * 3.14159 * 440 * t)

    @property
    def sample_rate(self):
        return self._sample_rate


class TestSpeedControl:
    def test_speed_2x_halves_duration(self):
        tts = FakeTTS(sr=16000)
        normal = tts.generate("Hello")
        fast = tts.generate("Hello", speed=2.0)

        assert normal is not None
        assert fast is not None
        # 2x speed should produce roughly half the duration
        ratio = fast.duration_sec / normal.duration_sec
        assert 0.3 < ratio < 0.7, f"Expected ~0.5 ratio, got {ratio}"

    def test_speed_05x_doubles_duration(self):
        tts = FakeTTS(sr=16000)
        normal = tts.generate("Hello")
        slow = tts.generate("Hello", speed=0.5)

        assert normal is not None
        assert slow is not None
        ratio = slow.duration_sec / normal.duration_sec
        assert 1.5 < ratio < 2.5, f"Expected ~2.0 ratio, got {ratio}"

    def test_speed_1x_unchanged(self):
        tts = FakeTTS(sr=16000)
        result = tts.generate("Hello", speed=1.0)
        assert result is not None
        assert abs(result.duration_sec - 1.0) < 0.2


class TestPitchControl:
    def test_pitch_shift_preserves_duration(self):
        tts = FakeTTS(sr=16000)
        normal = tts.generate("Hello")
        shifted = tts.generate("Hello", pitch_semitones=4.0)

        assert normal is not None
        assert shifted is not None
        # Pitch shift should preserve duration
        ratio = shifted.duration_sec / normal.duration_sec
        assert 0.8 < ratio < 1.2, f"Expected ~1.0 ratio, got {ratio}"

    def test_zero_pitch_unchanged(self):
        tts = FakeTTS(sr=16000)
        result = tts.generate("Hello", pitch_semitones=0.0)
        assert result is not None


class TestSpeedPitchApplyMethod:
    def test_apply_speed_pitch_noop(self):
        tts = FakeTTS()
        audio = torch.randn(16000)
        result = tts._apply_speed_pitch(audio, speed=1.0, pitch_semitones=0.0)
        assert torch.allclose(audio, result)
