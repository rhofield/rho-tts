"""Tests for sound decay validation and windowed normalization."""

import numpy as np
import torch
import pytest

from rho_tts.base_tts import BaseTTS
from rho_tts.cancellation import CancellationToken


class FakeTTS(BaseTTS):
    """Minimal concrete subclass for testing sound decay."""

    def __init__(self, sr=24000):
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
        t = torch.linspace(0, 1, self._sample_rate)
        return torch.sin(2 * 3.14159 * 440 * t) * 0.5

    @property
    def sample_rate(self):
        return self._sample_rate


class TestValidateSoundDecay:
    def test_constant_volume_passes(self):
        tts = FakeTTS()
        # Constant amplitude sine wave — no decay
        t = torch.linspace(0, 3, 3 * 24000)
        audio = torch.sin(2 * 3.14159 * 440 * t) * 0.5
        ratio, is_ok = tts._validate_sound_decay(audio)
        assert is_ok
        assert ratio > 0.9

    def test_severe_decay_fails(self):
        tts = FakeTTS()
        sr = 24000
        # Create audio that decays from 1.0 to near-zero — severe decay
        t = torch.linspace(0, 3, 3 * sr)
        envelope = torch.linspace(1.0, 0.01, 3 * sr)
        audio = torch.sin(2 * 3.14159 * 440 * t) * envelope
        ratio, is_ok = tts._validate_sound_decay(audio)
        assert not is_ok
        assert ratio < 0.3

    def test_mild_decay_passes(self):
        tts = FakeTTS()
        sr = 24000
        # Create audio that decays from 1.0 to 0.7 — mild, should pass
        t = torch.linspace(0, 3, 3 * sr)
        envelope = torch.linspace(1.0, 0.7, 3 * sr)
        audio = torch.sin(2 * 3.14159 * 440 * t) * envelope
        ratio, is_ok = tts._validate_sound_decay(audio)
        assert is_ok
        assert ratio > 0.3

    def test_empty_audio_passes(self):
        tts = FakeTTS()
        audio = torch.tensor([])
        ratio, is_ok = tts._validate_sound_decay(audio)
        assert is_ok
        assert ratio == 1.0

    def test_silent_audio_passes(self):
        tts = FakeTTS()
        audio = torch.zeros(24000)
        ratio, is_ok = tts._validate_sound_decay(audio)
        assert is_ok

    def test_custom_threshold(self):
        tts = FakeTTS()
        tts.sound_decay_threshold = 0.8  # very strict
        sr = 24000
        t = torch.linspace(0, 3, 3 * sr)
        envelope = torch.linspace(1.0, 0.6, 3 * sr)
        audio = torch.sin(2 * 3.14159 * 440 * t) * envelope
        ratio, is_ok = tts._validate_sound_decay(audio)
        assert not is_ok  # 0.6 ratio < 0.8 threshold


class TestWindowedNormalization:
    """Test the windowed normalization in QwenTTS._post_process_audio."""

    def test_corrects_decaying_audio(self):
        """Windowed normalization should even out volume decay."""
        # Import QwenTTS with mocked dependencies
        import sys
        from unittest.mock import MagicMock

        mock_qwen_tts = MagicMock()
        with pytest.MonkeyPatch.context() as m:
            m.setitem(sys.modules, "qwen_tts", mock_qwen_tts)

            from rho_tts.providers.qwen import QwenTTS

            tts = QwenTTS.__new__(QwenTTS)
            tts.qwen3_sr = 24000
            tts.device = "cpu"

            sr = 24000
            duration = 10  # seconds
            n_samples = sr * duration
            t = torch.linspace(0, duration, n_samples)

            # Create audio with severe decay: 1.0 -> 0.2 over 10 seconds
            envelope = torch.linspace(1.0, 0.2, n_samples)
            audio = torch.sin(2 * 3.14159 * 440 * t) * envelope

            # Measure decay before
            third = n_samples // 3
            first_rms_before = torch.sqrt(torch.mean(audio[:third] ** 2)).item()
            last_rms_before = torch.sqrt(torch.mean(audio[-third:] ** 2)).item()
            ratio_before = last_rms_before / first_rms_before

            # Apply post-processing
            result = tts._post_process_audio(audio.clone())

            # Measure decay after
            first_rms_after = torch.sqrt(torch.mean(result[:third] ** 2)).item()
            last_rms_after = torch.sqrt(torch.mean(result[-third:] ** 2)).item()
            ratio_after = last_rms_after / first_rms_after

            # The ratio should be much closer to 1.0 after correction
            assert ratio_before < 0.4, f"Precondition: audio should have decay, got ratio {ratio_before}"
            assert ratio_after > 0.6, f"Correction should improve ratio, got {ratio_after} (was {ratio_before})"

    def test_does_not_alter_constant_audio(self):
        """Constant-volume audio should not be changed by windowed normalization."""
        import sys
        from unittest.mock import MagicMock

        mock_qwen_tts = MagicMock()
        with pytest.MonkeyPatch.context() as m:
            m.setitem(sys.modules, "qwen_tts", mock_qwen_tts)

            from rho_tts.providers.qwen import QwenTTS

            tts = QwenTTS.__new__(QwenTTS)
            tts.qwen3_sr = 24000
            tts.device = "cpu"

            sr = 24000
            duration = 6
            n_samples = sr * duration
            t = torch.linspace(0, duration, n_samples)
            audio = torch.sin(2 * 3.14159 * 440 * t) * 0.5

            result = tts._post_process_audio(audio.clone())

            # Volume should be consistent (normalized to target)
            third = n_samples // 3
            first_rms = torch.sqrt(torch.mean(result[:third] ** 2)).item()
            last_rms = torch.sqrt(torch.mean(result[-third:] ** 2)).item()
            ratio = last_rms / first_rms

            assert 0.85 < ratio < 1.15, f"Constant audio should stay balanced, got ratio {ratio}"
