"""Tests for audio processing utilities: silence trimming, fades, DC offset removal."""
import torch

from ralph_tts.base_tts import BaseTTS


class ConcreteTTS:
    """Minimal stand-in to test BaseTTS audio methods without full init."""

    def __init__(self):
        self.device = "cpu"
        self.silence_threshold_db = -50.0
        self.crossfade_duration_sec = 0.05
        self.trim_silence = True
        self.fade_duration_sec = 0.02
        self.force_sentence_split = True
        self.inter_sentence_pause_sec = 0.1
        self._sample_rate = 16000

    @property
    def sample_rate(self):
        return self._sample_rate


# Borrow methods from BaseTTS for testing
ConcreteTTS._trim_silence = BaseTTS._trim_silence
ConcreteTTS._remove_dc_offset = BaseTTS._remove_dc_offset
ConcreteTTS._apply_fades = BaseTTS._apply_fades
ConcreteTTS._smooth_segment_join = BaseTTS._smooth_segment_join


class TestDCOffsetRemoval:
    def test_removes_offset(self):
        tts = ConcreteTTS()
        # Audio with DC offset of 0.5
        audio = torch.randn(16000) + 0.5
        result = tts._remove_dc_offset(audio)
        assert abs(result.mean().item()) < 0.01

    def test_empty_audio(self):
        tts = ConcreteTTS()
        audio = torch.tensor([])
        result = tts._remove_dc_offset(audio)
        assert result.numel() == 0

    def test_zero_audio_unchanged(self):
        tts = ConcreteTTS()
        audio = torch.zeros(100)
        result = tts._remove_dc_offset(audio)
        assert torch.allclose(result, audio)


class TestFades:
    def test_fade_in_starts_at_zero(self):
        tts = ConcreteTTS()
        audio = torch.ones(16000)
        result = tts._apply_fades(audio, fade_in=True, fade_out=False)
        # First sample should be near zero due to fade-in
        assert abs(result[0].item()) < 0.01

    def test_fade_out_ends_at_zero(self):
        tts = ConcreteTTS()
        audio = torch.ones(16000)
        result = tts._apply_fades(audio, fade_in=False, fade_out=True)
        # Last sample should be near zero due to fade-out
        assert abs(result[-1].item()) < 0.01

    def test_no_fade_unchanged(self):
        tts = ConcreteTTS()
        audio = torch.ones(16000)
        result = tts._apply_fades(audio, fade_in=False, fade_out=False)
        assert torch.allclose(result, audio)

    def test_short_audio_not_faded(self):
        tts = ConcreteTTS()
        # Audio shorter than 2 * fade_samples should be returned unchanged
        short_audio = torch.ones(10)
        result = tts._apply_fades(short_audio, fade_in=True, fade_out=True)
        assert torch.allclose(result, short_audio)

    def test_empty_audio(self):
        tts = ConcreteTTS()
        audio = torch.tensor([])
        result = tts._apply_fades(audio, fade_in=True, fade_out=True)
        assert result.numel() == 0


class TestSilenceTrimming:
    def test_trims_leading_silence(self):
        tts = ConcreteTTS()
        # 1 second of silence then 0.5 seconds of signal
        silence = torch.zeros(16000)
        signal = torch.randn(8000) * 0.5
        audio = torch.cat([silence, signal])
        result = tts._trim_silence(audio, from_start=True, from_end=False)
        # Result should be shorter than original (silence trimmed)
        assert result.shape[-1] < audio.shape[-1]

    def test_trims_trailing_silence(self):
        tts = ConcreteTTS()
        signal = torch.randn(8000) * 0.5
        silence = torch.zeros(16000)
        audio = torch.cat([signal, silence])
        result = tts._trim_silence(audio, from_start=False, from_end=True)
        assert result.shape[-1] < audio.shape[-1]

    def test_disabled_trimming(self):
        tts = ConcreteTTS()
        tts.trim_silence = False
        audio = torch.zeros(16000)
        result = tts._trim_silence(audio, from_start=True, from_end=True)
        assert result.shape == audio.shape

    def test_all_silent_audio(self):
        tts = ConcreteTTS()
        audio = torch.zeros(16000)
        result = tts._trim_silence(audio, from_start=True, from_end=True)
        # Should return at least a small portion
        assert result.numel() > 0


class TestSegmentJoining:
    def test_single_segment(self):
        tts = ConcreteTTS()
        segment = torch.randn(16000) * 0.3
        result = tts._smooth_segment_join([segment])
        assert result is not None
        assert result.numel() > 0

    def test_two_segments(self):
        tts = ConcreteTTS()
        seg1 = torch.randn(16000) * 0.3
        seg2 = torch.randn(16000) * 0.3
        result = tts._smooth_segment_join([seg1, seg2])
        assert result is not None
        # Result should be roughly the sum of both minus crossfade overlap
        assert result.shape[-1] > 16000

    def test_empty_list_returns_none(self):
        tts = ConcreteTTS()
        result = tts._smooth_segment_join([])
        assert result is None
