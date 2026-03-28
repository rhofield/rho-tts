"""Tests for smart max_chars_per_segment computation."""

from unittest.mock import patch, MagicMock, PropertyMock

import pytest
import torch

from rho_tts.base_tts import BaseTTS
from rho_tts.cancellation import CancellationToken


class FakeTTS(BaseTTS):
    """Minimal concrete subclass for smart segmentation tests."""

    MAX_MODEL_CHARS = 3000
    BYTES_PER_CHAR_ESTIMATE = 500_000

    def __init__(self, sr=16000, **kwargs):
        self.device = kwargs.get("device", "cpu")
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
        self.max_chars_per_segment = kwargs.get("max_chars_per_segment", 800)
        self.max_iterations = 1
        self.accent_drift_threshold = 0.17
        self.text_similarity_threshold = 0.85
        self.sound_decay_threshold = 0.3
        self.max_decay_retries = 3
        self.voice_id = None
        self.drift_model_path = None
        self._max_chars_explicit = kwargs.get("_max_chars_explicit", False)
        self._max_model_chars = kwargs.get("_max_model_chars", self.MAX_MODEL_CHARS)

    def _generate_audio(self, text, **kwargs):
        t = torch.linspace(0, 1, self._sample_rate)
        return torch.sin(2 * 3.14159 * 440 * t) * 0.5

    @property
    def sample_rate(self):
        return self._sample_rate


class TestExplicitOverride:
    def test_explicit_override_returns_unchanged(self):
        """When user explicitly sets max_chars_per_segment, smart sizing is bypassed."""
        tts = FakeTTS(max_chars_per_segment=500, _max_chars_explicit=True)
        assert tts._compute_max_chars() == 500


class TestModelMaxConstraint:
    def test_model_max_constrains_result(self):
        """Result can't exceed MAX_MODEL_CHARS (after 80% cap)."""
        tts = FakeTTS(_max_model_chars=2000)
        # Mock abundant memory so resource isn't the bottleneck
        with patch.object(tts, "_get_available_memory_bytes", return_value=100 * 1024**3):
            result = tts._compute_max_chars()
            # 80% of 2000 = 1600
            assert result == 1600


class TestResourceLimit:
    def test_resource_limit_constrains_result(self):
        """With low VRAM, resource limit should constrain the result."""
        tts = FakeTTS(_max_model_chars=4000)
        # 500MB available / 500KB per char = 1000 chars resource limit
        with patch.object(tts, "_get_available_memory_bytes", return_value=500 * 1024**2):
            result = tts._compute_max_chars()
            resource_max = int((500 * 1024**2) / 500_000)
            expected = int(min(4000, resource_max) * 0.8)
            assert result == expected


class TestEightyPercentCap:
    def test_eighty_percent_cap(self):
        """Result should be 80% of min(model, resource)."""
        tts = FakeTTS(_max_model_chars=1000)
        # 2GB available / 500KB = 4000 resource max; model max = 1000
        with patch.object(tts, "_get_available_memory_bytes", return_value=2 * 1024**3):
            result = tts._compute_max_chars()
            assert result == int(1000 * 0.8)  # 800


class TestFloor:
    def test_floor_at_200(self):
        """Even with very low memory, result should never go below 200."""
        tts = FakeTTS(_max_model_chars=4000)
        # Tiny memory: 10MB / 500KB = 20 chars, * 0.8 = 16 -> floor to 200
        with patch.object(tts, "_get_available_memory_bytes", return_value=10 * 1024**2):
            result = tts._compute_max_chars()
            assert result == 200


class TestMemoryDetection:
    def test_gpu_uses_vram(self):
        """On CUDA device, _get_available_memory_bytes should check GPU memory."""
        tts = FakeTTS(device="cuda")
        mock_mem_info = MagicMock(return_value=(4 * 1024**3, 8 * 1024**3))
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.mem_get_info", mock_mem_info):
            result = tts._get_available_memory_bytes()
            mock_mem_info.assert_called_once()
            assert result == 4 * 1024**3

    def test_cpu_uses_ram(self):
        """On CPU device, _get_available_memory_bytes should fall back to system RAM."""
        tts = FakeTTS(device="cpu")
        mock_vmem = MagicMock()
        mock_vmem.available = 16 * 1024**3
        with patch("psutil.virtual_memory", return_value=mock_vmem):
            result = tts._get_available_memory_bytes()
            assert result == 16 * 1024**3

    def test_cpu_fallback_without_psutil(self):
        """Without psutil, should fall back to os.sysconf."""
        tts = FakeTTS(device="cpu")
        with patch.dict("sys.modules", {"psutil": None}), \
             patch("os.sysconf", side_effect=lambda key: {
                 "SC_PAGE_SIZE": 4096,
                 "SC_AVPHYS_PAGES": 1024 * 1024,
             }[key]):
            result = tts._get_available_memory_bytes()
            assert result == 4096 * 1024 * 1024


class TestPipelineIntegration:
    def test_pipeline_uses_smart_sizing(self):
        """_run_pipeline should use _compute_max_chars for segmentation."""
        tts = FakeTTS(_max_model_chars=1000)

        with patch.object(tts, "_compute_max_chars", return_value=750) as mock_compute, \
             patch.object(tts, "_split_text_into_segments", wraps=tts._split_text_into_segments) as mock_split:
            token = CancellationToken()
            tts._run_pipeline(["Hello world."], token)

            mock_compute.assert_called()
            mock_split.assert_called_with("Hello world.", 750)

    def test_stream_uses_smart_sizing(self):
        """stream() should use _compute_max_chars for segmentation."""
        tts = FakeTTS(_max_model_chars=1000)

        with patch.object(tts, "_compute_max_chars", return_value=750) as mock_compute, \
             patch.object(tts, "_split_text_into_segments", wraps=tts._split_text_into_segments) as mock_split:
            results = list(tts.stream("Hello world."))

            mock_compute.assert_called()
            mock_split.assert_called_with("Hello world.", 750)
