"""Tests for auto-sort audio into training folders based on drift score."""

import os
import tempfile
from unittest.mock import patch

import torch
import pytest

from rho_tts.base_tts import BaseTTS
from rho_tts.cancellation import CancellationToken


class FakeTTS(BaseTTS):
    """Minimal concrete subclass for testing auto-sort."""

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
        # Auto-sort attributes
        self.auto_sort_good_threshold = None
        self.auto_sort_bad_threshold = None
        self.auto_sort_good_dir = None
        self.auto_sort_bad_dir = None

    def _generate_audio(self, text, **kwargs):
        t = torch.linspace(0, 1, self._sample_rate)
        return torch.sin(2 * 3.14159 * 440 * t) * 0.5

    @property
    def sample_rate(self):
        return self._sample_rate


class TestAutoSortAudio:
    """Test the _auto_sort_audio method directly."""

    def test_disabled_when_dirs_none(self, tmp_path):
        """No copy when both dirs are None."""
        tts = FakeTTS()
        src = tmp_path / "test.wav"
        src.write_bytes(b"fake audio")
        tts._auto_sort_audio(str(src), 0.05)
        # No good/bad dirs created
        assert not (tmp_path / "good").exists()
        assert not (tmp_path / "bad").exists()

    def test_routes_to_good(self, tmp_path):
        """Drift below good_threshold copies to good dir."""
        tts = FakeTTS()
        good_dir = str(tmp_path / "good")
        tts.auto_sort_good_dir = good_dir
        tts.auto_sort_good_threshold = 0.10
        tts.auto_sort_bad_dir = str(tmp_path / "bad")
        tts.auto_sort_bad_threshold = 0.30

        src = tmp_path / "test.wav"
        src.write_bytes(b"fake audio data")
        tts._auto_sort_audio(str(src), 0.05)

        assert os.path.isfile(os.path.join(good_dir, "test.wav"))
        assert not (tmp_path / "bad").exists()

    def test_routes_to_bad(self, tmp_path):
        """Drift above bad_threshold copies to bad dir."""
        tts = FakeTTS()
        bad_dir = str(tmp_path / "bad")
        tts.auto_sort_good_dir = str(tmp_path / "good")
        tts.auto_sort_good_threshold = 0.10
        tts.auto_sort_bad_dir = bad_dir
        tts.auto_sort_bad_threshold = 0.30

        src = tmp_path / "test.wav"
        src.write_bytes(b"fake audio data")
        tts._auto_sort_audio(str(src), 0.50)

        assert os.path.isfile(os.path.join(bad_dir, "test.wav"))
        assert not (tmp_path / "good").exists()

    def test_middle_zone_skipped(self, tmp_path):
        """Drift between thresholds copies to neither dir."""
        tts = FakeTTS()
        tts.auto_sort_good_dir = str(tmp_path / "good")
        tts.auto_sort_good_threshold = 0.10
        tts.auto_sort_bad_dir = str(tmp_path / "bad")
        tts.auto_sort_bad_threshold = 0.30

        src = tmp_path / "test.wav"
        src.write_bytes(b"fake audio data")
        tts._auto_sort_audio(str(src), 0.20)

        assert not (tmp_path / "good").exists()
        assert not (tmp_path / "bad").exists()

    def test_good_dir_only(self, tmp_path):
        """Only good dir configured; bad dir is None."""
        tts = FakeTTS()
        good_dir = str(tmp_path / "good")
        tts.auto_sort_good_dir = good_dir
        tts.auto_sort_good_threshold = 0.10

        src = tmp_path / "test.wav"
        src.write_bytes(b"fake audio data")
        tts._auto_sort_audio(str(src), 0.05)

        assert os.path.isfile(os.path.join(good_dir, "test.wav"))

    def test_bad_dir_only(self, tmp_path):
        """Only bad dir configured; good dir is None."""
        tts = FakeTTS()
        bad_dir = str(tmp_path / "bad")
        tts.auto_sort_bad_dir = bad_dir
        tts.auto_sort_bad_threshold = 0.30

        src = tmp_path / "test.wav"
        src.write_bytes(b"fake audio data")
        tts._auto_sort_audio(str(src), 0.50)

        assert os.path.isfile(os.path.join(bad_dir, "test.wav"))

    def test_creates_directories(self, tmp_path):
        """Dirs are created on demand with makedirs."""
        tts = FakeTTS()
        nested = str(tmp_path / "deep" / "nested" / "good")
        tts.auto_sort_good_dir = nested
        tts.auto_sort_good_threshold = 0.10

        src = tmp_path / "test.wav"
        src.write_bytes(b"fake audio data")
        tts._auto_sort_audio(str(src), 0.05)

        assert os.path.isfile(os.path.join(nested, "test.wav"))

    def test_good_threshold_none_skips_good(self, tmp_path):
        """Good dir set but threshold is None -> skip good routing."""
        tts = FakeTTS()
        tts.auto_sort_good_dir = str(tmp_path / "good")
        tts.auto_sort_good_threshold = None
        tts.auto_sort_bad_dir = str(tmp_path / "bad")
        tts.auto_sort_bad_threshold = 0.30

        src = tmp_path / "test.wav"
        src.write_bytes(b"fake audio data")
        tts._auto_sort_audio(str(src), 0.01)

        # Good threshold is None, so even low drift doesn't go to good
        assert not (tmp_path / "good").exists()
        # Drift 0.01 < 0.30 bad_threshold, so not bad either
        assert not (tmp_path / "bad").exists()


class TestAutoSortInPipeline:
    """Test that auto-sort integrates into _run_pipeline."""

    def test_auto_sort_runs_with_max_iterations_1(self, tmp_path):
        """With max_iterations=1, auto-sort still runs drift + copies."""
        tts = FakeTTS()
        tts.max_iterations = 1
        tts.auto_sort_good_dir = str(tmp_path / "good")
        tts.auto_sort_good_threshold = 0.50
        tts.auto_sort_bad_dir = str(tmp_path / "bad")
        tts.auto_sort_bad_threshold = 0.90

        with patch.object(tts, '_validate_accent_drift', return_value=(0.1, True)):
            token = CancellationToken()
            results = tts._run_pipeline(["Hello"], token)

        assert results[0] is not None
        good_files = os.listdir(str(tmp_path / "good"))
        assert len(good_files) == 1
        assert good_files[0].endswith(".wav")

    def test_no_auto_sort_when_dirs_none_max_iterations_1(self, tmp_path):
        """With max_iterations=1 and no dirs, drift detection is skipped."""
        tts = FakeTTS()
        tts.max_iterations = 1

        with patch.object(tts, '_validate_accent_drift') as mock_drift:
            token = CancellationToken()
            results = tts._run_pipeline(["Hello"], token)

        assert results[0] is not None
        mock_drift.assert_not_called()

    def test_auto_sort_runs_in_validation_loop(self, tmp_path):
        """With max_iterations>1, auto-sort is called during validation."""
        tts = FakeTTS()
        tts.max_iterations = 2
        tts.voice_cloning = True
        tts.auto_sort_good_dir = str(tmp_path / "good")
        tts.auto_sort_good_threshold = 0.50
        tts.auto_sort_bad_dir = str(tmp_path / "bad")
        tts.auto_sort_bad_threshold = 0.90

        with patch.object(tts, '_validate_accent_drift', return_value=(0.05, True)), \
             patch.object(tts, '_validate_text_match', return_value=(True, 0.95, "Hello")):
            token = CancellationToken()
            results = tts._run_pipeline(["Hello"], token)

        assert results[0] is not None
        good_files = os.listdir(str(tmp_path / "good"))
        assert len(good_files) == 1

    def test_auto_sort_bad_in_validation_loop(self, tmp_path):
        """Bad drift score routes to bad dir during validation loop."""
        tts = FakeTTS()
        tts.max_iterations = 2
        tts.voice_cloning = True
        tts.auto_sort_good_dir = str(tmp_path / "good")
        tts.auto_sort_good_threshold = 0.10
        tts.auto_sort_bad_dir = str(tmp_path / "bad")
        tts.auto_sort_bad_threshold = 0.30

        # First iteration: bad drift (0.5), second: good drift (0.05)
        drift_returns = iter([(0.5, False), (0.05, True)])
        with patch.object(tts, '_validate_accent_drift', side_effect=lambda p: next(drift_returns)), \
             patch.object(tts, '_validate_text_match', return_value=(True, 0.95, "Hello")):
            token = CancellationToken()
            results = tts._run_pipeline(["Hello"], token)

        assert results[0] is not None
        # First attempt should go to bad, second to good
        bad_files = os.listdir(str(tmp_path / "bad"))
        good_files = os.listdir(str(tmp_path / "good"))
        assert len(bad_files) == 1
        assert len(good_files) == 1
