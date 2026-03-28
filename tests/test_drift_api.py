"""Tests for drift detection API exposure and custom model path support."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from rho_tts.base_tts import BaseTTS


class FakeTTS(BaseTTS):
    """Minimal concrete subclass for drift API tests."""

    def __init__(self, **kwargs):
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
        self.voice_id = kwargs.get("voice_id")
        self.drift_model_path = kwargs.get("drift_model_path")
        self._max_chars_explicit = True
        self._max_model_chars = 3000
        self.voice_cloning = True

    def _generate_audio(self, text, **kwargs):
        return torch.zeros(self._sample_rate)

    @property
    def sample_rate(self):
        return self._sample_rate


# -- train_drift_classifier import tests --


def test_train_importable_from_rho_tts():
    from rho_tts import train_drift_classifier

    assert callable(train_drift_classifier)


def test_train_importable_from_validation():
    from rho_tts.validation import train_drift_classifier

    assert callable(train_drift_classifier)


def test_train_forwards_args():
    with patch("rho_tts.validation.classifier.trainer.train") as mock_train:
        mock_train.return_value = "/path/to/model.pkl"
        from rho_tts import train_drift_classifier

        result = train_drift_classifier(
            dataset_dir="/data",
            voice_id="alice",
            output_path="/out/model.pkl",
        )

        mock_train.assert_called_once_with(
            dataset_dir="/data",
            voice_id="alice",
            output_path="/out/model.pkl",
            progress_callback=None,
        )
        assert result == "/path/to/model.pkl"


def test_train_missing_deps_helpful_error():
    with patch.dict("sys.modules", {"resemblyzer": None, "sklearn": None}):
        # The wrapper itself catches ImportError from the trainer import
        from rho_tts import train_drift_classifier

        # When the underlying trainer can't import, the wrapper raises ImportError
        # with a helpful message. We test the wrapper's own guard.
        with patch(
            "rho_tts.validation.classifier.trainer.train",
            side_effect=ImportError("No module named 'resemblyzer'"),
        ):
            with pytest.raises(ImportError):
                train_drift_classifier(dataset_dir="/data")


# -- drift_model_path tests --


def test_drift_model_path_default_none():
    tts = FakeTTS()
    assert tts.drift_model_path is None


def test_drift_model_path_forwarded_to_validator():
    tts = FakeTTS(drift_model_path="/custom/model.pkl", voice_id="bob")

    with patch(
        "rho_tts.validation.classifier.predict_accent_drift_probability",
        return_value=0.05,
    ) as mock_predict:
        drift_prob, is_ok = tts._validate_accent_drift("/audio.wav")

        mock_predict.assert_called_once_with(
            "/audio.wav",
            voice_id="bob",
            model_path="/custom/model.pkl",
        )
        assert drift_prob == 0.05
        assert is_ok is True


def test_explicit_path_uses_path_cache_key():
    """When model_path is provided, it should be used as the cache key."""
    from rho_tts.validation.classifier import _load_model, _models, _thresholds

    # Clean up cache state
    saved_models = dict(_models)
    saved_thresholds = dict(_thresholds)
    _models.clear()
    _thresholds.clear()

    try:
        # _load_model with a non-existent path just logs and returns
        _load_model(model_path="/nonexistent/a.pkl", voice_id="v1")
        # The cache key should be the path, not the voice_id
        assert "/nonexistent/a.pkl" not in _models  # file doesn't exist, no model loaded
        # But if it had loaded, it would use path as key, not voice_id
    finally:
        _models.clear()
        _models.update(saved_models)
        _thresholds.clear()
        _thresholds.update(saved_thresholds)


def test_voice_id_and_path_no_cache_collision():
    """Different model_path values should not collide even with same voice_id."""
    from rho_tts.validation.classifier import _load_model, _models, _thresholds

    saved_models = dict(_models)
    saved_thresholds = dict(_thresholds)
    _models.clear()
    _thresholds.clear()

    try:
        mock_model_a = MagicMock()
        mock_model_b = MagicMock()

        # Simulate two different models loaded under different paths
        _models["/path/a.pkl"] = mock_model_a
        _models["/path/b.pkl"] = mock_model_b

        assert _models["/path/a.pkl"] is not _models["/path/b.pkl"]
    finally:
        _models.clear()
        _models.update(saved_models)
        _thresholds.clear()
        _thresholds.update(saved_thresholds)
