"""
Voice quality classifier for accent drift detection.

Predicts the probability that a generated audio sample has drifted
from the target voice/accent.
"""
import logging
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Per-voice model caches: voice_id -> (model, optimal_threshold)
_models: Dict[str, object] = {}
_thresholds: Dict[str, float] = {}

_DEFAULT_THRESHOLD = 0.18


def get_model_path(voice_id: str) -> str:
    """Return the default model path for a given voice_id."""
    return os.path.join(os.path.expanduser("~"), ".rho_tts", "models", f"{voice_id}_classifier.pkl")


def _load_model(model_path: Optional[str] = None, voice_id: Optional[str] = None):
    """Load the voice quality classifier model into the per-voice cache."""
    cache_key = voice_id or "__global__"

    if cache_key in _models:
        return

    try:
        import joblib
    except ImportError:
        raise ImportError(
            "joblib is required for the voice quality classifier. "
            "Install with: pip install rho-tts[validation]"
        )

    if model_path is None:
        if voice_id is not None:
            model_path = get_model_path(voice_id)
        else:
            model_path = os.environ.get(
                "RHO_TTS_CLASSIFIER_MODEL",
                os.path.join(os.path.dirname(__file__), "voice_quality_model.pkl"),
            )

    if not os.path.exists(model_path):
        if voice_id is not None:
            logger.debug(
                f"No per-voice model found for '{voice_id}' at {model_path}. "
                f"Accent drift validation will be skipped for this voice."
            )
        else:
            logger.warning(
                f"Voice quality model not found at {model_path}. "
                f"Accent drift validation will be unavailable. "
                f"Set RHO_TTS_CLASSIFIER_MODEL env var or train a model with the trainer module."
            )
        return

    model_data = joblib.load(model_path)
    if isinstance(model_data, dict):
        _models[cache_key] = model_data['model']
        _thresholds[cache_key] = model_data.get('optimal_threshold', _DEFAULT_THRESHOLD)
    else:
        _models[cache_key] = model_data
        _thresholds[cache_key] = _DEFAULT_THRESHOLD

    logger.info(f"Voice quality classifier loaded from {model_path}")


def get_optimal_threshold(voice_id: Optional[str] = None) -> float:
    """Get the optimal threshold from model metadata."""
    _load_model(voice_id=voice_id)
    cache_key = voice_id or "__global__"
    return _thresholds.get(cache_key, _DEFAULT_THRESHOLD)


def predict_accent_drift_probability(
    audio_path: str,
    voice_id: Optional[str] = None,
    model_path: Optional[str] = None,
) -> Optional[float]:
    """
    Predict the probability that audio has accent drift.

    Args:
        audio_path: Path to the audio file to evaluate
        voice_id: Voice ID to select a per-voice model. When given, looks up
            ~/.rho_tts/models/{voice_id}_classifier.pkl. When None, falls back
            to the legacy global model (backward compatible).
        model_path: Optional explicit path to classifier model file (overrides voice_id lookup)

    Returns:
        Probability of accent drift (0-1), or None if prediction fails
    """
    _load_model(model_path, voice_id)

    cache_key = voice_id or "__global__"
    model = _models.get(cache_key)

    if model is None:
        logger.debug("No classifier model loaded, skipping accent drift prediction")
        return None

    from .trainer import extract_features

    feat = extract_features(audio_path)
    if feat is None:
        return None

    prob = model.predict_proba([feat])[0][1]  # Probability of "bad"
    logger.info(f"Accent drift likelihood: {prob:.2f}")
    return prob
