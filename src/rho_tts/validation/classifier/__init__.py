"""
Voice quality classifier for accent drift detection.

Predicts the probability that a generated audio sample has drifted
from the target voice/accent.
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded model state
_model = None
_optimal_threshold = 0.18


def _load_model(model_path: Optional[str] = None):
    """Load the voice quality classifier model."""
    global _model, _optimal_threshold

    if _model is not None:
        return

    try:
        import joblib
    except ImportError:
        raise ImportError(
            "joblib is required for the voice quality classifier. "
            "Install with: pip install rho-tts[validation]"
        )

    if model_path is None:
        model_path = os.environ.get(
            "RHO_TTS_CLASSIFIER_MODEL",
            os.path.join(os.path.dirname(__file__), "voice_quality_model.pkl"),
        )

    if not os.path.exists(model_path):
        logger.warning(
            f"Voice quality model not found at {model_path}. "
            f"Accent drift validation will be unavailable. "
            f"Set RHO_TTS_CLASSIFIER_MODEL env var or train a model with the trainer module."
        )
        return

    model_data = joblib.load(model_path)
    if isinstance(model_data, dict):
        _model = model_data['model']
        _optimal_threshold = model_data.get('optimal_threshold', 0.18)
    else:
        _model = model_data
        _optimal_threshold = 0.18

    logger.info(f"Voice quality classifier loaded from {model_path}")


def get_optimal_threshold() -> float:
    """Get the optimal threshold from model metadata."""
    _load_model()
    return _optimal_threshold


def predict_accent_drift_probability(audio_path: str, model_path: Optional[str] = None) -> Optional[float]:
    """
    Predict the probability that audio has accent drift.

    Args:
        audio_path: Path to the audio file to evaluate
        model_path: Optional path to classifier model file

    Returns:
        Probability of accent drift (0-1), or None if prediction fails
    """
    _load_model(model_path)

    if _model is None:
        logger.debug("No classifier model loaded, skipping accent drift prediction")
        return None

    from .trainer import extract_features

    feat = extract_features(audio_path)
    if feat is None:
        return None

    prob = _model.predict_proba([feat])[0][1]  # Probability of "bad"
    logger.info(f"Accent drift likelihood: {prob:.2f}")
    return prob
