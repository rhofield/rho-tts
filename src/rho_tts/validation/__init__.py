"""Validation modules for TTS output quality checking."""


def __getattr__(name):
    """Lazy imports to avoid crashing when optional deps are missing."""
    if name == "predict_accent_drift_probability":
        from .classifier import predict_accent_drift_probability
        return predict_accent_drift_probability
    if name == "transcribe_audio":
        from .stt import transcribe_audio
        return transcribe_audio
    if name == "validate_audio_text_match":
        from .stt import validate_audio_text_match
        return validate_audio_text_match
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "predict_accent_drift_probability",
    "transcribe_audio",
    "validate_audio_text_match",
]
