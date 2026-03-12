"""
Custom exception hierarchy for rho-tts.

All rho-tts exceptions inherit from RhoTTSError, making it easy to
catch any library error with a single except clause.
"""


class RhoTTSError(Exception):
    """Base exception for all rho-tts errors."""
    pass


class ProviderNotFoundError(RhoTTSError):
    """Raised when a requested TTS provider is not registered."""
    pass


class ModelLoadError(RhoTTSError):
    """Raised when a TTS model fails to load."""
    pass


class AudioGenerationError(RhoTTSError):
    """Raised when audio generation fails."""
    pass


class FormatConversionError(RhoTTSError):
    """Raised when audio format conversion fails."""
    pass
