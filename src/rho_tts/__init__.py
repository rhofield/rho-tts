"""
rho-tts: Multi-provider text-to-speech with optional voice cloning and quality validation.

Quick start (default voice):
    from rho_tts import TTSFactory

    tts = TTSFactory.get_tts_instance(provider="qwen")
    result = tts.generate("Hello world!", "output.wav")

In-memory generation:
    result = tts.generate("Hello world!")
    result.audio   # torch.Tensor
    result.sample_rate  # int

Voice cloning:
    tts = TTSFactory.get_tts_instance(
        provider="qwen",
        reference_audio="voice_sample.wav",
        reference_text="Transcript of voice sample.",
    )
    result = tts.generate("Hello world!", "output.wav")

Context manager:
    with TTSFactory.get_tts_instance(provider="qwen") as tts:
        result = tts.generate("Hello world!", "output.wav")
"""

__version__ = "0.1.0"

from .base_tts import BaseTTS
from .cancellation import CancellationToken, CancelledException
from .exceptions import (
    AudioGenerationError,
    FormatConversionError,
    ModelLoadError,
    ProviderNotFoundError,
    RhoTTSError,
)
from .factory import TTSFactory
from .provider_info import ProviderInfo, VoiceInfo
from .result import GenerationResult

__all__ = [
    "BaseTTS",
    "CancellationToken",
    "CancelledException",
    "TTSFactory",
    "GenerationResult",
    "ProviderInfo",
    "VoiceInfo",
    "RhoTTSError",
    "ProviderNotFoundError",
    "ModelLoadError",
    "AudioGenerationError",
    "FormatConversionError",
    "__version__",
    "launch_ui",
]


def launch_ui(**kwargs):
    """Launch the Gradio web UI. Requires ``pip install rho-tts[ui]``."""
    from .ui import launch_ui as _launch_ui

    _launch_ui(**kwargs)
