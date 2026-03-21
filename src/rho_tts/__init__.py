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

__version__ = "1.0.7"

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
    "train_drift_classifier",
]


def launch_ui(**kwargs):
    """Launch the Gradio web UI. Requires ``pip install rho-tts[ui]``."""
    from .ui import launch_ui as _launch_ui

    _launch_ui(**kwargs)


def train_drift_classifier(
    dataset_dir: str,
    voice_id: str | None = None,
    output_path: str | None = None,
    progress_callback=None,
):
    """Train a drift-detection classifier. Requires ``pip install rho-tts[validation]``.

    Args:
        dataset_dir: Directory containing 'good/' and 'bad/' subdirectories with .wav files.
        voice_id: Voice ID to associate with this model.
        output_path: Explicit path to save the trained model.
        progress_callback: Optional callable receiving progress messages.
    """
    try:
        from .validation.classifier.trainer import train
    except ImportError:
        raise ImportError(
            "Training a drift classifier requires the validation extras. "
            "Install with: pip install rho-tts[validation]"
        )
    return train(
        dataset_dir=dataset_dir,
        voice_id=voice_id,
        output_path=output_path,
        progress_callback=progress_callback,
    )
