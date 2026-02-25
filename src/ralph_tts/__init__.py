"""
ralph-tts: Multi-provider text-to-speech with voice cloning and quality validation.

Quick start:
    from ralph_tts import TTSFactory

    tts = TTSFactory.get_tts_instance(
        provider="qwen",
        reference_audio="voice_sample.wav",
        reference_text="Transcript of voice sample.",
    )
    tts.generate_single("Hello world!", "output.wav")
"""

__version__ = "0.1.0"

from .base_tts import BaseTTS
from .cancellation import CancellationToken, CancelledException
from .factory import TTSFactory
from .generator import GenerateAudio

__all__ = [
    "BaseTTS",
    "CancellationToken",
    "CancelledException",
    "GenerateAudio",
    "TTSFactory",
    "__version__",
]
