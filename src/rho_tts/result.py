"""
GenerationResult dataclass returned by TTS generate() calls.

Provides both the file path (when saved to disk) and the raw audio
tensor (always present), along with generation metadata.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class GenerationResult:
    """Result of a single TTS generation.

    Attributes:
        path: File path where audio was saved, or None if in-memory only.
        audio: Raw audio tensor (always present on success).
        sample_rate: Sample rate of the audio in Hz.
        duration_sec: Duration of the audio in seconds.
        segments_count: Number of text segments that were generated and joined.
        format: Audio format (e.g. "wav", "mp3", "flac", "ogg").
    """
    path: Optional[str] = None
    audio: Optional[torch.Tensor] = None
    sample_rate: int = 0
    duration_sec: float = 0.0
    segments_count: int = 0
    format: str = "wav"
    drift_prob: Optional[float] = None
    text_similarity: Optional[float] = None
