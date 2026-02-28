"""
Chatterbox TTS provider implementation.

Used for single segment regeneration with comprehensive validation
including accent drift checking and STT validation.
Supports both default voice and voice cloning via reference audio.
"""
import copy
import logging
from typing import Dict, List, Optional, Union

import torch

from ..base_tts import BaseTTS

logger = logging.getLogger(__name__)


class ChatterboxTTS(BaseTTS):
    """
    Chatterbox TTS implementation with comprehensive validation.

    Args:
        device: Device to run the model on ('cuda' or 'cpu')
        seed: Random seed for consistent voice generation
        deterministic: If True, use deterministic CUDA operations
        reference_audio: Path to audio file for voice cloning (optional).
            If not provided, uses the model's default voice.
        implementation: "standard" or "faster" (rsxdalv optimizations)
        max_chars_per_segment: Max characters per text segment (default: 800)
        max_iterations: Maximum validation retry iterations (default: 50)
        accent_drift_threshold: Threshold for accent drift (default: 0.17)
        text_similarity_threshold: Min similarity for STT validation (default: 0.75)
        phonetic_mapping: Custom word-to-pronunciation mapping
        temperature: Sampling temperature for generation (default: 1.0)
        cfg_weight: Classifier-free guidance weight (default: 0.6)
    """

    def __init__(
        self,
        device: str = "cuda",
        seed: int = 789,
        deterministic: bool = False,
        reference_audio: Optional[str] = None,
        implementation: str = "standard",
        max_chars_per_segment: int = 800,
        max_iterations: int = 50,
        accent_drift_threshold: float = 0.17,
        text_similarity_threshold: float = 0.75,
        phonetic_mapping: Optional[Dict[str, str]] = None,
        temperature: float = 1.0,
        cfg_weight: float = 0.6,
    ):
        super().__init__(device, seed, deterministic, phonetic_mapping=phonetic_mapping)

        if implementation not in ("standard", "faster"):
            raise ValueError(f"Invalid implementation '{implementation}'. Must be 'standard' or 'faster'")

        self.reference_audio_path = reference_audio
        self.voice_cloning = reference_audio is not None
        self.implementation = implementation

        # Configurable thresholds
        self.max_chars_per_segment = max_chars_per_segment
        self.max_iterations = max_iterations
        self.accent_drift_threshold = accent_drift_threshold
        self.text_similarity_threshold = text_similarity_threshold

        # Initialize the base ChatterboxTTS model
        try:
            from chatterbox import ChatterboxTTS as ChatterboxModel
        except ImportError:
            raise ImportError(
                "chatterbox-tts is required for ChatterboxTTS. "
                "Install with: pip install rho-tts[chatterbox]"
            )

        # Detect broken perth watermarker (setuptools>=82 removes pkg_resources)
        import perth
        if perth.PerthImplicitWatermarker is None:
            raise ImportError(
                "The 'perth' audio watermarker failed to initialize "
                "(missing 'pkg_resources' — likely setuptools>=82). "
                "Fix: pip install 'setuptools<82'"
            )

        self.model = ChatterboxModel.from_pretrained(device=device)
        self._prompt_cache: Dict = {}
        self.temperature = temperature
        self.cfg_weight = cfg_weight

        if implementation == "faster":
            logger.info("Using 'faster' implementation with rsxdalv optimizations")

    def _generate_audio(
        self,
        text: Union[str, List[str]],
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Generate audio, with voice prompt caching when cloning."""
        if isinstance(text, list):
            return [self._generate_audio(t, **kwargs) for t in text]

        audio_prompt_path = self.reference_audio_path if self.voice_cloning else None

        if audio_prompt_path:
            if audio_prompt_path not in self._prompt_cache:
                conds = self.model.prepare_conditionals(audio_prompt_path)
                self._prompt_cache[audio_prompt_path] = conds

            self.model.conditionals = copy.deepcopy(self._prompt_cache[audio_prompt_path])
            audio_prompt_path = None

        gen_kwargs = dict(temperature=self.temperature, cfg_weight=self.cfg_weight)

        if self.implementation == "faster":
            gen_kwargs.setdefault('max_cache_len', 1500)
            gen_kwargs.setdefault('max_new_tokens', 1000)

        gen_kwargs.update(kwargs)

        return self.model.generate(text, audio_prompt_path=audio_prompt_path, **gen_kwargs)

    @property
    def sample_rate(self) -> int:
        """Get the sample rate for Chatterbox TTS."""
        return self.model.sr
