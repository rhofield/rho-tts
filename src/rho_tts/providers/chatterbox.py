"""
Chatterbox TTS provider implementation.

Used for single segment regeneration with comprehensive validation
including accent drift checking and STT validation.
Supports both default voice and voice cloning via reference audio.
"""
import copy
import logging
import time
import traceback
from typing import Dict, List, Optional, Union

import torch
import torchaudio as ta

from ..base_tts import BaseTTS
from ..cancellation import CancellationToken, CancelledException

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

        if implementation == "faster":
            logger.info("Using 'faster' implementation with rsxdalv optimizations")

    def _generate_audio(
        self,
        text: Union[str, List[str]],
        audio_prompt_path: Optional[str] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Generate audio, with voice prompt caching when cloning."""
        if isinstance(text, list):
            return [self._generate_audio(t, audio_prompt_path, **kwargs) for t in text]

        if audio_prompt_path:
            if audio_prompt_path not in self._prompt_cache:
                conds = self.model.prepare_conditionals(audio_prompt_path)
                self._prompt_cache[audio_prompt_path] = conds

            self.model.conditionals = copy.deepcopy(self._prompt_cache[audio_prompt_path])
            audio_prompt_path = None

        if self.implementation == "faster":
            if 'max_cache_len' not in kwargs:
                kwargs['max_cache_len'] = 1500
            if 'max_new_tokens' not in kwargs:
                kwargs['max_new_tokens'] = 1000

        return self.model.generate(text, audio_prompt_path=audio_prompt_path, **kwargs)

    def generate_single(
        self,
        text: str,
        output_path: str,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Optional[torch.Tensor]:
        """
        Generate audio with comprehensive validation loop.

        Implements accent drift checking and text accuracy validation via STT.
        """
        try:
            self._set_seeds()
            token = cancellation_token or CancellationToken()

            def audio_generation_loop(text: str, temp_output_path: Optional[str] = None):
                """Inner loop for generation with validation."""
                best_wav = None
                best_accent_drift = float('inf')

                for n_iterations in range(self.max_iterations):
                    if token.is_cancelled():
                        raise CancelledException("Audio generation was cancelled during retry loop")

                    temp = 1
                    cfg = 0.6

                    if n_iterations > 0:
                        random_seed = int(time.time() * 1000) % 100000
                        self.seed = random_seed
                        self._set_seeds()

                    logger.info(f"Iteration {n_iterations+1}: seed {self.seed} (temp={temp}, cfg={cfg})")

                    wav = self._generate_audio(
                        text,
                        audio_prompt_path=self.reference_audio_path if self.voice_cloning else None,
                        temperature=temp,
                        cfg_weight=cfg,
                    )

                    if temp_output_path:
                        ta.save(temp_output_path, wav, self.model.sr)

                        # Step 1: Check accent drift
                        drift_prob, is_voice_accurate = self._validate_accent_drift(temp_output_path)

                        if drift_prob < best_accent_drift:
                            best_accent_drift = drift_prob
                            best_wav = wav.clone()
                            logger.info(f"   New best: accent drift {best_accent_drift:.3f}")

                        # Step 2: Text accuracy
                        is_text_accurate = True
                        text_similarity = 1.0

                        if is_voice_accurate:
                            is_text_accurate, text_similarity, transcribed_text = self._validate_text_match(
                                temp_output_path, text
                            )
                            thresh = self.text_similarity_threshold
                            logger.info(f"Text similarity: {text_similarity:.3f} (threshold: {thresh})")
                            if transcribed_text:
                                logger.debug(f"   Original: {text}")
                                logger.debug(f"   Transcribed: {transcribed_text}")

                        is_valid = is_voice_accurate and is_text_accurate

                        if is_valid:
                            logger.info(f"Audio valid! Returning after {n_iterations + 1} iteration(s)")
                            return wav

                        failure_reasons = []
                        if not is_voice_accurate:
                            failure_reasons.append(f"accent drift (prob: {drift_prob:.3f})")
                        if not is_text_accurate:
                            failure_reasons.append(f"text mismatch (similarity: {text_similarity:.3f})")

                        logger.warning(
                            f"Audio invalid: {', '.join(failure_reasons)}, "
                            f"retrying... (iteration {n_iterations + 1}/{self.max_iterations})"
                        )
                    else:
                        return wav

                if best_wav is not None:
                    logger.warning(
                        f"Max iterations reached, returning best audio (drift: {best_accent_drift:.3f})"
                    )
                    return best_wav
                else:
                    logger.warning("Max iterations reached, returning last generated audio")
                    return wav

            segments = self._split_text_into_segments(text, self.max_chars_per_segment)
            audio_segments = []

            for i, segment in enumerate(segments):
                logger.info(f"Segment {i+1}/{len(segments)} ({len(segment)} chars)")

                if token.is_cancelled():
                    raise CancelledException("Audio generation was cancelled during segmentation")

                mapped_segment = self._apply_phonetic_mapping(segment)
                wav = audio_generation_loop(mapped_segment, output_path)
                audio_segments.append(wav)

            if not audio_segments:
                logger.error("No audio segments were generated successfully")
                return None

            final_wav = self._smooth_segment_join(audio_segments)

            if final_wav is None:
                return None

            if final_wav.device.type != 'cpu':
                final_wav = final_wav.cpu()

            if final_wav.dim() == 1:
                save_wav = final_wav.unsqueeze(0)
            else:
                save_wav = final_wav

            ta.save(output_path, save_wav, self.model.sr)
            logger.info(f"Audio saved: {output_path}")

            return final_wav

        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            traceback.print_exc()
            return None

    def generate(
        self,
        texts: List[str],
        output_base_path: str,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Optional[List[str]]:
        """Batch generation by calling generate_single() for each text."""
        try:
            token = cancellation_token or CancellationToken()
            output_files = []

            logger.info(f"Generating audio for {len(texts)} text item(s) with Chatterbox TTS...")

            for idx, text in enumerate(texts):
                if token.is_cancelled():
                    raise CancelledException(f"Cancelled during text item {idx}")

                output_path = f"{output_base_path}_{idx}.wav"
                result = self.generate_single(text, output_path, cancellation_token)

                if result is not None:
                    output_files.append(output_path)
                else:
                    logger.error(f"Failed to generate audio for text item {idx}")
                    output_files.append(None)

            successful = sum(1 for f in output_files if f is not None)
            failed = len(output_files) - successful

            if failed > 0:
                logger.warning(f"{failed}/{len(output_files)} text item(s) failed to generate")

            if successful == 0:
                logger.error("All text items failed to generate")
                return None

            logger.info(f"Successfully generated {successful}/{len(output_files)} audio file(s)")
            return output_files

        except CancelledException as e:
            logger.warning(f"Generation cancelled: {e}")
            return None
        except Exception as e:
            logger.error(f"Error in Chatterbox TTS: {e}")
            traceback.print_exc()
            return None

    @property
    def sample_rate(self) -> int:
        """Get the sample rate for Chatterbox TTS."""
        return self.model.sr
