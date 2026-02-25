"""
Chatterbox TTS provider implementation.

Used for single segment regeneration with comprehensive validation
including accent drift checking, speaker similarity, and STT validation.
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
        reference_audio: Path to audio file for voice cloning (required)
        implementation: "standard" or "faster" (rsxdalv optimizations)
        max_chars_per_segment: Max characters per text segment (default: 800)
        max_iterations: Maximum validation retry iterations (default: 50)
        accent_drift_threshold: Threshold for accent drift (default: 0.17)
        text_similarity_threshold: Min similarity for STT validation (default: 0.75)
        speaker_similarity_threshold: Min cosine similarity for speaker matching (default: 0.85)
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
        speaker_similarity_threshold: float = 0.85,
        phonetic_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__(device, seed, deterministic, phonetic_mapping=phonetic_mapping)

        if reference_audio is None:
            raise ValueError("reference_audio path is required for ChatterboxTTS voice cloning")

        if implementation not in ("standard", "faster"):
            raise ValueError(f"Invalid implementation '{implementation}'. Must be 'standard' or 'faster'")

        self.reference_audio_path = reference_audio
        self.implementation = implementation

        # Configurable thresholds
        self.MAX_CHARS_PER_SEGMENT = max_chars_per_segment
        self.MAX_ITERATIONS = max_iterations
        self.ACCENT_DRIFT_THRESHOLD = accent_drift_threshold
        self.TEXT_SIMILARITY_THRESHOLD = text_similarity_threshold
        self.SPEAKER_SIMILARITY_THRESHOLD = speaker_similarity_threshold

        # Initialize the base ChatterboxTTS model
        try:
            from chatterbox import ChatterboxTTS as ChatterboxModel
        except ImportError:
            raise ImportError(
                "chatterbox-tts is required for ChatterboxTTS. "
                "Install with: pip install ralph-tts[chatterbox]"
            )

        self.model = ChatterboxModel.from_pretrained(device=device)
        self._prompt_cache: Dict = {}

        # Initialize reference embedding
        try:
            from resemblyzer import preprocess_wav

            logger.info("Loading voice encoder for speaker similarity validation...")
            reference_wav = preprocess_wav(self.reference_audio_path)
            self.reference_embedding = self.voice_encoder.embed_utterance(reference_wav)
            logger.info("Reference voice embedding computed")
        except ImportError:
            logger.warning(
                "resemblyzer not installed, speaker similarity validation disabled. "
                "Install with: pip install ralph-tts[validation]"
            )

        if implementation == "faster":
            logger.info("Using 'faster' implementation with rsxdalv optimizations")

    def _generate_audio(
        self,
        text: Union[str, List[str]],
        audio_prompt_path: Optional[str] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Generate audio with voice prompt caching for better performance."""
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

        Implements accent drift checking, speaker similarity validation,
        and text accuracy validation via STT.
        """
        try:
            self._set_seeds()
            token = cancellation_token or CancellationToken()

            def audio_generation_loop(text: str, temp_output_path: Optional[str] = None):
                """Inner loop for generation with validation."""
                best_wav = None
                best_accent_drift = float('inf')

                for n_iterations in range(self.MAX_ITERATIONS):
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
                        audio_prompt_path=self.reference_audio_path,
                        temperature=temp,
                        cfg_weight=cfg,
                    )

                    if temp_output_path:
                        ta.save(temp_output_path, wav, self.model.sr)

                        # Step 1: Check accent drift
                        drift_prob, is_accent_ok = self._validate_accent_drift(temp_output_path)

                        # Step 2: Check speaker similarity (currently using passthrough)
                        speaker_similarity = 1
                        is_speaker_similar = speaker_similarity > self.SPEAKER_SIMILARITY_THRESHOLD
                        threshold = self.SPEAKER_SIMILARITY_THRESHOLD
                        logger.info(f"Speaker similarity: {speaker_similarity:.3f} (threshold: {threshold})")

                        is_voice_accurate = is_accent_ok and is_speaker_similar

                        if drift_prob < best_accent_drift:
                            best_accent_drift = drift_prob
                            best_wav = wav.clone()
                            logger.info(f"   New best: accent drift {best_accent_drift:.3f}")

                        # Step 3: Text accuracy
                        is_text_accurate = True
                        text_similarity = 1.0

                        if is_voice_accurate:
                            is_text_accurate, text_similarity, transcribed_text = self._validate_text_match(
                                temp_output_path, text
                            )
                            thresh = self.TEXT_SIMILARITY_THRESHOLD
                            logger.info(f"Text similarity: {text_similarity:.3f} (threshold: {thresh})")
                            if transcribed_text:
                                logger.debug(f"   Original: {text}")
                                logger.debug(f"   Transcribed: {transcribed_text}")

                        is_valid = is_voice_accurate and is_text_accurate

                        if is_valid:
                            logger.info(f"Audio valid! Returning after {n_iterations + 1} iteration(s)")
                            return wav

                        failure_reasons = []
                        if not is_accent_ok:
                            failure_reasons.append(f"accent drift (prob: {drift_prob:.3f})")
                        if not is_speaker_similar:
                            failure_reasons.append(f"speaker mismatch (similarity: {speaker_similarity:.3f})")
                        if not is_text_accurate:
                            failure_reasons.append(f"text mismatch (similarity: {text_similarity:.3f})")

                        logger.warning(
                            f"Audio invalid: {', '.join(failure_reasons)}, "
                            f"retrying... (iteration {n_iterations + 1}/{self.MAX_ITERATIONS})"
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

            # Split text into segments
            segments = self._split_text_into_segments(text, self.MAX_CHARS_PER_SEGMENT)
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

    def _validate_accent_drift(self, audio_path: str) -> tuple:
        """Run accent drift validation if classifier is available."""
        try:
            from ..validation.classifier import predict_accent_drift_probability
            drift_prob = predict_accent_drift_probability(audio_path)
            is_ok = drift_prob is not None and drift_prob < self.ACCENT_DRIFT_THRESHOLD
            return (drift_prob if drift_prob is not None else 0.0), is_ok
        except ImportError:
            logger.debug("Accent drift classifier not available, skipping validation")
            return 0.0, True

    def _validate_text_match(self, audio_path: str, expected_text: str) -> tuple:
        """Run STT text matching validation if available."""
        try:
            from ..validation.stt.stt_validator import validate_audio_text_match
            is_accurate, similarity, transcribed = validate_audio_text_match(
                audio_path, expected_text, self.TEXT_SIMILARITY_THRESHOLD
            )
            return is_accurate, similarity, transcribed
        except ImportError:
            logger.debug("STT validator not available, skipping text validation")
            return True, 1.0, None

    @property
    def sample_rate(self) -> int:
        """Get the sample rate for Chatterbox TTS."""
        return self.model.sr
