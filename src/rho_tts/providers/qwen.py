"""
Qwen3-TTS provider implementation.

Primary/default TTS provider for production use. Uses batch processing
and quality validation for high-quality audio generation. Supports both
default voice and voice cloning via reference audio.
"""
import logging
import os
import traceback
from typing import Dict, List, Optional, Union

import torch
import torchaudio as ta

from ..base_tts import BaseTTS
from ..cancellation import CancellationToken, CancelledException

logger = logging.getLogger(__name__)


class QwenTTS(BaseTTS):
    """
    Qwen3-TTS implementation with batch processing and validation.

    Args:
        device: Device to run the model on ('cuda' or 'cpu')
        seed: Random seed for consistent voice generation
        deterministic: If True, use deterministic CUDA operations
        reference_audio: Path to reference audio file for voice cloning (optional).
            If not provided, uses the model's default voice.
        reference_text: Transcript of the reference audio (required when reference_audio is set)
        model_path: Path to local model or HuggingFace model ID
            (default: "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        max_chars_per_text: Maximum characters per text segment (default: 6000)
        batch_size: Number of texts to process per batch (default: 5)
        max_iterations: Maximum validation retry iterations (default: 10)
        accent_drift_threshold: Threshold for accent drift validation (default: 0.17)
        text_similarity_threshold: Threshold for STT text matching (default: 0.85)
        sound_decay_threshold: Max ratio of final to initial RMS energy (default: 0.3)
        phonetic_mapping: Custom word-to-pronunciation mapping
    """

    def __init__(
        self,
        device: str = "cuda",
        seed: int = 789,
        deterministic: bool = False,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
        speaker: Optional[str] = None,
        model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        max_chars_per_text: int = 6000,
        batch_size: int = 5,
        max_iterations: int = 10,
        accent_drift_threshold: float = 0.17,
        text_similarity_threshold: float = 0.85,
        sound_decay_threshold: float = 0.3,
        phonetic_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__(device, seed, deterministic, phonetic_mapping=phonetic_mapping)

        if reference_audio is not None and reference_text is None:
            raise ValueError("reference_text (transcript of reference audio) is required when reference_audio is set")

        self.reference_audio_path = reference_audio
        self.reference_text = reference_text
        self.speaker = speaker
        self.voice_cloning = reference_audio is not None
        self.model_path = model_path

        # Configurable thresholds
        self.MAX_CHARS_PER_TEXT = max_chars_per_text
        self.BATCH_SIZE = batch_size
        self.MAX_ITERATIONS = max_iterations
        self.ACCENT_DRIFT_THRESHOLD = accent_drift_threshold
        self.TEXT_SIMILARITY_THRESHOLD = text_similarity_threshold
        self.SOUND_DECAY_THRESHOLD = sound_decay_threshold

        # Qwen3-TTS model (lazy loaded)
        self.qwen3_model = None
        self.qwen3_sr = None

        self._reference_embedding_initialized = False

    def _load_qwen3_model(self):
        """Lazy load Qwen3-TTS model on first use."""
        if self.qwen3_model is None:
            try:
                from qwen_tts import Qwen3TTSModel

                logger.info("Loading Qwen3-TTS model...")

                model_path = self.model_path
                if os.path.exists(model_path):
                    logger.info(f"Using local model from {model_path}")
                else:
                    logger.info(f"Will download model from HuggingFace: {model_path}")

                # Try flash_attention_2 first, fall back to sdpa
                try:
                    self.qwen3_model = Qwen3TTSModel.from_pretrained(
                        model_path,
                        device_map=self.device,
                        dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                    )
                    logger.info("Qwen3-TTS model loaded with flash_attention_2")
                except Exception as e:
                    logger.warning(f"flash_attention_2 failed ({e}), falling back to sdpa...")
                    self.qwen3_model = Qwen3TTSModel.from_pretrained(
                        model_path,
                        device_map=self.device,
                        dtype=torch.bfloat16,
                        attn_implementation="sdpa",
                    )
                    logger.info("Qwen3-TTS model loaded with sdpa attention")

            except ImportError as e:
                raise ImportError(
                    f"Failed to import qwen_tts: {e}. "
                    f"Install with: pip install rho-tts[qwen]"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load Qwen3-TTS model: {e}")

        return self.qwen3_model

    def _initialize_reference_embedding(self):
        """Initialize reference voice embedding after model is loaded."""
        if not self.voice_cloning:
            return
        if not self._reference_embedding_initialized and self.qwen3_sr is not None:
            try:
                from resemblyzer import preprocess_wav

                logger.info("Computing reference voice embedding for Qwen3-TTS...")
                reference_wav = preprocess_wav(self.reference_audio_path)
                self.reference_embedding = self.voice_encoder.embed_utterance(reference_wav)
                self._reference_embedding_initialized = True
                logger.info("Reference voice embedding computed")
            except ImportError:
                logger.warning(
                    "resemblyzer not installed, speaker similarity validation disabled. "
                    "Install with: pip install rho-tts[validation]"
                )

    def _generate_audio(self, text: Union[str, List[str]], **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Generate audio using Qwen3-TTS.

        Routes to the correct generation method based on model type:
        - CustomVoice model + speaker → generate_custom_voice()
        - Base model + reference audio → generate_voice_clone()
        - Base model + no reference audio → error
        """
        model = self._load_qwen3_model()

        is_single = isinstance(text, str)
        text_list = [text] if is_single else text

        model_type = getattr(model.model, "tts_model_type", "base")

        if model_type == "custom_voice" and self.speaker:
            wavs, sr = model.generate_custom_voice(
                text=text_list,
                speaker=self.speaker,
                language="English",
            )
        elif self.voice_cloning:
            wavs, sr = model.generate_voice_clone(
                text=text_list,
                language="English",
                ref_audio=self.reference_audio_path,
                ref_text=self.reference_text,
            )
        else:
            raise ValueError(
                "Qwen Base model requires reference audio for voice cloning. "
                "Use a CustomVoice model with a named speaker, or provide reference audio."
            )

        if self.qwen3_sr is None:
            self.qwen3_sr = sr
            if self.voice_cloning:
                self._initialize_reference_embedding()

        torch_wavs = [torch.from_numpy(wav).float() for wav in wavs]
        return torch_wavs[0] if is_single else torch_wavs

    @staticmethod
    def _normalize_audio_for_voice(
        audio_tensor: torch.Tensor,
        target_rms_db: float = -23.0,
        min_rms_db: float = -30.0,
        max_rms_db: float = -10.0,
    ) -> torch.Tensor:
        """
        Normalize audio to keep voice in an optimal loudness range.

        Uses dynamic range compression to make quiet parts louder while
        preventing loud parts from clipping.
        """
        original_shape = audio_tensor.shape
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()

        rms = torch.sqrt(torch.mean(audio_tensor ** 2))

        if rms < 1e-8:
            return audio_tensor.reshape(original_shape)

        current_rms_db = 20 * torch.log10(rms)
        gain_db = target_rms_db - current_rms_db.item()
        gain_linear = 10 ** (gain_db / 20)

        normalized = audio_tensor * gain_linear

        max_amplitude = 0.95
        normalized = torch.tanh(normalized / max_amplitude) * max_amplitude

        return normalized.reshape(original_shape)

    @staticmethod
    def _detect_sound_decay(
        audio_tensor: torch.Tensor,
        sample_rate: int,
        window_size: float = 1,
        decay_threshold: float = 0.3,
    ) -> tuple[bool, float]:
        """
        Detect if sound level decays significantly over time.

        Returns:
            Tuple of (has_decay, energy_ratio)
        """
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()

        min_samples = int(2 * window_size * sample_rate)
        if len(audio_tensor) < min_samples:
            return False, 1.0

        window_samples = int(window_size * sample_rate)

        initial_segment = audio_tensor[:window_samples]
        final_segment = audio_tensor[-window_samples:]

        initial_rms = torch.sqrt(torch.mean(initial_segment ** 2))
        final_rms = torch.sqrt(torch.mean(final_segment ** 2))

        if initial_rms < 1e-8:
            return False, 1.0

        energy_ratio = (final_rms / initial_rms).item()
        has_decay = energy_ratio < decay_threshold

        return has_decay, energy_ratio

    def generate(
        self,
        texts: List[str],
        output_base_path: str,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Optional[List[str]]:
        """
        Generate TTS audio with batch processing and quality validation.

        Args:
            texts: List of text strings to generate audio for
            output_base_path: Base path for output files (will append _{idx}.wav)
            cancellation_token: Optional token for cancellation

        Returns:
            List of output file paths (None for failed items), or None if completely failed
        """
        try:
            token = cancellation_token or CancellationToken()
            mapped_texts = [self._apply_phonetic_mapping(text) for text in texts]
            output_files = []

            logger.info(f"Generating audio for {len(mapped_texts)} text item(s) with Qwen3-TTS...")

            for idx, text in enumerate(mapped_texts):
                if len(text) > self.MAX_CHARS_PER_TEXT:
                    logger.warning(
                        f"Text item {idx} is long ({len(text)} chars, limit ~{self.MAX_CHARS_PER_TEXT}). May fail."
                    )

            # Validation loop
            best_audios = [None] * len(mapped_texts)
            best_drifts = [float('inf')] * len(mapped_texts)
            passed_items = [False] * len(mapped_texts)

            for iteration in range(self.MAX_ITERATIONS):
                if token.is_cancelled():
                    raise CancelledException(f"Cancelled during iteration {iteration}")

                failed_count = len([p for p in passed_items if not p])
                logger.info(f"Iteration {iteration + 1}/{self.MAX_ITERATIONS} - {failed_count} item(s) remaining...")

                failed_indices = [idx for idx in range(len(mapped_texts)) if not passed_items[idx]]

                if not failed_indices:
                    break

                for batch_start in range(0, len(failed_indices), self.BATCH_SIZE):
                    batch_end = min(batch_start + self.BATCH_SIZE, len(failed_indices))
                    items_to_generate = failed_indices[batch_start:batch_end]

                    if token.is_cancelled():
                        raise CancelledException(f"Cancelled before batch {batch_start}-{batch_end}")

                    try:
                        batch_texts = [mapped_texts[idx] for idx in items_to_generate]
                        audio_tensors = self._generate_audio(batch_texts)

                        for local_idx, global_idx in enumerate(items_to_generate):
                            if token.is_cancelled():
                                raise CancelledException(f"Cancelled during validation of item {global_idx}")

                            audio_tensor = audio_tensors[local_idx]
                            audio_tensor = self._normalize_audio_for_voice(audio_tensor)

                            temp_path = None

                            try:
                                temp_path = f"/tmp/rho_tts_validate_{global_idx}_{iteration}.wav"
                                segment_wav = audio_tensor.cpu() if audio_tensor.device.type != 'cpu' else audio_tensor
                                if segment_wav.dim() == 1:
                                    segment_wav = segment_wav.unsqueeze(0)
                                ta.save(temp_path, segment_wav, self.qwen3_sr)

                                # Validation 1: Accent drift
                                drift_prob, is_voice_accurate = self._validate_accent_drift(temp_path)

                                # Validation 2: Text matching
                                is_text_accurate = True
                                text_similarity = 1.0

                                if is_voice_accurate:
                                    is_text_accurate, text_similarity, _ = self._validate_text_match(
                                        temp_path, batch_texts[local_idx]
                                    )

                                is_valid = is_voice_accurate and is_text_accurate

                                if drift_prob is not None and drift_prob < best_drifts[global_idx]:
                                    best_drifts[global_idx] = drift_prob
                                    best_audios[global_idx] = audio_tensor.clone()

                                if is_valid:
                                    logger.info(
                                        f"  Item {global_idx + 1}: PASSED "
                                        f"(drift={drift_prob:.3f}, text={text_similarity:.3f})"
                                    )
                                    passed_items[global_idx] = True
                                else:
                                    reasons = []
                                    if not is_voice_accurate:
                                        reasons.append(f"drift={drift_prob:.3f}")
                                    if not is_text_accurate:
                                        reasons.append(f"text={text_similarity:.3f}")
                                    logger.warning(
                                        f"  Item {global_idx + 1}: FAILED ({', '.join(reasons)}), "
                                        f"best_drift={best_drifts[global_idx]:.3f}"
                                    )

                            except Exception as e:
                                logger.warning(f"  Item {global_idx + 1}: Error during validation ({e})")
                            finally:
                                if temp_path and os.path.exists(temp_path):
                                    try:
                                        os.remove(temp_path)
                                    except OSError:
                                        pass

                        del audio_tensors
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() or "length" in str(e).lower():
                            logger.error(f"Batch {batch_start}-{batch_end} OOM: {e}")
                        else:
                            raise
                    except Exception as e:
                        logger.warning(f"Batch {batch_start}-{batch_end}: Error ({e})")

            # Save final audio
            for idx in range(len(mapped_texts)):
                if best_audios[idx] is not None:
                    try:
                        output_path = f"{output_base_path}_{idx}.wav"
                        audio = best_audios[idx]
                        final_wav = audio.cpu() if audio.device.type != 'cpu' else audio
                        if final_wav.dim() == 1:
                            final_wav = final_wav.unsqueeze(0)
                        ta.save(output_path, final_wav, self.qwen3_sr)
                        output_files.append(output_path)

                        status = "PASSED" if passed_items[idx] else f"BEST (drift={best_drifts[idx]:.3f})"
                        logger.info(f"Item {idx + 1} saved: {output_path} [{status}]")

                        best_audios[idx] = None
                        del final_wav
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except Exception as e:
                        logger.error(f"Failed to save audio for item {idx}: {e}")
                        output_files.append(None)
                else:
                    logger.error(f"Item {idx + 1} failed to generate")
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
            logger.error(f"Error in Qwen3 TTS: {e}")
            traceback.print_exc()
            return None

    def generate_single(
        self,
        text: str,
        output_path: str,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Optional[torch.Tensor]:
        """Generate audio for a single text using the batch pipeline."""
        import shutil

        temp_dir = os.path.dirname(output_path)
        temp_base = os.path.join(temp_dir, "temp_single_gen")

        result = self.generate([text], temp_base, cancellation_token)

        if result is None or len(result) == 0 or result[0] is None:
            return None

        temp_file = result[0]
        if os.path.exists(temp_file):
            shutil.move(temp_file, output_path)
            logger.info(f"Single audio saved: {output_path}")
            wav, sr = ta.load(output_path)
            return wav.squeeze(0)

        return None

    @property
    def sample_rate(self) -> int:
        """Get the sample rate for Qwen3-TTS."""
        if self.qwen3_sr is None:
            self._generate_audio("test")
        return self.qwen3_sr
