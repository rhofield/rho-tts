"""
Qwen3-TTS provider implementation.

Primary/default TTS provider for production use. Uses batch processing
and quality validation for high-quality audio generation. Supports both
default voice and voice cloning via reference audio.
"""
import logging
import os
from typing import Dict, List, Optional, Union

import torch

from ..base_tts import BaseTTS

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
        language: str = "English",
        model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        max_chars_per_text: int = 1000,
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
        self.language = language
        self.voice_cloning = reference_audio is not None
        self.model_path = model_path

        # Configurable thresholds
        self.max_chars_per_segment = max_chars_per_text
        self.batch_size = batch_size
        self.force_sentence_split = False
        self.max_iterations = max_iterations
        self.accent_drift_threshold = accent_drift_threshold
        self.text_similarity_threshold = text_similarity_threshold
        self.sound_decay_threshold = sound_decay_threshold

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

        is_custom_voice = "CustomVoice" in self.model_path

        if is_custom_voice and self.speaker:
            wavs, sr = model.generate_custom_voice(
                text=text_list,
                speaker=self.speaker,
                language=self.language,
            )
        elif is_custom_voice:
            raise ValueError(
                "CustomVoice model requires a named speaker. "
                "Select a built-in voice (e.g. Vivian, Ryan) or provide reference audio with a Base model for voice cloning."
            )
        elif self.voice_cloning:
            wavs, sr = model.generate_voice_clone(
                text=text_list,
                language=self.language,
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

    def _post_process_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio loudness for consistent voice output.

        Uses RMS-based gain normalization with soft clipping (tanh) to keep
        voice in an optimal loudness range without harsh distortion.
        """
        target_rms_db = -23.0

        original_shape = audio.shape
        if audio.dim() > 1:
            audio = audio.squeeze()

        rms = torch.sqrt(torch.mean(audio ** 2))

        if rms < 1e-8:
            return audio.reshape(original_shape)

        current_rms_db = 20 * torch.log10(rms)
        gain_db = target_rms_db - current_rms_db.item()
        gain_linear = 10 ** (gain_db / 20)

        normalized = audio * gain_linear

        max_amplitude = 0.95
        normalized = torch.tanh(normalized / max_amplitude) * max_amplitude

        return normalized.reshape(original_shape)

    @property
    def sample_rate(self) -> int:
        """Get the sample rate for Qwen3-TTS."""
        if self.qwen3_sr is None:
            self._generate_audio("test")
        return self.qwen3_sr
