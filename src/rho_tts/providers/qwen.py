"""
Qwen3-TTS provider implementation.

Primary/default TTS provider for production use. Uses batch processing
and quality validation for high-quality audio generation. Supports both
default voice and voice cloning via reference audio.
"""
import logging
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from ..base_tts import BaseTTS
from ..provider_info import ProviderInfo, VoiceInfo

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
        max_chars_per_segment: Maximum characters per text segment (default: auto-computed)
        batch_size: Number of texts to process per batch (default: 5)
        max_iterations: Maximum validation retry iterations (default: 10)
        accent_drift_threshold: Threshold for accent drift validation (default: 0.17)
        text_similarity_threshold: Threshold for STT text matching (default: 0.85)
        sound_decay_threshold: Max ratio of final to initial RMS energy (default: 0.3)
        drift_model_path: Explicit path to a .pkl drift classifier model (overrides voice_id lookup)
        phonetic_mapping: Custom word-to-pronunciation mapping
    """

    MAX_MODEL_CHARS = 4000
    BYTES_PER_CHAR_ESTIMATE = 500_000

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
        max_chars_per_segment: Optional[int] = None,
        batch_size: int = 5,
        max_iterations: int = 10,
        accent_drift_threshold: float = 0.17,
        text_similarity_threshold: float = 0.85,
        sound_decay_threshold: float = 0.3,
        drift_model_path: Optional[str] = None,
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
        self.drift_model_path = drift_model_path

        # Configurable thresholds
        self._max_chars_explicit = max_chars_per_segment is not None
        self.max_chars_per_segment = max_chars_per_segment if max_chars_per_segment is not None else 1000
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

                self.qwen3_model = self._try_load_model(Qwen3TTSModel, model_path)

            except ImportError as e:
                raise ImportError(
                    f"Failed to import qwen_tts: {e}. "
                    f"Install with: pip install rho-tts[qwen]"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load Qwen3-TTS model: {e}")

            # Refine max model chars from config if available
            try:
                config = getattr(self.qwen3_model, 'config', None)
                if config is not None:
                    max_pos = getattr(config, 'max_position_embeddings', None)
                    if max_pos is not None:
                        self._max_model_chars = min(self.MAX_MODEL_CHARS, max_pos)
            except Exception:
                pass

    def _try_load_model(self, model_cls, model_path: str):
        """Try loading the model with attention fallback and CUDA fallback.

        Order: flash_attention_2 → sdpa → CPU (if CUDA failed).
        """
        devices_to_try = [self.device]
        if self.device == "cuda":
            devices_to_try.append("cpu")

        last_error = None
        for device in devices_to_try:
            if device != self.device:
                logger.warning("CUDA failed, falling back to CPU...")

            # Try flash_attention_2 first, then sdpa
            for attn in ("flash_attention_2", "sdpa"):
                try:
                    model = model_cls.from_pretrained(
                        model_path,
                        device_map=device,
                        dtype=torch.bfloat16,
                        attn_implementation=attn,
                    )
                    if device != self.device:
                        self.device = device
                    logger.info(f"Qwen3-TTS model loaded with {attn} on {device}")
                    return model
                except ImportError:
                    # flash_attn not installed — try next attn impl
                    last_error = None
                    continue
                except RuntimeError as e:
                    last_error = e
                    err_msg = str(e).lower()
                    if "cuda" in err_msg or "nvidia" in err_msg or "driver" in err_msg:
                        # CUDA-level failure — skip to CPU fallback
                        break
                    if attn == "flash_attention_2":
                        # Non-CUDA error on flash_attn — try sdpa
                        logger.warning(f"flash_attention_2 failed ({e}), trying sdpa...")
                        continue
                    # sdpa also failed with non-CUDA error — re-raise
                    raise
                except Exception as e:
                    last_error = e
                    if attn == "flash_attention_2":
                        logger.warning(f"flash_attention_2 failed ({e}), trying sdpa...")
                        continue
                    raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to load model: no compatible configuration found")

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
        """Normalize audio loudness with decay correction.

        Two-pass approach:
        1. Windowed normalization — corrects volume decay by computing
           per-window RMS and applying a smoothed gain envelope so the
           volume stays consistent across the full duration.
        2. Global normalization — brings overall loudness to target RMS.
        3. Soft clipping (tanh) to prevent harsh distortion.
        """
        target_rms_db = -23.0
        window_sec = 2.0
        max_gain_db = 12.0

        original_shape = audio.shape
        if audio.dim() > 1:
            audio = audio.squeeze()

        overall_rms = torch.sqrt(torch.mean(audio ** 2))
        if overall_rms < 1e-8:
            return audio.reshape(original_shape)

        # --- Pass 1: Windowed decay correction ---
        n_samples = audio.shape[0]
        sr = self.qwen3_sr or 24000
        window_samples = int(sr * window_sec)

        if n_samples > window_samples * 2:
            audio = self._apply_windowed_normalization(
                audio, window_samples, max_gain_db,
            )

        # --- Pass 2: Global normalization to target RMS ---
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms > 1e-8:
            current_rms_db = 20 * torch.log10(rms)
            gain_db = target_rms_db - current_rms_db.item()
            gain_linear = 10 ** (gain_db / 20)
            audio = audio * gain_linear

        # --- Pass 3: Soft clip ---
        max_amplitude = 0.95
        audio = torch.tanh(audio / max_amplitude) * max_amplitude

        return audio.reshape(original_shape)

    def _apply_windowed_normalization(
        self,
        audio: torch.Tensor,
        window_samples: int,
        max_gain_db: float,
    ) -> torch.Tensor:
        """Apply per-window gain to correct volume decay over time.

        Computes RMS in non-overlapping windows, derives a per-window gain
        to match the first window's energy, interpolates to a smooth
        sample-level gain envelope, and applies it.
        """
        n_samples = audio.shape[0]
        n_windows = n_samples // window_samples

        if n_windows < 2:
            return audio

        # Compute per-window RMS
        window_rms = []
        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples
            chunk = audio[start:end]
            rms = torch.sqrt(torch.mean(chunk ** 2)).item()
            window_rms.append(rms)

        # Use the first window as the reference level
        ref_rms = window_rms[0]
        if ref_rms < 1e-8:
            return audio

        max_gain_linear = 10 ** (max_gain_db / 20)

        # Compute per-window gain (capped)
        window_gains = []
        for rms in window_rms:
            if rms < 1e-8:
                window_gains.append(1.0)
            else:
                gain = ref_rms / rms
                gain = min(gain, max_gain_linear)
                window_gains.append(gain)

        # Only apply correction if there's actual decay to fix
        # (skip if all gains are close to 1.0)
        gain_range = max(window_gains) - min(window_gains)
        if gain_range < 0.05:
            return audio

        # Smooth the gain curve with a simple 3-tap moving average
        smoothed = list(window_gains)
        for _ in range(2):
            new_smoothed = list(smoothed)
            for i in range(1, len(smoothed) - 1):
                new_smoothed[i] = (smoothed[i - 1] + smoothed[i] + smoothed[i + 1]) / 3
            smoothed = new_smoothed

        # Interpolate to sample-level gain envelope using numpy (vectorized)
        centers = np.array([(i + 0.5) * window_samples for i in range(n_windows)])
        sample_indices = np.arange(n_samples, dtype=np.float64)
        gain_np = np.interp(sample_indices, centers, smoothed)
        gain_envelope = torch.from_numpy(gain_np).float().to(audio.device)

        return audio * gain_envelope

    def close(self) -> None:
        """Release model and free GPU memory."""
        if self.qwen3_model is not None:
            del self.qwen3_model
            self.qwen3_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def provider_info(cls) -> ProviderInfo:
        """Return Qwen provider metadata."""
        return ProviderInfo(
            name="qwen",
            supports_voice_cloning=True,
            supported_languages=["English", "Chinese", "Japanese", "Korean"],
            builtin_voices=[
                VoiceInfo(id="Chelsie", name="Chelsie", language="English"),
                VoiceInfo(id="Aidan", name="Aidan", language="English"),
                VoiceInfo(id="Vivian", name="Vivian", language="English"),
                VoiceInfo(id="Ryan", name="Ryan", language="English"),
                VoiceInfo(id="Aria", name="Aria", language="English"),
                VoiceInfo(id="Ethan", name="Ethan", language="English"),
                VoiceInfo(id="Luna", name="Luna", language="English"),
                VoiceInfo(id="Harper", name="Harper", language="English"),
                VoiceInfo(id="James", name="James", language="English"),
            ],
        )

    @property
    def sample_rate(self) -> int:
        """Get the sample rate for Qwen3-TTS."""
        if self.qwen3_sr is None:
            self._generate_audio("test")
        return self.qwen3_sr
