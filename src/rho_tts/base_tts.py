"""
Abstract base class for TTS implementations.

Provides shared functionality for audio processing, validation, and text handling
that can be reused across different TTS providers.
"""
import logging
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .cancellation import CancellationToken

logger = logging.getLogger(__name__)

# Default phonetic mapping - users can override via constructor
DEFAULT_PHONETIC_MAPPING: Dict[str, str] = {}


class BaseTTS(ABC):
    """Abstract base class for TTS implementations."""

    def __init__(
        self,
        device: str = "cuda",
        seed: int = 789,
        deterministic: bool = False,
        phonetic_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize base TTS configuration.

        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            seed: Random seed for consistent voice generation
            deterministic: If True, use deterministic CUDA operations (slower but reproducible)
            phonetic_mapping: Custom word-to-pronunciation mapping for improving TTS output.
                            Example: {"exocrine": "exo-crene"}
        """
        self.device = device
        self.seed = seed
        self.deterministic = deterministic
        self.phonetic_mapping = phonetic_mapping if phonetic_mapping is not None else DEFAULT_PHONETIC_MAPPING.copy()
        self._set_seeds()

        # Validation thresholds (subclasses override as needed)
        self.accent_drift_threshold = 0.17
        self.text_similarity_threshold = 0.85

        # Audio segment smoothing parameters
        self.silence_threshold_db = -50.0
        self.crossfade_duration_sec = 0.05
        self.trim_silence = True
        self.fade_duration_sec = 0.02
        self.force_sentence_split = True
        self.inter_sentence_pause_sec = 0.1

        # Voice encoder for speaker similarity validation (lazy loaded)
        self._voice_encoder = None
        self.reference_embedding = None

    @property
    def voice_encoder(self):
        """Lazy-load voice encoder to avoid import cost when not needed."""
        if self._voice_encoder is None:
            try:
                from resemblyzer import VoiceEncoder
                self._voice_encoder = VoiceEncoder()
            except ImportError:
                raise ImportError(
                    "resemblyzer is required for speaker similarity validation. "
                    "Install it with: pip install rho-tts[validation]"
                )
        return self._voice_encoder

    def _set_seeds(self):
        """Set random seeds for reproducible generation."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    def _apply_phonetic_mapping(self, text: str) -> str:
        """
        Apply phonetic mappings to improve TTS pronunciation.

        Args:
            text: Original text

        Returns:
            Text with phonetic mappings applied
        """
        updated_text = text
        for original, phonetic in self.phonetic_mapping.items():
            updated_text = updated_text.replace(original, phonetic)
        return updated_text

    def _validate_accent_drift(self, audio_path: str) -> tuple:
        """Run accent drift validation if classifier is available."""
        if not getattr(self, 'voice_cloning', False):
            return 0.0, True
        try:
            from .validation.classifier import predict_accent_drift_probability
            drift_prob = predict_accent_drift_probability(audio_path)
            if drift_prob is None:
                return 0.0, True
            return drift_prob, drift_prob < self.accent_drift_threshold
        except ImportError:
            logger.debug("Accent drift classifier not available, skipping validation")
            return 0.0, True

    def _validate_text_match(self, audio_path: str, expected_text: str) -> tuple:
        """Run STT text matching validation if available."""
        try:
            from .validation.stt.stt_validator import validate_audio_text_match
            is_accurate, similarity, transcribed = validate_audio_text_match(
                audio_path, expected_text, self.text_similarity_threshold
            )
            return is_accurate, similarity, transcribed
        except ImportError:
            logger.debug("STT validator not available, skipping text validation")
            return True, 1.0, None

    def _compute_speaker_similarity(self, wav_tensor: torch.Tensor) -> float:
        """
        Compute cosine similarity between generated audio and reference voice.

        Args:
            wav_tensor: Generated audio tensor

        Returns:
            Cosine similarity score (0-1, higher is more similar)
        """
        from resemblyzer import preprocess_wav

        wav_np = wav_tensor.cpu().numpy().flatten()
        preprocessed_wav = preprocess_wav(wav_np, source_sr=self.sample_rate)
        generated_embedding = self.voice_encoder.embed_utterance(preprocessed_wav)

        dot_product = np.dot(self.reference_embedding, generated_embedding)
        norm_reference = np.linalg.norm(self.reference_embedding)
        norm_generated = np.linalg.norm(generated_embedding)
        similarity = dot_product / (norm_reference * norm_generated)

        return similarity

    def _trim_silence(self, audio: torch.Tensor, from_start: bool = True, from_end: bool = True) -> torch.Tensor:
        """
        Trim silence from the beginning and/or end of an audio tensor.

        Args:
            audio: Audio tensor of shape (channels, samples) or (samples,)
            from_start: Whether to trim silence from the beginning
            from_end: Whether to trim silence from the end

        Returns:
            Trimmed audio tensor
        """
        if not self.trim_silence or audio.numel() == 0:
            return audio

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        window_size = int(self.sample_rate * 0.01)
        threshold = 10 ** (self.silence_threshold_db / 20)

        audio_squared = audio ** 2
        energy = torch.sqrt(torch.nn.functional.avg_pool1d(
            audio_squared,
            kernel_size=window_size,
            stride=window_size // 2,
            padding=window_size // 2
        ).mean(dim=0))

        non_silent = energy > threshold

        if not non_silent.any():
            return audio[:, :window_size]

        non_silent_indices = non_silent.nonzero(as_tuple=True)[0]
        first_non_silent = non_silent_indices[0].item()
        last_non_silent = non_silent_indices[-1].item()

        start_sample = (first_non_silent * window_size // 2) if from_start else 0
        end_sample = ((last_non_silent + 2) * window_size // 2) if from_end else audio.shape[-1]

        start_sample = max(0, min(start_sample, audio.shape[-1]))
        end_sample = max(start_sample, min(end_sample, audio.shape[-1]))

        return audio[:, start_sample:end_sample].squeeze(0)

    def _remove_dc_offset(self, audio: torch.Tensor) -> torch.Tensor:
        """Remove DC offset from audio."""
        if audio.numel() == 0:
            return audio
        dc_offset = audio.mean()
        return audio - dc_offset

    def _apply_fades(self, audio: torch.Tensor, fade_in: bool = True, fade_out: bool = True) -> torch.Tensor:
        """
        Apply fade-in and/or fade-out to prevent pops at audio boundaries.

        Args:
            audio: Audio tensor of shape (channels, samples) or (samples,)
            fade_in: Whether to apply fade-in at start
            fade_out: Whether to apply fade-out at end

        Returns:
            Audio with fades applied
        """
        if audio.numel() == 0:
            return audio

        original_shape = audio.shape
        if audio.dim() == 2:
            audio = audio.squeeze(0)

        fade_samples = int(self.sample_rate * self.fade_duration_sec)

        if audio.shape[-1] < fade_samples * 2:
            return audio.view(original_shape)

        if fade_in:
            fade_in_curve = 0.5 * (1 - torch.cos(torch.linspace(0, np.pi, fade_samples, device=audio.device)))
            audio[:fade_samples] = audio[:fade_samples] * fade_in_curve

        if fade_out:
            fade_out_curve = 0.5 * (1 + torch.cos(torch.linspace(0, np.pi, fade_samples, device=audio.device)))
            audio[-fade_samples:] = audio[-fade_samples:] * fade_out_curve

        return audio.view(original_shape)

    def _smooth_segment_join(self, audio_segments: list[torch.Tensor]) -> torch.Tensor:
        """
        Join audio segments with silence trimming and crossfading for smooth transitions.

        Args:
            audio_segments: List of audio tensors to join

        Returns:
            Single audio tensor with smoothly joined segments
        """
        if len(audio_segments) == 0:
            return None
        if len(audio_segments) == 1:
            audio = audio_segments[0]
            audio = self._trim_silence(audio, from_start=True, from_end=True)
            audio = self._remove_dc_offset(audio)
            audio = self._apply_fades(audio, fade_in=True, fade_out=True)
            return audio

        original_dim = audio_segments[0].dim()
        crossfade_samples = int(self.sample_rate * self.crossfade_duration_sec)

        processed_segments = []

        for i, segment in enumerate(audio_segments):
            if segment.device != torch.device(self.device):
                segment = segment.to(self.device)

            if segment.dim() != original_dim:
                if original_dim == 1 and segment.dim() == 2:
                    segment = segment.squeeze(0)
                elif original_dim == 2 and segment.dim() == 1:
                    segment = segment.unsqueeze(0)

            if i == 0:
                trimmed = self._trim_silence(segment, from_start=False, from_end=True)
            elif i == len(audio_segments) - 1:
                trimmed = self._trim_silence(segment, from_start=True, from_end=False)
            else:
                trimmed = self._trim_silence(segment, from_start=True, from_end=True)

            trimmed = self._remove_dc_offset(trimmed)
            processed_segments.append(trimmed)

        result_segments = []

        for i in range(len(processed_segments)):
            current_segment = processed_segments[i]

            if i == 0:
                if len(processed_segments) > 1 and current_segment.shape[-1] > crossfade_samples:
                    result_segments.append(current_segment[..., :-crossfade_samples])
                else:
                    result_segments.append(current_segment)
            else:
                prev_segment = processed_segments[i - 1]
                overlap_size = min(crossfade_samples, prev_segment.shape[-1], current_segment.shape[-1])

                if overlap_size > 10:
                    prev_tail = prev_segment[..., -overlap_size:]
                    curr_head = current_segment[..., :overlap_size]

                    fade_out = torch.cos(torch.linspace(0, np.pi / 2, overlap_size, device=self.device))
                    fade_in = torch.cos(torch.linspace(np.pi / 2, 0, overlap_size, device=self.device))

                    if prev_tail.dim() == 2:
                        fade_out = fade_out.unsqueeze(0)
                        fade_in = fade_in.unsqueeze(0)

                    crossfaded = prev_tail * fade_out + curr_head * fade_in
                    result_segments.append(crossfaded)

                    if i < len(processed_segments) - 1:
                        if current_segment.shape[-1] > (overlap_size + crossfade_samples):
                            remaining = current_segment[..., overlap_size:-crossfade_samples]
                        else:
                            remaining = current_segment[..., overlap_size:]
                    else:
                        remaining = current_segment[..., overlap_size:]

                    if remaining.shape[-1] > 0:
                        result_segments.append(remaining)

                    if self.inter_sentence_pause_sec > 0 and i < len(processed_segments) - 1:
                        pause_samples = int(self.sample_rate * self.inter_sentence_pause_sec)
                        silence_pause = torch.zeros(pause_samples, device=self.device)
                        result_segments.append(silence_pause)
                else:
                    result_segments.append(current_segment)

        if result_segments:
            try:
                final_audio = torch.cat(result_segments, dim=-1)
                final_audio = self._apply_fades(final_audio, fade_in=True, fade_out=True)
                return final_audio
            except Exception as e:
                logger.warning(f"Error during crossfade concatenation: {e}, falling back to direct concatenation")
                fallback = torch.cat(audio_segments, dim=-1)
                return self._apply_fades(fallback, fade_in=True, fade_out=True)
        else:
            fallback = torch.cat(audio_segments, dim=-1)
            return self._apply_fades(fallback, fade_in=True, fade_out=True)

    def _split_text_into_segments(self, text: str, max_chars: int) -> list[str]:
        """
        Split text into segments at natural break points.

        Args:
            text: Text to split
            max_chars: Maximum characters per segment

        Returns:
            List of text segments
        """
        sentences = text.split('. ')
        segments = []
        current_segment = ""

        for sentence in sentences:
            if sentence != sentences[-1]:
                sentence += ". "

            force_split = self.force_sentence_split and len(sentences) > 1

            if force_split or len(current_segment) + len(sentence) > max_chars:
                if current_segment:
                    segments.append(current_segment.strip())
                    current_segment = sentence
                else:
                    if len(sentence) > max_chars:
                        words = sentence.split()
                        current_segment = ""
                        for word in words:
                            if len(current_segment) + len(word) + 1 > max_chars:
                                if current_segment:
                                    segments.append(current_segment.strip())
                                    current_segment = word
                                else:
                                    segments.append(word[:max_chars])
                                    current_segment = ""
                            else:
                                current_segment += " " + word if current_segment else word
                    else:
                        segments.append(sentence.strip())
            else:
                current_segment += sentence

        if current_segment.strip():
            segments.append(current_segment.strip())

        return segments

    @abstractmethod
    def _generate_audio(self, text: Union[str, List[str]], **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Core audio generation method - implementation specific.

        Args:
            text: Text to synthesize (single string or list)
            **kwargs: Implementation-specific parameters

        Returns:
            Generated audio tensor(s)
        """
        pass

    @abstractmethod
    def generate(self, texts: List[str], output_base_path: str,
                 cancellation_token: Optional[CancellationToken] = None) -> Optional[List[str]]:
        """
        Batch generation - primary production API.

        Args:
            texts: List of text strings to generate audio for
            output_base_path: Base path for output files
            cancellation_token: Optional token for cancellation

        Returns:
            List of output file paths (None for failed items), or None if completely failed
        """
        pass

    @abstractmethod
    def generate_single(self, text: str, output_path: str,
                        cancellation_token: Optional[CancellationToken] = None) -> Optional[torch.Tensor]:
        """
        Single-text generation.

        Args:
            text: Text to synthesize
            output_path: Output file path
            cancellation_token: Optional token for cancellation

        Returns:
            Generated audio tensor, or None if failed
        """
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Get the sample rate for this TTS implementation."""
        pass
