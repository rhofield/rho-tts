"""
Runtime state management for the rho-tts web UI.

Manages the active TTS instance (cached by voice+model combo),
cancellation tokens, and config persistence.
"""

import logging
import threading
from typing import Optional

import torch

from ..cancellation import CancellationToken
from ..factory import TTSFactory
from ..base_tts import BaseTTS
from .config import (
    AppConfig,
    ModelConfig,
    VoiceProfile,
    get_phonetic_key,
    load_config,
    save_config,
)

logger = logging.getLogger(__name__)


class AppState:
    """
    Singleton-ish runtime state shared across all Gradio callbacks.

    Holds the current config, the active TTS instance (lazily created
    and cached by voice+model), and the current cancellation token.
    """

    def __init__(self, config: Optional[AppConfig] = None, config_path: Optional[str] = None):
        self.config = config or load_config(config_path)
        self._config_path = config_path
        self._active_tts: Optional[BaseTTS] = None
        self._cache_key: Optional[tuple] = None  # (voice_id, model_id)
        self._lock = threading.Lock()
        self.cancellation_token: Optional[CancellationToken] = None

    def save(self) -> None:
        save_config(self.config, self._config_path)

    def get_or_create_tts(
        self,
        model_config: ModelConfig,
        voice_profile: Optional[VoiceProfile] = None,
    ) -> BaseTTS:
        """
        Return a cached TTS instance, or create a new one if the
        voice/model combination has changed.

        When *voice_profile* is ``None`` the provider's default voice is used.
        """
        voice_id = voice_profile.id if voice_profile else None
        new_key = (voice_id, model_config.id)

        with self._lock:
            if self._active_tts is not None and self._cache_key == new_key:
                return self._active_tts

            # Tear down previous instance
            if self._active_tts is not None:
                logger.info(f"Releasing TTS instance for {self._cache_key}")
                if hasattr(self._active_tts, "shutdown"):
                    self._active_tts.shutdown()
                del self._active_tts
                self._active_tts = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Build kwargs for TTSFactory
            kwargs = {
                "device": self.config.device,
                **model_config.params,
            }

            if voice_profile:
                phonetic_key = get_phonetic_key(voice_profile.id, model_config.id)
                phonetic_mapping = self.config.phonetic_mappings.get(phonetic_key, {})

                if voice_profile.reference_audio:
                    kwargs["reference_audio"] = voice_profile.reference_audio
                if phonetic_mapping:
                    kwargs["phonetic_mapping"] = phonetic_mapping

                # Qwen CustomVoice named speaker
                if voice_profile.speaker:
                    kwargs["speaker"] = voice_profile.speaker

                # Pass language (defaults to English for user-created voices)
                kwargs["language"] = voice_profile.language

                # Qwen needs reference_text when doing voice cloning
                if model_config.provider == "qwen" and voice_profile.reference_text:
                    kwargs["reference_text"] = voice_profile.reference_text

            logger.info(f"Creating TTS instance: provider={model_config.provider}, voice={voice_id}")
            self._active_tts = TTSFactory.get_tts_instance(
                provider=model_config.provider,
                **kwargs,
            )
            self._cache_key = new_key
            return self._active_tts

    def invalidate_tts(self) -> None:
        """Force the next get_or_create_tts call to rebuild."""
        with self._lock:
            if self._active_tts is not None:
                if hasattr(self._active_tts, "shutdown"):
                    self._active_tts.shutdown()
                del self._active_tts
                self._active_tts = None
                self._cache_key = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def new_cancellation_token(self) -> CancellationToken:
        """Create and store a fresh cancellation token for a new generation run."""
        self.cancellation_token = CancellationToken()
        return self.cancellation_token

    def cancel_generation(self) -> None:
        """Cancel the current generation if one is running."""
        if self.cancellation_token is not None:
            self.cancellation_token.cancel()
