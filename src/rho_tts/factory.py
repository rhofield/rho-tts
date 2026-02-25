"""
Factory for creating TTS instances.

Provides a centralized way to create different TTS provider instances
with a consistent API. Supports dynamic provider registration.
"""
from typing import Dict, Type

from .base_tts import BaseTTS


class TTSFactory:
    """Factory for creating TTS provider instances."""

    _providers: Dict[str, Type[BaseTTS]] = {}
    _default_providers_registered = False

    @classmethod
    def _register_default_providers(cls):
        """Register built-in providers on first use."""
        if cls._default_providers_registered:
            return
        cls._default_providers_registered = True

        # Try to register Qwen provider
        try:
            from .providers.qwen import QwenTTS
            cls._providers["qwen"] = QwenTTS
        except ImportError:
            pass

        # Try to register Chatterbox provider
        try:
            from .providers.chatterbox import ChatterboxTTS
            cls._providers["chatterbox"] = ChatterboxTTS
        except ImportError:
            pass

    @classmethod
    def get_tts_instance(cls, provider: str = "qwen", **kwargs) -> BaseTTS:
        """
        Create a TTS instance for the specified provider.

        Args:
            provider: TTS provider name (default: "qwen")
            **kwargs: Additional arguments passed to TTS constructor

        Returns:
            BaseTTS instance for the specified provider

        Raises:
            ValueError: If provider is unknown
        """
        cls._register_default_providers()

        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys()) if cls._providers else "(none registered)"
            raise ValueError(
                f"Unknown TTS provider: '{provider}'. "
                f"Available providers: {available}. "
                f"Make sure the provider's dependencies are installed "
                f"(e.g., pip install rho-tts[qwen])"
            )

        tts_class = cls._providers[provider]
        return tts_class(**kwargs)

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseTTS]):
        """
        Register a new TTS provider.

        Args:
            name: Provider name (used in get_tts_instance)
            provider_class: TTS class (must inherit from BaseTTS)

        Raises:
            TypeError: If provider_class doesn't inherit from BaseTTS
        """
        if not issubclass(provider_class, BaseTTS):
            raise TypeError(f"{provider_class} must inherit from BaseTTS")
        cls._providers[name] = provider_class

    @classmethod
    def list_providers(cls) -> list[str]:
        """Get list of available TTS provider names."""
        cls._register_default_providers()
        return list(cls._providers.keys())
