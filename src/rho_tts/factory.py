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
    _isolated_providers: set[str] = set()
    _default_providers_registered = False

    @classmethod
    def _register_default_providers(cls):
        """Register built-in providers on first use.

        Providers whose dependencies aren't available are added to
        ``_isolated_providers`` — they'll be run in an auto-managed
        venv via ``ProviderProxy``.
        """
        if cls._default_providers_registered:
            return
        cls._default_providers_registered = True

        try:
            from .providers.qwen import QwenTTS
            cls._providers["qwen"] = QwenTTS
        except ImportError:
            cls._isolated_providers.add("qwen")

        try:
            from .providers.chatterbox import ChatterboxTTS
            cls._providers["chatterbox"] = ChatterboxTTS
        except ImportError:
            cls._isolated_providers.add("chatterbox")

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

        # Direct import available — fast path
        if provider in cls._providers:
            tts_class = cls._providers[provider]
            return tts_class(**kwargs)

        # Deps not locally available — delegate to isolated venv
        if provider in cls._isolated_providers:
            from .isolation import ProviderProxy
            return ProviderProxy(provider, **kwargs)

        available = ", ".join(cls.list_providers()) or "(none registered)"
        raise ValueError(
            f"Unknown TTS provider: '{provider}'. "
            f"Available providers: {available}. "
            f"Make sure the provider's dependencies are installed "
            f"(e.g., pip install rho-tts[qwen])"
        )

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
        """Get list of available TTS provider names (including isolated)."""
        cls._register_default_providers()
        return sorted(set(cls._providers.keys()) | cls._isolated_providers)
