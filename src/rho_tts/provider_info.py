"""
Provider and voice introspection dataclasses.

Used by BaseTTS.provider_info() and TTSFactory.get_provider_info() /
TTSFactory.list_voices() to expose provider capabilities without
requiring model initialization.
"""

from dataclasses import dataclass, field


@dataclass
class VoiceInfo:
    """Metadata for a single voice."""
    id: str
    name: str
    language: str = "English"
    is_builtin: bool = True


@dataclass
class ProviderInfo:
    """Metadata about a TTS provider's capabilities."""
    name: str
    supports_voice_cloning: bool = False
    supported_languages: list[str] = field(default_factory=list)
    builtin_voices: list[VoiceInfo] = field(default_factory=list)
