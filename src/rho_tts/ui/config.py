"""
Persistent configuration for the rho-tts web UI.

Defines data models for voice profiles, model configs, and phonetic mappings.
Persists to JSON at ~/.rho_tts/config.json (overridable via RHO_TTS_CONFIG
env var or --config CLI arg).
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = os.path.expanduser("~/.rho_tts")
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_CONFIG_DIR, "config.json")
DEFAULT_HISTORY_PATH = os.path.join(DEFAULT_CONFIG_DIR, "history.json")
DEFAULT_VOICES_DIR = os.path.join(DEFAULT_CONFIG_DIR, "voices")


# ---------------------------------------------------------------------------
# Provider model catalog — defines available models per provider
# ---------------------------------------------------------------------------

PROVIDER_MODELS: dict[str, list[dict]] = {
    "qwen": [
        {
            "display_name": "Qwen3-TTS 1.7B Base",
            "defaults": {
                "model_path": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "max_iterations": 10,
                "accent_drift_threshold": 0.17,
                "text_similarity_threshold": 0.85,
            },
        },
        {
            "display_name": "Qwen3-TTS 0.6B Base",
            "defaults": {
                "model_path": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                "max_iterations": 10,
                "accent_drift_threshold": 0.17,
                "text_similarity_threshold": 0.85,
            },
        },
        {
            "display_name": "Qwen3-TTS 1.7B CustomVoice",
            "defaults": {
                "model_path": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "max_iterations": 10,
                "accent_drift_threshold": 0.17,
                "text_similarity_threshold": 0.85,
            },
        },
        {
            "display_name": "Qwen3-TTS 0.6B CustomVoice",
            "defaults": {
                "model_path": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                "max_iterations": 10,
                "accent_drift_threshold": 0.17,
                "text_similarity_threshold": 0.85,
            },
        },
    ],
    "chatterbox": [
        {
            "display_name": "Chatterbox Standard",
            "defaults": {
                "implementation": "standard",
                "max_iterations": 50,
                "accent_drift_threshold": 0.17,
                "text_similarity_threshold": 0.75,
            },
        },
        {
            "display_name": "Chatterbox Faster",
            "defaults": {
                "implementation": "faster",
                "max_iterations": 50,
                "accent_drift_threshold": 0.17,
                "text_similarity_threshold": 0.75,
            },
        },
    ],
}


def is_model_cached(repo_id: str) -> bool:
    """Check if a HuggingFace model is in the local cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        result = try_to_load_from_cache(repo_id, "config.json")
        return isinstance(result, str)
    except Exception:
        return False


def get_provider_model_choices(provider: str) -> list[str]:
    """Return display names for all models under a provider."""
    return [m["display_name"] for m in PROVIDER_MODELS.get(provider, [])]


def get_provider_model_defaults(provider: str, display_name: str) -> dict:
    """Return default params for a specific provider model."""
    for m in PROVIDER_MODELS.get(provider, []):
        if m["display_name"] == display_name:
            return dict(m["defaults"])
    return {}


@dataclass
class VoiceProfile:
    id: str
    name: str
    reference_audio: Optional[str] = None
    reference_text: Optional[str] = None
    speaker: Optional[str] = None
    provider: Optional[str] = None  # set on builtins; None = works with any provider
    description: Optional[str] = None
    language: str = "English"

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "reference_audio": self.reference_audio,
            "reference_text": self.reference_text,
        }
        if self.speaker is not None:
            d["speaker"] = self.speaker
        if self.description is not None:
            d["description"] = self.description
        if self.language != "English":
            d["language"] = self.language
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceProfile":
        return cls(
            id=data["id"],
            name=data["name"],
            reference_audio=data.get("reference_audio"),
            reference_text=data.get("reference_text"),
            speaker=data.get("speaker"),
            description=data.get("description"),
            language=data.get("language", "English"),
        )


# ---------------------------------------------------------------------------
# Built-in voices — available without user-uploaded reference audio
# ---------------------------------------------------------------------------

_QWEN_CUSTOM_SPEAKERS: list[dict] = [
    {"name": "Vivian", "language": "Chinese", "description": "Bright, slightly edgy young female voice"},
    {"name": "Serena", "language": "Chinese", "description": "Warm, gentle young female voice"},
    {"name": "Uncle_Fu", "language": "Chinese", "description": "Seasoned male voice with a low, mellow timbre"},
    {"name": "Dylan", "language": "Chinese", "description": "Youthful Beijing male voice, clear and natural"},
    {"name": "Eric", "language": "Chinese", "description": "Lively Chengdu male voice, slightly husky"},
    {"name": "Ryan", "language": "English", "description": "Dynamic male voice with strong rhythmic drive"},
    {"name": "Aiden", "language": "English", "description": "Sunny American male voice with a clear midrange"},
    {"name": "Ono_Anna", "language": "Japanese", "description": "Playful Japanese female voice, light and nimble"},
    {"name": "Sohee", "language": "Korean", "description": "Warm Korean female voice with rich emotion"},
]

BUILTIN_VOICES: List[VoiceProfile] = [
    VoiceProfile(id="builtin:chatterbox_default", name="Chatterbox Default", provider="chatterbox"),
    *(
        VoiceProfile(
            id=f"builtin:qwen_{s['name'].lower()}",
            name=f"Qwen — {s['name']}",
            speaker=s["name"],
            provider="qwen",
            description=s["description"],
            language=s["language"],
        )
        for s in _QWEN_CUSTOM_SPEAKERS
    ),
]

_BUILTIN_VOICE_MAP: Dict[str, VoiceProfile] = {v.id: v for v in BUILTIN_VOICES}


def get_builtin_voice(voice_id: str) -> Optional[VoiceProfile]:
    """Look up a built-in voice by ID, or return None."""
    return _BUILTIN_VOICE_MAP.get(voice_id)


@dataclass
class ModelConfig:
    id: str
    name: str
    provider: str
    params: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(
            id=data["id"],
            name=data["name"],
            provider=data["provider"],
            params=data.get("params", {}),
        )


@dataclass
class GenerationRecord:
    """A single TTS generation event for the audio library."""

    id: str
    timestamp: float
    audio_path: str
    text: str
    model_id: str
    model_name: str
    voice_id: Optional[str]
    voice_name: str
    provider: str
    duration_sec: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "audio_path": self.audio_path,
            "text": self.text,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "voice_id": self.voice_id,
            "voice_name": self.voice_name,
            "provider": self.provider,
            "duration_sec": self.duration_sec,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationRecord":
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            audio_path=data["audio_path"],
            text=data["text"],
            model_id=data["model_id"],
            model_name=data["model_name"],
            voice_id=data.get("voice_id"),
            voice_name=data.get("voice_name", ""),
            provider=data["provider"],
            duration_sec=data.get("duration_sec"),
        )


@dataclass
class AppConfig:
    voices: Dict[str, VoiceProfile] = field(default_factory=dict)
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    phonetic_mappings: Dict[str, Dict[str, str]] = field(default_factory=dict)
    output_dir: str = "./rho_tts_output"
    device: str = "cuda"

    def to_dict(self) -> dict:
        return {
            "voices": {k: v.to_dict() for k, v in self.voices.items()},
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "phonetic_mappings": self.phonetic_mappings,
            "output_dir": self.output_dir,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        voices = {k: VoiceProfile.from_dict(v) for k, v in data.get("voices", {}).items()}
        models = {k: ModelConfig.from_dict(v) for k, v in data.get("models", {}).items()}
        return cls(
            voices=voices,
            models=models,
            phonetic_mappings=data.get("phonetic_mappings", {}),
            output_dir=data.get("output_dir", "./rho_tts_output"),
            device=data.get("device", "cuda"),
        )


def get_phonetic_key(voice_id: str, model_id: str) -> str:
    """Build the composite key used in AppConfig.phonetic_mappings."""
    return f"{voice_id}::{model_id}"


def get_config_path() -> str:
    """Resolve config file path from env var or default."""
    return os.environ.get("RHO_TTS_CONFIG", DEFAULT_CONFIG_PATH)


def load_config(path: Optional[str] = None) -> AppConfig:
    """
    Load config from JSON file.

    Returns a default AppConfig if the file doesn't exist or is invalid.
    """
    path = path or get_config_path()
    if not os.path.exists(path):
        logger.info(f"No config file at {path}, using defaults")
        return AppConfig()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded config from {path}")
        return AppConfig.from_dict(data)
    except Exception as e:
        logger.warning(f"Failed to load config from {path}: {e}, using defaults")
        return AppConfig()


def save_config(config: AppConfig, path: Optional[str] = None) -> None:
    """Persist config to JSON file, creating parent directories as needed."""
    path = path or get_config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Config saved to {path}")


def get_history_path() -> str:
    """Resolve history file path from env var or default."""
    return os.environ.get("RHO_TTS_HISTORY", DEFAULT_HISTORY_PATH)


def load_history(path: Optional[str] = None) -> List[GenerationRecord]:
    """Load generation history from JSON file.

    Returns an empty list if the file doesn't exist or is invalid.
    """
    path = path or get_history_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return [GenerationRecord.from_dict(r) for r in data]
    except Exception as e:
        logger.warning(f"Failed to load history from {path}: {e}")
        return []


def save_history(records: List[GenerationRecord], path: Optional[str] = None) -> None:
    """Persist generation history to JSON file."""
    path = path or get_history_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)


def copy_voice_audio(source_path: str, voice_id: str) -> str:
    """
    Copy a voice reference audio file into the managed voices directory.

    Returns the destination path inside ~/.rho_tts/voices/.
    """
    os.makedirs(DEFAULT_VOICES_DIR, exist_ok=True)
    ext = os.path.splitext(source_path)[1] or ".wav"
    dest = os.path.join(DEFAULT_VOICES_DIR, f"{voice_id}{ext}")
    shutil.copy2(source_path, dest)
    return dest
