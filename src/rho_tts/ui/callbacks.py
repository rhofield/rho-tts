"""
Gradio event-handler callbacks for the rho-tts web UI.

Each function here is wired to a Gradio component event in app.py.
They bridge between Gradio's string/list/DataFrame world and the
rho-tts backend (TTSFactory, BaseTTS, CancellationToken).
"""

import logging
import os
import time
import uuid
from typing import Optional

from ..cancellation import CancelledException
from .config import (
    BUILTIN_VOICES,
    AppConfig,
    ModelConfig,
    VoiceProfile,
    copy_voice_audio,
    get_builtin_voice,
    get_phonetic_key,
    get_provider_model_defaults,
)
from .state import AppState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generate tab
# ---------------------------------------------------------------------------

def generate_audio(
    state: AppState,
    model_id: str,
    voice_id: str,
    text: str,
) -> tuple[Optional[str], str]:
    """
    Run TTS generation for the given text.

    Returns:
        (output_wav_path_or_None, status_message)
    """
    if not model_id:
        return None, "Please select a model."
    if not text.strip():
        return None, "Please enter text to generate."

    model_cfg = state.config.models.get(model_id)
    voice_cfg: Optional[VoiceProfile] = None
    if voice_id:
        voice_cfg = get_builtin_voice(voice_id) or state.config.voices.get(voice_id)

    if model_cfg is None:
        return None, f"Model '{model_id}' not found in config."

    try:
        tts = state.get_or_create_tts(model_cfg, voice_cfg)
    except Exception as e:
        return None, f"Failed to initialize TTS: {e}"

    token = state.new_cancellation_token()

    os.makedirs(state.config.output_dir, exist_ok=True)
    timestamp = int(time.time())
    voice_label = voice_id or "default"
    output_path = os.path.join(state.config.output_dir, f"gen_{voice_label}_{timestamp}.wav")

    try:
        result = tts.generate_single(text, output_path, cancellation_token=token)
        if result is None:
            return None, "Generation failed — no audio produced. Check logs for details."
        return output_path, f"Generated: {os.path.basename(output_path)}"
    except CancelledException:
        return None, "Generation cancelled."
    except Exception as e:
        logger.exception("Generation error")
        return None, f"Generation error: {e}"


def cancel_generation(state: AppState) -> str:
    """Cancel the currently running generation."""
    state.cancel_generation()
    return "Cancelling..."


# ---------------------------------------------------------------------------
# Phonetic mapping
# ---------------------------------------------------------------------------

def load_phonetic_mapping(
    state: AppState,
    voice_id: str,
    model_id: str,
) -> tuple[list[list[str]], str]:
    """
    Load phonetic mappings for a voice+model pair.

    Returns:
        (rows_for_dataframe, label_string)
    """
    if not voice_id or not model_id:
        return [], ""

    key = get_phonetic_key(voice_id, model_id)
    mapping = state.config.phonetic_mappings.get(key, {})
    rows = [[word, pron] for word, pron in mapping.items()]

    voice_name = voice_id
    model_name = model_id
    if voice_id in state.config.voices:
        voice_name = state.config.voices[voice_id].name
    if model_id in state.config.models:
        model_name = state.config.models[model_id].name

    label = f"Mappings for: {voice_name} + {model_name}"
    return rows, label


def save_phonetic_mapping(
    state: AppState,
    voice_id: str,
    model_id: str,
    dataframe_data: list[list[str]],
) -> str:
    """Save edited phonetic mappings back to config and persist to disk."""
    if not voice_id or not model_id:
        return "Select a voice and model first."

    key = get_phonetic_key(voice_id, model_id)
    mapping = {}
    for row in dataframe_data:
        if len(row) >= 2 and row[0] and row[0].strip():
            mapping[row[0].strip()] = row[1].strip() if row[1] else ""

    state.config.phonetic_mappings[key] = mapping
    state.save()

    # Invalidate cached TTS so next generation picks up new mappings
    state.invalidate_tts()

    return f"Saved {len(mapping)} mapping(s)."


# ---------------------------------------------------------------------------
# Voice management
# ---------------------------------------------------------------------------

def add_voice(
    state: AppState,
    voice_id: str,
    display_name: str,
    audio_path: Optional[str],
    reference_text: str,
) -> tuple[list[list[str]], str]:
    """
    Add or update a voice profile.

    Returns:
        (updated_voices_table, status_message)
    """
    if not voice_id or not voice_id.strip():
        return _voices_table(state.config), "Voice ID is required."
    if not audio_path:
        return _voices_table(state.config), "Reference audio is required."

    voice_id = voice_id.strip()
    display_name = display_name.strip() if display_name else voice_id

    # Copy audio into managed directory
    try:
        managed_path = copy_voice_audio(audio_path, voice_id)
    except Exception as e:
        return _voices_table(state.config), f"Failed to copy audio: {e}"

    profile = VoiceProfile(
        id=voice_id,
        name=display_name,
        reference_audio=managed_path,
        reference_text=reference_text.strip() if reference_text else None,
    )
    state.config.voices[voice_id] = profile
    state.save()
    state.invalidate_tts()

    return _voices_table(state.config), f"Voice '{display_name}' saved."


def delete_voice(state: AppState, voice_id: str) -> tuple[list[list[str]], str]:
    """Delete a voice profile by ID."""
    if not voice_id or voice_id not in state.config.voices:
        return _voices_table(state.config), "Select a voice to delete."

    profile = state.config.voices.pop(voice_id)

    # Remove associated phonetic mappings
    keys_to_remove = [k for k in state.config.phonetic_mappings if k.startswith(f"{voice_id}::")]
    for k in keys_to_remove:
        del state.config.phonetic_mappings[k]

    state.save()
    state.invalidate_tts()
    return _voices_table(state.config), f"Deleted voice '{profile.name}'."


def _voices_table(config: AppConfig) -> list[list[str]]:
    """Build the voices table data for Gradio Dataframe."""
    return [
        [v.id, v.name, v.reference_audio, v.reference_text or ""]
        for v in config.voices.values()
    ]


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def add_model(
    state: AppState,
    provider: str,
    model_name: str,
    max_iterations: Optional[int],
    accent_drift_threshold: Optional[float],
    text_similarity_threshold: Optional[float],
) -> tuple[list[list[str]], str]:
    """
    Add a model configuration from the provider catalog.

    Generates a UUID as the internal model ID. Merges catalog defaults
    with any user-overridden threshold values.

    Returns:
        (updated_models_table, status_message)
    """
    if not provider:
        return _models_table(state.config), "Please select a provider."
    if not model_name:
        return _models_table(state.config), "Please select a model."

    # Check for duplicate: same provider + model name already saved
    for existing in state.config.models.values():
        if existing.provider == provider and existing.name == model_name:
            return _models_table(state.config), f"'{model_name}' is already added."

    # Start from catalog defaults, then override thresholds
    params = get_provider_model_defaults(provider, model_name)
    if max_iterations is not None and max_iterations > 0:
        params["max_iterations"] = int(max_iterations)
    if accent_drift_threshold is not None:
        params["accent_drift_threshold"] = float(accent_drift_threshold)
    if text_similarity_threshold is not None:
        params["text_similarity_threshold"] = float(text_similarity_threshold)

    model_id = uuid.uuid4().hex[:12]

    model_cfg = ModelConfig(
        id=model_id,
        name=model_name,
        provider=provider,
        params=params,
    )
    state.config.models[model_id] = model_cfg
    state.save()
    state.invalidate_tts()

    return _models_table(state.config), f"Model '{model_name}' added."


def delete_model(state: AppState, model_id: str) -> tuple[list[list[str]], str]:
    """Delete a model config by ID."""
    if not model_id or model_id not in state.config.models:
        return _models_table(state.config), "Select a model to delete."

    model_cfg = state.config.models.pop(model_id)

    # Remove associated phonetic mappings
    keys_to_remove = [k for k in state.config.phonetic_mappings if k.endswith(f"::{model_id}")]
    for k in keys_to_remove:
        del state.config.phonetic_mappings[k]

    state.save()
    state.invalidate_tts()
    return _models_table(state.config), f"Deleted model '{model_cfg.name}'."


def _models_table(config: AppConfig) -> list[list[str]]:
    """Build the models table data for Gradio Dataframe."""
    return [
        [m.name, m.provider, str(m.params)]
        for m in config.models.values()
    ]


# ---------------------------------------------------------------------------
# Dropdown helpers
# ---------------------------------------------------------------------------

def voice_choices(config: AppConfig) -> list[tuple[str, str]]:
    """Return (display_label, value) pairs for voice dropdown.

    Built-in voices appear first, followed by user-created voices.
    """
    builtin = [(v.name, v.id) for v in BUILTIN_VOICES]
    user = [(v.name, v.id) for v in config.voices.values()]
    return builtin + user


def voice_choices_for_model(
    config: AppConfig, model_id: Optional[str]
) -> list[tuple[str, str]]:
    """Return voice choices filtered by the selected model's provider.

    Built-in voices are filtered to only those matching the model's
    provider. User-created voices (provider=None) are always included
    since they carry reference audio usable by any provider.
    """
    if not model_id:
        return voice_choices(config)

    model_cfg = config.models.get(model_id)
    if model_cfg is None:
        return voice_choices(config)

    provider = model_cfg.provider
    builtin = [(v.name, v.id) for v in BUILTIN_VOICES if v.provider == provider]
    user = [(v.name, v.id) for v in config.voices.values()]
    return builtin + user


def model_choices(config: AppConfig) -> list[tuple[str, str]]:
    """Return (display_label, value) pairs for model dropdown."""
    return [(m.name, m.id) for m in config.models.values()]
