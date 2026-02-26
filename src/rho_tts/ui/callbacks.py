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
    is_model_cached,
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

    # Validate: Qwen Base models need reference audio (voice cloning).
    if (
        model_cfg.provider == "qwen"
        and not _is_custom_voice_model(model_cfg)
        and (voice_cfg is None or voice_cfg.reference_audio is None)
    ):
        return None, (
            "This Qwen Base model requires a voice with reference audio. "
            "Add a custom voice with reference audio, or switch to a CustomVoice model."
        )

    # Validate: Qwen CustomVoice models need a named speaker, not reference audio.
    if (
        model_cfg.provider == "qwen"
        and _is_custom_voice_model(model_cfg)
        and voice_cfg is not None
        and not voice_cfg.speaker
    ):
        return None, (
            "This CustomVoice model requires a built-in speaker voice (e.g. Vivian, Ryan). "
            "Voice cloning with reference audio is only supported on Base models."
        )

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
    display_name: str,
    audio_path: Optional[str],
    reference_text: str,
) -> tuple[list[list[str]], str]:
    """
    Add or update a voice profile.

    Returns:
        (updated_voices_table, status_message)
    """
    if not display_name or not display_name.strip():
        return _voices_table(state.config), "Display name is required."
    if not audio_path:
        return _voices_table(state.config), "Reference audio is required."

    voice_id = uuid.uuid4().hex[:12]
    display_name = display_name.strip()

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
    custom_name: Optional[str] = None,
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

    # Start from catalog defaults, then override thresholds
    params = get_provider_model_defaults(provider, model_name)

    # Identity-based duplicate check: compare model_path (Qwen) or implementation (Chatterbox)
    identity_key = params.get("model_path") or params.get("implementation")
    if identity_key:
        for existing in state.config.models.values():
            existing_identity = existing.params.get("model_path") or existing.params.get("implementation")
            if existing.provider == provider and existing_identity == identity_key:
                return _models_table(state.config), f"This model is already added as '{existing.name}'."

    if max_iterations is not None and max_iterations > 0:
        params["max_iterations"] = int(max_iterations)
    if accent_drift_threshold is not None:
        params["accent_drift_threshold"] = float(accent_drift_threshold)
    if text_similarity_threshold is not None:
        params["text_similarity_threshold"] = float(text_similarity_threshold)

    display_name = custom_name.strip() if custom_name and custom_name.strip() else model_name
    model_id = uuid.uuid4().hex[:12]

    model_cfg = ModelConfig(
        id=model_id,
        name=display_name,
        provider=provider,
        params=params,
    )
    state.config.models[model_id] = model_cfg
    state.save()
    state.invalidate_tts()

    return _models_table(state.config), f"Model '{display_name}' added."


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


def load_model_for_edit(
    state: AppState, model_id: str,
) -> tuple[str, int, float, float]:
    """
    Load a saved model's current values into the edit form.

    Returns:
        (name, max_iterations, accent_drift_threshold, text_similarity_threshold)
    """
    if not model_id or model_id not in state.config.models:
        return "", 10, 0.17, 0.85

    cfg = state.config.models[model_id]
    return (
        cfg.name,
        cfg.params.get("max_iterations", 10),
        cfg.params.get("accent_drift_threshold", 0.17),
        cfg.params.get("text_similarity_threshold", 0.85),
    )


def edit_model(
    state: AppState,
    model_id: str,
    new_name: Optional[str],
    max_iterations: Optional[int],
    accent_drift_threshold: Optional[float],
    text_similarity_threshold: Optional[float],
) -> tuple[list[list[str]], str]:
    """
    Update an existing saved model's name and parameters.

    Returns:
        (updated_models_table, status_message)
    """
    if not model_id or model_id not in state.config.models:
        return _models_table(state.config), "Select a model to edit."

    cfg = state.config.models[model_id]

    if new_name and new_name.strip():
        cfg.name = new_name.strip()

    if max_iterations is not None and max_iterations > 0:
        cfg.params["max_iterations"] = int(max_iterations)
    if accent_drift_threshold is not None:
        cfg.params["accent_drift_threshold"] = float(accent_drift_threshold)
    if text_similarity_threshold is not None:
        cfg.params["text_similarity_threshold"] = float(text_similarity_threshold)

    state.save()
    state.invalidate_tts()

    return _models_table(state.config), f"Model '{cfg.name}' updated."


def download_model(provider: str, model_name: str) -> tuple[str, dict]:
    """Download a HuggingFace model to the local cache.

    Returns:
        (status_message, download_button_update)
    """
    import gradio as gr

    defaults = get_provider_model_defaults(provider, model_name)
    model_path = defaults.get("model_path")
    if not model_path:
        return "No download needed.", gr.Button(visible=False)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_path)
        return "Downloaded", gr.Button(visible=False)
    except Exception as e:
        return f"Download failed: {e}", gr.Button(visible=True)


def _models_table(config: AppConfig) -> list[list[str]]:
    """Build the models table data for Gradio Dataframe."""
    return [
        [m.name, m.provider, str(m.params)]
        for m in config.models.values()
    ]


# ---------------------------------------------------------------------------
# Dropdown helpers
# ---------------------------------------------------------------------------

def _voice_display_label(v: VoiceProfile) -> str:
    """Build a display label for a voice, appending its description if present."""
    if v.description:
        return f"{v.name} — {v.description}"
    return v.name


def voice_choices(config: AppConfig) -> list[tuple[str, str]]:
    """Return (display_label, value) pairs for voice dropdown.

    Built-in voices appear first, followed by user-created voices.
    """
    builtin = [(_voice_display_label(v), v.id) for v in BUILTIN_VOICES]
    user = [(_voice_display_label(v), v.id) for v in config.voices.values()]
    return builtin + user


def user_voice_choices(config: AppConfig) -> list[tuple[str, str]]:
    """Return (display_label, value) pairs for user-created voices only."""
    return [(_voice_display_label(v), v.id) for v in config.voices.values()]


def _is_custom_voice_model(model_cfg: ModelConfig) -> bool:
    """Check whether a model config refers to a Qwen CustomVoice model."""
    model_path = model_cfg.params.get("model_path", "")
    return "CustomVoice" in model_path


def status_for_model_voice(
    config: AppConfig, model_id: Optional[str], voice_choices: list
) -> str:
    """Return a proactive status message when no compatible voices exist."""
    if not model_id or voice_choices:
        return ""
    model_cfg = config.models.get(model_id)
    if model_cfg is None:
        return ""
    if model_cfg.provider == "qwen" and not _is_custom_voice_model(model_cfg):
        return (
            "This Qwen Base model requires voice cloning — add a custom voice "
            "with reference audio in the Voices tab, or switch to a CustomVoice model."
        )
    if model_cfg.provider == "qwen" and _is_custom_voice_model(model_cfg):
        return (
            "This CustomVoice model requires a built-in speaker voice (e.g. Vivian, Ryan). "
            "Voice cloning with reference audio is only supported on Base models."
        )
    return ""


def voice_choices_for_model(
    config: AppConfig, model_id: Optional[str]
) -> list[tuple[str, str]]:
    """Return voice choices filtered by the selected model's provider.

    Built-in voices are filtered to only those matching the model's
    provider.  For Qwen Base models, built-in speaker voices are hidden
    (they require a CustomVoice model).  For Qwen CustomVoice models,
    user-created voice-cloning voices are hidden (they require a Base
    model).
    """
    if not model_id:
        return voice_choices(config)

    model_cfg = config.models.get(model_id)
    if model_cfg is None:
        return voice_choices(config)

    provider = model_cfg.provider
    is_custom = provider == "qwen" and _is_custom_voice_model(model_cfg)
    is_qwen_base = provider == "qwen" and not is_custom

    # Qwen Base models only support voice cloning via reference audio,
    # so hide built-in named-speaker voices (they have no reference audio).
    if is_qwen_base:
        builtin = []
    else:
        builtin = [(_voice_display_label(v), v.id) for v in BUILTIN_VOICES if v.provider == provider]

    # Qwen CustomVoice models only support named speakers, so hide
    # user-created voices that rely on reference audio (voice cloning).
    if is_custom:
        user = [(_voice_display_label(v), v.id) for v in config.voices.values() if v.speaker]
    else:
        user = [(_voice_display_label(v), v.id) for v in config.voices.values()]

    return builtin + user


def model_choices(config: AppConfig) -> list[tuple[str, str]]:
    """Return (display_label, value) pairs for model dropdown."""
    return [(m.name, m.id) for m in config.models.values()]
