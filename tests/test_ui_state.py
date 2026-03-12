"""Tests for rho_tts.ui.state.AppState runtime state management."""

import pytest
from unittest.mock import MagicMock, patch, call

from rho_tts.ui.config import (
    AppConfig,
    GenerationRecord,
    ModelConfig,
    VoiceProfile,
    get_phonetic_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_config(provider="chatterbox", model_id="model-1", params=None):
    return ModelConfig(
        id=model_id, name="Test Model", provider=provider, params=params or {}
    )


def _make_voice_profile(
    voice_id="voice-1",
    reference_audio=None,
    reference_text=None,
    speaker=None,
    language="English",
):
    return VoiceProfile(
        id=voice_id,
        name="Test Voice",
        reference_audio=reference_audio,
        reference_text=reference_text,
        speaker=speaker,
        language=language,
    )


def _make_record(record_id="rec-1"):
    return GenerationRecord(
        id=record_id,
        timestamp=1000.0,
        audio_path="/tmp/audio.wav",
        text="Hello",
        model_id="m1",
        model_name="Model",
        voice_id="v1",
        voice_name="Voice",
        provider="chatterbox",
    )


# ---------------------------------------------------------------------------
# Patch targets — all live inside the state module's namespace
# ---------------------------------------------------------------------------

_STATE = "rho_tts.ui.state"
_PATCH_TORCH = f"{_STATE}.torch"
_PATCH_FACTORY = f"{_STATE}.TTSFactory"
_PATCH_LOAD_CONFIG = f"{_STATE}.load_config"
_PATCH_SAVE_CONFIG = f"{_STATE}.save_config"
_PATCH_LOAD_HISTORY = f"{_STATE}.load_history"
_PATCH_SAVE_HISTORY = f"{_STATE}.save_history"


@pytest.fixture(autouse=True)
def _mock_torch():
    """Prevent real CUDA calls in every test."""
    with patch(f"{_PATCH_TORCH}.cuda") as mock_cuda:
        mock_cuda.is_available.return_value = False
        yield mock_cuda


@pytest.fixture
def mock_factory():
    with patch(_PATCH_FACTORY) as factory:
        factory.get_tts_instance.return_value = MagicMock()
        yield factory


@pytest.fixture
def mock_load_history():
    with patch(_PATCH_LOAD_HISTORY, return_value=[]) as m:
        yield m


@pytest.fixture
def mock_save_history():
    with patch(_PATCH_SAVE_HISTORY) as m:
        yield m


# Import AppState after fixtures are defined so the module-level `import torch`
# doesn't blow up in environments without torch — the autouse fixture covers
# runtime calls, and we rely on torch being installed (it is in the dev venv).
from rho_tts.ui.state import AppState


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAppStateConstructor:
    """__init__: default config, custom config, config_path."""

    def test_default_config_calls_load_config(self):
        with patch(_PATCH_LOAD_CONFIG, return_value=AppConfig()) as mock_lc:
            state = AppState()
            mock_lc.assert_called_once_with(None)
            assert isinstance(state.config, AppConfig)

    def test_custom_config_skips_load(self):
        cfg = AppConfig(device="cpu")
        with patch(_PATCH_LOAD_CONFIG) as mock_lc:
            state = AppState(config=cfg)
            mock_lc.assert_not_called()
            assert state.config is cfg

    def test_config_path_forwarded(self):
        with patch(_PATCH_LOAD_CONFIG, return_value=AppConfig()) as mock_lc:
            state = AppState(config_path="/tmp/custom.json")
            mock_lc.assert_called_once_with("/tmp/custom.json")
            assert state._config_path == "/tmp/custom.json"

    def test_initial_state_is_clean(self):
        cfg = AppConfig()
        state = AppState(config=cfg)
        assert state._active_tts is None
        assert state._cache_key is None
        assert state.cancellation_token is None
        assert state._history is None


class TestSave:
    def test_save_calls_save_config(self):
        cfg = AppConfig(device="cpu")
        state = AppState(config=cfg, config_path="/tmp/cfg.json")
        with patch(_PATCH_SAVE_CONFIG) as mock_sc:
            state.save()
            mock_sc.assert_called_once_with(cfg, "/tmp/cfg.json")


class TestGetOrCreateTTS:
    """get_or_create_tts: caching, teardown, kwargs forwarding."""

    def test_creates_new_instance(self, mock_factory):
        state = AppState(config=AppConfig(device="cpu"))
        model = _make_model_config()
        tts = state.get_or_create_tts(model)

        mock_factory.get_tts_instance.assert_called_once()
        assert tts is mock_factory.get_tts_instance.return_value

    def test_returns_cached_when_same_key(self, mock_factory):
        state = AppState(config=AppConfig(device="cpu"))
        model = _make_model_config()
        voice = _make_voice_profile()

        first = state.get_or_create_tts(model, voice)
        second = state.get_or_create_tts(model, voice)

        assert first is second
        assert mock_factory.get_tts_instance.call_count == 1

    def test_recreates_when_key_changes(self, mock_factory):
        state = AppState(config=AppConfig(device="cpu"))
        model = _make_model_config()
        voice_a = _make_voice_profile(voice_id="a")
        voice_b = _make_voice_profile(voice_id="b")

        first = state.get_or_create_tts(model, voice_a)
        second = state.get_or_create_tts(model, voice_b)

        assert mock_factory.get_tts_instance.call_count == 2
        # Old instance should have been shut down
        first.shutdown.assert_called_once()

    def test_teardown_calls_shutdown_on_old_instance(self, mock_factory):
        state = AppState(config=AppConfig(device="cpu"))
        model_a = _make_model_config(model_id="m-a")
        model_b = _make_model_config(model_id="m-b")

        old_tts = MagicMock()
        mock_factory.get_tts_instance.side_effect = [old_tts, MagicMock()]

        state.get_or_create_tts(model_a)
        state.get_or_create_tts(model_b)

        old_tts.shutdown.assert_called_once()

    def test_no_voice_profile_passes_device_and_model_params(self, mock_factory):
        cfg = AppConfig(device="cpu")
        state = AppState(config=cfg)
        model = _make_model_config(params={"model_path": "some/path"})

        state.get_or_create_tts(model)

        mock_factory.get_tts_instance.assert_called_once_with(
            provider="chatterbox",
            device="cpu",
            model_path="some/path",
        )

    def test_voice_profile_passes_reference_audio(self, mock_factory):
        state = AppState(config=AppConfig(device="cpu"))
        model = _make_model_config()
        voice = _make_voice_profile(reference_audio="/tmp/ref.wav")

        state.get_or_create_tts(model, voice)

        _, kwargs = mock_factory.get_tts_instance.call_args
        assert kwargs["reference_audio"] == "/tmp/ref.wav"

    def test_voice_profile_passes_speaker(self, mock_factory):
        state = AppState(config=AppConfig(device="cpu"))
        model = _make_model_config()
        voice = _make_voice_profile(speaker="Ryan")

        state.get_or_create_tts(model, voice)

        _, kwargs = mock_factory.get_tts_instance.call_args
        assert kwargs["speaker"] == "Ryan"

    def test_qwen_passes_language(self, mock_factory):
        state = AppState(config=AppConfig(device="cpu"))
        model = _make_model_config(provider="qwen")
        voice = _make_voice_profile(language="Chinese")

        state.get_or_create_tts(model, voice)

        _, kwargs = mock_factory.get_tts_instance.call_args
        assert kwargs["language"] == "Chinese"

    def test_non_qwen_omits_language(self, mock_factory):
        state = AppState(config=AppConfig(device="cpu"))
        model = _make_model_config(provider="chatterbox")
        voice = _make_voice_profile(language="Chinese")

        state.get_or_create_tts(model, voice)

        _, kwargs = mock_factory.get_tts_instance.call_args
        assert "language" not in kwargs

    def test_qwen_passes_reference_text(self, mock_factory):
        state = AppState(config=AppConfig(device="cpu"))
        model = _make_model_config(provider="qwen")
        voice = _make_voice_profile(reference_text="Hello world")

        state.get_or_create_tts(model, voice)

        _, kwargs = mock_factory.get_tts_instance.call_args
        assert kwargs["reference_text"] == "Hello world"

    def test_non_qwen_omits_reference_text(self, mock_factory):
        state = AppState(config=AppConfig(device="cpu"))
        model = _make_model_config(provider="chatterbox")
        voice = _make_voice_profile(reference_text="Hello world")

        state.get_or_create_tts(model, voice)

        _, kwargs = mock_factory.get_tts_instance.call_args
        assert "reference_text" not in kwargs

    def test_phonetic_mapping_passed_when_present(self, mock_factory):
        voice = _make_voice_profile()
        model = _make_model_config()
        key = get_phonetic_key(voice.id, model.id)
        cfg = AppConfig(
            device="cpu",
            phonetic_mappings={key: {"th": "d"}},
        )
        state = AppState(config=cfg)

        state.get_or_create_tts(model, voice)

        _, kwargs = mock_factory.get_tts_instance.call_args
        assert kwargs["phonetic_mapping"] == {"th": "d"}

    def test_chatterbox_only_params_passed_for_chatterbox(self, mock_factory):
        voice = _make_voice_profile()
        model = _make_model_config(provider="chatterbox")
        key = get_phonetic_key(voice.id, model.id)
        cfg = AppConfig(
            device="cpu",
            model_voice_params={key: {"temperature": 0.8, "cfg_weight": 3.0, "speed": 1.2}},
        )
        state = AppState(config=cfg)

        state.get_or_create_tts(model, voice)

        _, kwargs = mock_factory.get_tts_instance.call_args
        assert kwargs["temperature"] == 0.8
        assert kwargs["cfg_weight"] == 3.0
        assert kwargs["speed"] == 1.2

    def test_chatterbox_only_params_filtered_for_other_providers(self, mock_factory):
        voice = _make_voice_profile()
        model = _make_model_config(provider="qwen")
        key = get_phonetic_key(voice.id, model.id)
        cfg = AppConfig(
            device="cpu",
            model_voice_params={key: {"temperature": 0.8, "cfg_weight": 3.0, "speed": 1.2}},
        )
        state = AppState(config=cfg)

        state.get_or_create_tts(model, voice)

        _, kwargs = mock_factory.get_tts_instance.call_args
        assert "temperature" not in kwargs
        assert "cfg_weight" not in kwargs
        # Non-chatterbox-only params still pass through
        assert kwargs["speed"] == 1.2

    def test_voice_id_set_on_instance(self, mock_factory):
        state = AppState(config=AppConfig(device="cpu"))
        model = _make_model_config()
        voice = _make_voice_profile(voice_id="my-voice")

        tts = state.get_or_create_tts(model, voice)
        assert tts.voice_id == "my-voice"

    def test_voice_id_none_when_no_profile(self, mock_factory):
        state = AppState(config=AppConfig(device="cpu"))
        model = _make_model_config()

        tts = state.get_or_create_tts(model)
        assert tts.voice_id is None


class TestInvalidateTTS:
    def test_clears_active_tts_and_cache_key(self, mock_factory):
        state = AppState(config=AppConfig(device="cpu"))
        model = _make_model_config()
        tts = state.get_or_create_tts(model)

        state.invalidate_tts()

        assert state._active_tts is None
        assert state._cache_key is None
        tts.shutdown.assert_called_once()

    def test_noop_when_no_active_tts(self):
        state = AppState(config=AppConfig(device="cpu"))
        # Should not raise
        state.invalidate_tts()
        assert state._active_tts is None

    def test_cuda_cache_cleared_when_available(self, mock_factory, _mock_torch):
        _mock_torch.is_available.return_value = True
        state = AppState(config=AppConfig(device="cpu"))
        state.get_or_create_tts(_make_model_config())

        state.invalidate_tts()

        _mock_torch.empty_cache.assert_called()


class TestHistory:
    def test_lazy_loads_on_first_access(self, mock_load_history):
        state = AppState(config=AppConfig())
        assert state._history is None

        _ = state.history

        mock_load_history.assert_called_once()
        assert state._history is not None

    def test_second_access_does_not_reload(self, mock_load_history):
        state = AppState(config=AppConfig())

        _ = state.history
        _ = state.history

        mock_load_history.assert_called_once()


class TestAddGenerationRecord:
    def test_appends_and_saves(self, mock_load_history, mock_save_history):
        state = AppState(config=AppConfig())
        record = _make_record("r1")

        state.add_generation_record(record)

        assert record in state._history
        mock_save_history.assert_called_once_with([record])


class TestDeleteGenerationRecord:
    def test_removes_existing_record(self, mock_load_history, mock_save_history):
        record = _make_record("r1")
        mock_load_history.return_value = [record]
        state = AppState(config=AppConfig())

        result = state.delete_generation_record("r1")

        assert result is True
        assert len(state._history) == 0
        mock_save_history.assert_called_once_with([])

    def test_returns_false_for_unknown_id(self, mock_load_history, mock_save_history):
        record = _make_record("r1")
        mock_load_history.return_value = [record]
        state = AppState(config=AppConfig())

        result = state.delete_generation_record("nonexistent")

        assert result is False
        mock_save_history.assert_not_called()


class TestClearHistory:
    def test_clears_and_returns_count(self, mock_load_history, mock_save_history):
        records = [_make_record("r1"), _make_record("r2"), _make_record("r3")]
        mock_load_history.return_value = list(records)
        state = AppState(config=AppConfig())

        count = state.clear_history()

        assert count == 3
        assert state._history == []
        mock_save_history.assert_called_once_with([])

    def test_returns_zero_when_already_empty(self, mock_load_history, mock_save_history):
        state = AppState(config=AppConfig())

        count = state.clear_history()

        assert count == 0


class TestCancellation:
    def test_new_cancellation_token_creates_fresh(self):
        state = AppState(config=AppConfig())

        token = state.new_cancellation_token()

        assert token is state.cancellation_token
        assert not token.is_cancelled()

    def test_new_token_replaces_old(self):
        state = AppState(config=AppConfig())

        old = state.new_cancellation_token()
        new = state.new_cancellation_token()

        assert new is not old
        assert state.cancellation_token is new

    def test_cancel_generation_cancels_token(self):
        state = AppState(config=AppConfig())
        token = state.new_cancellation_token()

        state.cancel_generation()

        assert token.is_cancelled()

    def test_cancel_generation_noop_when_no_token(self):
        state = AppState(config=AppConfig())
        assert state.cancellation_token is None
        # Should not raise
        state.cancel_generation()
