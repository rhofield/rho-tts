"""Tests for Breeze's UI wiring: catalog, voice filtering, and kwargs routing."""

from unittest.mock import MagicMock, patch

import pytest

from rho_tts.ui.callbacks import PARAM_DEFAULTS, load_model_voice_params
from rho_tts.ui.config import (
    BUILTIN_VOICES,
    PROVIDER_MODELS,
    PROVIDER_ONLY_PARAMS,
    AppConfig,
    ModelConfig,
    VoiceProfile,
    get_phonetic_key,
    get_provider_model_choices,
)

_STATE = "rho_tts.ui.state"


@pytest.fixture
def mock_factory():
    with patch(f"{_STATE}.TTSFactory") as factory:
        factory.get_tts_instance.return_value = MagicMock()
        yield factory


def _model(provider="breeze", model_id="m-breeze", params=None):
    return ModelConfig(id=model_id, name="Breeze", provider=provider, params=params or {})


def _voice(**kw):
    kw.setdefault("id", "voice-1")
    kw.setdefault("name", "Test Voice")
    return VoiceProfile(**kw)


class TestBreezeCatalog:
    def test_breeze_is_a_selectable_provider(self):
        assert "breeze" in PROVIDER_MODELS

    def test_offers_cloning_and_direction_models(self):
        choices = get_provider_model_choices("breeze")
        assert len(choices) == 2
        assert any("Cloning" in c for c in choices)
        assert any("Direction" in c for c in choices)

    def test_direction_preset_actually_steers(self):
        """cfg_scale 1.0 disables steering, so the direction preset must be higher."""
        by_name = {m["display_name"]: m["defaults"] for m in PROVIDER_MODELS["breeze"]}
        cloning = next(v for k, v in by_name.items() if "Cloning" in k)
        direction = next(v for k, v in by_name.items() if "Direction" in k)
        assert cloning["cfg_scale"] == 1.0
        assert direction["cfg_scale"] > 1.0

    def test_has_a_reference_free_builtin_voice(self):
        """Voice design needs no recording — unlike every other builtin."""
        voices = [v for v in BUILTIN_VOICES if v.provider == "breeze"]
        assert len(voices) == 1
        assert voices[0].reference_audio is None


class TestBreezeParamLoading:
    def _cfg(self):
        cfg = AppConfig()
        cfg.models["m-breeze"] = ModelConfig(
            id="m-breeze", name="Breeze", provider="breeze",
            params={"instruction": "Sound weary.", "cfg_scale": 4.0},
        )
        cfg.models["m-cb"] = ModelConfig(id="m-cb", name="CB", provider="chatterbox", params={})
        return cfg

    def test_breeze_controls_flagged_visible(self):
        r = load_model_voice_params(None, "builtin:breeze_design", "m-breeze", config=self._cfg())
        instruction, cfg_scale, is_chatterbox, is_breeze = r[6], r[7], r[9], r[10]
        assert instruction == "Sound weary."
        assert cfg_scale == 4.0
        assert is_breeze is True
        assert is_chatterbox is False

    def test_non_breeze_model_hides_them_and_uses_defaults(self):
        r = load_model_voice_params(None, "builtin:breeze_design", "m-cb", config=self._cfg())
        assert r[6] == PARAM_DEFAULTS["instruction"]
        assert r[10] is False
        assert r[9] is True

    def test_missing_selection_returns_full_tuple(self):
        """The UI unpacks a fixed-width tuple; early returns must match."""
        assert len(load_model_voice_params(None, None, None, config=AppConfig())) == 11


class TestBreezeKwargsRouting:
    def test_reference_text_is_forwarded(self, mock_factory):
        """Breeze aligns reference audio against its exact transcript."""
        from rho_tts.ui.state import AppState

        state = AppState(config=AppConfig(device="cpu"))
        voice = _voice(reference_audio="/tmp/ref.wav", reference_text="the transcript")

        state.get_or_create_tts(_model(), voice)

        _, kwargs = mock_factory.get_tts_instance.call_args
        assert kwargs["reference_text"] == "the transcript"

    def test_breeze_params_reach_breeze(self, mock_factory):
        from rho_tts.ui.state import AppState

        cfg = AppConfig(device="cpu")
        voice = _voice()
        cfg.model_voice_params[get_phonetic_key(voice.id, "m-breeze")] = {
            "instruction": "Sound tired.", "cfg_scale": 4.0,
            "temperature": 1.0, "cfg_weight": 0.6,
        }
        state = AppState(config=cfg)

        state.get_or_create_tts(_model(), voice)

        _, kwargs = mock_factory.get_tts_instance.call_args
        assert kwargs["instruction"] == "Sound tired."
        assert kwargs["cfg_scale"] == 4.0
        # chatterbox-only params must not leak into Breeze's constructor
        assert "temperature" not in kwargs
        assert "cfg_weight" not in kwargs

    def test_breeze_params_do_not_leak_to_other_providers(self, mock_factory):
        """A saved param set contains every key; foreign ones must be dropped."""
        from rho_tts.ui.state import AppState

        cfg = AppConfig(device="cpu")
        voice = _voice()
        cfg.model_voice_params[get_phonetic_key(voice.id, "m-cb")] = {
            "instruction": "Sound tired.", "cfg_scale": 4.0,
            "temperature": 1.2, "cfg_weight": 0.5,
        }
        state = AppState(config=cfg)

        state.get_or_create_tts(_model(provider="chatterbox", model_id="m-cb"), voice)

        _, kwargs = mock_factory.get_tts_instance.call_args
        assert "instruction" not in kwargs
        assert "cfg_scale" not in kwargs
        assert kwargs["temperature"] == 1.2

    def test_every_provider_only_param_is_a_real_constructor_arg(self):
        """Guards against a typo silently dropping a param forever."""
        import inspect

        from rho_tts.providers.breeze import BreezeTTS

        sig = inspect.signature(BreezeTTS.__init__).parameters
        for param in PROVIDER_ONLY_PARAMS["breeze"]:
            assert param in sig, f"{param} is not a BreezeTTS constructor argument"
