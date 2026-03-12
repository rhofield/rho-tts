"""Tests for the rho_tts.ui.config module."""

import json
import os
from unittest.mock import patch

import pytest

from rho_tts.ui.config import (
    BUILTIN_VOICES,
    DEFAULT_CONFIG_PATH,
    DEFAULT_HISTORY_PATH,
    DEFAULT_VOICES_DIR,
    PROVIDER_MODELS,
    AppConfig,
    GenerationRecord,
    ModelConfig,
    VoiceProfile,
    copy_voice_audio,
    get_builtin_voice,
    get_config_path,
    get_history_path,
    get_phonetic_key,
    get_provider_model_choices,
    get_provider_model_defaults,
    is_model_cached,
    load_config,
    load_history,
    save_config,
    save_history,
)


# ---------------------------------------------------------------------------
# VoiceProfile
# ---------------------------------------------------------------------------


class TestVoiceProfile:
    def test_to_dict_from_dict_roundtrip(self):
        vp = VoiceProfile(
            id="v1",
            name="Test Voice",
            reference_audio="/tmp/ref.wav",
            reference_text="Hello world",
            speaker="Alice",
            description="A test voice",
            language="French",
        )
        d = vp.to_dict()
        restored = VoiceProfile.from_dict(d)
        assert restored.id == vp.id
        assert restored.name == vp.name
        assert restored.reference_audio == vp.reference_audio
        assert restored.reference_text == vp.reference_text
        assert restored.speaker == vp.speaker
        assert restored.description == vp.description
        assert restored.language == vp.language

    def test_optional_fields_excluded_when_none(self):
        vp = VoiceProfile(id="v2", name="Minimal")
        d = vp.to_dict()
        assert "speaker" not in d
        assert "description" not in d
        # language defaults to English, which is also excluded
        assert "language" not in d

    def test_language_default_is_english(self):
        vp = VoiceProfile(id="v3", name="Default Lang")
        assert vp.language == "English"

    def test_language_included_when_non_english(self):
        vp = VoiceProfile(id="v4", name="JP Voice", language="Japanese")
        d = vp.to_dict()
        assert d["language"] == "Japanese"

    def test_from_dict_missing_optionals(self):
        data = {"id": "v5", "name": "Bare"}
        vp = VoiceProfile.from_dict(data)
        assert vp.reference_audio is None
        assert vp.reference_text is None
        assert vp.speaker is None
        assert vp.description is None
        assert vp.language == "English"

    def test_reference_audio_and_text_always_in_to_dict(self):
        vp = VoiceProfile(id="v6", name="Has Nones")
        d = vp.to_dict()
        assert "reference_audio" in d
        assert "reference_text" in d
        assert d["reference_audio"] is None
        assert d["reference_text"] is None


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_to_dict_from_dict_roundtrip(self):
        mc = ModelConfig(
            id="m1",
            name="Test Model",
            provider="qwen",
            params={"model_path": "org/repo", "max_iterations": 5},
        )
        d = mc.to_dict()
        restored = ModelConfig.from_dict(d)
        assert restored.id == mc.id
        assert restored.name == mc.name
        assert restored.provider == mc.provider
        assert restored.params == mc.params

    def test_default_params_empty_dict(self):
        mc = ModelConfig(id="m2", name="No Params", provider="chatterbox")
        assert mc.params == {}

    def test_from_dict_missing_params(self):
        data = {"id": "m3", "name": "Bare Model", "provider": "qwen"}
        mc = ModelConfig.from_dict(data)
        assert mc.params == {}


# ---------------------------------------------------------------------------
# GenerationRecord
# ---------------------------------------------------------------------------


class TestGenerationRecord:
    def _make_record(self, **overrides):
        defaults = dict(
            id="rec1",
            timestamp=1700000000.0,
            audio_path="/out/rec1.wav",
            text="Hello world",
            model_id="m1",
            model_name="Test Model",
            voice_id="v1",
            voice_name="Test Voice",
            provider="qwen",
            duration_sec=2.5,
            format="wav",
        )
        defaults.update(overrides)
        return GenerationRecord(**defaults)

    def test_to_dict_from_dict_roundtrip(self):
        rec = self._make_record()
        d = rec.to_dict()
        restored = GenerationRecord.from_dict(d)
        assert restored.id == rec.id
        assert restored.timestamp == rec.timestamp
        assert restored.audio_path == rec.audio_path
        assert restored.text == rec.text
        assert restored.model_id == rec.model_id
        assert restored.model_name == rec.model_name
        assert restored.voice_id == rec.voice_id
        assert restored.voice_name == rec.voice_name
        assert restored.provider == rec.provider
        assert restored.duration_sec == rec.duration_sec
        assert restored.format == rec.format

    def test_optional_fields_defaults(self):
        rec = GenerationRecord(
            id="rec2",
            timestamp=1700000001.0,
            audio_path="/out/rec2.wav",
            text="Test",
            model_id="m1",
            model_name="M",
            voice_id=None,
            voice_name="V",
            provider="chatterbox",
        )
        assert rec.duration_sec is None
        assert rec.format == "wav"

    def test_from_dict_missing_optionals(self):
        data = {
            "id": "rec3",
            "timestamp": 1700000002.0,
            "audio_path": "/out/rec3.wav",
            "text": "Minimal",
            "model_id": "m1",
            "model_name": "M",
            "provider": "qwen",
        }
        rec = GenerationRecord.from_dict(data)
        assert rec.voice_id is None
        assert rec.voice_name == ""
        assert rec.duration_sec is None
        assert rec.format == "wav"


# ---------------------------------------------------------------------------
# AppConfig
# ---------------------------------------------------------------------------


class TestAppConfig:
    def test_to_dict_from_dict_roundtrip(self):
        voice = VoiceProfile(id="v1", name="V1", reference_audio="/ref.wav")
        model = ModelConfig(id="m1", name="M1", provider="qwen", params={"x": 1})
        cfg = AppConfig(
            voices={"v1": voice},
            models={"m1": model},
            phonetic_mappings={"v1::m1": {"hello": "heh-low"}},
            model_voice_params={"v1::m1": {"speed": 1.0}},
            output_dir="/custom/out",
            device="cpu",
        )
        d = cfg.to_dict()
        restored = AppConfig.from_dict(d)
        assert "v1" in restored.voices
        assert restored.voices["v1"].name == "V1"
        assert "m1" in restored.models
        assert restored.models["m1"].provider == "qwen"
        assert restored.phonetic_mappings == cfg.phonetic_mappings
        assert restored.model_voice_params == cfg.model_voice_params
        assert restored.output_dir == "/custom/out"
        assert restored.device == "cpu"

    def test_empty_config(self):
        cfg = AppConfig()
        assert cfg.voices == {}
        assert cfg.models == {}
        assert cfg.phonetic_mappings == {}
        assert cfg.model_voice_params == {}
        assert cfg.output_dir == "./rho_tts_output"
        assert cfg.device == "cuda"

    def test_from_dict_empty(self):
        cfg = AppConfig.from_dict({})
        assert cfg.voices == {}
        assert cfg.models == {}
        assert cfg.output_dir == "./rho_tts_output"
        assert cfg.device == "cuda"

    def test_nested_voices_and_models(self):
        v1 = VoiceProfile(id="v1", name="Voice One")
        v2 = VoiceProfile(id="v2", name="Voice Two", language="Korean")
        m1 = ModelConfig(id="m1", name="Model A", provider="qwen")
        m2 = ModelConfig(id="m2", name="Model B", provider="chatterbox")
        cfg = AppConfig(
            voices={"v1": v1, "v2": v2},
            models={"m1": m1, "m2": m2},
        )
        d = cfg.to_dict()
        restored = AppConfig.from_dict(d)
        assert len(restored.voices) == 2
        assert len(restored.models) == 2
        assert restored.voices["v2"].language == "Korean"
        assert restored.models["m2"].provider == "chatterbox"


# ---------------------------------------------------------------------------
# get_provider_model_choices / get_provider_model_defaults
# ---------------------------------------------------------------------------


class TestProviderModels:
    def test_choices_known_provider(self):
        choices = get_provider_model_choices("qwen")
        assert isinstance(choices, list)
        assert len(choices) > 0
        for c in choices:
            assert isinstance(c, str)

    def test_choices_unknown_provider(self):
        assert get_provider_model_choices("nonexistent") == []

    def test_defaults_known_model(self):
        # Pick the first qwen model display name
        display_name = PROVIDER_MODELS["qwen"][0]["display_name"]
        defaults = get_provider_model_defaults("qwen", display_name)
        assert isinstance(defaults, dict)
        assert len(defaults) > 0
        assert "model_path" in defaults

    def test_defaults_unknown_model(self):
        assert get_provider_model_defaults("qwen", "No Such Model") == {}

    def test_defaults_unknown_provider(self):
        assert get_provider_model_defaults("nonexistent", "anything") == {}

    def test_defaults_returns_copy(self):
        """Mutating the returned dict must not affect the catalog."""
        display_name = PROVIDER_MODELS["chatterbox"][0]["display_name"]
        d1 = get_provider_model_defaults("chatterbox", display_name)
        d1["extra_key"] = "injected"
        d2 = get_provider_model_defaults("chatterbox", display_name)
        assert "extra_key" not in d2


# ---------------------------------------------------------------------------
# get_builtin_voice
# ---------------------------------------------------------------------------


class TestGetBuiltinVoice:
    def test_valid_id(self):
        voice = get_builtin_voice("builtin:chatterbox_default")
        assert voice is not None
        assert voice.name == "Chatterbox Default"
        assert voice.provider == "chatterbox"

    def test_invalid_id(self):
        assert get_builtin_voice("nonexistent_voice") is None

    def test_qwen_builtin(self):
        voice = get_builtin_voice("builtin:qwen_ryan")
        assert voice is not None
        assert voice.speaker == "Ryan"
        assert voice.language == "English"

    def test_builtin_voices_list_populated(self):
        assert len(BUILTIN_VOICES) > 0


# ---------------------------------------------------------------------------
# get_phonetic_key
# ---------------------------------------------------------------------------


class TestGetPhoneticKey:
    def test_composite_key_format(self):
        assert get_phonetic_key("voice_1", "model_a") == "voice_1::model_a"

    def test_key_with_special_characters(self):
        assert get_phonetic_key("builtin:qwen_ryan", "m:1") == "builtin:qwen_ryan::m:1"


# ---------------------------------------------------------------------------
# get_config_path / get_history_path
# ---------------------------------------------------------------------------


class TestPathResolution:
    def test_get_config_path_default(self):
        with patch.dict(os.environ, {}, clear=False):
            # Remove the env var if set
            env = os.environ.copy()
            env.pop("RHO_TTS_CONFIG", None)
            with patch.dict(os.environ, env, clear=True):
                assert get_config_path() == DEFAULT_CONFIG_PATH

    def test_get_config_path_env_override(self):
        with patch.dict(os.environ, {"RHO_TTS_CONFIG": "/custom/config.json"}):
            assert get_config_path() == "/custom/config.json"

    def test_get_history_path_default(self):
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("RHO_TTS_HISTORY", None)
            with patch.dict(os.environ, env, clear=True):
                assert get_history_path() == DEFAULT_HISTORY_PATH

    def test_get_history_path_env_override(self):
        with patch.dict(os.environ, {"RHO_TTS_HISTORY": "/custom/history.json"}):
            assert get_history_path() == "/custom/history.json"


# ---------------------------------------------------------------------------
# load_config / save_config
# ---------------------------------------------------------------------------


class TestConfigPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "config.json")
        voice = VoiceProfile(id="v1", name="Voice 1", reference_audio="/ref.wav")
        model = ModelConfig(id="m1", name="Model 1", provider="qwen", params={"a": 1})
        cfg = AppConfig(
            voices={"v1": voice},
            models={"m1": model},
            phonetic_mappings={"v1::m1": {"hi": "hai"}},
            output_dir="/my/output",
            device="cpu",
        )
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded.voices["v1"].name == "Voice 1"
        assert loaded.models["m1"].params == {"a": 1}
        assert loaded.phonetic_mappings == {"v1::m1": {"hi": "hai"}}
        assert loaded.output_dir == "/my/output"
        assert loaded.device == "cpu"

    def test_load_missing_file_returns_default(self, tmp_path):
        path = str(tmp_path / "does_not_exist.json")
        cfg = load_config(path)
        assert isinstance(cfg, AppConfig)
        assert cfg.voices == {}
        assert cfg.models == {}

    def test_load_invalid_json_returns_default(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("{not valid json!!!")
        cfg = load_config(path)
        assert isinstance(cfg, AppConfig)
        assert cfg.voices == {}

    def test_save_creates_parent_directories(self, tmp_path):
        path = str(tmp_path / "nested" / "deep" / "config.json")
        save_config(AppConfig(), path)
        assert os.path.isfile(path)

    def test_saved_file_is_valid_json(self, tmp_path):
        path = str(tmp_path / "config.json")
        save_config(AppConfig(), path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "voices" in data
        assert "models" in data


# ---------------------------------------------------------------------------
# load_history / save_history
# ---------------------------------------------------------------------------


class TestHistoryPersistence:
    def _make_record(self, rec_id="rec1"):
        return GenerationRecord(
            id=rec_id,
            timestamp=1700000000.0,
            audio_path=f"/out/{rec_id}.wav",
            text="Hello",
            model_id="m1",
            model_name="Model 1",
            voice_id="v1",
            voice_name="Voice 1",
            provider="qwen",
            duration_sec=1.5,
            format="wav",
        )

    def test_save_and_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "history.json")
        records = [self._make_record("r1"), self._make_record("r2")]
        save_history(records, path)
        loaded = load_history(path)
        assert len(loaded) == 2
        assert loaded[0].id == "r1"
        assert loaded[1].id == "r2"
        assert loaded[0].text == "Hello"

    def test_load_missing_file_returns_empty(self, tmp_path):
        path = str(tmp_path / "no_such_history.json")
        assert load_history(path) == []

    def test_load_invalid_json_returns_empty(self, tmp_path):
        path = str(tmp_path / "bad_history.json")
        with open(path, "w") as f:
            f.write("NOT JSON")
        assert load_history(path) == []

    def test_save_creates_parent_directories(self, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "history.json")
        save_history([], path)
        assert os.path.isfile(path)

    def test_empty_history_roundtrip(self, tmp_path):
        path = str(tmp_path / "empty.json")
        save_history([], path)
        assert load_history(path) == []


# ---------------------------------------------------------------------------
# copy_voice_audio
# ---------------------------------------------------------------------------


class TestCopyVoiceAudio:
    def test_copies_file_correctly(self, tmp_path):
        # Create a source file
        src = tmp_path / "source.wav"
        src.write_bytes(b"RIFF fake wav data")

        with patch("rho_tts.ui.config.DEFAULT_VOICES_DIR", str(tmp_path / "voices")):
            dest = copy_voice_audio(str(src), "my_voice")
            assert os.path.isfile(dest)
            assert dest.endswith("my_voice.wav")
            with open(dest, "rb") as f:
                assert f.read() == b"RIFF fake wav data"

    def test_preserves_extension(self, tmp_path):
        src = tmp_path / "source.mp3"
        src.write_bytes(b"mp3 data")

        with patch("rho_tts.ui.config.DEFAULT_VOICES_DIR", str(tmp_path / "voices")):
            dest = copy_voice_audio(str(src), "voice2")
            assert dest.endswith("voice2.mp3")

    def test_default_extension_when_missing(self, tmp_path):
        src = tmp_path / "noext"
        src.write_bytes(b"audio data")

        with patch("rho_tts.ui.config.DEFAULT_VOICES_DIR", str(tmp_path / "voices")):
            dest = copy_voice_audio(str(src), "voice3")
            assert dest.endswith("voice3.wav")

    def test_creates_voices_directory(self, tmp_path):
        src = tmp_path / "source.wav"
        src.write_bytes(b"data")
        voices_dir = str(tmp_path / "new_voices_dir")

        with patch("rho_tts.ui.config.DEFAULT_VOICES_DIR", voices_dir):
            copy_voice_audio(str(src), "voice4")
            assert os.path.isdir(voices_dir)


# ---------------------------------------------------------------------------
# is_model_cached
# ---------------------------------------------------------------------------


class TestIsModelCached:
    def test_returns_true_when_cached(self):
        with patch(
            "rho_tts.ui.config.try_to_load_from_cache",
            return_value="/path/to/cached/config.json",
            create=True,
        ):
            # We need to patch the import inside the function
            mock_func = lambda repo_id, filename: "/path/to/cached/config.json"
            with patch.dict(
                "sys.modules",
                {"huggingface_hub": type("M", (), {"try_to_load_from_cache": staticmethod(mock_func)})()},
            ):
                assert is_model_cached("some/repo") is True

    def test_returns_false_when_not_cached(self):
        mock_func = lambda repo_id, filename: None
        with patch.dict(
            "sys.modules",
            {"huggingface_hub": type("M", (), {"try_to_load_from_cache": staticmethod(mock_func)})()},
        ):
            assert is_model_cached("some/repo") is False

    def test_returns_false_on_import_error(self):
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            assert is_model_cached("some/repo") is False


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------


class TestConstants:
    def test_provider_models_has_expected_providers(self):
        assert "qwen" in PROVIDER_MODELS
        assert "chatterbox" in PROVIDER_MODELS

    def test_provider_models_entries_have_required_keys(self):
        for provider, models in PROVIDER_MODELS.items():
            for m in models:
                assert "display_name" in m, f"Missing display_name in {provider}"
                assert "defaults" in m, f"Missing defaults in {provider}"

    def test_default_paths_contain_rho_tts(self):
        assert ".rho_tts" in DEFAULT_CONFIG_PATH
        assert ".rho_tts" in DEFAULT_HISTORY_PATH
        assert ".rho_tts" in DEFAULT_VOICES_DIR
