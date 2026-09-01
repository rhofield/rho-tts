"""Tests for the Breeze provider's configuration and registration.

The model itself is never loaded — BreezeTTS._load_model is patched out — so
these run without the isolated venv, the ~7 GB of weights, or a GPU.
"""

from unittest.mock import patch

import pytest

from rho_tts.factory import TTSFactory
from rho_tts.provider_info import ProviderInfo


def _make(**kwargs):
    """Construct a BreezeTTS with model loading stubbed out."""
    from rho_tts.providers.breeze import BreezeTTS

    with patch.object(BreezeTTS, "_load_model", lambda self: None):
        return BreezeTTS(device="cpu", **kwargs)


class TestBreezeValidation:
    """Constructor argument validation — the mode-selection contract."""

    def test_reference_audio_without_text_raises(self):
        with pytest.raises(ValueError, match="reference_text"):
            _make(reference_audio="/tmp/ref.wav")

    def test_no_reference_and_no_instruction_raises(self):
        with pytest.raises(ValueError, match="reference_audio.*or instruction"):
            _make()

    def test_instruction_only_is_voice_design(self):
        tts = _make(instruction="A warm, older narrator.")
        assert tts.voice_cloning is False
        assert tts.instruction == "A warm, older narrator."

    def test_reference_with_text_is_cloning(self):
        tts = _make(reference_audio="/tmp/ref.wav", reference_text="hello there")
        assert tts.voice_cloning is True
        assert tts.reference_text == "hello there"

    def test_cloning_defaults_to_neutral_instruction(self):
        from rho_tts.providers.breeze import NEUTRAL_INSTRUCTION

        tts = _make(reference_audio="/tmp/ref.wav", reference_text="hi")
        assert tts.instruction == NEUTRAL_INSTRUCTION

    @pytest.mark.parametrize("bad", [0, -1.0])
    def test_non_positive_cfg_scale_raises(self, bad):
        with pytest.raises(ValueError, match="cfg_scale"):
            _make(instruction="x", cfg_scale=bad)

    def test_defaults_to_eager_attention(self):
        """The shipped config asks for flash_attention_2, which upstream does
        not install; defaulting to eager keeps a clean install working."""
        assert _make(instruction="x").attn_implementation == "eager"

    def test_fast_path_off_by_default(self):
        assert _make(instruction="x").fast is False


class TestBreezeTemplateSelection:
    """Mode -> upstream template mapping."""

    def test_design_uses_instruction_template(self):
        from rho_tts.providers.breeze import TEMPLATE_INSTRUCTION

        tts = _make(instruction="A calm voice.")
        request, template = tts._build_request("hello world")
        assert template == TEMPLATE_INSTRUCTION
        assert "ref_audio_path" not in request
        assert request["instruction"] == "A calm voice."

    def test_cloning_uses_reference_template(self):
        from rho_tts.providers.breeze import TEMPLATE_REFERENCE

        tts = _make(reference_audio="/tmp/ref.wav", reference_text="transcript here")
        tts._ref_audio_24k = "/tmp/ref24k.wav"
        request, template = tts._build_request("hello world")
        assert template == TEMPLATE_REFERENCE
        assert request["ref_audio_path"] == "/tmp/ref24k.wav"
        assert request["ref_text"] == "transcript here"

    def test_direction_keeps_reference_and_instruction(self):
        """Voice direction: steering composes with cloning rather than replacing it."""
        from rho_tts.providers.breeze import TEMPLATE_REFERENCE

        tts = _make(
            reference_audio="/tmp/ref.wav",
            reference_text="transcript",
            instruction="Sound tired and resigned.",
            cfg_scale=4.0,
        )
        tts._ref_audio_24k = "/tmp/ref24k.wav"
        request, template = tts._build_request("hello")
        assert template == TEMPLATE_REFERENCE
        assert request["ref_audio_path"] == "/tmp/ref24k.wav"
        assert request["instruction"] == "Sound tired and resigned."
        assert tts.cfg_scale == 4.0

    def test_request_carries_text(self):
        tts = _make(instruction="x")
        request, _ = tts._build_request("the quick brown fox")
        assert request["text"] == "the quick brown fox"


class TestBreezeSampleRate:
    def test_falls_back_to_constant_before_load(self):
        from rho_tts.providers.breeze import SAMPLE_RATE

        assert _make(instruction="x").sample_rate == SAMPLE_RATE
        assert SAMPLE_RATE == 24000


class TestBreezeFactoryRegistration:
    """Factory tests must save/restore class-level state."""

    def setup_method(self):
        self._providers = TTSFactory._providers.copy()
        self._isolated = TTSFactory._isolated_providers.copy()
        self._registered = TTSFactory._default_providers_registered

    def teardown_method(self):
        TTSFactory._providers = self._providers
        TTSFactory._isolated_providers = self._isolated
        TTSFactory._default_providers_registered = self._registered

    def test_breeze_is_listed(self):
        assert "breeze" in TTSFactory.list_providers()

    def test_isolated_when_upstream_absent(self):
        """Without breeze_infer on the path, breeze must resolve to a proxy."""
        TTSFactory.list_providers()
        import importlib.util

        if importlib.util.find_spec("breeze_infer") is None:
            assert "breeze" in TTSFactory._isolated_providers
            assert "breeze" not in TTSFactory._providers
        else:
            assert "breeze" in TTSFactory._providers

    def test_registers_directly_when_upstream_present(self):
        """Inside the provisioned venv breeze must register as a real provider,
        not a proxy — otherwise the worker recurses into another subprocess."""
        with patch("importlib.util.find_spec", return_value=object()):
            TTSFactory._providers = {}
            TTSFactory._isolated_providers = set()
            TTSFactory._default_providers_registered = False
            TTSFactory.list_providers()
        assert "breeze" in TTSFactory._providers
        assert "breeze" not in TTSFactory._isolated_providers

    def test_provider_info_without_subprocess(self):
        info = TTSFactory.get_provider_info("breeze")
        assert isinstance(info, ProviderInfo)
        assert info.name == "breeze"
        assert info.supports_voice_cloning is True

    def test_static_info_matches_class_info(self):
        from rho_tts.providers.breeze import BreezeTTS

        static = TTSFactory.get_provider_info("breeze")
        live = BreezeTTS.provider_info()
        assert static.name == live.name
        assert static.supports_voice_cloning == live.supports_voice_cloning
        assert static.supported_languages == live.supported_languages
        assert [v.id for v in static.builtin_voices] == [v.id for v in live.builtin_voices]


class TestBreezeVenvWiring:
    def test_breeze_has_an_extras_key(self):
        from rho_tts.isolation.venv_manager import PROVIDER_EXTRAS

        assert PROVIDER_EXTRAS["breeze"] == "breeze"

    def test_breeze_repo_is_pinned_to_a_full_sha(self):
        from rho_tts.isolation.venv_manager import PROVIDER_REPOS

        url, sha = PROVIDER_REPOS["breeze"]
        assert url.endswith("breeze-tts.git")
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)

    def test_pip_only_providers_have_no_repo(self):
        from rho_tts.isolation.venv_manager import PROVIDER_REPOS

        assert "qwen" not in PROVIDER_REPOS
        assert "chatterbox" not in PROVIDER_REPOS


class TestBreezeModelPinning:
    """The weights repo is untagged and its main branch has moved mid-development,
    so the revision must be pinned rather than tracking main."""

    def test_model_revision_is_a_full_sha(self):
        from rho_tts.providers.breeze import MODEL_REVISION

        assert len(MODEL_REVISION) == 40
        assert all(c in "0123456789abcdef" for c in MODEL_REVISION)

    def test_snapshot_download_is_called_with_the_pin(self):
        """Guards against a refactor dropping revision= and silently following main."""
        import inspect

        from rho_tts.providers import breeze

        src = inspect.getsource(breeze.BreezeTTS._load_model)
        assert "revision=MODEL_REVISION" in src
