"""Tests that rho-tts works correctly as a pip-installed library.

Validates the public API surface, exports, entry points, optional dependency
error messages, and that core functionality is usable without provider extras.
"""

import os

import torch
import pytest

from rho_tts import BaseTTS, TTSFactory, CancellationToken, CancelledException, __version__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StubTTS(BaseTTS):
    """Minimal concrete subclass for testing BaseTTS utilities."""

    def __init__(self, **kwargs):
        super().__init__(device="cpu", **kwargs)
        self._sample_rate = 16000

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _generate_audio(self, text, **kwargs):
        return torch.randn(self._sample_rate)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

class TestPublicExports:
    def test_version_is_string(self):
        assert isinstance(__version__, str)
        assert __version__  # non-empty

    def test_all_exports_importable(self):
        import rho_tts
        for name in rho_tts.__all__:
            assert hasattr(rho_tts, name), f"__all__ lists '{name}' but it is missing"

    def test_base_tts_is_abstract(self):
        with pytest.raises(TypeError):
            BaseTTS(device="cpu")

    def test_launch_ui_is_callable(self):
        from rho_tts import launch_ui
        assert callable(launch_ui)


# ---------------------------------------------------------------------------
# Factory as library entry point
# ---------------------------------------------------------------------------

class TestFactoryAsLibrary:
    def setup_method(self):
        self._orig_providers = TTSFactory._providers.copy()
        self._orig_isolated = TTSFactory._isolated_providers.copy()

    def teardown_method(self):
        TTSFactory._providers = self._orig_providers
        TTSFactory._isolated_providers = self._orig_isolated

    def test_list_providers_returns_sorted_list(self):
        providers = TTSFactory.list_providers()
        assert isinstance(providers, list)
        assert providers == sorted(providers)

    def test_register_and_instantiate_custom_provider(self):
        TTSFactory.register_provider("stub", StubTTS)
        tts = TTSFactory.get_tts_instance("stub")
        assert isinstance(tts, StubTTS)
        assert tts.sample_rate == 16000

    def test_unknown_provider_message_is_helpful(self):
        with pytest.raises(ValueError, match="pip install rho-tts"):
            TTSFactory.get_tts_instance(provider="does_not_exist_xyz")

    def test_kwargs_forwarded_to_provider(self):
        TTSFactory.register_provider("stub", StubTTS)
        tts = TTSFactory.get_tts_instance("stub", seed=123)
        assert tts.seed == 123


# ---------------------------------------------------------------------------
# CancellationToken standalone usage
# ---------------------------------------------------------------------------

class TestCancellationTokenStandalone:
    def test_lifecycle(self):
        token = CancellationToken()
        assert not token.is_cancelled()
        token.cancel()
        assert token.is_cancelled()
        with pytest.raises(CancelledException):
            token.raise_if_cancelled()


# ---------------------------------------------------------------------------
# BaseTTS audio utilities (the library's value-add)
# ---------------------------------------------------------------------------

class TestBaseTTSUtilities:
    @pytest.fixture()
    def tts(self):
        return StubTTS()

    def test_text_splitting(self, tts):
        text = "First sentence. Second sentence. Third."
        segments = tts._split_text_into_segments(text, max_chars=200)
        assert len(segments) == 3
        assert segments[0] == "First sentence."

    def test_trim_silence(self, tts):
        audio = torch.randn(1, 16000)
        trimmed = tts._trim_silence(audio)
        assert trimmed.numel() > 0
        assert trimmed.numel() <= audio.numel()

    def test_apply_fades(self, tts):
        audio = torch.randn(1, 16000)
        faded = tts._apply_fades(audio)
        assert faded.shape == audio.shape

    def test_remove_dc_offset(self, tts):
        audio = torch.randn(16000) + 0.5  # intentional DC offset
        cleaned = tts._remove_dc_offset(audio)
        assert abs(cleaned.mean().item()) < 0.01

    def test_smooth_segment_join(self, tts):
        segments = [torch.randn(8000) for _ in range(3)]
        joined = tts._smooth_segment_join(segments)
        assert joined.dim() == 1
        assert joined.numel() > 0

    def test_phonetic_mapping(self, tts):
        tts.phonetic_mapping = {"rho": "row"}
        result = tts._apply_phonetic_mapping("rho is greek")
        assert result == "row is greek"


# ---------------------------------------------------------------------------
# Optional dependency error messages
# ---------------------------------------------------------------------------

class GeneratingTTS(BaseTTS):
    """Stub that does NOT override generate(), exercising the base class logic."""

    def __init__(self, **kwargs):
        super().__init__(device="cpu", **kwargs)
        self._sample_rate = 16000

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _generate_audio(self, text, **kwargs):
        # Return 0.5 seconds of deterministic audio
        return torch.randn(self._sample_rate // 2)


class TestBaseTTSGenerate:
    """Test BaseTTS.generate() without provider override."""

    @pytest.fixture()
    def tts(self):
        return GeneratingTTS()

    def test_single_string_returns_path(self, tts, tmp_path):
        out = str(tmp_path / "out.wav")
        result = tts.generate("Hello world", out)
        assert result == out
        assert os.path.exists(out)

    def test_single_string_none_on_failure(self, tts, tmp_path):
        tts._generate_audio = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        result = tts.generate("Hello", str(tmp_path / "out.wav"))
        assert result is None

    def test_list_returns_list_of_paths(self, tts, tmp_path):
        out = str(tmp_path / "batch")
        result = tts.generate(["First text", "Second text"], out)
        assert isinstance(result, list)
        assert len(result) == 2
        for path in result:
            assert path is not None
            assert os.path.exists(path)

    def test_phonetic_mapping_applied(self, tts, tmp_path):
        tts.phonetic_mapping = {"AI": "A.I."}
        calls = []
        original = tts._generate_audio

        def tracking_generate(text, **kwargs):
            calls.append(text)
            return original(text, **kwargs)

        tts._generate_audio = tracking_generate
        tts.generate("AI is great", str(tmp_path / "out.wav"))
        assert any("A.I." in c for c in calls)

    def test_cancellation_returns_none(self, tts, tmp_path):
        token = CancellationToken()
        token.cancel()
        result = tts.generate("Hello", str(tmp_path / "out.wav"), cancellation_token=token)
        assert result is None


class TestOptionalDependencyErrors:
    def test_voice_encoder_missing_gives_install_hint(self):
        tts = StubTTS()
        tts._voice_encoder = None
        try:
            _ = tts.voice_encoder
            pytest.skip("resemblyzer is installed, can't test missing-dep path")
        except ImportError as e:
            assert "pip install rho-tts[validation]" in str(e)
