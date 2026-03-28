"""Tests for provider introspection."""

import pytest

from rho_tts.provider_info import ProviderInfo, VoiceInfo
from rho_tts.base_tts import BaseTTS
from rho_tts.factory import TTSFactory
from rho_tts.exceptions import ProviderNotFoundError

import torch


class FakeTTS(BaseTTS):
    """Minimal subclass for testing."""

    def __init__(self, **kwargs):
        self.device = "cpu"
        self.seed = 42
        self.deterministic = False
        self.phonetic_mapping = {}
        self.silence_threshold_db = -50.0
        self.crossfade_duration_sec = 0.05
        self.trim_silence = True
        self.fade_duration_sec = 0.02
        self.force_sentence_split = True
        self.inter_sentence_pause_sec = 0.1
        self._voice_encoder = None
        self.reference_embedding = None
        self._sample_rate = 16000
        self.max_chars_per_segment = 800
        self.max_iterations = 1
        self.accent_drift_threshold = 0.17
        self.text_similarity_threshold = 0.85
        self.sound_decay_threshold = 0.3
        self.max_decay_retries = 3
        self.voice_id = None
        self.drift_model_path = None
        self._max_chars_explicit = True
        self._max_model_chars = 3000

    def _generate_audio(self, text, **kwargs):
        return torch.zeros(self._sample_rate)

    @property
    def sample_rate(self):
        return self._sample_rate


class TestVoiceInfo:
    def test_defaults(self):
        v = VoiceInfo(id="test", name="Test Voice")
        assert v.language == "English"
        assert v.is_builtin is True

    def test_custom(self):
        v = VoiceInfo(id="jp1", name="Japanese 1", language="Japanese", is_builtin=False)
        assert v.language == "Japanese"
        assert v.is_builtin is False


class TestProviderInfo:
    def test_defaults(self):
        p = ProviderInfo(name="test")
        assert p.supports_voice_cloning is False
        assert p.supported_languages == []
        assert p.builtin_voices == []

    def test_with_voices(self):
        voices = [VoiceInfo(id="v1", name="Voice 1")]
        p = ProviderInfo(
            name="test",
            supports_voice_cloning=True,
            supported_languages=["English"],
            builtin_voices=voices,
        )
        assert len(p.builtin_voices) == 1
        assert p.supports_voice_cloning is True


class TestBaseTTSProviderInfo:
    def test_default_provider_info(self):
        info = FakeTTS.provider_info()
        assert isinstance(info, ProviderInfo)
        assert info.name == "FakeTTS"


class TestFactoryIntrospection:
    def setup_method(self):
        self._orig_providers = TTSFactory._providers.copy()
        self._orig_isolated = TTSFactory._isolated_providers.copy()
        self._orig_registered = TTSFactory._default_providers_registered

    def teardown_method(self):
        TTSFactory._providers = self._orig_providers
        TTSFactory._isolated_providers = self._orig_isolated
        TTSFactory._default_providers_registered = self._orig_registered

    def test_get_provider_info_registered(self):
        TTSFactory.register_provider("fake", FakeTTS)
        info = TTSFactory.get_provider_info("fake")
        assert isinstance(info, ProviderInfo)

    def test_get_provider_info_isolated(self):
        TTSFactory._isolated_providers.add("qwen")
        info = TTSFactory.get_provider_info("qwen")
        assert info.name == "qwen"
        assert info.supports_voice_cloning is True

    def test_get_provider_info_unknown_raises(self):
        with pytest.raises(ProviderNotFoundError):
            TTSFactory.get_provider_info("nonexistent_xyz")

    def test_list_voices(self):
        TTSFactory._isolated_providers.add("qwen")
        voices = TTSFactory.list_voices("qwen")
        assert len(voices) == 9
        assert all(isinstance(v, VoiceInfo) for v in voices)

    def test_list_voices_unknown_raises(self):
        with pytest.raises(ProviderNotFoundError):
            TTSFactory.list_voices("nonexistent_xyz")
