"""Tests for TTSFactory provider registration and instantiation."""

import pytest
import torch

from rho_tts.base_tts import BaseTTS
from rho_tts.factory import TTSFactory


class MockTTS(BaseTTS):
    """Mock TTS for testing factory registration."""

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
        self.kwargs = kwargs

    def _generate_audio(self, text, **kwargs):
        return torch.zeros(self._sample_rate)

    def generate(self, texts, output_base_path, cancellation_token=None):
        return [f"{output_base_path}_{i}.wav" for i in range(len(texts))]

    def generate_single(self, text, output_path, cancellation_token=None):
        return torch.zeros(self._sample_rate)

    @property
    def sample_rate(self):
        return self._sample_rate


class TestTTSFactory:
    def setup_method(self):
        """Clean up registered providers before each test."""
        # Save and restore original providers
        self._original_providers = TTSFactory._providers.copy()

    def teardown_method(self):
        TTSFactory._providers = self._original_providers

    def test_register_provider(self):
        TTSFactory.register_provider("mock", MockTTS)
        assert "mock" in TTSFactory.list_providers()

    def test_get_registered_instance(self):
        TTSFactory.register_provider("mock", MockTTS)
        instance = TTSFactory.get_tts_instance(provider="mock")
        assert isinstance(instance, MockTTS)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            TTSFactory.get_tts_instance(provider="nonexistent_provider_xyz")

    def test_register_non_subclass_raises(self):
        class NotATTS:
            pass

        with pytest.raises(TypeError):
            TTSFactory.register_provider("bad", NotATTS)

    def test_kwargs_passed_to_constructor(self):
        TTSFactory.register_provider("mock", MockTTS)
        instance = TTSFactory.get_tts_instance(provider="mock", custom_param="test_value")
        assert instance.kwargs.get("custom_param") == "test_value"

    def test_list_providers_returns_list(self):
        result = TTSFactory.list_providers()
        assert isinstance(result, list)

    def test_overwrite_existing_provider(self):
        TTSFactory.register_provider("mock", MockTTS)

        class MockTTS2(MockTTS):
            pass

        TTSFactory.register_provider("mock", MockTTS2)
        instance = TTSFactory.get_tts_instance(provider="mock")
        assert isinstance(instance, MockTTS2)
