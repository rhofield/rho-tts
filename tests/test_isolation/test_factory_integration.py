"""Tests for TTSFactory integration with the isolation layer."""

from unittest.mock import MagicMock, patch

import pytest

from rho_tts.factory import TTSFactory
from rho_tts.isolation.protocol import READY


class TestFactoryIsolation:
    def setup_method(self):
        self._orig_providers = TTSFactory._providers.copy()
        self._orig_isolated = TTSFactory._isolated_providers.copy()
        self._orig_registered = TTSFactory._default_providers_registered

    def teardown_method(self):
        TTSFactory._providers = self._orig_providers
        TTSFactory._isolated_providers = self._orig_isolated
        TTSFactory._default_providers_registered = self._orig_registered

    def test_isolated_provider_in_list(self):
        TTSFactory._default_providers_registered = True
        TTSFactory._isolated_providers = {"qwen"}
        TTSFactory._providers = {}

        providers = TTSFactory.list_providers()
        assert "qwen" in providers

    def test_isolated_and_local_in_list(self):
        TTSFactory._default_providers_registered = True
        TTSFactory._isolated_providers = {"chatterbox"}

        # Simulate qwen available locally
        mock_cls = MagicMock()
        TTSFactory._providers = {"qwen": mock_cls}

        providers = TTSFactory.list_providers()
        assert "qwen" in providers
        assert "chatterbox" in providers

    def test_local_provider_preferred(self):
        """When deps are available locally, don't use isolation."""
        TTSFactory._default_providers_registered = True
        mock_cls = MagicMock()
        TTSFactory._providers = {"qwen": mock_cls}
        TTSFactory._isolated_providers = set()

        TTSFactory.get_tts_instance(provider="qwen")
        mock_cls.assert_called_once()

    def test_isolated_provider_returns_proxy(self):
        """When deps are NOT available locally, return ProviderProxy."""
        TTSFactory._default_providers_registered = True
        TTSFactory._providers = {}
        TTSFactory._isolated_providers = {"qwen"}

        mock_proxy = MagicMock()
        with patch("rho_tts.isolation.proxy.ProviderProxy", return_value=mock_proxy) as mock_cls, \
             patch.dict("sys.modules", {}):
            # Patch at the import target — factory imports ProviderProxy lazily
            with patch("rho_tts.isolation.ProviderProxy", mock_cls):
                result = TTSFactory.get_tts_instance(provider="qwen", device="cpu")
                mock_cls.assert_called_once_with("qwen", device="cpu")
                assert result is mock_proxy

    def test_unknown_provider_still_raises(self):
        TTSFactory._default_providers_registered = True
        TTSFactory._providers = {}
        TTSFactory._isolated_providers = set()

        with pytest.raises(ValueError, match="Unknown TTS provider"):
            TTSFactory.get_tts_instance(provider="totally_fake")
