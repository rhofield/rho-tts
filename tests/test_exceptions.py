"""Tests for the custom exception hierarchy."""

import pytest

from rho_tts.exceptions import (
    AudioGenerationError,
    FormatConversionError,
    ModelLoadError,
    ProviderNotFoundError,
    RhoTTSError,
)
from rho_tts.cancellation import CancelledException


class TestExceptionHierarchy:
    """Verify all exceptions inherit from RhoTTSError."""

    def test_base_exception(self):
        assert issubclass(RhoTTSError, Exception)

    @pytest.mark.parametrize("exc_class", [
        ProviderNotFoundError,
        ModelLoadError,
        AudioGenerationError,
        FormatConversionError,
        CancelledException,
    ])
    def test_subclass_of_rho_tts_error(self, exc_class):
        assert issubclass(exc_class, RhoTTSError)

    @pytest.mark.parametrize("exc_class", [
        ProviderNotFoundError,
        ModelLoadError,
        AudioGenerationError,
        FormatConversionError,
        CancelledException,
    ])
    def test_catchable_as_base(self, exc_class):
        with pytest.raises(RhoTTSError):
            raise exc_class("test message")

    def test_message_preserved(self):
        msg = "provider 'foo' not found"
        exc = ProviderNotFoundError(msg)
        assert str(exc) == msg

    def test_cancelled_still_catchable_as_original(self):
        """CancelledException should still be catchable as CancelledException."""
        with pytest.raises(CancelledException):
            raise CancelledException("cancelled")
