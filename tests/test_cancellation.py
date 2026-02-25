"""Tests for CancellationToken behavior."""
import threading
import time

import pytest

from ralph_tts.cancellation import CancellationToken, CancelledException


class TestCancellationToken:
    def test_initial_state_not_cancelled(self):
        token = CancellationToken()
        assert not token.is_cancelled()

    def test_cancel_sets_state(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled()

    def test_cancel_is_idempotent(self):
        token = CancellationToken()
        token.cancel()
        token.cancel()
        assert token.is_cancelled()

    def test_raise_if_cancelled_does_nothing_when_not_cancelled(self):
        token = CancellationToken()
        token.raise_if_cancelled()  # Should not raise

    def test_raise_if_cancelled_raises_when_cancelled(self):
        token = CancellationToken()
        token.cancel()
        with pytest.raises(CancelledException):
            token.raise_if_cancelled()

    def test_raise_if_cancelled_custom_message(self):
        token = CancellationToken()
        token.cancel()
        with pytest.raises(CancelledException, match="custom stop"):
            token.raise_if_cancelled("custom stop")

    def test_reset(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled()
        token.reset()
        assert not token.is_cancelled()

    def test_thread_safety(self):
        """Verify cancel() is visible from another thread."""
        token = CancellationToken()
        seen = []

        def worker():
            while not token.is_cancelled():
                time.sleep(0.01)
            seen.append(True)

        t = threading.Thread(target=worker)
        t.start()
        time.sleep(0.05)
        token.cancel()
        t.join(timeout=1)
        assert seen == [True]

    def test_cancelled_exception_is_exception(self):
        assert issubclass(CancelledException, Exception)
