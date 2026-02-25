"""
Thread-safe cancellation token for cooperative task cancellation.

Provides a clean mechanism for signaling and checking cancellation
across threads during long-running TTS generation tasks.
"""

import threading
from typing import Optional


class CancelledException(Exception):
    """Exception raised when a task is cancelled."""
    pass


class CancellationToken:
    """
    A thread-safe cancellation token for cooperative task cancellation.

    Usage:
        token = CancellationToken()

        # In worker thread:
        for item in items:
            token.raise_if_cancelled()
            process(item)

        # In controller thread:
        token.cancel()
    """

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._cancelled = False

    def cancel(self) -> None:
        """Signal that the task should be cancelled. Thread-safe."""
        with self._lock:
            if not self._cancelled:
                self._cancelled = True
                self._event.set()

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._event.is_set()

    def raise_if_cancelled(self, message: Optional[str] = None) -> None:
        """
        Raise CancelledException if cancellation has been requested.

        Args:
            message: Optional custom message for the exception.
        """
        if self.is_cancelled():
            raise CancelledException(message or "Task was cancelled")

    def reset(self) -> None:
        """Reset the token to allow reuse. Prefer creating a new token instead."""
        with self._lock:
            self._cancelled = False
            self._event.clear()
