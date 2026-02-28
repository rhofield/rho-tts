"""
ProviderProxy — duck-types the BaseTTS interface and delegates to a
long-running worker subprocess in an isolated venv.

Does NOT subclass BaseTTS to avoid importing torch/heavy deps in the
main process when they may not be installed.
"""

import logging
import threading
from typing import List, Optional, Union

from .process_manager import WorkerProcess
from .protocol import (
    CANCELLED,
    ERROR,
    GENERATE,
    INIT,
    READY,
    RESULT,
)
from .venv_manager import VenvManager

logger = logging.getLogger(__name__)

# How often (seconds) the cancellation forwarder polls the token
_CANCEL_POLL_INTERVAL = 0.1


class ProviderProxy:
    """Proxy that forwards TTS calls to an isolated worker subprocess.

    Quacks like BaseTTS but runs inference in a separate venv.
    """

    def __init__(self, provider: str, **kwargs):
        self._provider = provider
        self._kwargs = kwargs
        self._sample_rate: Optional[int] = None
        self._worker: Optional[WorkerProcess] = None
        self._shutting_down = False

        # Create venv (no-op after first time)
        mgr = VenvManager(provider)
        python_path = mgr.ensure_venv()

        # Start worker
        self._worker = WorkerProcess(python_path)
        self._worker.start()

        # Send init
        resp = self._worker.send(INIT, provider=provider, kwargs=kwargs)
        if resp.get("type") == READY:
            self._sample_rate = resp["sample_rate"]
            logger.info(
                "Isolated provider '%s' ready (sample_rate=%d)",
                provider, self._sample_rate,
            )
        elif resp.get("type") == ERROR:
            self._cleanup()
            raise RuntimeError(
                f"Failed to initialize isolated provider '{provider}': "
                f"{resp.get('message', 'unknown error')}"
            )
        else:
            self._cleanup()
            raise RuntimeError(
                f"Unexpected response from worker during init: {resp}"
            )

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            raise RuntimeError("Provider not initialized")
        return self._sample_rate

    def generate(
        self,
        texts: Union[str, List[str]],
        output_path: str,
        cancellation_token=None,
    ) -> Union[Optional[str], Optional[List[str]]]:
        """Generate audio. Accepts a single string or list of strings."""
        _single_mode = isinstance(texts, str)

        cancel_stop = threading.Event()
        if cancellation_token is not None:
            self._start_cancel_forwarder(cancellation_token, cancel_stop)

        try:
            if _single_mode:
                resp = self._worker.send(
                    GENERATE,
                    text=texts,
                    output_path=output_path,
                )
            else:
                resp = self._worker.send(
                    GENERATE,
                    texts=texts,
                    output_base_path=output_path,
                )
        finally:
            cancel_stop.set()

        if resp.get("type") == RESULT:
            if _single_mode:
                if not resp.get("success", False):
                    return None
                return resp.get("output_path")
            return resp.get("output_paths")
        elif resp.get("type") == CANCELLED:
            return None
        elif resp.get("type") == ERROR:
            raise RuntimeError(f"Worker error: {resp.get('message')}")
        else:
            raise RuntimeError(f"Unexpected response: {resp}")

    def shutdown(self) -> None:
        """Gracefully shut down the worker subprocess."""
        if self._shutting_down:
            return
        self._shutting_down = True
        if self._worker is not None:
            self._worker.shutdown()
            self._worker = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    # -- Internal helpers --------------------------------------------------

    def _start_cancel_forwarder(
        self,
        cancellation_token,
        stop_event: threading.Event,
    ) -> None:
        """Poll cancellation_token and forward cancel to worker."""
        def _poll():
            while not stop_event.is_set():
                if cancellation_token.is_cancelled():
                    if self._worker is not None:
                        self._worker.send_cancel()
                    return
                stop_event.wait(_CANCEL_POLL_INTERVAL)

        t = threading.Thread(target=_poll, daemon=True, name="cancel-forwarder")
        t.start()

    def _cleanup(self) -> None:
        """Kill the worker without grace."""
        if self._worker is not None:
            self._worker.kill()
            self._worker = None
