"""
Subprocess entry point for isolated provider execution.

Run as:  ``<venv-python> -m rho_tts.isolation.worker``

Reads JSON-line commands from stdin, executes them against the real
TTS provider, and writes JSON-line responses to stdout.

**stdout is reserved for protocol messages** — all logging goes to stderr.

Architecture: A reader thread owns stdin and routes messages:
  - ``cancel`` / ``ping``: handled immediately by reader
  - Everything else: queued for the main thread
The main thread blocks on inference and pulls commands from the queue.
"""

import logging
import queue
import sys
import threading
from typing import Optional

from rho_tts.cancellation import CancellationToken
from rho_tts.factory import TTSFactory
from rho_tts.isolation.protocol import (
    CANCEL,
    CANCELLED,
    ERROR,
    GENERATE,
    INIT,
    PING,
    PONG,
    READY,
    RESULT,
    SHUTDOWN,
    decode_message,
    encode_message,
)

# --- Redirect all logging to stderr (stdout = protocol only) ---
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("rho_tts.worker")


class Worker:
    """Encapsulates the worker state and message loop."""

    def __init__(self):
        self._tts = None
        self._cancel_token: Optional[CancellationToken] = None
        self._cancel_lock = threading.Lock()
        self._write_lock = threading.Lock()

    # -- Protocol helpers --------------------------------------------------

    def _write(self, msg_type: str, **payload) -> None:
        """Write a single JSON line to stdout (thread-safe)."""
        with self._write_lock:
            sys.stdout.write(encode_message(msg_type, **payload))
            sys.stdout.flush()

    # -- Command handlers --------------------------------------------------

    def _handle_init(self, msg: dict) -> None:
        provider = msg.get("provider", "qwen")
        kwargs = msg.get("kwargs", {})

        logger.info("Initializing provider '%s'...", provider)
        try:
            self._tts = TTSFactory.get_tts_instance(provider=provider, **kwargs)
            self._write(READY, sample_rate=self._tts.sample_rate)
            logger.info("Provider '%s' ready (sample_rate=%d)", provider, self._tts.sample_rate)
        except Exception as exc:
            logger.error("Init failed: %s", exc)
            self._write(ERROR, message=str(exc))

    def _handle_generate(self, msg: dict) -> None:
        texts = msg.get("texts") or msg.get("text")
        output_path = msg.get("output_base_path") or msg.get("output_path")

        with self._cancel_lock:
            self._cancel_token = CancellationToken()
        token = self._cancel_token

        try:
            result = self._tts.generate(texts, output_path, cancellation_token=token)
            if token.is_cancelled():
                self._write(CANCELLED)
            elif isinstance(result, str):
                self._write(RESULT, output_path=result, success=True)
            elif isinstance(result, list):
                self._write(RESULT, output_paths=result)
            else:
                self._write(RESULT, success=False)
        except Exception as exc:
            if token.is_cancelled():
                self._write(CANCELLED)
            else:
                logger.error("generate failed: %s", exc)
                self._write(ERROR, message=str(exc))
        finally:
            with self._cancel_lock:
                self._cancel_token = None

    # -- Main loop ---------------------------------------------------------

    def run(self) -> None:
        """Blocking main loop. Expects ``init`` first, then enter command loop."""
        # Phase 1: wait for init on stdin directly (reader thread not started yet)
        line = sys.stdin.readline()
        if not line:
            return
        msg = decode_message(line)
        if msg.get("type") != INIT:
            self._write(ERROR, message="Expected 'init' as first message")
            return

        self._handle_init(msg)
        if self._tts is None:
            return  # init failed

        # Phase 2: start reader thread + command queue
        cmd_queue: queue.Queue = queue.Queue()
        shutdown_event = threading.Event()

        def reader():
            """Read stdin, route cancel/ping immediately, queue the rest."""
            while not shutdown_event.is_set():
                try:
                    raw = sys.stdin.readline()
                    if not raw:
                        cmd_queue.put(None)
                        break
                    incoming = decode_message(raw)
                except Exception:
                    cmd_queue.put(None)
                    break

                incoming_type = incoming.get("type")
                if incoming_type == CANCEL:
                    with self._cancel_lock:
                        if self._cancel_token is not None:
                            self._cancel_token.cancel()
                            logger.info("Cancellation requested")
                elif incoming_type == PING:
                    self._write(PONG)
                else:
                    cmd_queue.put(incoming)

        threading.Thread(target=reader, daemon=True, name="worker-reader").start()

        # Main command dispatch
        while True:
            msg = cmd_queue.get()
            if msg is None:
                break

            msg_type = msg.get("type")
            if msg_type == SHUTDOWN:
                logger.info("Shutdown received")
                break
            elif msg_type == GENERATE:
                self._handle_generate(msg)
            else:
                self._write(ERROR, message=f"Unknown command: {msg_type}")

        shutdown_event.set()
        logger.info("Worker exiting")


def main():
    worker = Worker()
    worker.run()


if __name__ == "__main__":
    main()
