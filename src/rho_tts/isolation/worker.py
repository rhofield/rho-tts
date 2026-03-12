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
import os
import queue
import sys
import tempfile
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
    SEGMENT_RESULT,
    SHUTDOWN,
    STREAM,
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
        fmt = msg.get("format", "wav")
        speed = msg.get("speed", 1.0)
        pitch_semitones = msg.get("pitch_semitones", 0.0)

        with self._cancel_lock:
            self._cancel_token = CancellationToken()
        token = self._cancel_token

        try:
            result = self._tts.generate(
                texts, output_path,
                cancellation_token=token,
                format=fmt,
                speed=speed,
                pitch_semitones=pitch_semitones,
            )
            if token.is_cancelled():
                self._write(CANCELLED)
            elif result is None:
                self._write(RESULT, success=False)
            elif hasattr(result, 'path'):
                # Single GenerationResult
                self._write(
                    RESULT,
                    output_path=result.path,
                    success=result.audio is not None,
                    duration_sec=result.duration_sec,
                    segments_count=result.segments_count,
                    format=result.format,
                )
            elif isinstance(result, list):
                # List of GenerationResult
                paths = []
                durations = []
                seg_counts = []
                for r in result:
                    if r is None:
                        paths.append(None)
                        durations.append(0.0)
                        seg_counts.append(0)
                    else:
                        paths.append(r.path)
                        durations.append(r.duration_sec)
                        seg_counts.append(r.segments_count)
                self._write(
                    RESULT,
                    output_paths=paths,
                    durations=durations,
                    seg_counts=seg_counts,
                    format=fmt,
                )
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

    def _handle_stream(self, msg: dict) -> None:
        text = msg.get("text", "")
        temp_dir = msg.get("temp_dir")
        speed = msg.get("speed", 1.0)
        pitch_semitones = msg.get("pitch_semitones", 0.0)

        with self._cancel_lock:
            self._cancel_token = CancellationToken()
        token = self._cancel_token

        try:
            seg_idx = 0
            for result in self._tts.stream(
                text,
                cancellation_token=token,
                speed=speed,
                pitch_semitones=pitch_semitones,
            ):
                if token.is_cancelled():
                    self._write(CANCELLED)
                    return

                # Save segment to temp file
                import torchaudio
                seg_path = os.path.join(temp_dir or tempfile.gettempdir(), f"seg_{seg_idx}.wav")
                audio = result.audio
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                torchaudio.save(seg_path, audio.cpu(), result.sample_rate)

                self._write(
                    SEGMENT_RESULT,
                    path=seg_path,
                    duration_sec=result.duration_sec,
                )
                seg_idx += 1

            self._write(RESULT, success=True, segments=seg_idx)
        except Exception as exc:
            if token.is_cancelled():
                self._write(CANCELLED)
            else:
                logger.error("stream failed: %s", exc)
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
            elif msg_type == STREAM:
                self._handle_stream(msg)
            else:
                self._write(ERROR, message=f"Unknown command: {msg_type}")

        shutdown_event.set()
        logger.info("Worker exiting")


def main():
    worker = Worker()
    worker.run()


if __name__ == "__main__":
    main()
