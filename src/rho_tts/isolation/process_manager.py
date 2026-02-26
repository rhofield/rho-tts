"""
Manages the lifecycle of a long-running worker subprocess.

The worker communicates via JSON lines on stdin/stdout.  Stderr is
forwarded to the main process logger via a daemon thread.
"""

import logging
import subprocess
import sys
import threading
from typing import Optional

from .protocol import decode_message, encode_message, SHUTDOWN, PING, PONG

logger = logging.getLogger(__name__)

MAX_RESTARTS = 2


class WorkerProcess:
    """Manages a single worker subprocess."""

    def __init__(self, python_path: str):
        self._python = python_path
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()  # Protects _proc access
        self._restart_count = 0

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        """Spawn the worker subprocess."""
        cmd = [self._python, "-m", "rho_tts.isolation.worker"]
        logger.debug("Starting worker: %s", " ".join(cmd))

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered
        )

        # Daemon thread to forward stderr
        self._stderr_thread = threading.Thread(
            target=self._forward_stderr,
            daemon=True,
            name="worker-stderr",
        )
        self._stderr_thread.start()

    def send(self, msg_type: str, **payload) -> dict:
        """Send a message and wait for the response. Thread-safe.

        Raises RuntimeError on communication failure.
        """
        with self._lock:
            return self._send_locked(msg_type, **payload)

    def _send_locked(self, msg_type: str, **payload) -> dict:
        """Send/receive with crash recovery (must hold _lock)."""
        try:
            return self._do_send(msg_type, **payload)
        except (BrokenPipeError, OSError, ValueError) as exc:
            if self._restart_count >= MAX_RESTARTS:
                raise RuntimeError(
                    f"Worker crashed {self._restart_count + 1} times, giving up"
                ) from exc

            logger.warning(
                "Worker communication failed (%s), restarting (%d/%d)...",
                exc, self._restart_count + 1, MAX_RESTARTS,
            )
            self._restart_count += 1
            self._kill_unlocked()
            self.start()
            # Re-raise so caller knows the original request was lost
            raise RuntimeError(
                f"Worker crashed and was restarted. Original error: {exc}"
            ) from exc

    def _do_send(self, msg_type: str, **payload) -> dict:
        """Low-level send + receive."""
        if not self.alive:
            raise RuntimeError("Worker is not running")

        line = encode_message(msg_type, **payload)
        self._proc.stdin.write(line)
        self._proc.stdin.flush()

        resp_line = self._proc.stdout.readline()
        if not resp_line:
            raise BrokenPipeError("Worker closed stdout (crashed?)")

        return decode_message(resp_line)

    def send_cancel(self) -> None:
        """Non-blocking cancel — best-effort, does not wait for response."""
        try:
            if self.alive:
                line = encode_message("cancel")
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            logger.debug("Could not send cancel (worker already dead?)")

    def ping(self, timeout: float = 5.0) -> bool:
        """Check if the worker is responsive."""
        try:
            resp = self.send(PING)
            return resp.get("type") == PONG
        except Exception:
            return False

    def shutdown(self) -> None:
        """Graceful shutdown — send shutdown message, then wait."""
        with self._lock:
            if not self.alive:
                return
            try:
                line = encode_message(SHUTDOWN)
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
                logger.debug("Worker shut down gracefully")
            except Exception:
                logger.warning("Graceful shutdown failed, killing worker")
                self._kill_unlocked()

    def kill(self) -> None:
        """Force-kill the worker."""
        with self._lock:
            self._kill_unlocked()

    def _kill_unlocked(self) -> None:
        """Kill without holding lock (caller must hold _lock)."""
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None

    def _forward_stderr(self) -> None:
        """Read worker stderr and log it."""
        try:
            while self._proc and self._proc.stderr:
                line = self._proc.stderr.readline()
                if not line:
                    break
                line = line.rstrip("\n")
                if line:
                    logger.info("[worker] %s", line)
        except Exception:
            pass
