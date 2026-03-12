"""
ProviderProxy — duck-types the BaseTTS interface and delegates to a
long-running worker subprocess in an isolated venv.

Does NOT subclass BaseTTS to avoid importing torch/heavy deps in the
main process when they may not be installed.
"""

import asyncio
import logging
import os
import tempfile
import threading
from typing import Generator, List, Optional, Union

from .process_manager import WorkerProcess
from .protocol import (
    CANCELLED,
    ERROR,
    GENERATE,
    INIT,
    READY,
    RESULT,
    SEGMENT_RESULT,
    STREAM,
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
        from ..exceptions import ModelLoadError

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
            raise ModelLoadError(
                f"Failed to initialize isolated provider '{provider}': "
                f"{resp.get('message', 'unknown error')}"
            )
        else:
            self._cleanup()
            raise ModelLoadError(
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
        output_path: Optional[str] = None,
        cancellation_token=None,
        format: str = "wav",
        speed: float = 1.0,
        pitch_semitones: float = 0.0,
        progress_callback=None,
    ):
        """Generate audio. Accepts a single string or list of strings.

        Returns GenerationResult (or list thereof) when possible, falling
        back to path strings for backward compat with the worker protocol.
        """
        _single_mode = isinstance(texts, str)

        cancel_stop = threading.Event()
        if cancellation_token is not None:
            self._start_cancel_forwarder(cancellation_token, cancel_stop)

        # For in-memory mode (no output_path), use a temp directory
        use_temp = output_path is None
        temp_dir = None
        if use_temp:
            temp_dir = tempfile.mkdtemp(prefix="rho_tts_proxy_")
            effective_path = os.path.join(temp_dir, "output.wav")
        else:
            effective_path = output_path

        try:
            if _single_mode:
                resp = self._worker.send(
                    GENERATE,
                    text=texts,
                    output_path=effective_path,
                    format=format,
                    speed=speed,
                    pitch_semitones=pitch_semitones,
                )
            else:
                resp = self._worker.send(
                    GENERATE,
                    texts=texts,
                    output_base_path=effective_path,
                    format=format,
                    speed=speed,
                    pitch_semitones=pitch_semitones,
                )
        finally:
            cancel_stop.set()

        if resp.get("type") == RESULT:
            return self._build_results(resp, _single_mode, use_temp, temp_dir)
        elif resp.get("type") == CANCELLED:
            self._cleanup_temp(temp_dir)
            return None
        elif resp.get("type") == ERROR:
            self._cleanup_temp(temp_dir)
            raise RuntimeError(f"Worker error: {resp.get('message')}")
        else:
            self._cleanup_temp(temp_dir)
            raise RuntimeError(f"Unexpected response: {resp}")

    def _build_results(self, resp, single_mode, use_temp, temp_dir):
        """Build GenerationResult(s) from worker response."""
        from ..result import GenerationResult

        if single_mode:
            if not resp.get("success", False):
                self._cleanup_temp(temp_dir)
                return None

            result_path = resp.get("output_path")
            result = GenerationResult(
                sample_rate=self._sample_rate,
                duration_sec=resp.get("duration_sec", 0.0),
                segments_count=resp.get("segments_count", 0),
                format=resp.get("format", "wav"),
            )

            if use_temp and result_path:
                # Read audio into tensor, clean up temp
                result.audio = self._load_audio_tensor(result_path)
                self._cleanup_temp(temp_dir)
                result.path = None
            else:
                result.path = result_path

            return result
        else:
            output_paths = resp.get("output_paths", [])
            durations = resp.get("durations", [])
            seg_counts = resp.get("seg_counts", [])
            results = []
            for i, path in enumerate(output_paths):
                if path is None:
                    results.append(None)
                    continue
                r = GenerationResult(
                    sample_rate=self._sample_rate,
                    duration_sec=durations[i] if i < len(durations) else 0.0,
                    segments_count=seg_counts[i] if i < len(seg_counts) else 0,
                    format=resp.get("format", "wav"),
                )
                if use_temp:
                    r.audio = self._load_audio_tensor(path)
                    r.path = None
                else:
                    r.path = path
                results.append(r)

            self._cleanup_temp(temp_dir)
            if all(r is None for r in results):
                return None
            return results

    def _load_audio_tensor(self, path):
        """Load audio from a file path into a torch tensor."""
        try:
            import torchaudio
            audio, sr = torchaudio.load(path)
            return audio.squeeze(0)
        except Exception:
            return None

    def stream(
        self,
        text: str,
        cancellation_token=None,
        speed: float = 1.0,
        pitch_semitones: float = 0.0,
    ) -> Generator:
        """Yield GenerationResult per text segment as generated."""
        from ..result import GenerationResult

        cancel_stop = threading.Event()
        if cancellation_token is not None:
            self._start_cancel_forwarder(cancellation_token, cancel_stop)

        temp_dir = tempfile.mkdtemp(prefix="rho_tts_stream_")

        try:
            self._worker.send_nowait(
                STREAM,
                text=text,
                temp_dir=temp_dir,
                speed=speed,
                pitch_semitones=pitch_semitones,
            )

            while True:
                resp = self._worker.receive()
                if resp is None:
                    break

                resp_type = resp.get("type")
                if resp_type == SEGMENT_RESULT:
                    audio = self._load_audio_tensor(resp.get("path"))
                    if audio is not None:
                        yield GenerationResult(
                            audio=audio,
                            sample_rate=self._sample_rate,
                            duration_sec=resp.get("duration_sec", 0.0),
                            segments_count=1,
                            format="wav",
                        )
                    # Clean up temp segment file
                    seg_path = resp.get("path")
                    if seg_path and os.path.exists(seg_path):
                        try:
                            os.remove(seg_path)
                        except OSError:
                            pass
                elif resp_type == RESULT:
                    break
                elif resp_type == CANCELLED:
                    break
                elif resp_type == ERROR:
                    break
        finally:
            cancel_stop.set()
            self._cleanup_temp(temp_dir)

    async def async_generate(
        self,
        texts: Union[str, List[str]],
        output_path: Optional[str] = None,
        cancellation_token=None,
        format: str = "wav",
        speed: float = 1.0,
        pitch_semitones: float = 0.0,
        progress_callback=None,
    ):
        """Async wrapper around generate()."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(
                texts,
                output_path=output_path,
                cancellation_token=cancellation_token,
                format=format,
                speed=speed,
                pitch_semitones=pitch_semitones,
            ),
        )

    # -- Context manager / cleanup ---------------------------------------------

    def close(self) -> None:
        """Gracefully shut down the worker subprocess."""
        self.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

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

    @staticmethod
    def _cleanup_temp(temp_dir):
        """Remove a temporary directory and its contents."""
        if temp_dir is None:
            return
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
