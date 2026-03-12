"""Tests for isolation layer streaming — proxy.stream() and worker._handle_stream()."""

import io
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from rho_tts.isolation.protocol import (
    CANCELLED,
    ERROR,
    INIT,
    READY,
    RESULT,
    SEGMENT_RESULT,
    SHUTDOWN,
    STREAM,
    encode_message,
)
from rho_tts.isolation.proxy import ProviderProxy
from rho_tts.result import GenerationResult


# =============================================================================
# Worker._handle_stream() tests
# =============================================================================


class TestWorkerHandleStream:
    """Test the worker's STREAM command handler with mocked TTS."""

    def _run_worker(self, input_messages: list[str]) -> list[dict]:
        """Run the worker with canned stdin, capture stdout messages."""
        stdin_data = "".join(input_messages)
        captured_stdout = io.StringIO()

        # Build a mock TTS that yields GenerationResult from stream()
        mock_tts = MagicMock()
        mock_tts.sample_rate = 16000

        def _mock_stream(text, cancellation_token=None, speed=1.0, pitch_semitones=0.0):
            # Yield two segments
            for i in range(2):
                yield GenerationResult(
                    audio=torch.randn(16000),
                    sample_rate=16000,
                    duration_sec=1.0,
                    segments_count=1,
                    format="wav",
                )

        mock_tts.stream.side_effect = _mock_stream

        mock_ta = MagicMock()

        with patch("sys.stdin", io.StringIO(stdin_data)), \
             patch("sys.stdout", captured_stdout), \
             patch("rho_tts.isolation.worker.TTSFactory") as mock_factory, \
             patch.dict("sys.modules", {"torchaudio": mock_ta}):
            mock_factory.get_tts_instance.return_value = mock_tts

            from rho_tts.isolation.worker import Worker
            worker = Worker()
            worker.run()

        output = captured_stdout.getvalue()
        responses = []
        for line in output.strip().split("\n"):
            if line:
                responses.append(json.loads(line))
        return responses

    def test_stream_yields_segment_results(self):
        with tempfile.TemporaryDirectory() as td:
            messages = [
                encode_message(INIT, provider="qwen", kwargs={}),
                encode_message(STREAM, text="Hello world. Test.", temp_dir=td),
                encode_message(SHUTDOWN),
            ]
            responses = self._run_worker(messages)

        assert responses[0]["type"] == READY
        # Should have segment results followed by a final result
        seg_results = [r for r in responses if r["type"] == SEGMENT_RESULT]
        final_results = [r for r in responses if r["type"] == RESULT]

        assert len(seg_results) == 2
        for seg in seg_results:
            assert "path" in seg
            assert "duration_sec" in seg

        assert len(final_results) == 1
        assert final_results[0].get("success") is True
        assert final_results[0].get("segments") == 2

    def test_stream_empty_text(self):
        """Streaming empty text should yield no segments."""
        mock_tts = MagicMock()
        mock_tts.sample_rate = 16000
        mock_tts.stream.return_value = iter([])  # empty generator

        with tempfile.TemporaryDirectory() as td:
            messages = [
                encode_message(INIT, provider="qwen", kwargs={}),
                encode_message(STREAM, text="", temp_dir=td),
                encode_message(SHUTDOWN),
            ]
            stdin_data = "".join(messages)
            captured_stdout = io.StringIO()

            with patch("sys.stdin", io.StringIO(stdin_data)), \
                 patch("sys.stdout", captured_stdout), \
                 patch("rho_tts.isolation.worker.TTSFactory") as mock_factory:
                mock_factory.get_tts_instance.return_value = mock_tts

                from rho_tts.isolation.worker import Worker
                worker = Worker()
                worker.run()

            output = captured_stdout.getvalue()
            responses = [json.loads(line) for line in output.strip().split("\n") if line]

        assert responses[0]["type"] == READY
        final = [r for r in responses if r["type"] == RESULT]
        assert len(final) == 1
        assert final[0]["segments"] == 0

    def test_stream_error_in_tts(self):
        """If stream() throws, worker sends ERROR."""
        mock_tts = MagicMock()
        mock_tts.sample_rate = 16000
        mock_tts.stream.side_effect = RuntimeError("model crashed")

        with tempfile.TemporaryDirectory() as td:
            messages = [
                encode_message(INIT, provider="qwen", kwargs={}),
                encode_message(STREAM, text="Hello", temp_dir=td),
                encode_message(SHUTDOWN),
            ]
            stdin_data = "".join(messages)
            captured_stdout = io.StringIO()

            with patch("sys.stdin", io.StringIO(stdin_data)), \
                 patch("sys.stdout", captured_stdout), \
                 patch("rho_tts.isolation.worker.TTSFactory") as mock_factory:
                mock_factory.get_tts_instance.return_value = mock_tts

                from rho_tts.isolation.worker import Worker
                worker = Worker()
                worker.run()

            output = captured_stdout.getvalue()
            responses = [json.loads(line) for line in output.strip().split("\n") if line]

        errors = [r for r in responses if r["type"] == ERROR]
        assert len(errors) == 1
        assert "model crashed" in errors[0]["message"]


# =============================================================================
# ProviderProxy.stream() tests
# =============================================================================


class TestProxyStream:
    """Test ProviderProxy.stream() with mocked worker."""

    def _make_proxy(self, init_response):
        """Create a ProviderProxy with a mocked WorkerProcess."""
        mock_worker = MagicMock()
        mock_worker.send = MagicMock(return_value=init_response)
        mock_worker.send_nowait = MagicMock()
        mock_worker.send_cancel = MagicMock()
        mock_worker.shutdown = MagicMock()
        mock_worker.start = MagicMock()
        mock_worker.kill = MagicMock()

        mock_mgr = MagicMock()
        mock_mgr.ensure_venv.return_value = "/fake/python"

        with patch("rho_tts.isolation.proxy.VenvManager", return_value=mock_mgr), \
             patch("rho_tts.isolation.proxy.WorkerProcess", return_value=mock_worker):
            proxy = ProviderProxy("qwen", device="cuda")

        return proxy, mock_worker

    def test_stream_yields_results(self, tmp_path):
        proxy, worker = self._make_proxy({"type": READY, "sample_rate": 24000})

        # Create temp wav files for the responses
        seg_path_0 = str(tmp_path / "seg_0.wav")
        seg_path_1 = str(tmp_path / "seg_1.wav")

        # Write minimal wav files
        import torchaudio
        for p in [seg_path_0, seg_path_1]:
            torchaudio.save(p, torch.randn(1, 24000), 24000)

        responses = iter([
            {"type": SEGMENT_RESULT, "path": seg_path_0, "duration_sec": 1.0},
            {"type": SEGMENT_RESULT, "path": seg_path_1, "duration_sec": 1.0},
            {"type": RESULT, "success": True, "segments": 2},
        ])
        worker.receive = MagicMock(side_effect=lambda: next(responses))

        results = list(proxy.stream("Hello world. Test sentence."))

        assert len(results) == 2
        for r in results:
            assert isinstance(r, GenerationResult)
            assert r.audio is not None
            assert r.sample_rate == 24000
            assert r.segments_count == 1

    def test_stream_sends_stream_command(self):
        proxy, worker = self._make_proxy({"type": READY, "sample_rate": 24000})

        worker.receive = MagicMock(return_value={"type": RESULT, "success": True, "segments": 0})

        list(proxy.stream("Hello"))

        worker.send_nowait.assert_called_once()
        call_kwargs = worker.send_nowait.call_args
        assert call_kwargs[0][0] == STREAM

    def test_stream_cancelled(self):
        proxy, worker = self._make_proxy({"type": READY, "sample_rate": 24000})

        worker.receive = MagicMock(return_value={"type": CANCELLED})

        results = list(proxy.stream("Hello"))
        assert results == []

    def test_stream_error(self):
        proxy, worker = self._make_proxy({"type": READY, "sample_rate": 24000})

        worker.receive = MagicMock(return_value={"type": ERROR, "message": "failed"})

        results = list(proxy.stream("Hello"))
        assert results == []

    def test_stream_none_response_stops(self):
        proxy, worker = self._make_proxy({"type": READY, "sample_rate": 24000})

        worker.receive = MagicMock(return_value=None)

        results = list(proxy.stream("Hello"))
        assert results == []

    def test_stream_with_cancellation_token(self):
        proxy, worker = self._make_proxy({"type": READY, "sample_rate": 24000})

        mock_token = MagicMock()
        mock_token.is_cancelled.return_value = False

        worker.receive = MagicMock(return_value={"type": RESULT, "success": True, "segments": 0})

        list(proxy.stream("Hello", cancellation_token=mock_token))
        # Cancel forwarder should have been started
        # (verified by no crash with the token)

    def test_stream_with_speed_pitch(self):
        proxy, worker = self._make_proxy({"type": READY, "sample_rate": 24000})

        worker.receive = MagicMock(return_value={"type": RESULT, "success": True, "segments": 0})

        list(proxy.stream("Hello", speed=1.5, pitch_semitones=2.0))

        call_kwargs = worker.send_nowait.call_args
        assert call_kwargs[1].get("speed") == 1.5
        assert call_kwargs[1].get("pitch_semitones") == 2.0

    def test_stream_cleans_up_temp_files(self, tmp_path):
        proxy, worker = self._make_proxy({"type": READY, "sample_rate": 24000})

        seg_path = str(tmp_path / "seg_0.wav")
        import torchaudio
        torchaudio.save(seg_path, torch.randn(1, 24000), 24000)

        responses = iter([
            {"type": SEGMENT_RESULT, "path": seg_path, "duration_sec": 1.0},
            {"type": RESULT, "success": True, "segments": 1},
        ])
        worker.receive = MagicMock(side_effect=lambda: next(responses))

        results = list(proxy.stream("Hello"))
        assert len(results) == 1
        # The segment file should have been cleaned up
        assert not os.path.exists(seg_path)
