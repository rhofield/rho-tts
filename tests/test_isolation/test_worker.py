"""Tests for the worker subprocess entry point (mocked inference)."""

import json
import io
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest
import torch

from rho_tts.isolation.protocol import (
    CANCELLED,
    ERROR,
    GENERATE,
    INIT,
    READY,
    RESULT,
    SHUTDOWN,
    encode_message,
)
from rho_tts.result import GenerationResult


class TestWorkerProtocol:
    """Test the worker's message handling with mocked TTS."""

    def _run_worker(self, input_messages: list[str]) -> list[dict]:
        """Run the worker with canned stdin and capture stdout."""
        stdin_data = "".join(input_messages)
        captured_stdout = io.StringIO()

        # Mock TTS instance
        mock_tts = MagicMock()
        mock_tts.sample_rate = 24000

        def _mock_generate(texts, output_path=None, cancellation_token=None,
                          format="wav", speed=1.0, pitch_semitones=0.0):
            if isinstance(texts, str):
                return GenerationResult(
                    path=output_path,
                    audio=torch.zeros(24000),
                    sample_rate=24000,
                    duration_sec=1.0,
                    segments_count=1,
                    format=format,
                )
            results = []
            for i in range(len(texts)):
                p = f"{output_path}_{i}.wav" if output_path else None
                results.append(GenerationResult(
                    path=p,
                    audio=torch.zeros(24000),
                    sample_rate=24000,
                    duration_sec=1.0,
                    segments_count=1,
                    format=format,
                ))
            return results

        mock_tts.generate.side_effect = _mock_generate

        with patch("sys.stdin", io.StringIO(stdin_data)), \
             patch("sys.stdout", captured_stdout), \
             patch("rho_tts.isolation.worker.TTSFactory") as mock_factory:
            mock_factory.get_tts_instance.return_value = mock_tts

            from rho_tts.isolation.worker import Worker
            worker = Worker()
            worker.run()

        # Parse output lines
        output = captured_stdout.getvalue()
        responses = []
        for line in output.strip().split("\n"):
            if line:
                responses.append(json.loads(line))
        return responses

    def test_init_then_shutdown(self):
        messages = [
            encode_message(INIT, provider="qwen", kwargs={}),
            encode_message(SHUTDOWN),
        ]
        responses = self._run_worker(messages)

        assert len(responses) >= 1
        assert responses[0]["type"] == READY
        assert responses[0]["sample_rate"] == 24000

    def test_init_failure(self):
        messages = [
            encode_message(INIT, provider="qwen", kwargs={}),
        ]

        stdin_data = "".join(messages)
        captured_stdout = io.StringIO()

        with patch("sys.stdin", io.StringIO(stdin_data)), \
             patch("sys.stdout", captured_stdout), \
             patch("rho_tts.isolation.worker.TTSFactory") as mock_factory:
            mock_factory.get_tts_instance.side_effect = RuntimeError("No GPU")

            from rho_tts.isolation.worker import Worker
            worker = Worker()
            worker.run()

        output = captured_stdout.getvalue()
        responses = [json.loads(line) for line in output.strip().split("\n") if line]
        assert responses[0]["type"] == ERROR
        assert "No GPU" in responses[0]["message"]

    def test_generate_single(self):
        messages = [
            encode_message(INIT, provider="qwen", kwargs={}),
            encode_message(GENERATE, text="Hello", output_path="/tmp/test.wav"),
            encode_message(SHUTDOWN),
        ]
        responses = self._run_worker(messages)

        assert responses[0]["type"] == READY
        assert responses[1]["type"] == RESULT
        assert responses[1]["success"] is True
        assert responses[1]["output_path"] == "/tmp/test.wav"

    def test_bad_first_message(self):
        messages = [
            encode_message(SHUTDOWN),  # Not INIT
        ]

        stdin_data = "".join(messages)
        captured_stdout = io.StringIO()

        with patch("sys.stdin", io.StringIO(stdin_data)), \
             patch("sys.stdout", captured_stdout):
            from rho_tts.isolation.worker import Worker
            worker = Worker()
            worker.run()

        output = captured_stdout.getvalue()
        responses = [json.loads(line) for line in output.strip().split("\n") if line]
        assert responses[0]["type"] == ERROR
        assert "init" in responses[0]["message"].lower()


class TestProtocolStreamIsolation:
    """stdout is the JSON-line protocol channel; stray library prints must not
    reach it. Regression test: Breeze prints model-loading progress to stdout,
    which corrupted the stream and surfaced as a bogus 'worker crashed'."""

    def test_write_goes_to_the_explicit_protocol_stream(self):
        from rho_tts.isolation.worker import Worker

        protocol = io.StringIO()
        stray = io.StringIO()
        worker = Worker(protocol_out=protocol)

        with patch("sys.stdout", stray):
            worker._write("pong")
            print("stray library output")

        assert '"pong"' in protocol.getvalue()
        assert "stray" not in protocol.getvalue()
        assert "stray library output" in stray.getvalue()

    def test_main_redirects_sys_stdout_away_from_protocol(self):
        """main() must hand the worker the real stdout, then repoint sys.stdout."""
        from rho_tts.isolation import worker as worker_mod

        real_stdout = io.StringIO()
        fake_stderr = io.StringIO()
        captured = {}

        class _StubWorker:
            def __init__(self, protocol_out=None):
                captured["protocol_out"] = protocol_out

            def run(self):
                captured["stdout_during_run"] = worker_mod.sys.stdout

        with patch.object(worker_mod, "Worker", _StubWorker), \
             patch.object(worker_mod.sys, "stdout", real_stdout), \
             patch.object(worker_mod.sys, "stderr", fake_stderr):
            worker_mod.main()

        assert captured["protocol_out"] is real_stdout
        assert captured["stdout_during_run"] is fake_stderr

    def test_module_import_does_not_mutate_sys_stdout(self):
        """Importing the worker must be side-effect free — it is imported by
        tests and tooling, not only by the subprocess entry point."""
        import subprocess
        import sys as _sys

        result = subprocess.run(
            [_sys.executable, "-c",
             "import sys; before = sys.stdout;"
             " import rho_tts.isolation.worker;"
             " print('SAME' if sys.stdout is before else 'MUTATED')"],
            capture_output=True, text=True, timeout=120,
        )
        assert "SAME" in result.stdout, result.stdout + result.stderr
