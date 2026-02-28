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


class TestWorkerProtocol:
    """Test the worker's message handling with mocked TTS."""

    def _run_worker(self, input_messages: list[str]) -> list[dict]:
        """Run the worker with canned stdin and capture stdout."""
        stdin_data = "".join(input_messages)
        captured_stdout = io.StringIO()

        # Mock TTS instance
        mock_tts = MagicMock()
        mock_tts.sample_rate = 24000

        def _mock_generate(texts, output_path, cancellation_token=None):
            if isinstance(texts, str):
                return output_path
            return [f"{output_path}_{i}.wav" for i in range(len(texts))]

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
