"""Tests for WorkerProcess — subprocess lifecycle management."""

import json
import subprocess
import sys
import threading
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from rho_tts.isolation.process_manager import WorkerProcess


def _make_mock_proc(responses: list[str]):
    """Create a mock Popen that returns canned JSON line responses."""
    proc = MagicMock()
    proc.poll.return_value = None  # "alive"
    proc.stdin = MagicMock()
    proc.stderr = MagicMock()
    proc.stderr.readline = MagicMock(return_value="")

    # Queue up stdout responses
    response_lines = [json.dumps(r) + "\n" for r in responses]
    proc.stdout = MagicMock()
    proc.stdout.readline = MagicMock(side_effect=response_lines)
    return proc


class TestWorkerProcess:
    def test_alive_when_running(self):
        wp = WorkerProcess("/usr/bin/python")
        assert not wp.alive  # Not started yet

    @patch("rho_tts.isolation.process_manager.subprocess.Popen")
    def test_start_spawns_process(self, mock_popen):
        mock_proc = _make_mock_proc([{"type": "pong"}])
        mock_popen.return_value = mock_proc

        wp = WorkerProcess("/usr/bin/python")
        wp.start()

        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert "-m" in cmd
        assert "rho_tts.isolation.worker" in cmd

    @patch("rho_tts.isolation.process_manager.subprocess.Popen")
    def test_send_writes_and_reads(self, mock_popen):
        mock_proc = _make_mock_proc([{"type": "pong"}])
        mock_popen.return_value = mock_proc

        wp = WorkerProcess("/usr/bin/python")
        wp.start()
        resp = wp.send("ping")

        assert resp["type"] == "pong"
        mock_proc.stdin.write.assert_called_once()
        mock_proc.stdin.flush.assert_called_once()

    @patch("rho_tts.isolation.process_manager.subprocess.Popen")
    def test_send_cancel_nonblocking(self, mock_popen):
        mock_proc = _make_mock_proc([])
        mock_popen.return_value = mock_proc

        wp = WorkerProcess("/usr/bin/python")
        wp.start()
        wp.send_cancel()  # Should not raise

        mock_proc.stdin.write.assert_called_once()

    @patch("rho_tts.isolation.process_manager.subprocess.Popen")
    def test_shutdown_sends_shutdown_message(self, mock_popen):
        mock_proc = _make_mock_proc([])
        mock_popen.return_value = mock_proc
        mock_proc.wait.return_value = 0

        wp = WorkerProcess("/usr/bin/python")
        wp.start()
        wp.shutdown()

        # Should have written a shutdown message
        written = mock_proc.stdin.write.call_args[0][0]
        msg = json.loads(written)
        assert msg["type"] == "shutdown"

    @patch("rho_tts.isolation.process_manager.subprocess.Popen")
    def test_kill_terminates_process(self, mock_popen):
        mock_proc = _make_mock_proc([])
        mock_popen.return_value = mock_proc
        mock_proc.wait.return_value = 0

        wp = WorkerProcess("/usr/bin/python")
        wp.start()
        wp.kill()

        mock_proc.kill.assert_called_once()

    @patch("rho_tts.isolation.process_manager.subprocess.Popen")
    def test_send_when_not_running_raises(self, mock_popen):
        wp = WorkerProcess("/usr/bin/python")
        with pytest.raises(RuntimeError, match="not running"):
            wp.send("ping")

    @patch("rho_tts.isolation.process_manager.subprocess.Popen")
    def test_broken_pipe_triggers_restart(self, mock_popen):
        """If stdout returns empty (worker crashed), should attempt restart."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdin = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.readline = MagicMock(return_value="")
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline = MagicMock(return_value="")  # EOF = crash
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        wp = WorkerProcess("/usr/bin/python")
        wp.start()

        with pytest.raises(RuntimeError, match="crashed and was restarted"):
            wp.send("ping")

        # Should have attempted restart (called Popen twice total)
        assert mock_popen.call_count == 2

    @patch("rho_tts.isolation.process_manager.subprocess.Popen")
    def test_ping_returns_true_on_pong(self, mock_popen):
        mock_proc = _make_mock_proc([{"type": "pong"}])
        mock_popen.return_value = mock_proc

        wp = WorkerProcess("/usr/bin/python")
        wp.start()
        assert wp.ping() is True

    @patch("rho_tts.isolation.process_manager.subprocess.Popen")
    def test_ping_returns_false_when_dead(self, mock_popen):
        wp = WorkerProcess("/usr/bin/python")
        assert wp.ping() is False
