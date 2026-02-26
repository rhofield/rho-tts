"""Tests for ProviderProxy — the duck-typed BaseTTS proxy."""

import json
import threading
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from rho_tts.isolation.proxy import ProviderProxy
from rho_tts.isolation.protocol import CANCELLED, ERROR, READY, RESULT


class TestProviderProxy:
    """Test ProviderProxy with mocked VenvManager and WorkerProcess."""

    def _make_proxy(self, worker_responses: list[dict]):
        """Create a ProviderProxy with mocked internals."""
        resp_iter = iter(worker_responses)

        mock_worker = MagicMock()
        mock_worker.send = MagicMock(side_effect=lambda *a, **kw: next(resp_iter))
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

    def test_init_success(self):
        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
        ])
        assert proxy.sample_rate == 24000

    def test_init_error_raises(self):
        with pytest.raises(RuntimeError, match="Failed to initialize"):
            self._make_proxy([
                {"type": ERROR, "message": "torch not found"},
            ])

    def test_generate_single_success(self, tmp_path):
        output_path = str(tmp_path / "test.wav")

        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
            {"type": RESULT, "output_path": output_path, "success": True},
        ])

        # Mock torchaudio.load since we don't have a real WAV file
        import torch
        mock_ta = MagicMock()
        mock_ta.load.return_value = (torch.zeros(1, 24000), 24000)
        with patch.dict("sys.modules", {"torchaudio": mock_ta}):
            result = proxy.generate_single("Hello", output_path)

        assert result is not None
        assert result.shape == (1, 24000)

    def test_generate_single_failure_returns_none(self):
        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
            {"type": RESULT, "output_path": "/tmp/test.wav", "success": False},
        ])

        result = proxy.generate_single("Hello", "/tmp/test.wav")
        assert result is None

    def test_generate_single_cancelled(self):
        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
            {"type": CANCELLED},
        ])

        result = proxy.generate_single("Hello", "/tmp/test.wav")
        assert result is None

    def test_generate_single_error_raises(self):
        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
            {"type": ERROR, "message": "OOM"},
        ])

        with pytest.raises(RuntimeError, match="OOM"):
            proxy.generate_single("Hello", "/tmp/test.wav")

    def test_generate_batch_success(self):
        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
            {"type": RESULT, "output_paths": ["/tmp/0.wav", "/tmp/1.wav"]},
        ])

        result = proxy.generate(["Hello", "World"], "/tmp/base")
        assert result == ["/tmp/0.wav", "/tmp/1.wav"]

    def test_generate_batch_cancelled(self):
        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
            {"type": CANCELLED},
        ])

        result = proxy.generate(["Hello"], "/tmp/base")
        assert result is None

    def test_shutdown_calls_worker_shutdown(self):
        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
        ])

        proxy.shutdown()
        worker.shutdown.assert_called_once()

    def test_shutdown_idempotent(self):
        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
        ])

        proxy.shutdown()
        proxy.shutdown()  # Should not error
        worker.shutdown.assert_called_once()

    def test_cancellation_forwarding(self):
        """Verify cancel forwarder sends cancel to worker."""
        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
            {"type": RESULT, "output_path": "/tmp/test.wav", "success": True},
        ])

        # Create a mock cancellation token that is immediately cancelled
        mock_token = MagicMock()
        mock_token.is_cancelled.return_value = True

        import torch
        mock_ta = MagicMock()
        mock_ta.load.return_value = (torch.zeros(1, 24000), 24000)
        with patch.dict("sys.modules", {"torchaudio": mock_ta}):
            proxy.generate_single("Hello", "/tmp/test.wav", cancellation_token=mock_token)

        # Give the forwarder thread a moment to send
        import time
        time.sleep(0.2)
        worker.send_cancel.assert_called()
