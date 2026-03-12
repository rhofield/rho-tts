"""Tests for ProviderProxy — the duck-typed BaseTTS proxy."""

from unittest.mock import MagicMock, patch

import pytest

from rho_tts.exceptions import ModelLoadError
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
        with pytest.raises(ModelLoadError, match="Failed to initialize"):
            self._make_proxy([
                {"type": ERROR, "message": "torch not found"},
            ])

    def test_generate_single_success(self, tmp_path):
        output_path = str(tmp_path / "test.wav")

        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
            {"type": RESULT, "output_path": output_path, "success": True,
             "duration_sec": 1.0, "segments_count": 1, "format": "wav"},
        ])

        result = proxy.generate("Hello", output_path)
        assert result is not None
        assert result.path == output_path

    def test_generate_single_failure_returns_none(self):
        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
            {"type": RESULT, "output_path": "/tmp/test.wav", "success": False},
        ])

        result = proxy.generate("Hello", "/tmp/test.wav")
        assert result is None

    def test_generate_single_cancelled(self):
        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
            {"type": CANCELLED},
        ])

        result = proxy.generate("Hello", "/tmp/test.wav")
        assert result is None

    def test_generate_single_error_raises(self):
        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
            {"type": ERROR, "message": "OOM"},
        ])

        with pytest.raises(RuntimeError, match="OOM"):
            proxy.generate("Hello", "/tmp/test.wav")

    def test_generate_batch_success(self):
        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
            {"type": RESULT, "output_paths": ["/tmp/0.wav", "/tmp/1.wav"],
             "durations": [1.0, 1.5], "seg_counts": [1, 2], "format": "wav"},
        ])

        result = proxy.generate(["Hello", "World"], "/tmp/base")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].path == "/tmp/0.wav"
        assert result[1].path == "/tmp/1.wav"

    def test_generate_batch_cancelled(self):
        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
            {"type": CANCELLED},
        ])

        result = proxy.generate(["Hello"], "/tmp/base")
        assert result is None

    def test_generate_in_memory(self):
        """When no output_path, should use temp dir."""
        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
            {"type": RESULT, "output_path": None, "success": True,
             "duration_sec": 1.0, "segments_count": 1, "format": "wav"},
        ])

        result = proxy.generate("Hello")
        # Even without a real temp file, should return result
        assert result is not None
        assert result.path is None

    def test_context_manager(self):
        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
        ])

        with proxy as p:
            assert p is proxy
        worker.shutdown.assert_called_once()

    def test_close_aliases_shutdown(self):
        proxy, worker = self._make_proxy([
            {"type": READY, "sample_rate": 24000},
        ])

        proxy.close()
        worker.shutdown.assert_called_once()

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
            {"type": RESULT, "output_path": "/tmp/test.wav", "success": True,
             "duration_sec": 1.0, "segments_count": 1, "format": "wav"},
        ])

        # Create a mock cancellation token that is immediately cancelled
        mock_token = MagicMock()
        mock_token.is_cancelled.return_value = True

        proxy.generate("Hello", "/tmp/test.wav", cancellation_token=mock_token)

        # Give the forwarder thread a moment to send
        import time
        time.sleep(0.2)
        worker.send_cancel.assert_called()
