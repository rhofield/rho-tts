"""Tests for VenvManager — venv creation and caching."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rho_tts.isolation.venv_manager import MARKER_FILE, VenvManager, _version_hash


class TestVersionHash:
    def test_returns_string(self):
        h = _version_hash()
        assert isinstance(h, str)
        assert len(h) > 0

    def test_deterministic(self):
        assert _version_hash() == _version_hash()


class TestVenvManager:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            VenvManager("nonexistent")

    def test_python_path(self, tmp_path):
        mgr = VenvManager("qwen", venvs_root=tmp_path)
        python = mgr.python
        assert "qwen" in python
        assert python.endswith("python") or python.endswith("python.exe")

    def test_ensure_venv_skips_when_marker_current(self, tmp_path):
        """If the marker file matches the current hash, skip install."""
        mgr = VenvManager("qwen", venvs_root=tmp_path)
        venv_dir = tmp_path / "qwen"
        venv_dir.mkdir(parents=True)

        marker = venv_dir / MARKER_FILE
        marker.write_text(_version_hash())

        # Should return immediately without calling venv.create or pip
        with patch("rho_tts.isolation.venv_manager.venv") as mock_venv:
            result = mgr.ensure_venv()
            mock_venv.create.assert_not_called()

        assert result == mgr.python

    @patch("rho_tts.isolation.venv_manager.subprocess.run")
    @patch("rho_tts.isolation.venv_manager.venv")
    def test_ensure_venv_creates_and_installs(self, mock_venv_mod, mock_run, tmp_path):
        """First-time setup should create venv and run pip install."""
        mgr = VenvManager("qwen", venvs_root=tmp_path)

        # Simulate pip success
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        result = mgr.ensure_venv()

        # venv.create should have been called
        mock_venv_mod.create.assert_called_once()

        # pip install should have been called with the qwen extras
        mock_run.assert_called_once()
        pip_cmd = mock_run.call_args[0][0]
        assert any("qwen" in arg for arg in pip_cmd)

        # Marker file should exist
        marker = tmp_path / "qwen" / MARKER_FILE
        assert marker.exists()
        assert marker.read_text().strip() == _version_hash()

    @patch("rho_tts.isolation.venv_manager.subprocess.run")
    @patch("rho_tts.isolation.venv_manager.venv")
    def test_ensure_venv_pip_failure_raises(self, mock_venv_mod, mock_run, tmp_path):
        """Failed pip install should raise RuntimeError."""
        mgr = VenvManager("chatterbox", venvs_root=tmp_path)
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="ERROR: Could not find package",
            stdout="",
        )

        with pytest.raises(RuntimeError, match="Failed to install"):
            mgr.ensure_venv()

        # Marker should NOT have been written
        marker = tmp_path / "chatterbox" / MARKER_FILE
        assert not marker.exists()

    @patch("rho_tts.isolation.venv_manager.subprocess.run")
    @patch("rho_tts.isolation.venv_manager.venv")
    def test_ensure_venv_reinstalls_on_hash_change(self, mock_venv_mod, mock_run, tmp_path):
        """If marker hash doesn't match, reinstall."""
        mgr = VenvManager("qwen", venvs_root=tmp_path)
        venv_dir = tmp_path / "qwen"
        venv_dir.mkdir(parents=True)

        # Write a stale marker
        marker = venv_dir / MARKER_FILE
        marker.write_text("stale_hash")

        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        mgr.ensure_venv()

        # Should have called pip install
        mock_run.assert_called_once()
        # Marker should be updated
        assert marker.read_text().strip() == _version_hash()
