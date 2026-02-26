"""
Creates and caches per-provider virtual environments.

Venvs live at ``~/.rho_tts/venvs/<provider>/`` and are lazily created
on first use.  A marker file tracks the installed version so we skip
reinstallation on subsequent runs.
"""

import hashlib
import logging
import os
import subprocess
import sys
import venv
from pathlib import Path

logger = logging.getLogger(__name__)

# Maps provider name -> pyproject.toml extras key
PROVIDER_EXTRAS: dict[str, str] = {
    "qwen": "qwen",
    "chatterbox": "chatterbox",
}

VENVS_ROOT = Path.home() / ".rho_tts" / "venvs"
MARKER_FILE = ".rho_tts_installed"


def _version_hash() -> str:
    """Return a short hash representing the current package source.

    For editable installs we hash the pyproject.toml so that dependency
    changes trigger a reinstall.  For packaged installs we use the
    package version.
    """
    try:
        pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]
    except Exception:
        pass

    # Fallback: use package version
    try:
        from rho_tts import __version__
        return hashlib.sha256(__version__.encode()).hexdigest()[:16]
    except Exception:
        return "unknown"


def _find_project_root() -> Path | None:
    """Walk up from this file to find the directory containing pyproject.toml."""
    current = Path(__file__).resolve().parent
    for _ in range(6):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return None


class VenvManager:
    """Manages per-provider virtual environments."""

    def __init__(self, provider: str, venvs_root: Path | None = None):
        if provider not in PROVIDER_EXTRAS:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Supported: {', '.join(PROVIDER_EXTRAS)}"
            )
        self.provider = provider
        self.extras_key = PROVIDER_EXTRAS[provider]
        self.venv_dir = (venvs_root or VENVS_ROOT) / provider

    @property
    def python(self) -> str:
        """Path to the venv's Python interpreter."""
        if sys.platform == "win32":
            return str(self.venv_dir / "Scripts" / "python.exe")
        return str(self.venv_dir / "bin" / "python")

    def ensure_venv(self) -> str:
        """Create the venv and install deps if needed. Returns the python path."""
        marker = self.venv_dir / MARKER_FILE
        current_hash = _version_hash()

        if marker.exists() and marker.read_text().strip() == current_hash:
            logger.debug("Venv for '%s' is up to date", self.provider)
            return self.python

        logger.info(
            "Setting up isolated environment for '%s' provider "
            "(this only happens once)...",
            self.provider,
        )

        # Create venv with pip available
        if not self.venv_dir.exists():
            logger.info("Creating venv at %s", self.venv_dir)
            self.venv_dir.mkdir(parents=True, exist_ok=True)
            venv.create(str(self.venv_dir), with_pip=True, clear=True)
        elif not Path(self.python).exists():
            venv.create(str(self.venv_dir), with_pip=True, clear=True)

        # Install the package with provider extras
        self._install_package()

        # Write marker
        marker.write_text(current_hash)
        logger.info("Isolated environment for '%s' is ready", self.provider)
        return self.python

    def _install_package(self) -> None:
        """Install rho-tts with the provider's extras into the venv."""
        project_root = _find_project_root()

        if project_root is not None:
            # Editable install from source tree
            install_spec = f"-e {project_root}[{self.extras_key}]"
            cmd = [self.python, "-m", "pip", "install", "-e", f"{project_root}[{self.extras_key}]"]
        else:
            # Packaged install — install from PyPI
            install_spec = f"rho-tts[{self.extras_key}]"
            cmd = [self.python, "-m", "pip", "install", f"rho-tts[{self.extras_key}]"]

        logger.info("Installing %s (this may take a few minutes)...", install_spec)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for large installs
        )

        if result.returncode != 0:
            logger.error("pip install failed:\n%s", result.stderr)
            raise RuntimeError(
                f"Failed to install dependencies for '{self.provider}' provider.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr: {result.stderr[-500:]}"
            )

        logger.info("Installation complete for '%s'", self.provider)
