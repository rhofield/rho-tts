"""
Creates and caches per-provider virtual environments.

Venvs live at ``~/.rho_tts/venvs/<provider>/`` and are lazily created
on first use.  A marker file tracks the installed version so we skip
reinstallation on subsequent runs.
"""

import hashlib
import logging
import subprocess
import sys
import venv
from pathlib import Path

logger = logging.getLogger(__name__)

# Maps provider name -> pyproject.toml extras key
PROVIDER_EXTRAS: dict[str, str] = {
    "qwen": "qwen",
    "chatterbox": "chatterbox",
    "breeze": "breeze",
}

# Providers whose upstream code is not installable from PyPI and must be cloned.
# Keyed by provider name; the repo is cloned to REPOS_ROOT/<provider> at the
# pinned commit, then added to the venv's sys.path via a .pth file.
#
# breeze-tts ships no pyproject.toml/setup.py — its ``models`` and
# ``breeze_infer`` packages live at the repo root — so pip cannot install it.
# The SHA is pinned deliberately: upstream has no tags or releases.
PROVIDER_REPOS: dict[str, tuple[str, str]] = {
    "breeze": (
        "https://github.com/breezeblue-ai/breeze-tts.git",
        "ca632ce6c4d05f7985da4eab29b1a5d445b43f7b",
    ),
}

REPOS_ROOT = Path.home() / ".rho_tts" / "repos"

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
        return hashlib.sha256((__version__ + ":validation-v1").encode()).hexdigest()[:16]
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
        self.extras_key = PROVIDER_EXTRAS[provider] + ",validation"
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

        # Clone any non-pip-installable upstream source and expose it
        if self.provider in PROVIDER_REPOS:
            self._install_repo()

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
            from rho_tts import __version__
            install_spec = f"rho-tts[{self.extras_key}]=={__version__}"
            cmd = [self.python, "-m", "pip", "install", install_spec]

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

    # -- Non-pip-installable upstream source ----------------------------------

    def _clone_dir(self) -> Path:
        """Path the provider's upstream repo is cloned to."""
        return REPOS_ROOT / self.provider

    def _install_repo(self) -> None:
        """Clone the provider's upstream repo and put it on the venv's path.

        Used for providers whose upstream ships no packaging metadata. The repo
        root is added via a .pth file in site-packages rather than by mutating
        sys.path at import time, so the worker can import the upstream packages
        normally. This is safe only because each provider gets its own venv —
        breeze-tts exposes a top-level ``models`` package that would otherwise
        shadow unrelated code.
        """
        url, sha = PROVIDER_REPOS[self.provider]
        clone = self._clone_dir()

        if not (clone / ".git").is_dir():
            logger.info("Cloning %s for '%s'...", url, self.provider)
            clone.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--quiet", url, str(clone)],
                check=True, capture_output=True, text=True, timeout=600,
            )

        # Pin to the recorded commit. Fetch first so a re-pin to a newer SHA works.
        subprocess.run(
            ["git", "-C", str(clone), "fetch", "--quiet", "origin"],
            check=False, capture_output=True, text=True, timeout=600,
        )
        result = subprocess.run(
            ["git", "-C", str(clone), "checkout", "--quiet", sha],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to check out {sha[:8]} of {url} for '{self.provider}'.\n"
                f"stderr: {result.stderr[-500:]}"
            )

        self._write_path_file(clone)
        logger.info("Upstream source for '%s' pinned at %s", self.provider, sha[:8])

    def _write_path_file(self, clone: Path) -> None:
        """Add *clone* to the venv's import path via a .pth file."""
        result = subprocess.run(
            [self.python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            capture_output=True, text=True, check=True, timeout=60,
        )
        site_packages = Path(result.stdout.strip())
        (site_packages / f"rho_tts_{self.provider}.pth").write_text(f"{clone}\n")
