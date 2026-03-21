"""
Per-session context for multi-user deployments (e.g. HF Spaces).

Each browser tab gets its own SessionContext via gr.State, providing
isolated cancellation tokens, generation history, and output directories.
The shared AppState retains only the TTS instance and read-only config.
"""

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Optional

from ..cancellation import CancellationToken
from .config import AppConfig, GenerationRecord

logger = logging.getLogger(__name__)


@dataclass
class SessionContext:
    """Per-session state that isolates users from each other."""

    session_id: str
    config: Optional[AppConfig] = None
    cancellation_token: Optional[CancellationToken] = None
    history: list[GenerationRecord] = field(default_factory=list)
    _output_dir: Optional[str] = field(default=None, repr=False)

    def save(self) -> None:
        """No-op — session config is not persisted to disk."""
        pass

    def new_cancellation_token(self) -> CancellationToken:
        """Create and store a fresh cancellation token for this session."""
        self.cancellation_token = CancellationToken()
        return self.cancellation_token

    def cancel_generation(self) -> None:
        """Cancel this session's current generation."""
        if self.cancellation_token is not None:
            self.cancellation_token.cancel()

    @property
    def output_dir(self) -> str:
        """Lazy-create a temporary output directory for this session."""
        if self._output_dir is None:
            self._output_dir = tempfile.mkdtemp(
                prefix=f"rho_{self.session_id[:8]}_",
            )
            logger.info("Session %s output dir: %s", self.session_id[:8], self._output_dir)
        return self._output_dir

    def add_generation_record(self, record: GenerationRecord) -> None:
        """Append a generation record (in-memory only, no disk persistence)."""
        self.history.append(record)

    def delete_generation_record(self, record_id: str) -> bool:
        """Remove a generation record by ID. Returns True if found."""
        before = len(self.history)
        self.history = [r for r in self.history if r.id != record_id]
        return len(self.history) < before

    def clear_history(self) -> int:
        """Remove all generation records. Returns count removed."""
        count = len(self.history)
        self.history = []
        return count

    def cleanup(self) -> None:
        """Remove temporary output directory. Called on session end."""
        if self._output_dir is not None:
            try:
                shutil.rmtree(self._output_dir)
                logger.info("Cleaned up session %s output dir", self.session_id[:8])
            except Exception:
                logger.warning(
                    "Failed to clean up session dir: %s", self._output_dir,
                    exc_info=True,
                )
            self._output_dir = None
