"""
Provider isolation — auto-managed venvs for providers with conflicting deps.

When a provider's dependencies aren't importable in the main process,
``TTSFactory`` returns a ``ProviderProxy`` that transparently manages
a worker subprocess in a dedicated venv.
"""

from .proxy import ProviderProxy

__all__ = ["ProviderProxy"]
