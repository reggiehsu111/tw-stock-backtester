"""Bundled strategies.

Two, both on daily TAIFEX data, and both deliberately unremarkable. They exist to
exercise the engine and to show what the output looks like when a result is merely
interesting rather than real. A backtester that can only display a winner is not
telling you the truth about the winner.

Neither clears the five criteria at the end of docs/METHOD.md. The tearsheet says so
in its own output rather than leaving the reader to work it out.
"""

from .institutional_flow import InstitutionalFlow
from .momentum import Momentum

REGISTRY = {
    "institutional_flow": InstitutionalFlow,
    "momentum": Momentum,
}


def get(name: str):
    """Look up a strategy class by its CLI name."""
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name]


def available() -> list[str]:
    return sorted(REGISTRY)


__all__ = ["InstitutionalFlow", "Momentum", "REGISTRY", "get", "available"]
