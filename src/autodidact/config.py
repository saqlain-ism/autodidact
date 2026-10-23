"""Settings, read from environment variables and an optional ``.env`` file.

API keys live only in ``.env`` (which git ignores) or the real environment.
Nothing in the code base or the results ever contains a key.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader: ``KEY=value`` lines, ``#`` comments, optional quotes.

    Real environment variables win over the file, so CI and shells can override.
    """
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or default


def mask(secret: str | None) -> str:
    """Show that a key is set without revealing it: ``gsk_…9f2a``."""
    if not secret:
        return "(not set)"
    return f"{secret[:4]}…{secret[-4:]}" if len(secret) > 12 else "(set)"
