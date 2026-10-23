"""On-disk cache of model responses.

Free API tiers have daily limits, so a request should never be paid for twice.
The cache key is a hash of the complete request (model, messages, tools,
settings) plus a ``sample`` number that separates deliberately repeated runs.
Re-running an experiment, or resuming one after a quota stop, replays cached
responses instantly and for free.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


def request_key(body: dict[str, Any], sample: int) -> str:
    canonical = json.dumps({"body": body, "sample": sample}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT)")
        self._db.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False)),
            )
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
