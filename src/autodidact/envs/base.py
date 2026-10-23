"""Shared building blocks for sandbox environments.

Every sandbox app keeps its state in an in-memory SQLite database. That gives
us three things the experiments depend on:

* **Reset**: a world is rebuilt from a seed, so every episode starts identical.
* **Snapshots**: the whole state can be dumped to plain dicts.
* **Diffs**: comparing two snapshots shows exactly what an agent changed, which
  is how checkers catch side effects ("you restocked the right product, but you
  also archived two others").
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any


class AppError(Exception):
    """A business-rule error. Shown to the agent as ``CODE: message``."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# --------------------------------------------------------------------------- money


def round_half_up(value: Decimal | float) -> int:
    """Round to the nearest integer, halves away from zero (how shops round paise)."""
    return int(Decimal(str(value)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def paise_to_rupees(paise: int) -> Decimal:
    return (Decimal(paise) / 100).quantize(Decimal("0.01"))


# ----------------------------------------------------------------------- snapshots

Snapshot = dict[str, dict[tuple, dict[str, Any]]]


def snapshot(conn: sqlite3.Connection, keys: dict[str, tuple[str, ...]]) -> Snapshot:
    """Dump every table in ``keys`` to ``{table: {primary_key_tuple: row_dict}}``."""
    snap: Snapshot = {}
    for table, key_cols in keys.items():
        cur = conn.execute(f"SELECT * FROM {table}")
        cols = [c[0] for c in cur.description]
        rows = {}
        for raw in cur.fetchall():
            row = dict(zip(cols, raw, strict=True))
            rows[tuple(row[k] for k in key_cols)] = row
        snap[table] = rows
    return snap


@dataclass
class TableDiff:
    added: dict[tuple, dict] = field(default_factory=dict)
    removed: dict[tuple, dict] = field(default_factory=dict)
    changed: dict[tuple, tuple[dict, dict]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.added or self.removed or self.changed)


def diff(before: Snapshot, after: Snapshot) -> dict[str, TableDiff]:
    """Per-table diff of two snapshots. Tables with no change are omitted."""
    out: dict[str, TableDiff] = {}
    for table in before.keys() | after.keys():
        b, a = before.get(table, {}), after.get(table, {})
        td = TableDiff(
            added={k: a[k] for k in a.keys() - b.keys()},
            removed={k: b[k] for k in b.keys() - a.keys()},
            changed={k: (b[k], a[k]) for k in a.keys() & b.keys() if a[k] != b[k]},
        )
        if td:
            out[table] = td
    return out


def describe_diff(d: dict[str, TableDiff], limit: int = 8) -> str:
    """Short human-readable summary, used in checker failure messages."""
    parts = []
    for table, td in sorted(d.items()):
        bits = []
        if td.added:
            bits.append(f"+{len(td.added)}")
        if td.removed:
            bits.append(f"-{len(td.removed)}")
        if td.changed:
            keys = ", ".join(str(k[0] if len(k) == 1 else k) for k in list(td.changed)[:limit])
            bits.append(f"~{len(td.changed)} [{keys}]")
        parts.append(f"{table}: {' '.join(bits)}")
    return "; ".join(parts) or "no changes"
