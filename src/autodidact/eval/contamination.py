"""How close did self-generated practice come to the hidden exam?

The curriculum never sees the hidden tasks, but it practises in the same kind
of application, so some overlap in *kind* of task is expected ("restock
something" appears in both). What would be a problem is near-copies. This
reports, for every practice exercise, its closest hidden task by word overlap
(Jaccard similarity), so the README can state the number instead of asking
for trust.
"""

from __future__ import annotations

import re
from typing import Any

from autodidact.tasks import Task

NEAR_COPY = 0.6


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def similarity(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def report(exercises: list[str], hidden: list[Task]) -> dict[str, Any]:
    rows = []
    for ex in exercises:
        best = max(hidden, key=lambda t: similarity(ex, t.prompt))
        rows.append(
            {
                "exercise": ex,
                "closest_task": best.id,
                "similarity": round(similarity(ex, best.prompt), 3),
            }
        )
    scores = [r["similarity"] for r in rows]
    return {
        "exercises": len(rows),
        "max_similarity": max(scores, default=0.0),
        "mean_similarity": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "near_copies": sum(s >= NEAR_COPY for s in scores),
        "rows": rows,
    }
