"""Summaries of evaluation runs, with honest error bars.

Pass rates come with a 95% Wilson confidence interval. With 60 tasks, a jump
from 40% to 45% is within noise; the interval makes that visible instead of
letting a lucky run look like progress.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TIERS = ("easy", "medium", "hard")


def load(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "results.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def wilson(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = passed / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    passed = sum(r["passed"] for r in records)
    by_tier: dict[str, list] = defaultdict(list)
    by_template: dict[str, list] = defaultdict(list)
    for r in records:
        by_tier[r["tier"]].append(r["passed"])
        by_template[r["template"]].append(r["passed"])

    def mean(key: str) -> float:
        return sum(r[key] for r in records) / n if n else 0.0

    return {
        "n": n,
        "passed": passed,
        "rate": passed / n if n else 0.0,
        "ci95": wilson(passed, n),
        "tiers": {t: (sum(v), len(v)) for t, v in by_tier.items()},
        "templates": {t: (sum(v), len(v)) for t, v in sorted(by_template.items())},
        "avg_turns": mean("turns"),
        "avg_tool_calls": mean("tool_calls"),
        "avg_tool_errors": mean("tool_errors"),
        "avg_tokens": mean("prompt_tokens") + mean("completion_tokens"),
        "avg_requests": mean("requests"),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in records)),
        "models": sorted({r["model"] for r in records}),
        "conditions": sorted({r["condition"] for r in records}),
    }


def _pct(passed: int, n: int) -> str:
    return f"{100 * passed / n:5.1f}% ({passed}/{n})" if n else "   -"


def format_report(run_dirs: list[str | Path]) -> str:
    """Side-by-side table for one or more run folders."""
    summaries = [(Path(d).name, summarize(load(d))) for d in run_dirs]
    width = max(22, *(len(name) for name, _ in summaries))
    rows: list[tuple[str, list[str]]] = []

    def add(label: str, values: list[str]) -> None:
        rows.append((label, values))

    add("model", [", ".join(s["models"]) or "-" for _, s in summaries])
    add("attempts", [str(s["n"]) for _, s in summaries])
    add("PASS RATE", [_pct(s["passed"], s["n"]) for _, s in summaries])
    add(
        "  95% interval",
        [f"{100 * s['ci95'][0]:.0f}%-{100 * s['ci95'][1]:.0f}%" for _, s in summaries],
    )
    for tier in TIERS:
        add(f"  {tier}", [_pct(*s["tiers"].get(tier, (0, 0))) for _, s in summaries])
    add("avg turns", [f"{s['avg_turns']:.1f}" for _, s in summaries])
    add("avg tool calls", [f"{s['avg_tool_calls']:.1f}" for _, s in summaries])
    add("avg tool errors", [f"{s['avg_tool_errors']:.1f}" for _, s in summaries])
    add("avg tokens / task", [f"{s['avg_tokens']:,.0f}" for _, s in summaries])
    add("avg API requests / task", [f"{s['avg_requests']:.1f}" for _, s in summaries])
    templates = sorted({t for _, s in summaries for t in s["templates"]})
    add("", ["" for _ in summaries])
    for t in templates:
        add(t, [_pct(*s["templates"].get(t, (0, 0))) for _, s in summaries])

    head = " " * 24 + "".join(name.ljust(width + 2) for name, _ in summaries)
    lines = [head.rstrip()]
    for label, values in rows:
        lines.append(label.ljust(24) + "".join(v.ljust(width + 2) for v in values).rstrip())
    reasons = [f"{name}: {s['stop_reasons']}" for name, s in summaries]
    lines.append("")
    lines.append("stop reasons  " + " | ".join(reasons))
    return "\n".join(lines)
