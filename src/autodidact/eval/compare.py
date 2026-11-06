"""Paired before/after comparison of two evaluation runs.

Two runs on the same hidden tasks are compared attempt by attempt: the same
task and the same repeat number in both. That pairing matters. Treating the
runs as two independent samples throws away the fact that some tasks are hard
for both; the paired view asks only "on the attempts where the outcome
changed, which way did it change?".

The test is McNemar's exact test: among attempts whose outcome differs
between the runs, how surprising is the split between "fixed" (failed before,
passed after) and "broken" (passed before, failed after) if learning made no
difference (each equally likely)?
"""

from __future__ import annotations

from collections.abc import Iterable
from math import comb
from typing import Any

Record = dict[str, Any]


def mcnemar_exact(fixed: int, broken: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on the discordant pairs)."""
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def pair(before: Iterable[Record], after: Iterable[Record]) -> list[tuple[Record, Record]]:
    b = {(r["task_id"], r["run"]): r for r in before}
    a = {(r["task_id"], r["run"]): r for r in after}
    return [(b[k], a[k]) for k in sorted(b.keys() & a.keys())]


def _tokens(r: Record) -> int:
    return r["prompt_tokens"] + r["completion_tokens"]


def summarize_pairs(pairs: list[tuple[Record, Record]]) -> dict[str, Any]:
    n = len(pairs)
    fixed = [(b["task_id"], b["run"]) for b, a in pairs if not b["passed"] and a["passed"]]
    broken = [(b["task_id"], b["run"]) for b, a in pairs if b["passed"] and not a["passed"]]
    both = [(b, a) for b, a in pairs if b["passed"] and a["passed"]]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "attempts": n,
        "passed_before": sum(b["passed"] for b, _ in pairs),
        "passed_after": sum(a["passed"] for _, a in pairs),
        "fixed": fixed,
        "broken": broken,
        "p_value": mcnemar_exact(len(fixed), len(broken)),
        "tokens_before": mean([_tokens(b) for b, _ in pairs]),
        "tokens_after": mean([_tokens(a) for _, a in pairs]),
        "tokens_both_passed_before": mean([_tokens(b) for b, _ in both]),
        "tokens_both_passed_after": mean([_tokens(a) for _, a in both]),
        "requests_before": mean([b["requests"] for b, _ in pairs]),
        "requests_after": mean([a["requests"] for _, a in pairs]),
    }


def compare(
    before: list[Record], after: list[Record], groups: dict[str, set[str]] | None = None
) -> dict[str, Any]:
    """Overall, per template, and per named group of templates."""
    pairs = pair(before, after)
    templates = sorted({b["template"] for b, _ in pairs})
    out = {
        "overall": summarize_pairs(pairs),
        "templates": {
            t: summarize_pairs([(b, a) for b, a in pairs if b["template"] == t]) for t in templates
        },
        "groups": {},
    }
    for name, members in (groups or {}).items():
        out["groups"][name] = summarize_pairs(
            [(b, a) for b, a in pairs if b["template"] in members]
        )
    return out


def format_comparison(result: dict[str, Any], before_name: str, after_name: str) -> str:
    def line(label: str, s: dict[str, Any]) -> str:
        n = s["attempts"]
        return (
            f"{label:<24} {s['passed_before']:>3}/{n:<3} -> {s['passed_after']:>3}/{n:<3} "
            f"fixed {len(s['fixed']):>2}  broken {len(s['broken']):>2}  p = {s['p_value']:.3f}"
        )

    rows = [f"paired comparison: {before_name}  ->  {after_name}", ""]
    rows.append(line("OVERALL", result["overall"]))
    for name, s in result["groups"].items():
        rows.append(line(name, s))
    rows.append("")
    for t, s in result["templates"].items():
        if s["fixed"] or s["broken"]:
            rows.append(line(t, s))
    o = result["overall"]
    rows += [
        "",
        f"avg tokens per attempt: {o['tokens_before']:,.0f} -> {o['tokens_after']:,.0f}",
        f"  on attempts both passed: {o['tokens_both_passed_before']:,.0f} -> "
        f"{o['tokens_both_passed_after']:,.0f}",
        f"avg API requests per attempt: {o['requests_before']:.1f} -> {o['requests_after']:.1f}",
        "",
        "p is McNemar's exact test on the attempts whose outcome changed.",
    ]
    return "\n".join(rows)
