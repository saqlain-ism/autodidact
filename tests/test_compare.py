"""Paired comparison and McNemar's exact test."""

import asyncio
import json

import pytest
from test_eval_runner import STOCK, StockPolicy

from autodidact import cli
from autodidact.eval.compare import compare, mcnemar_exact, pair
from autodidact.eval.runner import evaluate


@pytest.mark.parametrize(
    ("fixed", "broken", "expected"),
    [
        (7, 0, 0.015625),
        (7, 2, 0.1796875),
        (3, 0, 0.25),
        (1, 1, 1.0),
        (0, 0, 1.0),
        (10, 0, 0.001953125),
    ],
)
def test_mcnemar_exact_known_values(fixed, broken, expected):
    assert mcnemar_exact(fixed, broken) == pytest.approx(expected)
    assert mcnemar_exact(broken, fixed) == pytest.approx(expected)  # symmetric


def rec(task, run, passed, template="t", tokens=100):
    return {
        "task_id": task,
        "run": run,
        "passed": passed,
        "template": template,
        "prompt_tokens": tokens,
        "completion_tokens": 0,
        "requests": 1,
    }


def test_pairs_by_task_and_repeat_and_counts_changes():
    before = [
        rec("a", 0, False, "x"),
        rec("a", 1, True, "x"),
        rec("b", 0, True, "y"),
        rec("c", 0, False),
    ]
    after = [rec("a", 0, True, "x"), rec("a", 1, True, "x"), rec("b", 0, False, "y")]
    assert len(pair(before, after)) == 3  # c has no partner
    result = compare(before, after, groups={"xy": {"x", "y"}})
    o = result["overall"]
    assert (o["passed_before"], o["passed_after"]) == (2, 2)
    assert o["fixed"] == [("a", 0)] and o["broken"] == [("b", 0)]
    assert result["templates"]["x"]["fixed"] == [("a", 0)]
    assert result["groups"]["xy"]["attempts"] == 3


def test_compare_command(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    asyncio.run(evaluate(StockPolicy(), STOCK, a, condition="docs-only"))
    asyncio.run(evaluate(StockPolicy(), STOCK, b, condition="learned"))
    assert cli.main(["compare", str(a), str(b), "--json", str(tmp_path / "c.json")]) == 0
    assert "OVERALL" in capsys.readouterr().out
    assert json.loads((tmp_path / "c.json").read_text())["overall"]["attempts"] == 4


def test_meta_records_every_task_and_repeat_after_resuming(tmp_path):
    asyncio.run(evaluate(StockPolicy(), STOCK[:2], tmp_path, condition="docs-only"))
    asyncio.run(evaluate(StockPolicy(), STOCK[1:3], tmp_path, condition="docs-only", runs=2))
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["runs"] == 2 and meta["task_ids"] == [t.id for t in STOCK[:3]]
