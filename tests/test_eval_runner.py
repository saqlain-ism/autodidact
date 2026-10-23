"""The evaluation runner: recording, resuming after a quota stop, and reporting."""

import asyncio
import json
import re

import pytest

from autodidact.envs.inventory.tasks import hidden_test_tasks
from autodidact.eval.report import format_report, load, summarize, wilson
from autodidact.eval.runner import RunMismatch, evaluate, select_tasks
from autodidact.llm.base import LLMResponse, QuotaExhausted, ToolCallRequest, Usage

ALL = hidden_test_tasks()
STOCK = [t for t in ALL if t.template == "stock_lookup"]


class StockPolicy:
    """Solves stock-lookup tasks properly: read the SKU, look it up, answer."""

    name = "policy:stock"

    def __init__(self, fail_after: int | None = None):
        self.requests = 0
        self.fail_after = fail_after

    async def chat(self, messages, tools, *, sample=0):
        self.requests += 1
        if self.fail_after is not None and self.requests > self.fail_after:
            raise QuotaExhausted("daily limit reached")
        last = messages[-1]
        if last["role"] == "user":
            sku = re.search(r"[A-Z]{3}-\d{4}", last["content"]).group()
            call = ToolCallRequest("a", "get_product", {"sku": sku})
        else:
            stock = json.loads(last["content"])["stock"]
            call = ToolCallRequest("b", "finish", {"answer": str(stock)})
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
            ],
        }
        return LLMResponse(None, [call], msg, Usage(500, 20, 1))


def test_select_tasks():
    assert len(select_tasks(ALL, per_template=2)) == 30
    assert len(select_tasks(ALL, per_template=1, tiers={"hard"})) == 5
    assert [t.id for t in select_tasks(ALL, ids={"inv-stock_lookup-2"})] == ["inv-stock_lookup-2"]


def test_results_and_transcripts_are_recorded(tmp_path):
    out = asyncio.run(evaluate(StockPolicy(), STOCK, tmp_path, condition="docs-only", runs=2))
    assert out.completed == 8 and not out.stopped_early
    records = load(tmp_path)
    assert len(records) == 8 and all(r["passed"] for r in records)
    assert {r["run"] for r in records} == {0, 1}
    assert records[0]["turns"] == 2 and records[0]["requests"] == 2
    episode = json.loads((tmp_path / "episodes" / f"{STOCK[0].id}__run0.json").read_text())
    assert episode["episode"]["turns"][0]["calls"][0]["name"] == "get_product"


def test_quota_stop_saves_progress_and_resume_finishes(tmp_path):
    first = asyncio.run(evaluate(StockPolicy(fail_after=5), STOCK, tmp_path, condition="docs-only"))
    assert first.stopped_early and first.completed == 2
    assert len(load(tmp_path)) == 2
    second = asyncio.run(evaluate(StockPolicy(), STOCK, tmp_path, condition="docs-only"))
    assert not second.stopped_early and second.completed == 2 and second.skipped == 2
    assert sorted(r["task_id"] for r in load(tmp_path)) == sorted(t.id for t in STOCK)


def test_run_folder_refuses_a_different_model(tmp_path):
    asyncio.run(evaluate(StockPolicy(), STOCK[:1], tmp_path, condition="docs-only"))

    class Other(StockPolicy):
        name = "policy:other"

    with pytest.raises(RunMismatch):
        asyncio.run(evaluate(Other(), STOCK[:1], tmp_path, condition="docs-only"))


def test_wilson_interval():
    lo, hi = wilson(30, 60)
    assert 0.37 < lo < 0.38 and 0.62 < hi < 0.63
    assert wilson(0, 0) == (0.0, 0.0)


def test_report_compares_runs(tmp_path):
    a, b = tmp_path / "before", tmp_path / "after"
    asyncio.run(evaluate(StockPolicy(), STOCK, a, condition="docs-only"))
    asyncio.run(evaluate(StockPolicy(), STOCK[:2], b, condition="learned"))
    s = summarize(load(a))
    assert s["rate"] == 1.0 and s["tiers"]["easy"] == (4, 4)
    text = format_report([a, b])
    assert "before" in text and "after" in text and "100.0% (4/4)" in text and "(2/2)" in text
