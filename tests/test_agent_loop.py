"""The agent loop, driven by a scripted model against the real StockRoom over MCP."""

import asyncio

from conftest import ScriptedLLM, last_tool_result

from autodidact.agent.loop import SYSTEM_PROMPT, compact, run_episode
from autodidact.envs.inventory.tasks import hidden_test_tasks
from autodidact.envs.session import ToolSession
from autodidact.llm.base import LLMError

TASKS = {t.id: t for t in hidden_test_tasks()}


def episode(llm, prompt, seed=1, **kw):
    async def go():
        async with ToolSession(seed) as s:
            before = s.app.snapshot()
            ep = await run_episode(llm, s, prompt, **kw)
            return ep, before, s.app.snapshot()

    return asyncio.run(go())


def test_scripted_agent_solves_a_hidden_task_and_the_checker_agrees():
    task = TASKS["inv-stock_lookup-1"]
    sku = task.prompt.split("units of ")[1].split(" ")[0]
    llm = ScriptedLLM(
        [
            {"content": "Let me look it up.", "calls": [("get_product", {"sku": sku})]},
            lambda msgs: {"calls": [("finish", {"answer": str(last_tool_result(msgs)["stock"])})]},
        ]
    )
    ep, before, after = episode(llm, task.prompt, seed=task.world_seed)
    assert ep.stop_reason == "finished"
    assert task.check(before, after, ep.answer).passed
    assert ep.tool_calls == 2 and ep.tool_errors == 0 and len(ep.turns) == 2


def test_tool_errors_are_shown_to_the_model():
    llm = ScriptedLLM(
        [
            {"calls": [("adjust_stock", {"sku": "BEV-0001", "delta": -1, "reason": "broken"})]},
            {"calls": [("finish", {"answer": "gave up"})]},
        ]
    )
    ep, *_ = episode(llm, "Record one broken unit of BEV-0001.")
    tool_msg = [m for m in llm.requests[1] if m["role"] == "tool"][0]
    assert (
        tool_msg["content"].startswith("ERROR: E_NOT_FOUND")
        or "E_INVALID_REASON" in tool_msg["content"]
    )
    assert ep.tool_errors == 1


def test_assistant_messages_go_back_unchanged_including_thought_signatures():
    llm = ScriptedLLM(
        [
            {"calls": [("list_categories", {}), ("list_adjustment_reasons", {})]},
            {"calls": [("finish", {"answer": "ok"})]},
        ]
    )
    episode(llm, "Look around.")
    second = llm.requests[1]
    assistant = next(m for m in second if m["role"] == "assistant")
    assert [
        tc["extra_content"]["google"]["thought_signature"] for tc in assistant["tool_calls"]
    ] == ["sig-1-0", "sig-1-1"]
    tool_ids = [m["tool_call_id"] for m in second if m["role"] == "tool"]
    assert tool_ids == [tc["id"] for tc in assistant["tool_calls"]]


def test_text_reply_ends_the_episode_with_that_answer():
    ep, *_ = episode(ScriptedLLM([{"content": "There are 12."}]), "How many?")
    assert ep.stop_reason == "text_reply" and ep.answer == "There are 12."


def test_turn_budget_is_enforced():
    llm = ScriptedLLM([{"calls": [("list_categories", {})]}], repeat_last=True)
    ep, *_ = episode(llm, "Loop forever.", max_turns=4)
    assert ep.stop_reason == "max_turns" and len(ep.turns) == 4 and len(llm.requests) == 4


def test_malformed_arguments_are_reported_without_calling_the_tool():
    llm = ScriptedLLM(
        [
            {"calls": [("create_supplier", "{oops")]},
            {"calls": [("finish", {"answer": "x"})]},
        ]
    )
    ep, before, after = episode(llm, "Add a supplier.")
    assert ep.tool_errors == 1 and before == after
    assert "not valid JSON" in [m for m in llm.requests[1] if m["role"] == "tool"][0]["content"]


def test_knowledge_goes_into_the_system_prompt_and_baseline_has_no_app_hints():
    for word in ("paise", "StockRoom", "cursor", "archived", "moq", "draft"):
        assert word.lower() not in SYSTEM_PROMPT.lower()
    llm = ScriptedLLM([{"calls": [("finish", {"answer": "x"})]}])
    episode(llm, "Hi", knowledge="- Prices are in paise.")
    system = llm.requests[0][0]["content"]
    assert system.startswith(SYSTEM_PROMPT) and "Prices are in paise." in system
    names = [t["function"]["name"] for t in llm.tools_seen[0]]
    assert len(names) == 24 and names[-1] == "finish"


def test_model_failure_ends_episode_gracefully():
    class Broken:
        name = "broken"

        async def chat(self, *a, **k):
            raise LLMError("HTTP 400: bad request")

    ep, *_ = episode(Broken(), "Anything")
    assert ep.stop_reason == "llm_error" and "400" in ep.error


def test_compaction_only_shortens_old_tool_results():
    msgs = [{"role": "system", "content": "s"}]
    for i in range(12):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": str(i)}]})
        msgs.append({"role": "tool", "tool_call_id": str(i), "content": "x" * 1000})
    out = compact(msgs)
    tool_lengths = [len(m["content"]) for m in out if m["role"] == "tool"]
    assert all(n < 500 for n in tool_lengths[:4]) and all(n == 1000 for n in tool_lengths[4:])
    assert [m for m in out if m["role"] == "assistant"] == [
        m for m in msgs if m["role"] == "assistant"
    ]
    assert msgs[2]["content"] == "x" * 1000  # the original history is untouched
