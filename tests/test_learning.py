"""The learning loop, offline: a simulated learner model plays every role."""

import asyncio
import json
import re

import pytest

from autodidact.envs.inventory.world import TEST_WORLD_SEED
from autodidact.eval.contamination import report
from autodidact.learn import prompts
from autodidact.learn.jsonreply import extract_json
from autodidact.learn.knowledge import KnowledgeBase, data_specific
from autodidact.learn.session import State, learn, practice_seed
from autodidact.llm.base import LLMResponse, QuotaExhausted, ToolCallRequest, Usage

PACK_LESSON = (
    "create_purchase_order qty counts packs: units received = qty x pack_size from the quote."
)


class FakeLearner:
    """Plays curriculum, agent, reflector and consolidator.

    As the agent it buys stock once, receives it and reads the product back,
    exactly the read-back that exposes the case-pack trap. As the reflector it
    only writes the pack lesson if the transcript really shows stock rising by
    more than the ordered qty.
    """

    name = "fake:learner"

    def __init__(self, quota_after: int | None = None):
        self.calls = 0
        self.quota_after = quota_after
        self.exercises = 0

    def _reply(self, content=None, calls=()):
        msg = {"role": "assistant", "content": content}
        reqs = []
        if calls:
            msg["tool_calls"] = []
            for i, (name, args) in enumerate(calls):
                cid = f"c{self.calls}_{i}"
                msg["tool_calls"].append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                )
                reqs.append(ToolCallRequest(cid, name, args))
        return LLMResponse(content, reqs, msg, Usage(300, 30, 1))

    async def chat(self, messages, tools, *, sample=0, effort=None):
        self.efforts = getattr(self, "efforts", []) + [effort]
        self.calls += 1
        if self.quota_after is not None and self.calls > self.quota_after:
            raise QuotaExhausted("daily limit reached")
        system = messages[0]["content"]
        if system == prompts.CURRICULUM_SYSTEM:
            self.exercises += 1
            return self._reply(
                json.dumps(
                    {
                        "exercise": "Restock a low product, then confirm its stock.",
                        "focus": "how purchase orders change stock",
                    }
                )
            )
        if system == prompts.REFLECT_SYSTEM:
            return self._reply(json.dumps(self._reflect(messages[1]["content"])))
        if system == prompts.CONSOLIDATE_SYSTEM:
            ids = [int(x) for x in re.findall(r"^\[(\d+)\]", messages[1]["content"], re.M)]
            return self._reply(
                "```json\n"
                + json.dumps(
                    {"lessons": [{"text": PACK_LESSON, "kind": "rule", "replaces": ids[:2]}]}
                )
                + "\n```"
            )
        return self._agent(messages)

    def _reflect(self, user: str) -> dict:
        lessons = [
            {"text": "SUP-01 always has the cheapest quote.", "kind": "rule"}
        ]  # must be rejected
        rose = re.findall(r'"qty": (\d+), "pack_size": (\d+)', user)
        if any(int(p) > 1 for _, p in rose):
            lessons.append({"text": PACK_LESSON, "kind": "rule"})
        lessons.append({"text": "Page size is capped at 20; follow next_cursor.", "kind": "rule"})
        existing = [int(x) for x in re.findall(r"^\[(\d+)\]", user, re.M)]
        return {
            "outcome": "success",
            "evidence": "stock read back",
            "new_lessons": lessons,
            "confirmed": existing[:1],
            "contradicted": [],
        }

    def _agent(self, messages):
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        step = len(tool_msgs)
        if step == 0:
            return self._reply(calls=[("list_products", {"page_size": 20})])
        if step == 1:
            items = json.loads(tool_msgs[0]["content"])["items"]
            self.sku = items[0]["sku"]
            return self._reply(calls=[("get_supplier_quotes", {"sku": self.sku})])
        if step == 2:
            quotes = json.loads(tool_msgs[1]["content"])["quotes"]
            q = max(quotes, key=lambda x: x["pack_size"])
            self.q = q
            return self._reply(calls=[("get_supplier", {"supplier_id": q["supplier_id"]})])
        if step == 3:
            if json.loads(tool_msgs[2]["content"])["status"] != "active":
                return self._reply(calls=[("finish", {"answer": "supplier inactive"})])
            return self._reply(
                calls=[
                    (
                        "create_purchase_order",
                        {
                            "supplier_id": self.q["supplier_id"],
                            "lines": [{"sku": self.sku, "qty": self.q["moq"]}],
                        },
                    )
                ]
            )
        if step == 4:
            po = json.loads(tool_msgs[3]["content"])
            self.po = po["id"]
            return self._reply(calls=[("submit_purchase_order", {"po_id": self.po})])
        if step == 5:
            return self._reply(calls=[("receive_purchase_order", {"po_id": self.po})])
        if step == 6:
            return self._reply(calls=[("get_product", {"sku": self.sku})])
        return self._reply(calls=[("finish", {"answer": "Stock rose by more than I ordered."})])


def run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------- knowledge


@pytest.mark.parametrize(
    "text",
    [
        "SUP-01 always has the cheapest quote.",
        "Order BEV-0412 from PO-5003 again.",
        "Satpura Supplies has stopped trading.",
        "Arjun Pillai orders most often.",
        "Nimbus products are expensive.",
    ],
)
def test_data_specific_lessons_are_rejected(text):
    assert data_specific(text)


@pytest.mark.parametrize(
    "text",
    [
        "Page size is capped at 20; follow next_cursor until it is null.",
        "Filter list_products by category code (e.g. DRY), not the category name.",
        "update_product_price expects paise: Rs 78 is 7800.",
        PACK_LESSON,
    ],
)
def test_general_lessons_are_accepted(text):
    assert data_specific(text) is None


def test_contradicted_lessons_are_retired_and_supported_ones_survive():
    kb = KnowledgeBase()
    a = kb.add("Search returns every match.", "rule", 0)
    b = kb.add("Search returns at most 5 matches and sets truncated.", "rule", 0)
    assert kb.add("Search  returns at most 5 matches and sets truncated.", "rule", 1) is None  # dup
    kb.judge(confirmed=[b.id], contradicted=[a.id])
    assert a.active
    retired = kb.judge(confirmed=[b.id], contradicted=[a.id])
    assert retired == [a] and not a.active and b.active and b.confirmed == 2
    assert "Search returns every match." not in kb.render()


def test_consolidation_keeps_evidence_and_ignores_unknown_ids():
    kb = KnowledgeBase()
    x = kb.add("PO qty is in packs.", "rule", 0)
    y = kb.add("Receiving a PO adds qty times pack size.", "rule", 1)
    z = kb.add("Use calculate for totals.", "procedure", 2)
    kb.judge([x.id, y.id], [])
    kb.replace_active(
        [
            {"text": PACK_LESSON, "kind": "rule", "replaces": [x.id, y.id]},
            {"text": "invented lesson", "kind": "rule", "replaces": [999]},
        ]
    )
    texts = [x.text for x in kb.active()]
    assert PACK_LESSON in texts and z.text in texts and "invented lesson" not in texts
    merged = next(x for x in kb.active() if x.text == PACK_LESSON)
    assert merged.confirmed == 3 and merged.sources == [0, 1]


def test_json_extraction():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": {"b": 2}} done') == {"a": {"b": 2}}
    assert extract_json("no json here") is None


# --------------------------------------------------------------- the loop


def test_practice_never_uses_the_test_world():
    assert all(practice_seed(i) != TEST_WORLD_SEED for i in range(5000))
    with pytest.raises(ValueError):
        practice_seed(TEST_WORLD_SEED - 1000)


def test_learning_discovers_the_pack_trap_from_its_own_read_back(tmp_path):
    outcome = run(learn(FakeLearner(), tmp_path, episodes=3, consolidate_every=2))
    assert outcome.stopped == "done" and outcome.episodes_done == 3
    state = State.load(tmp_path / "state.json")
    knowledge = (tmp_path / "knowledge.md").read_text()
    assert "packs" in knowledge and "SUP-01" not in knowledge
    assert any("SUP-01" in r["text"] for r in state.kb.rejected)
    assert [h["seed"] for h in state.history] == [1000, 1001, 1002]
    assert state.tool_usage["receive_purchase_order"][0] >= 1
    assert (tmp_path / "episodes" / "ep002.json").exists()
    naive = (tmp_path / "naive_memory.md").read_text()
    assert "create_purchase_order" in naive and "Practice mode" not in naive


def test_quota_stop_then_resume_finishes_without_duplicates(tmp_path):
    first = run(learn(FakeLearner(quota_after=15), tmp_path, episodes=3))
    assert first.stopped == "quota" and first.episodes_done == 1
    second = run(learn(FakeLearner(), tmp_path, episodes=3))
    assert second.stopped == "done"
    state = State.load(tmp_path / "state.json")
    assert [h["episode"] for h in state.history] == [0, 1, 2]


def test_request_budget_pauses_the_session(tmp_path):
    outcome = run(learn(FakeLearner(), tmp_path, episodes=10, max_requests=12))
    assert outcome.stopped == "budget" and 1 <= outcome.episodes_done < 10


def test_notebook_is_tied_to_one_model(tmp_path):
    run(learn(FakeLearner(), tmp_path, episodes=1))

    class Other(FakeLearner):
        name = "fake:other"

    with pytest.raises(ValueError, match="belongs to model"):
        run(learn(Other(), tmp_path, episodes=2))


def test_contamination_report():
    from autodidact.envs.inventory.tasks import hidden_test_tasks

    tasks = list(hidden_test_tasks())
    r = report(["Look around the application.", tasks[0].prompt], tasks)
    assert r["exercises"] == 2 and r["near_copies"] == 1 and r["max_similarity"] == 1.0


# ------------------------------------------------------------- learner v2


@pytest.mark.parametrize(
    "prompt",
    [
        prompts.CURRICULUM_SYSTEM,
        prompts.CURRICULUM_USER,
        prompts.PRACTICE_SUFFIX,
        prompts.REFLECT_SYSTEM,
        prompts.REFLECT_USER,
        prompts.CONSOLIDATE_SYSTEM,
    ],
)
def test_prompts_give_no_hints_about_the_app(prompt):
    banned = [
        "pack",
        "paise",
        "rupee",
        "cursor",
        "paginat",
        "inventory",
        "stock",
        "supplier",
        "categor",
        "archiv",
        "sku",
        "truncat",
        "purchase",
        "sales order",
    ]
    lowered = prompt.lower()
    assert [w for w in banned if w in lowered] == []


def test_practice_asks_for_expectations_and_reflection_sees_the_docs(tmp_path):
    seen = []

    class Recording(FakeLearner):
        async def chat(self, messages, tools, *, sample=0, effort=None):
            seen.append(
                (
                    messages[0]["content"],
                    messages[-1]["content"] if messages else "",
                    effort,
                    messages,
                )
            )
            return await super().chat(messages, tools, sample=sample, effort=effort)

    run(learn(Recording(), tmp_path, episodes=1))
    reflect = next(m for system, _, _, m in seen if system == prompts.REFLECT_SYSTEM)
    assert "TOOL DESCRIPTIONS" in reflect[1]["content"]
    assert "create_purchase_order(supplier_id, lines)" in reflect[1]["content"]
    agent_first = next(m for system, _, _, m in seen if system.startswith("You are an operations"))
    assert "EXPECTED vs ACTUAL" in agent_first[1]["content"]
    efforts = {system[:30]: effort for system, _, effort, _ in seen}
    assert efforts[prompts.CURRICULUM_SYSTEM[:30]] == "medium"
    assert efforts[prompts.REFLECT_SYSTEM[:30]] == "medium"
    assert efforts["You are an operations assistan"] is None  # agent steps use the default


def test_a_notebook_from_the_old_learner_is_not_continued(tmp_path):
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "episodes_done": 10,
                "history": [],
                "tool_usage": {},
                "requests": 90,
                "tokens": 1,
                "model": "fake:learner",
                "knowledge": {"lessons": [], "next_id": 1, "rejected": []},
            }
        )
    )
    with pytest.raises(ValueError, match="learner-v1"):
        run(learn(FakeLearner(), tmp_path, episodes=11))
