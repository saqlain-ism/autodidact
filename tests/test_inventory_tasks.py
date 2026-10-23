"""The hidden test set is fair: every task is solvable and every checker is honest.

* A scripted reference solution, using only the MCP tools, passes every task.
* Doing nothing fails every task that asks for a change.
* A wrong answer fails every task that asks a question.
* Sabotage (a correct solution plus one unrelated change) fails, proving the
  checkers look at side effects and not just at the target.
"""

import asyncio
from collections import Counter

import pytest

from autodidact.envs.inventory.tasks import hidden_test_tasks
from autodidact.envs.session import ToolSession
from autodidact.tasks import parse_number

TASKS = hidden_test_tasks()


async def _episode(task, solve):
    async with ToolSession(task.world_seed) as s:
        before = s.app.snapshot()
        answer = await solve(s)
        return task.check(before, s.app.snapshot(), answer)


def test_shape_of_the_test_set():
    assert len(TASKS) == 60
    assert Counter(t.tier for t in TASKS) == {"easy": 20, "medium": 20, "hard": 20}
    assert len({t.id for t in TASKS}) == 60
    assert len({t.prompt for t in TASKS}) == 60


def test_test_set_is_deterministic():
    hidden_test_tasks.cache_clear()
    again = hidden_test_tasks()
    assert [t.prompt for t in again] == [t.prompt for t in TASKS]


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_reference_solution_passes(task):
    result = asyncio.run(_episode(task, task.reference))
    assert result.passed, result.reason


@pytest.mark.parametrize("task", [t for t in TASKS if not t.expects_answer], ids=lambda t: t.id)
def test_doing_nothing_fails(task):
    async def nothing(s):
        return "Done."

    assert not asyncio.run(_episode(task, nothing)).passed


@pytest.mark.parametrize("task", [t for t in TASKS if t.expects_answer], ids=lambda t: t.id)
def test_wrong_answer_fails(task):
    async def wrong(s):
        answer = await task.reference(s)
        number = parse_number(answer)
        if number is not None and "-" not in answer:
            return str(number + 1)
        return "Mumbai" if answer != "Mumbai" else "Delhi"  # wrong city / wrong SKU

    assert not asyncio.run(_episode(task, wrong)).passed


@pytest.mark.parametrize("task", TASKS[::3], ids=lambda t: t.id)
def test_side_effects_are_caught(task):
    async def sabotage(s):
        answer = await task.reference(s)
        victim = next(
            p
            for p in s.app.snapshot()["products"].values()
            if p["sku"] not in str(answer) and not p["archived"]
        )
        await s.must("set_product_archived", sku=victim["sku"], archived=True)
        return answer

    assert not asyncio.run(_episode(task, sabotage)).passed


def test_self_cancelled_mistakes_are_forgiven():
    """An agent that makes a wrong order and cancels it should not be penalised."""
    task = next(t for t in TASKS if t.template == "place_order")

    async def mistake_then_fix(s):
        p = next(p for p in await s.collect("list_products") if p["stock"] >= 1)
        wrong = await s.must(
            "create_sales_order", customer_name="Oops", lines=[{"sku": p["sku"], "qty": 1}]
        )
        await s.must("cancel_sales_order", order_id=wrong["id"])
        return await task.reference(s)

    assert asyncio.run(_episode(task, mistake_then_fix)).passed
