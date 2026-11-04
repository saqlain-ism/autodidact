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


# ---------------------------------------------------------------------------
# Each silent trap decides the outcome: the naive strategy fails the task.


def _task(template, n=1):
    return next(t for t in TASKS if t.id == f"inv-{template}-{n}")


async def _sku_for(s, prompt):
    products = await s.collect("list_products")
    return next(p["sku"] for p in products if p["name"] in prompt)


def test_naive_units_instead_of_packs_fails():
    task = _task("restock_one", 3)  # cheapest supplier sells in packs

    async def naive(s):
        sku = await _sku_for(s, task.prompt)
        target = int(task.prompt.split("at least ")[1].split(" ")[0])
        prod = await s.must("get_product", sku=sku)
        quotes = (await s.must("get_supplier_quotes", sku=sku))["quotes"]
        active = []
        for q in quotes:
            if (await s.must("get_supplier", supplier_id=q["supplier_id"]))["status"] == "active":
                active.append(q)
        best = min(active, key=lambda q: q["unit_cost_paise"])
        units = max(target - prod["stock"], best["moq"])  # treats qty as units
        po = await s.must(
            "create_purchase_order",
            supplier_id=best["supplier_id"],
            lines=[{"sku": sku, "qty": units}],
        )
        await s.must("submit_purchase_order", po_id=po["id"])
        await s.must("receive_purchase_order", po_id=po["id"])

    assert not asyncio.run(_episode(task, naive)).passed


def test_naive_cheapest_quote_from_inactive_supplier_fails():
    task = _task("restock_one", 1)  # cheapest quote is from a supplier that stopped trading

    async def naive(s):
        sku = await _sku_for(s, task.prompt)
        quotes = (await s.must("get_supplier_quotes", sku=sku))["quotes"]
        best = min(quotes, key=lambda q: q["unit_cost_paise"])
        po = await s.must(
            "create_purchase_order",
            supplier_id=best["supplier_id"],
            lines=[{"sku": sku, "qty": best["moq"] * 10}],
        )
        refused = await s.call("submit_purchase_order", {"po_id": po["id"]})
        assert refused.error.startswith("E_SUPPLIER_UNAVAILABLE")

    assert not asyncio.run(_episode(task, naive)).passed


def test_naive_price_in_rupees_fails():
    task = _task("set_price")

    async def naive(s):
        sku = task.prompt.split("price of ")[1].split(" ")[0]
        rupees = float(task.prompt.split("₹")[1].rstrip("."))
        await s.must("update_product_price", sku=sku, price=round(rupees))

    assert not asyncio.run(_episode(task, naive)).passed


def test_naive_category_name_filter_fails():
    task = _task("count_active")

    async def naive(s):
        category = task.prompt.split("in the ")[1].split(" category")[0]
        return str(len(await s.collect("list_products", category=category)))

    assert not asyncio.run(_episode(task, naive)).passed
