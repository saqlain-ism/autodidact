"""The StockRoom app: determinism, quirks and business rules, all over MCP."""

import asyncio

from autodidact.envs.inventory.app import InventoryApp
from autodidact.envs.inventory.world import TEST_WORLD_SEED
from autodidact.envs.session import ToolSession


def run(coro):
    return asyncio.run(coro)


def test_same_seed_same_world_different_seed_different_world():
    a, b = InventoryApp.from_seed(3).snapshot(), InventoryApp.from_seed(3).snapshot()
    assert a == b
    assert InventoryApp.from_seed(4).snapshot()["products"] != a["products"]


def test_test_world_shares_no_product_data_with_practice_worlds():
    test = InventoryApp.from_seed(TEST_WORLD_SEED).snapshot()["products"]
    test_rows = {(r["sku"], r["name"], r["unit_price_paise"]) for r in test.values()}
    for seed in range(1, 21):
        practice = InventoryApp.from_seed(seed).snapshot()["products"]
        practice_rows = {(r["sku"], r["name"], r["unit_price_paise"]) for r in practice.values()}
        assert not test_rows & practice_rows


def test_server_advertises_tools_with_schemas():
    async def go():
        async with ToolSession(1) as s:
            specs = {t["name"]: t for t in await s.tool_specs()}
            assert len(specs) == 23
            assert "lines" in specs["create_purchase_order"]["input_schema"]["properties"]
            assert all(t["description"] for t in specs.values())

    run(go())


def test_pagination_caps_page_size_and_follows_cursor():
    async def go():
        async with ToolSession(1) as s:
            assert not (await s.call("list_products", {"page_size": 50})).ok
            first = await s.must("list_products")
            assert len(first["items"]) == 10 and first["next_cursor"]
            everything = await s.collect("list_products")
            assert len(everything) == len({p["sku"] for p in everything}) > 20
            bad = await s.call("list_products", {"cursor": "garbage"})
            assert "E_BAD_CURSOR" in bad.error

    run(go())


def test_archived_products_hidden_unless_requested():
    async def go():
        async with ToolSession(1) as s:
            archived = [p for p in s.app.snapshot()["products"].values() if p["archived"]]
            assert archived
            active = {p["sku"] for p in await s.collect("list_products")}
            every = {p["sku"] for p in await s.collect("list_products", include_archived=True)}
            assert archived[0]["sku"] not in active and archived[0]["sku"] in every

    run(go())


def test_search_is_capped_at_five():
    async def go():
        async with ToolSession(1) as s:
            short = await s.call("search_products", {"query": "a"})
            assert short.error == "E_VALIDATION: query too short"
            res = await s.must("search_products", query="Kaveri")
            assert len(res["items"]) == 5
            assert res["truncated"] is True

    run(go())


def test_sku_must_match_category_code():
    async def go():
        async with ToolSession(1) as s:
            bad = await s.call(
                "create_product",
                {
                    "sku": "SNK-9000",
                    "name": "Test Tea",
                    "category": "Beverages",
                    "supplier_id": "SUP-01",
                    "unit_price_paise": 1000,
                },
            )
            assert bad.error == "E_INVALID_SKU: sku rejected"
            ok = await s.must(
                "create_product",
                sku="BEV-9000",
                name="Test Tea",
                category="beverages",
                supplier_id="SUP-01",
                unit_price_paise=1000,
            )
            assert ok["stock"] == 0 and ok["category"] == "Beverages"

    run(go())


def test_adjustment_reason_must_come_from_hidden_list():
    async def go():
        async with ToolSession(1) as s:
            sku = (await s.must("list_products"))["items"][0]["sku"]
            assert (
                await s.call("adjust_stock", {"sku": sku, "delta": -1, "reason": "broken"})
            ).error == "E_INVALID_REASON: reason not accepted"
            reasons = (await s.must("list_adjustment_reasons"))["reasons"]
            assert "damage" in reasons
            await s.must("adjust_stock", sku=sku, delta=-1, reason="damage")

    run(go())


def _quote(snap, *, supplier_status="active", min_moq=1, min_pack=1):
    """First quote for an active product matching the filters."""
    return next(
        q
        for q in snap["quotes"].values()
        if q["moq"] >= min_moq
        and q["pack_size"] >= min_pack
        and not snap["products"][(q["sku"],)]["archived"]
        and snap["suppliers"][(q["supplier_id"],)]["status"] == supplier_status
    )


def test_purchase_order_lifecycle_moq_and_case_packs():
    async def go():
        async with ToolSession(1) as s:
            q = _quote(s.app.snapshot(), min_moq=2, min_pack=6)
            sku, sid, moq, pack = q["sku"], q["supplier_id"], q["moq"], q["pack_size"]
            low = await s.call(
                "create_purchase_order",
                {"supplier_id": sid, "lines": [{"sku": sku, "qty": moq - 1}]},
            )
            assert low.error == "E_BELOW_MOQ: quantity rejected for one or more lines"
            before = (await s.must("get_product", sku=sku))["stock"]
            po = await s.must(
                "create_purchase_order", supplier_id=sid, lines=[{"sku": sku, "qty": moq}]
            )
            assert po["status"] == "draft"
            assert po["lines"] == [
                {"sku": sku, "qty": moq, "pack_size": pack, "unit_cost_paise": q["unit_cost_paise"]}
            ]
            assert po["total_cost_paise"] == moq * pack * q["unit_cost_paise"]
            assert (
                "E_INVALID_STATE"
                in (await s.call("receive_purchase_order", {"po_id": po["id"]})).error
            )
            await s.must("submit_purchase_order", po_id=po["id"])
            assert (await s.must("get_product", sku=sku))["stock"] == before  # not yet delivered
            await s.must("receive_purchase_order", po_id=po["id"])
            # Silent trap: quantities are packs, so far more units arrive than "qty".
            assert (await s.must("get_product", sku=sku))["stock"] == before + moq * pack

    run(go())


def test_inactive_supplier_accepts_a_draft_but_refuses_submission():
    async def go():
        async with ToolSession(1) as s:
            q = _quote(s.app.snapshot(), supplier_status="inactive")
            quotes = (await s.must("get_supplier_quotes", sku=q["sku"]))["quotes"]
            assert q["supplier_id"] in {x["supplier_id"] for x in quotes}  # still quoted
            supplier = await s.must("get_supplier", supplier_id=q["supplier_id"])
            assert supplier["status"] == "inactive"
            po = await s.must(
                "create_purchase_order",
                supplier_id=q["supplier_id"],
                lines=[{"sku": q["sku"], "qty": q["moq"]}],
            )
            refused = await s.call("submit_purchase_order", {"po_id": po["id"]})
            assert refused.error == "E_SUPPLIER_UNAVAILABLE: supplier is not accepting orders"
            assert (await s.must("get_purchase_order", po_id=po["id"]))["status"] == "draft"

    run(go())


def test_every_product_has_its_main_supplier_active():
    snap = InventoryApp.from_seed(TEST_WORLD_SEED).snapshot()
    for p in snap["products"].values():
        assert snap["suppliers"][(p["supplier_id"],)]["status"] == "active"


def test_price_update_takes_paise_without_saying_so():
    async def go():
        async with ToolSession(1) as s:
            sku = (await s.must("list_products"))["items"][0]["sku"]
            await s.must("update_product_price", sku=sku, price=78)  # meant as Rs 78
            assert (await s.must("get_product", sku=sku))["unit_price_paise"] == 78  # Rs 0.78

    run(go())


def test_category_filter_takes_the_code_and_a_name_silently_matches_nothing():
    async def go():
        async with ToolSession(1) as s:
            by_name = await s.must("list_products", category="Dairy")
            assert by_name == {"items": [], "next_cursor": None}
            by_code = await s.collect("list_products", category="DRY")
            assert by_code and {p["category"] for p in by_code} == {"Dairy"}

    run(go())


def test_sales_orders_deduct_and_cancellation_restores():
    async def go():
        async with ToolSession(1) as s:
            p = next(p for p in await s.collect("list_products") if p["stock"] >= 2)
            order = await s.must(
                "create_sales_order", customer_name="Test", lines=[{"sku": p["sku"], "qty": 2}]
            )
            assert (await s.must("get_product", sku=p["sku"]))["stock"] == p["stock"] - 2
            too_many = await s.call(
                "create_sales_order",
                {"customer_name": "Test", "lines": [{"sku": p["sku"], "qty": 10_000}]},
            )
            assert "E_INSUFFICIENT_STOCK" in too_many.error
            await s.must("cancel_sales_order", order_id=order["id"])
            assert (await s.must("get_product", sku=p["sku"]))["stock"] == p["stock"]
            orders = await s.collect("list_sales_orders")
            assert orders[-1]["id"] == order["id"]  # oldest first: newest is last
            assert orders[-1]["created_at"] > orders[0]["created_at"]

    run(go())


def test_failed_calls_change_nothing():
    async def go():
        async with ToolSession(1) as s:
            before = s.app.snapshot()
            await s.call(
                "create_sales_order",
                {
                    "customer_name": "X",
                    "lines": [{"sku": "BEV-0001", "qty": 1}, {"sku": "NOPE-0000", "qty": 1}],
                },
            )
            await s.call(
                "create_product",
                {
                    "sku": "bad",
                    "name": "x",
                    "category": "Dairy",
                    "supplier_id": "SUP-01",
                    "unit_price_paise": 1,
                },
            )
            assert s.app.snapshot() == before

    run(go())
