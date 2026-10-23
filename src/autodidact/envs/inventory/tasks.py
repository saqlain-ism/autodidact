"""The 60 hidden test tasks for StockRoom.

15 templates x 4 instances, in three tiers:

* **easy**: one or two lookups or a single change.
* **medium**: needs a quirk (pagination, archived products, search cap,
  SKU rules, oldest-first ordering) or several coordinated calls.
* **hard**: multi-step workflows (purchase-order lifecycle, minimum order
  quantities, cheapest supplier, conditional logic, bulk updates).

All tasks run on the test world (``TEST_WORLD_SEED``), which the agent never
sees while practising. Parameters are drawn from that world with a fixed seed,
so the task list is identical on every run and every machine.
"""

from __future__ import annotations

import random
import re
from decimal import Decimal
from functools import cache

from autodidact.envs.base import (
    Snapshot,
    TableDiff,
    describe_diff,
    diff,
    paise_to_rupees,
    round_half_up,
)
from autodidact.envs.inventory.app import InventoryApp
from autodidact.envs.inventory.world import BRANDS, CATEGORIES, TEST_WORLD_SEED
from autodidact.envs.session import ToolSession
from autodidact.tasks import Checker, CheckResult, Reference, Task, number_matches

APP = "inventory"

NEW_CUSTOMERS = [
    "Ritu Agarwal",
    "Salim Ansari",
    "Gayatri Kulkarni",
    "Naveen Choudhary",
    "Ayesha Siddiqui",
    "Harish Menon",
    "Divya Bansal",
    "Tanvir Alam",
]

NEW_SUPPLIERS = [
    ("Godavari Foods", "Rajahmundry", 4),
    ("Tapti Traders", "Jalgaon", 6),
    ("Malabar Provisions", "Kozhikode", 3),
    ("Chambal Agencies", "Kota", 8),
]

NEW_ITEMS = {
    "BEV": ["Kesar Badam Drink 200ml", "Tulsi Tea 100g", "Aam Panna 500ml", "Coconut Water 200ml"],
    "SNK": ["Ragi Chips 100g", "Peanut Ladoo 200g", "Baked Samosa 250g", "Jowar Puffs 100g"],
    "STP": ["Brown Rice 1kg", "Masoor Dal 500g", "Jaggery 1kg", "Millet Flour 1kg"],
    "PRC": [
        "Hibiscus Hair Oil 100ml",
        "Charcoal Soap 100g",
        "Rose Water 200ml",
        "Multani Mitti 100g",
    ],
    "HSH": ["Lemon Dish Gel 500ml", "Cotton Mop", "Room Freshener 200ml", "Steel Wool Pack of 6"],
    "DRY": ["Mishti Doi 200g", "Flavoured Milk 200ml", "Rabri 200g", "Malai Paneer 500g"],
}

SKU_IN_TEXT = re.compile(r"[A-Z]{3}-\d{4}")

# What a template yields per task: (tier, prompt, expects_answer, check, reference).
Spec = tuple[str, str, bool, Checker, Reference]


# ===================================================================== world view


class View:
    """Convenient read-only access to a snapshot."""

    def __init__(self, snap: Snapshot):
        self.products = {k[0]: r for k, r in snap["products"].items()}
        self.suppliers = {k[0]: r for k, r in snap["suppliers"].items()}
        self.quotes: dict[str, list[dict]] = {}
        for (sku, _), row in sorted(snap["quotes"].items()):
            self.quotes.setdefault(sku, []).append(row)
        self.orders = {k[0]: r for k, r in snap["sales_orders"].items()}
        self.order_lines: dict[str, list[dict]] = {}
        for (oid, _), row in sorted(snap["so_lines"].items()):
            self.order_lines.setdefault(oid, []).append(row)

    def active(self, category: str | None = None) -> list[dict]:
        return [
            p
            for sku, p in sorted(self.products.items())
            if not p["archived"] and (category is None or p["category"] == category)
        ]

    def cheapest(self, sku: str) -> dict:
        return min(self.quotes[sku], key=lambda q: q["unit_cost_paise"])

    def is_low(self, p: dict) -> bool:
        return p["stock"] < p["reorder_level"]


# ===================================================================== check helpers


def effective_diff(before: Snapshot, after: Snapshot) -> dict[str, TableDiff]:
    """Diff that forgives honest mistakes the agent fully undid.

    New sales or purchase orders that the agent itself cancelled have no net
    business effect, so they (and their lines and stock movements) are ignored.
    Everything else counts.
    """
    d = diff(before, after)
    new_po = d.get("purchase_orders", TableDiff()).added.values()
    new_so = d.get("sales_orders", TableDiff()).added.values()
    dead_po = {r["id"] for r in new_po if r["status"] == "cancelled"}
    dead_so = {r["id"] for r in new_so if r["status"] == "cancelled"}

    def prune(table: str, dead: set, col: str) -> None:
        if table in d:
            d[table].added = {k: r for k, r in d[table].added.items() if r[col] not in dead}

    prune("purchase_orders", dead_po, "id")
    prune("po_lines", dead_po, "po_id")
    prune("sales_orders", dead_so, "id")
    prune("so_lines", dead_so, "order_id")
    prune("stock_movements", dead_so, "ref")
    return {t: td for t, td in d.items() if td}


def first_failure(*results: CheckResult | None) -> CheckResult:
    for r in results:
        if r is not None and not r.passed:
            return r
    return CheckResult.ok()


def only(d: dict[str, TableDiff], *tables: str) -> CheckResult | None:
    extra = set(d) - set(tables)
    if extra:
        return CheckResult.fail("unexpected changes: " + describe_diff({t: d[t] for t in extra}))
    return None


def products_changed_exactly(
    d: dict[str, TableDiff], expected: dict[str, dict]
) -> CheckResult | None:
    """Exactly these products changed, exactly these fields, to exactly these values."""
    td = d.get("products", TableDiff())
    if td.added or td.removed:
        return CheckResult.fail("products were created or deleted")
    changed = {k[0]: pair for k, pair in td.changed.items()}
    if set(changed) != set(expected):
        return CheckResult.fail(f"changed products {sorted(changed)}, expected {sorted(expected)}")
    for sku, (b, a) in changed.items():
        fields = {f for f in a if a[f] != b[f]}
        if fields != set(expected[sku]):
            return CheckResult.fail(
                f"{sku}: changed {sorted(fields)}, expected {sorted(expected[sku])}"
            )
        for f, v in expected[sku].items():
            if a[f] != v:
                return CheckResult.fail(f"{sku}.{f} is {a[f]}, expected {v}")
    return None


def new_rows(d: dict[str, TableDiff], table: str) -> list[dict]:
    return list(d.get(table, TableDiff()).added.values())


def no_existing_rows_changed(d: dict[str, TableDiff], table: str) -> CheckResult | None:
    td = d.get(table, TableDiff())
    if td.changed or td.removed:
        return CheckResult.fail(f"existing {table} were modified")
    return None


def lines_of(d: dict[str, TableDiff], table: str, col: str, parent_id: str) -> set[tuple]:
    return {(r["sku"], r["qty"]) for r in new_rows(d, table) if r[col] == parent_id}


def same_name(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


def rupees(paise: int) -> str:
    return f"₹{paise_to_rupees(paise)}"


# ======================================================================== templates
# Each template returns 4 tasks. Default arguments bind loop variables.


def t_price_lookup(v: View, rng: random.Random) -> list[Spec]:
    tasks = []
    for p in rng.sample(v.active(), 4):

        def check(b, a, ans, price=p["unit_price_paise"]):
            d = effective_diff(b, a)
            return first_failure(only(d), number_matches(ans, paise_to_rupees(price), "0.005"))

        async def ref(s: ToolSession, name=p["name"]):
            res = await s.must("search_products", query=name)
            item = next(i for i in res["items"] if i["name"] == name)
            return rupees(item["unit_price_paise"])

        tasks.append(
            (
                "easy",
                f"What is the selling price of {p['name']}? Reply in rupees.",
                True,
                check,
                ref,
            )
        )
    return tasks


def t_stock_lookup(v: View, rng: random.Random) -> list[Spec]:
    tasks = []
    for p in rng.sample(v.active(), 4):

        def check(b, a, ans, stock=p["stock"]):
            return first_failure(only(effective_diff(b, a)), number_matches(ans, stock))

        async def ref(s: ToolSession, sku=p["sku"]):
            return str((await s.must("get_product", sku=sku))["stock"])

        tasks.append(
            ("easy", f"How many units of {p['sku']} are in stock right now?", True, check, ref)
        )
    return tasks


def t_set_price(v: View, rng: random.Random) -> list[Spec]:
    tasks = []
    for p in rng.sample(v.active(), 4):
        factor = Decimal(rng.choice(["0.85", "0.9", "1.1", "1.2"]))
        new = round_half_up(p["unit_price_paise"] * factor / 50) * 50
        if new == p["unit_price_paise"]:
            new += 500

        def check(b, a, ans, sku=p["sku"], new=new):
            d = effective_diff(b, a)
            return first_failure(
                only(d, "products"), products_changed_exactly(d, {sku: {"unit_price_paise": new}})
            )

        async def ref(s: ToolSession, sku=p["sku"], new=new):
            await s.must("update_product_price", sku=sku, unit_price_paise=new)

        tasks.append(
            ("easy", f"Change the selling price of {p['sku']} to {rupees(new)}.", False, check, ref)
        )
    return tasks


def t_record_damage(v: View, rng: random.Random) -> list[Spec]:
    tasks = []
    for p in rng.sample([p for p in v.active() if p["stock"] >= 5], 4):
        n = rng.randint(2, 4)

        def check(b, a, ans, sku=p["sku"], n=n, stock=p["stock"]):
            d = effective_diff(b, a)
            moves = new_rows(d, "stock_movements")
            bad = [m for m in moves if m["sku"] != sku or m["reason"] != "damage"]
            return first_failure(
                only(d, "products", "stock_movements"),
                products_changed_exactly(d, {sku: {"stock": stock - n}}),
                CheckResult.fail(f"adjustments not recorded as damage: {bad}") if bad else None,
            )

        async def ref(s: ToolSession, name=p["name"], n=n):
            res = await s.must("search_products", query=name)
            sku = next(i["sku"] for i in res["items"] if i["name"] == name)
            await s.must("adjust_stock", sku=sku, delta=-n, reason="damage")

        tasks.append(
            (
                "easy",
                f"{n} units of {p['name']} were found damaged in the storeroom. "
                "Record this in the inventory.",
                False,
                check,
                ref,
            )
        )
    return tasks


def t_supplier_city(v: View, rng: random.Random) -> list[Spec]:
    tasks = []
    for p in rng.sample(v.active(), 4):
        city = v.suppliers[p["supplier_id"]]["city"]

        def check(b, a, ans, city=city):
            found = bool(ans) and city.casefold() in ans.casefold()
            return first_failure(
                only(effective_diff(b, a)),
                None if found else CheckResult.fail(f"expected {city}, got {ans!r}"),
            )

        async def ref(s: ToolSession, sku=p["sku"]):
            prod = await s.must("get_product", sku=sku)
            return (await s.must("get_supplier", supplier_id=prod["supplier_id"]))["city"]

        tasks.append(
            ("easy", f"Which city is the main supplier of {p['sku']} based in?", True, check, ref)
        )
    return tasks


def t_new_product(v: View, rng: random.Random) -> list[Spec]:
    tasks = []
    codes = rng.sample(list(CATEGORIES), 4)
    for code, (sup, city, lead) in zip(codes, NEW_SUPPLIERS, strict=True):
        category = CATEGORIES[code]
        name = f"Pragati {rng.choice(NEW_ITEMS[code])}"
        price = rng.randrange(2000, 40000, 50)
        top = max(int(s.split("-")[1]) for s, p in v.products.items() if p["category"] == category)
        sku = f"{code}-{top + 1:04d}"

        def check(
            b,
            a,
            ans,
            sku=sku,
            name=name,
            category=category,
            price=price,
            sup=sup,
            city=city,
            lead=lead,
        ):
            d = effective_diff(b, a)
            sups, prods = new_rows(d, "suppliers"), new_rows(d, "products")
            if len(sups) != 1 or len(prods) != 1:
                return CheckResult.fail(f"created {len(sups)} suppliers and {len(prods)} products")
            s_, p_ = sups[0], prods[0]
            problems = [
                not same_name(s_["name"], sup) and "supplier name",
                not same_name(s_["city"], city) and "supplier city",
                s_["lead_time_days"] != lead and "lead time",
                p_["sku"] != sku and f"sku {p_['sku']} (expected {sku})",
                not same_name(p_["name"], name) and "product name",
                p_["category"] != category and "category",
                p_["supplier_id"] != s_["id"] and "product not linked to the new supplier",
                p_["unit_price_paise"] != price and f"price {p_['unit_price_paise']} paise",
            ]
            problems = [x for x in problems if x]
            return first_failure(
                only(d, "suppliers", "products"),
                no_existing_rows_changed(d, "suppliers"),
                no_existing_rows_changed(d, "products"),
                CheckResult.fail("wrong: " + ", ".join(problems)) if problems else None,
            )

        async def ref(
            s: ToolSession,
            name=name,
            category=category,
            price=price,
            sup=sup,
            city=city,
            lead=lead,
            code=code,
        ):
            new_sup = await s.must("create_supplier", name=sup, city=city, lead_time_days=lead)
            rows = await s.collect("list_products", category=category, include_archived=True)
            nxt = max(int(r["sku"].split("-")[1]) for r in rows) + 1
            await s.must(
                "create_product",
                sku=f"{code}-{nxt:04d}",
                name=name,
                category=category,
                supplier_id=new_sup["id"],
                unit_price_paise=price,
            )

        tasks.append(
            (
                "medium",
                f"We're adding a new product: {name}, in {category}, selling at "
                f"{rupees(price)}. It comes from a new supplier, {sup} in {city}, with a "
                f"lead time of {lead} days. Register the supplier and create the product "
                "using the next unused SKU number in that category.",
                False,
                check,
                ref,
            )
        )
    return tasks


def t_count_active(v: View, rng: random.Random) -> list[Spec]:
    tasks = []
    for code in rng.sample(list(CATEGORIES), 4):
        category = CATEGORIES[code]
        count = len(v.active(category))

        def check(b, a, ans, count=count):
            return first_failure(only(effective_diff(b, a)), number_matches(ans, count))

        async def ref(s: ToolSession, category=category):
            return str(len(await s.collect("list_products", category=category)))

        tasks.append(
            (
                "medium",
                f"How many active (not archived) products are there in the {category} category?",
                True,
                check,
                ref,
            )
        )
    return tasks


def t_place_order(v: View, rng: random.Random) -> list[Spec]:
    tasks = []
    stocked = [p for p in v.active() if p["stock"] >= 3]
    for cust in rng.sample(NEW_CUSTOMERS, 4):
        p1, p2 = rng.sample(stocked, 2)
        q1, q2 = rng.randint(1, 3), rng.randint(1, 3)

        def check(
            b,
            a,
            ans,
            cust=cust,
            lines={(p1["sku"], q1), (p2["sku"], q2)},
            expected={
                p1["sku"]: {"stock": p1["stock"] - q1},
                p2["sku"]: {"stock": p2["stock"] - q2},
            },
        ):
            d = effective_diff(b, a)
            orders = new_rows(d, "sales_orders")
            if len(orders) != 1:
                return CheckResult.fail(f"{len(orders)} new sales orders, expected 1")
            o = orders[0]
            got = lines_of(d, "so_lines", "order_id", o["id"])
            return first_failure(
                only(d, "sales_orders", "so_lines", "products", "stock_movements"),
                no_existing_rows_changed(d, "sales_orders"),
                None if same_name(o["customer_name"], cust) else CheckResult.fail("wrong customer"),
                None if o["status"] == "placed" else CheckResult.fail("order not placed"),
                None
                if got == lines
                else CheckResult.fail(f"lines {sorted(got)}, expected {sorted(lines)}"),
                products_changed_exactly(d, expected),
            )

        async def ref(s: ToolSession, cust=cust, n1=p1["name"], n2=p2["name"], q1=q1, q2=q2):
            lines = []
            for name, qty in ((n1, q1), (n2, q2)):
                res = await s.must("search_products", query=name)
                lines.append(
                    {"sku": next(i["sku"] for i in res["items"] if i["name"] == name), "qty": qty}
                )
            await s.must("create_sales_order", customer_name=cust, lines=lines)

        tasks.append(
            (
                "medium",
                f"{cust} wants to buy {q1} x {p1['name']} and {q2} x {p2['name']}. "
                "Place the order for them.",
                False,
                check,
                ref,
            )
        )
    return tasks


def t_cancel_latest(v: View, rng: random.Random) -> list[Spec]:
    by_customer: dict[str, list[dict]] = {}
    for o in v.orders.values():
        by_customer.setdefault(o["customer_name"], []).append(o)
    eligible = []
    for cust, orders in sorted(by_customer.items()):
        latest = max(orders, key=lambda o: o["created_at"])
        if len(orders) >= 2 and latest["status"] == "placed":
            eligible.append((cust, latest))
    tasks = []
    for cust, latest in rng.sample(eligible, 4):
        expected = {}
        for ln in v.order_lines[latest["id"]]:
            expected[ln["sku"]] = {"stock": v.products[ln["sku"]]["stock"] + ln["qty"]}

        def check(b, a, ans, oid=latest["id"], expected=expected):
            d = effective_diff(b, a)
            changed = {
                k[0]: after for k, (_, after) in d.get("sales_orders", TableDiff()).changed.items()
            }
            return first_failure(
                only(d, "sales_orders", "products", "stock_movements"),
                CheckResult.fail("new sales orders were created")
                if new_rows(d, "sales_orders")
                else None,
                None
                if set(changed) == {oid} and changed[oid]["status"] == "cancelled"
                else CheckResult.fail(f"cancelled {sorted(changed)}, expected {oid}"),
                products_changed_exactly(d, expected),
            )

        async def ref(s: ToolSession, cust=cust):
            orders = [o for o in await s.collect("list_sales_orders") if o["customer_name"] == cust]
            latest = max(orders, key=lambda o: o["created_at"])
            await s.must("cancel_sales_order", order_id=latest["id"])

        tasks.append(("medium", f"Cancel {cust}'s most recent order.", False, check, ref))
    return tasks


def t_brand_max(v: View, rng: random.Random) -> list[Spec]:
    eligible = []
    for brand in BRANDS:
        matches = [p for p in v.active() if brand.casefold() in p["name"].casefold()]
        prices = sorted((p["unit_price_paise"] for p in matches), reverse=True)
        if len(matches) > 5 and prices[0] != prices[1]:
            eligible.append((brand, max(matches, key=lambda p: p["unit_price_paise"])))
    tasks = []
    for brand, top in rng.sample(eligible, 4):

        def check(b, a, ans, sku=top["sku"]):
            found = set(SKU_IN_TEXT.findall((ans or "").upper()))
            return first_failure(
                only(effective_diff(b, a)),
                None
                if found == {sku}
                else CheckResult.fail(f"answered {sorted(found)}, expected {sku}"),
            )

        async def ref(s: ToolSession, brand=brand):
            rows = [
                p
                for p in await s.collect("list_products")
                if brand.casefold() in p["name"].casefold()
            ]
            return max(rows, key=lambda p: p["unit_price_paise"])["sku"]

        tasks.append(
            (
                "medium",
                f'Among active products with "{brand}" in their name, which one '
                "has the highest selling price? Reply with its SKU.",
                True,
                check,
                ref,
            )
        )
    return tasks


def _restock_qty(v: View, p: dict, target: int) -> tuple[dict, int]:
    best = v.cheapest(p["sku"])
    return best, max(target - p["stock"], best["moq"])


async def _ref_restock(s: ToolSession, sku: str, target: int, receive: bool) -> None:
    prod = await s.must("get_product", sku=sku)
    quotes = (await s.must("get_supplier_quotes", sku=sku))["quotes"]
    best = min(quotes, key=lambda q: q["unit_cost_paise"])
    qty = max(target - prod["stock"], best["moq"])
    po = await s.must(
        "create_purchase_order", supplier_id=best["supplier_id"], lines=[{"sku": sku, "qty": qty}]
    )
    await s.must("submit_purchase_order", po_id=po["id"])
    if receive:
        await s.must("receive_purchase_order", po_id=po["id"])


def _check_pos(d: dict[str, TableDiff], expected: set[tuple], status: str) -> CheckResult | None:
    """New purchase orders hold exactly the ``expected`` (supplier, sku, qty) lines,
    and every one of them is in ``status``."""
    pos = {r["id"]: r for r in new_rows(d, "purchase_orders")}
    got = {
        (pos[r["po_id"]]["supplier_id"], r["sku"], r["qty"])
        for r in new_rows(d, "po_lines")
        if r["po_id"] in pos
    }
    wrong_state = sorted(pid for pid, r in pos.items() if r["status"] != status)
    return first_failure(
        no_existing_rows_changed(d, "purchase_orders"),
        CheckResult.fail(f"purchase orders {wrong_state} not {status}") if wrong_state else None,
        None
        if got == expected
        else CheckResult.fail(f"ordered {sorted(got)}, expected {sorted(expected)}"),
    )


def t_restock_one(v: View, rng: random.Random) -> list[Spec]:
    tasks = []
    for p in rng.sample([p for p in v.active() if v.is_low(p)], 4):
        target = 2 * p["reorder_level"]
        best, qty = _restock_qty(v, p, target)

        def check(b, a, ans, sku=p["sku"], sid=best["supplier_id"], qty=qty, stock=p["stock"]):
            d = effective_diff(b, a)
            return first_failure(
                only(d, "purchase_orders", "po_lines", "products", "stock_movements"),
                _check_pos(d, {(sid, sku, qty)}, "received"),
                products_changed_exactly(d, {sku: {"stock": stock + qty}}),
            )

        async def ref(s: ToolSession, name=p["name"], target=target):
            res = await s.must("search_products", query=name)
            sku = next(i["sku"] for i in res["items"] if i["name"] == name)
            await _ref_restock(s, sku, target, receive=True)

        tasks.append(
            (
                "hard",
                f"{p['name']} is running low. Buy enough from whichever supplier "
                f"offers the lowest unit cost to bring stock up to at least {target} units, "
                "ordering as few units as that supplier allows, and record the delivery "
                "as received.",
                False,
                check,
                ref,
            )
        )
    return tasks


def t_restock_category(v: View, rng: random.Random) -> list[Spec]:
    eligible = [c for c in CATEGORIES.values() if sum(v.is_low(p) for p in v.active(c)) >= 2]
    tasks = []
    for category in rng.sample(eligible, 4):
        expected_pos, expected_stock = set(), {}
        for p in v.active(category):
            if v.is_low(p):
                best, qty = _restock_qty(v, p, 2 * p["reorder_level"])
                expected_pos.add((best["supplier_id"], p["sku"], qty))
                expected_stock[p["sku"]] = {"stock": p["stock"] + qty}

        def check(b, a, ans, pos=frozenset(expected_pos), stock=expected_stock):
            d = effective_diff(b, a)
            return first_failure(
                only(d, "purchase_orders", "po_lines", "products", "stock_movements"),
                _check_pos(d, set(pos), "received"),
                products_changed_exactly(d, stock),
            )

        async def ref(s: ToolSession, category=category):
            for row in await s.collect("list_products", category=category):
                prod = await s.must("get_product", sku=row["sku"])
                if prod["stock"] < prod["reorder_level"]:
                    await _ref_restock(s, prod["sku"], 2 * prod["reorder_level"], receive=True)

        tasks.append(
            (
                "hard",
                f"Restock the {category} category: every active product in it that "
                "is below its reorder level should be brought up to at least twice its "
                "reorder level. Buy each one from its cheapest supplier, order as few units "
                "as allowed, and receive the deliveries.",
                False,
                check,
                ref,
            )
        )
    return tasks


def t_stock_value(v: View, rng: random.Random) -> list[Spec]:
    tasks = []
    for code in rng.sample(list(CATEGORIES), 4):
        category = CATEGORIES[code]
        total = sum(p["unit_price_paise"] * p["stock"] for p in v.active(category))

        def check(b, a, ans, total=total):
            return first_failure(
                only(effective_diff(b, a)), number_matches(ans, paise_to_rupees(total), "0.005")
            )

        async def ref(s: ToolSession, category=category):
            rows = await s.collect("list_products", category=category)
            return rupees(sum(r["unit_price_paise"] * r["stock"] for r in rows))

        tasks.append(
            (
                "hard",
                f"What is the total retail value of the stock we hold for active "
                f"products in {category} (selling price x units in stock, summed over "
                "products)? Answer in rupees.",
                True,
                check,
                ref,
            )
        )
    return tasks


def t_price_hike(v: View, rng: random.Random) -> list[Spec]:
    eligible = [
        sid for sid in sorted(v.suppliers) if sum(p["supplier_id"] == sid for p in v.active()) >= 3
    ]
    tasks = []
    for sid, pct in zip(rng.sample(eligible, 4), rng.sample([5, 8, 10, 12, 15], 4), strict=True):
        name = v.suppliers[sid]["name"]
        expected = {
            p["sku"]: {
                "unit_price_paise": round_half_up(
                    Decimal(p["unit_price_paise"]) * (100 + pct) / 100
                )
            }
            for p in v.active()
            if p["supplier_id"] == sid
        }

        def check(b, a, ans, expected=expected):
            d = effective_diff(b, a)
            return first_failure(only(d, "products"), products_changed_exactly(d, expected))

        async def ref(s: ToolSession, name=name, pct=pct):
            sups = await s.collect("list_suppliers")
            sid = next(x["id"] for x in sups if x["name"] == name)
            for row in await s.collect("list_products"):
                if row["supplier_id"] == sid:
                    new = round_half_up(Decimal(row["unit_price_paise"]) * (100 + pct) / 100)
                    await s.must("update_product_price", sku=row["sku"], unit_price_paise=new)

        tasks.append(
            (
                "hard",
                f"{name} has raised its prices. Increase the selling price of every "
                f"active product whose main supplier is {name} by {pct}%, rounding to the "
                "nearest paisa.",
                False,
                check,
                ref,
            )
        )
    return tasks


def t_fulfil_or_backorder(v: View, rng: random.Random) -> list[Spec]:
    enough = rng.sample([p for p in v.active() if p["stock"] >= 4], 2)
    short = rng.sample([p for p in v.active() if 1 <= p["stock"] <= 8], 2)
    custs = rng.sample(NEW_CUSTOMERS, 4)
    cases = [(enough[0], False), (short[0], True), (enough[1], False), (short[1], True)]
    tasks = []
    for (p, is_short), cust in zip(cases, custs, strict=True):
        q = p["stock"] + rng.randint(3, 10) if is_short else rng.randint(2, min(p["stock"], 6))
        sold = p["stock"] if is_short else q
        best = v.cheapest(p["sku"])
        po = (
            {(best["supplier_id"], p["sku"], max(q - p["stock"], best["moq"]))}
            if is_short
            else set()
        )

        def check(
            b, a, ans, cust=cust, sku=p["sku"], sold=sold, stock=p["stock"], po=frozenset(po)
        ):
            d = effective_diff(b, a)
            orders = new_rows(d, "sales_orders")
            if len(orders) != 1:
                return CheckResult.fail(f"{len(orders)} new sales orders, expected 1")
            o = orders[0]
            got = lines_of(d, "so_lines", "order_id", o["id"])
            return first_failure(
                only(
                    d,
                    "sales_orders",
                    "so_lines",
                    "products",
                    "stock_movements",
                    *(("purchase_orders", "po_lines") if po else ()),
                ),
                None if same_name(o["customer_name"], cust) else CheckResult.fail("wrong customer"),
                None
                if got == {(sku, sold)}
                else CheckResult.fail(f"sold {sorted(got)}, expected {sku} x {sold}"),
                products_changed_exactly(d, {sku: {"stock": stock - sold}}),
                _check_pos(d, set(po), "submitted") if po else None,
            )

        async def ref(s: ToolSession, cust=cust, name=p["name"], q=q):
            res = await s.must("search_products", query=name)
            sku = next(i["sku"] for i in res["items"] if i["name"] == name)
            prod = await s.must("get_product", sku=sku)
            sell = min(q, prod["stock"])
            await s.must(
                "create_sales_order", customer_name=cust, lines=[{"sku": sku, "qty": sell}]
            )
            if sell < q:
                quotes = (await s.must("get_supplier_quotes", sku=sku))["quotes"]
                best = min(quotes, key=lambda x: x["unit_cost_paise"])
                order = await s.must(
                    "create_purchase_order",
                    supplier_id=best["supplier_id"],
                    lines=[{"sku": sku, "qty": max(q - sell, best["moq"])}],
                )
                await s.must("submit_purchase_order", po_id=order["id"])

        tasks.append(
            (
                "hard",
                f"{cust} wants {q} x {p['name']}. If we have enough stock, just place "
                "the order. If not, place an order for everything we have, and raise a "
                "purchase order with the cheapest supplier for the shortfall (as few units "
                "as that supplier allows). Submit it, but it has not been delivered yet.",
                False,
                check,
                ref,
            )
        )
    return tasks


TEMPLATES = [
    t_price_lookup,
    t_stock_lookup,
    t_set_price,
    t_record_damage,
    t_supplier_city,
    t_new_product,
    t_count_active,
    t_place_order,
    t_cancel_latest,
    t_brand_max,
    t_restock_one,
    t_restock_category,
    t_stock_value,
    t_price_hike,
    t_fulfil_or_backorder,
]


@cache
def hidden_test_tasks() -> tuple[Task, ...]:
    """The hidden test set: 60 tasks on the test world, identical on every run."""
    view = View(InventoryApp.from_seed(TEST_WORLD_SEED).snapshot())
    out = []
    for template in TEMPLATES:
        name = template.__name__.removeprefix("t_")
        rng = random.Random(f"{TEST_WORLD_SEED}:{name}")
        for i, (tier, prompt, expects_answer, check, ref) in enumerate(
            template(view, rng), start=1
        ):
            out.append(
                Task(
                    id=f"inv-{name}-{i}",
                    app=APP,
                    tier=tier,
                    template=name,
                    prompt=prompt,
                    world_seed=TEST_WORLD_SEED,
                    expects_answer=expects_answer,
                    check=check,
                    reference=ref,
                )
            )
    return tuple(out)
