"""StockRoom: a small inventory app the agent has never seen.

This is the "unfamiliar software" Autodidact must teach itself. It behaves like
a real internal tool, including the parts that trip people up. The quirks are
deliberate, and each one is something the agent can only learn by exploring:

1. **Money is in paise.** ``unit_price_paise=24950`` means ₹249.50.
2. **Cursor pagination.** Lists return at most 20 items and a ``next_cursor``.
3. **Archived products are hidden** from lists and search unless asked for.
4. **Search is capped at 5 results** and only says ``truncated: true``.
5. **SKUs must match the category code** (``BEV-0042`` for Beverages). The error
   just says the SKU was rejected.
6. **Stock adjustments need a reason** from a fixed list that is not in the
   tool descriptions; a separate tool lists them.
7. **Purchase orders have a lifecycle:** draft, then submitted, then received.
   Stock only changes on receipt.
8. **Suppliers have minimum order quantities.** Ordering less is rejected with
   a vague message; the MOQ is visible in the supplier quotes.
9. **The primary supplier is not always the cheapest.**
10. **Sales orders list oldest first**, so "most recent" is on the last page.

All methods return plain dicts and raise ``AppError`` for business-rule errors.
"""

from __future__ import annotations

import base64
import re
import sqlite3
from typing import Any

from autodidact.envs.base import AppError, Snapshot, snapshot
from autodidact.envs.inventory.world import build_world

MAX_PAGE = 20
ADJUSTMENT_REASONS = ["damage", "expired", "theft", "count_correction", "customer_return"]
SKU_RE = re.compile(r"^[A-Z]{3}-\d{4}$")

TABLE_KEYS = {
    "categories": ("code",),
    "suppliers": ("id",),
    "products": ("sku",),
    "quotes": ("sku", "supplier_id"),
    "purchase_orders": ("id",),
    "po_lines": ("po_id", "sku"),
    "sales_orders": ("id",),
    "so_lines": ("order_id", "sku"),
    "stock_movements": ("id",),
}


class InventoryApp:
    """Business logic over one world's database."""

    def __init__(self, conn: sqlite3.Connection):
        self.db = conn
        self.db.row_factory = sqlite3.Row

    @classmethod
    def from_seed(cls, seed: int) -> InventoryApp:
        return cls(build_world(seed))

    def snapshot(self) -> Snapshot:
        self.db.row_factory = None
        try:
            return snapshot(self.db, TABLE_KEYS)
        finally:
            self.db.row_factory = sqlite3.Row

    # ------------------------------------------------------------------ helpers

    def _one(self, sql: str, args: tuple = ()) -> sqlite3.Row | None:
        return self.db.execute(sql, args).fetchone()

    def _all(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        return self.db.execute(sql, args).fetchall()

    def _product(self, sku: str) -> sqlite3.Row:
        row = self._one("SELECT * FROM products WHERE sku = ?", (sku,))
        if row is None:
            raise AppError("E_NOT_FOUND", f"no product with sku '{sku}'")
        return row

    def _supplier(self, supplier_id: str) -> sqlite3.Row:
        row = self._one("SELECT * FROM suppliers WHERE id = ?", (supplier_id,))
        if row is None:
            raise AppError("E_NOT_FOUND", f"no supplier with id '{supplier_id}'")
        return row

    def _category_name(self, category: str) -> str:
        row = self._one("SELECT name FROM categories WHERE lower(name) = lower(?)", (category,))
        if row is None:
            raise AppError("E_NOT_FOUND", f"unknown category '{category}'")
        return row["name"]

    def _now(self) -> str:
        """Deterministic clock: each write moves time forward by 7 minutes from 1 Nov 2026."""
        minutes = int(self._one("SELECT value FROM meta WHERE key = 'clock_minutes'")[0]) + 7
        self.db.execute("UPDATE meta SET value = ? WHERE key = 'clock_minutes'", (str(minutes),))
        day, rem = divmod(9 * 60 + minutes, 24 * 60)
        return f"2026-11-{1 + day:02d}T{rem // 60:02d}:{rem % 60:02d}:00"

    def _next_id(self, table: str, prefix: str, start: int) -> str:
        rows = self._all(f"SELECT id FROM {table}")
        nums = [int(r["id"].split("-")[1]) for r in rows]
        return f"{prefix}-{max(nums, default=start - 1) + 1}"

    @staticmethod
    def _page(rows: list[dict], cursor: str | None, page_size: int) -> dict[str, Any]:
        if not 1 <= page_size <= MAX_PAGE:
            raise AppError("E_VALIDATION", "invalid page_size")
        offset = 0
        if cursor:
            try:
                kind, value = base64.urlsafe_b64decode(cursor.encode()).decode().split(":")
                assert kind == "o"
                offset = int(value)
            except Exception as exc:
                raise AppError("E_BAD_CURSOR", "cursor is invalid or expired") from exc
        items = rows[offset : offset + page_size]
        end = offset + page_size
        nxt = base64.urlsafe_b64encode(f"o:{end}".encode()).decode() if end < len(rows) else None
        return {"items": items, "next_cursor": nxt}

    def _move(self, sku: str, delta: int, reason: str, ref: str | None) -> None:
        self.db.execute("UPDATE products SET stock = stock + ? WHERE sku = ?", (delta, sku))
        self.db.execute(
            "INSERT INTO stock_movements (sku, delta, reason, ref) VALUES (?, ?, ?, ?)",
            (sku, delta, reason, ref),
        )

    @staticmethod
    def _check_lines(lines: list[dict]) -> None:
        if not lines:
            raise AppError("E_VALIDATION", "an order needs at least one line")
        skus = [ln["sku"] for ln in lines]
        if len(set(skus)) != len(skus):
            raise AppError("E_VALIDATION", "each sku may appear only once per order")
        if any(ln["qty"] <= 0 for ln in lines):
            raise AppError("E_VALIDATION", "quantities must be positive")

    # --------------------------------------------------------------- catalogue

    def list_categories(self) -> dict:
        return {
            "items": [dict(r) for r in self._all("SELECT code, name FROM categories ORDER BY name")]
        }

    def list_products(
        self,
        cursor: str | None = None,
        page_size: int = 10,
        category: str | None = None,
        include_archived: bool = False,
    ) -> dict:
        sql = (
            "SELECT sku, name, category, supplier_id, unit_price_paise, stock FROM products "
            "WHERE 1=1"
        )
        args: list = []
        if category is not None:
            sql += " AND category = ?"
            args.append(self._category_name(category))
        if not include_archived:
            sql += " AND archived = 0"
        rows = [dict(r) for r in self._all(sql + " ORDER BY sku", tuple(args))]
        return self._page(rows, cursor, page_size)

    def get_product(self, sku: str) -> dict:
        row = dict(self._product(sku))
        row["archived"] = bool(row["archived"])
        return row

    def search_products(self, query: str) -> dict:
        if len(query.strip()) < 2:
            raise AppError("E_VALIDATION", "query too short")
        rows = self._all(
            "SELECT sku, name, category, unit_price_paise, stock FROM products "
            "WHERE archived = 0 AND instr(lower(name), lower(?)) > 0 ORDER BY sku",
            (query.strip(),),
        )
        return {"items": [dict(r) for r in rows[:5]], "truncated": len(rows) > 5}

    def create_product(
        self,
        sku: str,
        name: str,
        category: str,
        supplier_id: str,
        unit_price_paise: int,
        reorder_level: int = 10,
    ) -> dict:
        cat = self._category_name(category)
        code = self._one("SELECT code FROM categories WHERE name = ?", (cat,))["code"]
        if not SKU_RE.match(sku) or not sku.startswith(code + "-"):
            raise AppError("E_INVALID_SKU", "sku rejected")
        if self._one("SELECT 1 FROM products WHERE sku = ?", (sku,)):
            raise AppError("E_DUPLICATE", "sku already exists")
        if self._one("SELECT 1 FROM products WHERE lower(name) = lower(?)", (name.strip(),)):
            raise AppError("E_DUPLICATE", "a product with this name already exists")
        self._supplier(supplier_id)
        if unit_price_paise <= 0 or reorder_level < 0:
            raise AppError("E_VALIDATION", "invalid price or reorder level")
        self.db.execute(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, 0, ?, 0)",
            (sku, name.strip(), cat, supplier_id, unit_price_paise, reorder_level),
        )
        return self.get_product(sku)

    def update_product_price(self, sku: str, unit_price_paise: int) -> dict:
        self._product(sku)
        if unit_price_paise <= 0:
            raise AppError("E_VALIDATION", "invalid price")
        self.db.execute(
            "UPDATE products SET unit_price_paise = ? WHERE sku = ?", (unit_price_paise, sku)
        )
        return self.get_product(sku)

    def set_product_archived(self, sku: str, archived: bool) -> dict:
        self._product(sku)
        self.db.execute("UPDATE products SET archived = ? WHERE sku = ?", (int(archived), sku))
        return self.get_product(sku)

    def list_adjustment_reasons(self) -> dict:
        return {"reasons": list(ADJUSTMENT_REASONS)}

    def adjust_stock(self, sku: str, delta: int, reason: str) -> dict:
        product = self._product(sku)
        if reason not in ADJUSTMENT_REASONS:
            raise AppError("E_INVALID_REASON", "reason not accepted")
        if delta == 0:
            raise AppError("E_VALIDATION", "delta must not be zero")
        if product["stock"] + delta < 0:
            raise AppError("E_INSUFFICIENT_STOCK", "adjustment would make stock negative")
        self._move(sku, delta, reason, None)
        return {"sku": sku, "stock": product["stock"] + delta}

    # --------------------------------------------------------------- suppliers

    def list_suppliers(self, cursor: str | None = None, page_size: int = 10) -> dict:
        rows = [dict(r) for r in self._all("SELECT * FROM suppliers ORDER BY id")]
        return self._page(rows, cursor, page_size)

    def get_supplier(self, supplier_id: str) -> dict:
        return dict(self._supplier(supplier_id))

    def create_supplier(self, name: str, city: str, lead_time_days: int) -> dict:
        if self._one("SELECT 1 FROM suppliers WHERE lower(name) = lower(?)", (name.strip(),)):
            raise AppError("E_DUPLICATE", "supplier already exists")
        if not 1 <= lead_time_days <= 60 or not city.strip():
            raise AppError("E_VALIDATION", "invalid supplier details")
        nums = [int(r["id"].split("-")[1]) for r in self._all("SELECT id FROM suppliers")]
        sid = f"SUP-{max(nums, default=0) + 1:02d}"
        self.db.execute(
            "INSERT INTO suppliers VALUES (?, ?, ?, ?)",
            (sid, name.strip(), city.strip(), lead_time_days),
        )
        return self.get_supplier(sid)

    def get_supplier_quotes(self, sku: str) -> dict:
        self._product(sku)
        rows = self._all(
            "SELECT q.supplier_id, s.name AS supplier_name, q.unit_cost_paise, q.moq "
            "FROM quotes q JOIN suppliers s ON s.id = q.supplier_id "
            "WHERE q.sku = ? ORDER BY q.supplier_id",
            (sku,),
        )
        return {"sku": sku, "quotes": [dict(r) for r in rows]}

    # --------------------------------------------------------- purchase orders

    def create_purchase_order(self, supplier_id: str, lines: list[dict]) -> dict:
        self._supplier(supplier_id)
        self._check_lines(lines)
        priced = []
        for ln in lines:
            product = self._product(ln["sku"])
            if product["archived"]:
                raise AppError("E_ARCHIVED", f"product '{ln['sku']}' is archived")
            quote = self._one(
                "SELECT * FROM quotes WHERE sku = ? AND supplier_id = ?", (ln["sku"], supplier_id)
            )
            if quote is None:
                raise AppError("E_NO_QUOTE", f"supplier does not supply '{ln['sku']}'")
            priced.append((ln["sku"], ln["qty"], quote["unit_cost_paise"], quote["moq"]))
        if any(qty < moq for _, qty, _, moq in priced):
            raise AppError("E_BELOW_MOQ", "quantity rejected for one or more lines")
        pid = self._next_id("purchase_orders", "PO", 5001)
        self.db.execute("INSERT INTO purchase_orders VALUES (?, ?, 'draft')", (pid, supplier_id))
        for sku, qty, cost, _ in priced:
            self.db.execute("INSERT INTO po_lines VALUES (?, ?, ?, ?)", (pid, sku, qty, cost))
        return self.get_purchase_order(pid)

    def _po(self, po_id: str) -> sqlite3.Row:
        row = self._one("SELECT * FROM purchase_orders WHERE id = ?", (po_id,))
        if row is None:
            raise AppError("E_NOT_FOUND", f"no purchase order '{po_id}'")
        return row

    def get_purchase_order(self, po_id: str) -> dict:
        po = dict(self._po(po_id))
        po["lines"] = [
            dict(r)
            for r in self._all(
                "SELECT sku, qty, unit_cost_paise FROM po_lines WHERE po_id = ? ORDER BY sku",
                (po_id,),
            )
        ]
        po["total_cost_paise"] = sum(ln["qty"] * ln["unit_cost_paise"] for ln in po["lines"])
        return po

    def list_purchase_orders(self, status: str | None = None) -> dict:
        rows = self._all("SELECT id FROM purchase_orders ORDER BY id")
        items = [self.get_purchase_order(r["id"]) for r in rows]
        if status is not None:
            items = [p for p in items if p["status"] == status]
        return {"items": items}

    def _transition(self, po_id: str, allowed: tuple[str, ...], new: str) -> sqlite3.Row:
        po = self._po(po_id)
        if po["status"] not in allowed:
            raise AppError("E_INVALID_STATE", "purchase order cannot do that in its current state")
        self.db.execute("UPDATE purchase_orders SET status = ? WHERE id = ?", (new, po_id))
        return po

    def submit_purchase_order(self, po_id: str) -> dict:
        self._transition(po_id, ("draft",), "submitted")
        return self.get_purchase_order(po_id)

    def receive_purchase_order(self, po_id: str) -> dict:
        self._transition(po_id, ("submitted",), "received")
        for ln in self._all("SELECT sku, qty FROM po_lines WHERE po_id = ?", (po_id,)):
            self._move(ln["sku"], ln["qty"], "restock", po_id)
        return self.get_purchase_order(po_id)

    def cancel_purchase_order(self, po_id: str) -> dict:
        self._transition(po_id, ("draft", "submitted"), "cancelled")
        return self.get_purchase_order(po_id)

    # ------------------------------------------------------------ sales orders

    def create_sales_order(self, customer_name: str, lines: list[dict]) -> dict:
        if not customer_name.strip():
            raise AppError("E_VALIDATION", "customer_name is required")
        self._check_lines(lines)
        for ln in lines:
            product = self._product(ln["sku"])
            if product["archived"]:
                raise AppError("E_ARCHIVED", f"product '{ln['sku']}' is archived")
            if product["stock"] < ln["qty"]:
                raise AppError("E_INSUFFICIENT_STOCK", f"not enough stock for '{ln['sku']}'")
        oid = self._next_id("sales_orders", "SO", 1001)
        self.db.execute(
            "INSERT INTO sales_orders VALUES (?, ?, 'placed', ?)",
            (oid, customer_name.strip(), self._now()),
        )
        for ln in lines:
            price = self._product(ln["sku"])["unit_price_paise"]
            self.db.execute(
                "INSERT INTO so_lines VALUES (?, ?, ?, ?)", (oid, ln["sku"], ln["qty"], price)
            )
            self._move(ln["sku"], -ln["qty"], "sale", oid)
        return self.get_sales_order(oid)

    def _so(self, order_id: str) -> sqlite3.Row:
        row = self._one("SELECT * FROM sales_orders WHERE id = ?", (order_id,))
        if row is None:
            raise AppError("E_NOT_FOUND", f"no sales order '{order_id}'")
        return row

    def get_sales_order(self, order_id: str) -> dict:
        so = dict(self._so(order_id))
        so["lines"] = [
            dict(r)
            for r in self._all(
                "SELECT sku, qty, unit_price_paise FROM so_lines WHERE order_id = ? ORDER BY sku",
                (order_id,),
            )
        ]
        so["total_paise"] = sum(ln["qty"] * ln["unit_price_paise"] for ln in so["lines"])
        return so

    def list_sales_orders(
        self, cursor: str | None = None, page_size: int = 10, status: str | None = None
    ) -> dict:
        sql = "SELECT id FROM sales_orders"
        args: tuple = ()
        if status is not None:
            sql, args = sql + " WHERE status = ?", (status,)
        ids = [r["id"] for r in self._all(sql + " ORDER BY id", args)]
        page = self._page(ids, cursor, page_size)
        summaries = []
        for oid in page["items"]:
            so = self.get_sales_order(oid)
            del so["lines"]
            summaries.append(so)
        page["items"] = summaries
        return page

    def cancel_sales_order(self, order_id: str) -> dict:
        so = self._so(order_id)
        if so["status"] != "placed":
            raise AppError("E_INVALID_STATE", "order cannot be cancelled in its current state")
        self.db.execute("UPDATE sales_orders SET status = 'cancelled' WHERE id = ?", (order_id,))
        for ln in self._all("SELECT sku, qty FROM so_lines WHERE order_id = ?", (order_id,)):
            self._move(ln["sku"], ln["qty"], "sale_cancelled", order_id)
        return self.get_sales_order(order_id)
