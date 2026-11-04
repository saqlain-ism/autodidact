"""StockRoom exposed as an MCP server.

The tool descriptions below are the only documentation the agent gets. They
are written like real internal API docs: accurate, but terse, and they leave
out the quirks listed in ``app.py``. Discovering those is the agent's job.

The same server object is used two ways:

* **In-process** (``mcp.Client(server)``): full MCP protocol over memory. Used
  for experiments because it is fast and lets checkers inspect the database.
* **Stdio** (``autodidact-inventory-server``): a normal MCP server process that
  any MCP client can connect to.
"""

from __future__ import annotations

import argparse
import functools
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

from autodidact.envs.base import AppError
from autodidact.envs.inventory.app import InventoryApp
from autodidact.envs.inventory.world import TEST_WORLD_SEED

INSTRUCTIONS = "StockRoom inventory API: catalogue, suppliers, purchase orders and sales orders."


class OrderLine(BaseModel):
    sku: str
    qty: int


def _lines(lines: list[OrderLine]) -> list[dict]:
    return [ln.model_dump() for ln in lines]


def build_server(app: InventoryApp) -> MCPServer:
    """Create an MCP server whose tools operate on ``app``."""
    server = MCPServer("stockroom", instructions=INSTRUCTIONS, version="1.4.2")

    def tool(fn):
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except AppError as exc:
                raise ToolError(str(exc)) from exc

        server.add_tool(wrapper, name=fn.__name__, description=fn.__doc__)
        return fn

    # ------------------------------------------------------------- catalogue

    @tool
    def list_categories() -> dict:
        """List product categories."""
        return app.list_categories()

    @tool
    def list_products(
        cursor: str | None = None,
        page_size: int = 10,
        category: str | None = None,
        include_archived: bool = False,
    ) -> dict:
        """List products, optionally filtered by category. Returns one page of
        results; pass next_cursor back to get the next page."""
        return app.list_products(cursor, page_size, category, include_archived)

    @tool
    def get_product(sku: str) -> dict:
        """Get full details of a product by SKU."""
        return app.get_product(sku)

    @tool
    def search_products(query: str) -> dict:
        """Search products by name."""
        return app.search_products(query)

    @tool
    def create_product(
        sku: str,
        name: str,
        category: str,
        supplier_id: str,
        unit_price_paise: int,
        reorder_level: int = 10,
    ) -> dict:
        """Create a product. New products start with zero stock."""
        return app.create_product(sku, name, category, supplier_id, unit_price_paise, reorder_level)

    @tool
    def update_product_price(sku: str, price: int) -> dict:
        """Set a product's selling price."""
        return app.update_product_price(sku, price)

    @tool
    def set_product_archived(sku: str, archived: bool) -> dict:
        """Archive or unarchive a product."""
        return app.set_product_archived(sku, archived)

    @tool
    def adjust_stock(sku: str, delta: int, reason: str) -> dict:
        """Manually adjust stock up or down, with a reason."""
        return app.adjust_stock(sku, delta, reason)

    @tool
    def list_adjustment_reasons() -> dict:
        """Reasons accepted for manual stock adjustments."""
        return app.list_adjustment_reasons()

    # ------------------------------------------------------------- suppliers

    @tool
    def list_suppliers(cursor: str | None = None, page_size: int = 10) -> dict:
        """List suppliers. Paginated like list_products."""
        return app.list_suppliers(cursor, page_size)

    @tool
    def get_supplier(supplier_id: str) -> dict:
        """Get a supplier by id."""
        return app.get_supplier(supplier_id)

    @tool
    def create_supplier(name: str, city: str, lead_time_days: int) -> dict:
        """Register a new supplier."""
        return app.create_supplier(name, city, lead_time_days)

    @tool
    def get_supplier_quotes(sku: str) -> dict:
        """Supplier quotes for a product: which suppliers sell it and at what cost."""
        return app.get_supplier_quotes(sku)

    # ------------------------------------------------------- purchase orders

    @tool
    def create_purchase_order(supplier_id: str, lines: list[OrderLine]) -> dict:
        """Create a purchase order to buy stock from a supplier."""
        return app.create_purchase_order(supplier_id, _lines(lines))

    @tool
    def submit_purchase_order(po_id: str) -> dict:
        """Send a purchase order to the supplier."""
        return app.submit_purchase_order(po_id)

    @tool
    def receive_purchase_order(po_id: str) -> dict:
        """Mark a purchase order as delivered."""
        return app.receive_purchase_order(po_id)

    @tool
    def cancel_purchase_order(po_id: str) -> dict:
        """Cancel a purchase order."""
        return app.cancel_purchase_order(po_id)

    @tool
    def get_purchase_order(po_id: str) -> dict:
        """Get a purchase order with its lines."""
        return app.get_purchase_order(po_id)

    @tool
    def list_purchase_orders(status: str | None = None) -> dict:
        """List purchase orders, optionally by status."""
        return app.list_purchase_orders(status)

    # ---------------------------------------------------------- sales orders

    @tool
    def create_sales_order(customer_name: str, lines: list[OrderLine]) -> dict:
        """Place a customer order. Stock is deducted immediately."""
        return app.create_sales_order(customer_name, _lines(lines))

    @tool
    def list_sales_orders(
        cursor: str | None = None, page_size: int = 10, status: str | None = None
    ) -> dict:
        """List sales orders. Paginated like list_products."""
        return app.list_sales_orders(cursor, page_size, status)

    @tool
    def get_sales_order(order_id: str) -> dict:
        """Get a sales order with its lines."""
        return app.get_sales_order(order_id)

    @tool
    def cancel_sales_order(order_id: str) -> dict:
        """Cancel a sales order and return its stock."""
        return app.cancel_sales_order(order_id)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the StockRoom sandbox as an MCP server.")
    parser.add_argument("--seed", type=int, default=1, help="world seed (default: 1)")
    parser.add_argument(
        "--allow-test-world",
        action="store_true",
        help=f"allow the hidden test world (seed {TEST_WORLD_SEED})",
    )
    args = parser.parse_args()
    if args.seed == TEST_WORLD_SEED and not args.allow_test_world:
        parser.error("that seed is reserved for hidden evaluation; pass --allow-test-world")
    build_server(InventoryApp.from_seed(args.seed)).run("stdio")


if __name__ == "__main__":
    main()
