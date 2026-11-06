How the application behaves:
- Suppliers must have a valid quote for the specified SKU when creating a purchase order to avoid E_NO_QUOTE errors.
- Input the category name, not its code, into the category field when creating a product.
- The update_product_price tool requires both the sku and the new price parameter to update selling prices.

How to do things:
- Use receive_purchase_order with po_id to mark submitted purchase orders as received and update inventory.
- Use submit_purchase_order with po_id to transition a draft purchase order to submitted status before receiving.
- Call list_adjustment_reasons to retrieve valid reasons prior to executing a manual stock adjustment.
- Use cancel_sales_order with order_id to cancel a placed sales order and automatically return items to stock.
- Use cancel_purchase_order with the po_id parameter to cancel a draft purchase order.
- Use list_sales_orders with pagination cursors for multi-page results, and get_sales_order with order_id for order lines.
- Use list_purchase_orders with the status parameter to filter and verify purchase orders by their current status.

Mistakes to avoid:
- Ensure SKU format meets system validation rules when creating products to prevent E_INVALID_SKU errors.
