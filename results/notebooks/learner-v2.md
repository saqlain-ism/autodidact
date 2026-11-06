How the application behaves:
- In create_purchase_order, line item quantity specifies packs, increasing stock by quantity multiplied by supplier pack size.
- The create_product tool requires a category name, and the SKU prefix must match the category code.
- The create_purchase_order tool fails with E_BELOW_MOQ if a line item quantity is below the supplier's minimum order quantity.
- The create_sales_order tool rejects archived products, returning an E_ARCHIVED error.
- The submit_purchase_order tool fails with E_SUPPLIER_UNAVAILABLE if the target supplier is inactive.
- The receive_purchase_order tool fails with E_INVALID_STATE if called on a draft purchase order.
- The cancel_purchase_order tool successfully cancels purchase orders in the draft state, not just those in the submitted state.
- The cancel_purchase_order tool successfully transitions purchase orders from the 'submitted' state to 'cancelled'.
