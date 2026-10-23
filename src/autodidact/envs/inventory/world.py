"""Seeded generator for inventory worlds.

A *world* is the full starting state of the app: categories, suppliers,
products, supplier quotes, and some order history. The same seed always gives
the same world, byte for byte.

Two kinds of worlds are used, and keeping them apart is part of the anti-cheating
design:

* **Practice worlds** (any seed except the test seed) are where the agent explores
  and invents its own practice tasks.
* **The test world** (``TEST_WORLD_SEED``) is only ever used for the hidden test
  tasks. The agent never sees its data while learning, so it cannot memorise
  answers such as "BEV-0412 costs ₹85" - it has to learn how the app works.

All brands, suppliers and customers are fictional.
"""

from __future__ import annotations

import random
import sqlite3

TEST_WORLD_SEED = 7001

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE categories (code TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE suppliers (
    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, city TEXT NOT NULL,
    lead_time_days INTEGER NOT NULL
);
CREATE TABLE products (
    sku TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, category TEXT NOT NULL,
    supplier_id TEXT NOT NULL, unit_price_paise INTEGER NOT NULL,
    stock INTEGER NOT NULL, reorder_level INTEGER NOT NULL, archived INTEGER NOT NULL
);
CREATE TABLE quotes (
    sku TEXT NOT NULL, supplier_id TEXT NOT NULL, unit_cost_paise INTEGER NOT NULL,
    moq INTEGER NOT NULL, PRIMARY KEY (sku, supplier_id)
);
CREATE TABLE purchase_orders (id TEXT PRIMARY KEY, supplier_id TEXT NOT NULL, status TEXT NOT NULL);
CREATE TABLE po_lines (
    po_id TEXT NOT NULL, sku TEXT NOT NULL, qty INTEGER NOT NULL,
    unit_cost_paise INTEGER NOT NULL, PRIMARY KEY (po_id, sku)
);
CREATE TABLE sales_orders (
    id TEXT PRIMARY KEY, customer_name TEXT NOT NULL, status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE so_lines (
    order_id TEXT NOT NULL, sku TEXT NOT NULL, qty INTEGER NOT NULL,
    unit_price_paise INTEGER NOT NULL, PRIMARY KEY (order_id, sku)
);
CREATE TABLE stock_movements (
    id INTEGER PRIMARY KEY AUTOINCREMENT, sku TEXT NOT NULL, delta INTEGER NOT NULL,
    reason TEXT NOT NULL, ref TEXT
);
"""

CATEGORIES = {
    "BEV": "Beverages",
    "SNK": "Snacks",
    "STP": "Staples",
    "PRC": "Personal Care",
    "HSH": "Household",
    "DRY": "Dairy",
}

ITEMS = {
    "BEV": [
        "Masala Tea",
        "Green Tea",
        "Filter Coffee",
        "Instant Coffee",
        "Mango Drink",
        "Lemon Squash",
        "Rose Sharbat",
        "Cold Coffee",
        "Ginger Tea",
        "Kokum Syrup",
        "Elaichi Tea",
        "Jaljeera Mix",
    ],
    "SNK": [
        "Banana Chips",
        "Masala Peanuts",
        "Aloo Bhujia",
        "Khakhra",
        "Chakli",
        "Murukku",
        "Rusk",
        "Coconut Cookies",
        "Moong Dal",
        "Mathri",
        "Chikki",
        "Makhana",
    ],
    "STP": [
        "Basmati Rice",
        "Sona Masoori Rice",
        "Toor Dal",
        "Chana Dal",
        "Atta",
        "Besan",
        "Sugar",
        "Rock Salt",
        "Poha",
        "Rava",
        "Groundnut Oil",
        "Mustard Oil",
    ],
    "PRC": [
        "Neem Soap",
        "Sandal Soap",
        "Herbal Shampoo",
        "Coconut Hair Oil",
        "Clove Toothpaste",
        "Aloe Face Wash",
        "Talcum Powder",
        "Lip Balm",
        "Body Lotion",
        "Shaving Cream",
        "Turmeric Cream",
        "Hand Wash",
    ],
    "HSH": [
        "Dishwash Bar",
        "Floor Cleaner",
        "Detergent Powder",
        "Agarbatti",
        "Camphor Tablets",
        "Glass Cleaner",
        "Toilet Cleaner",
        "Scrub Pad",
        "Garbage Bags",
        "Mosquito Coil",
        "Matchbox Pack",
        "Naphthalene Balls",
    ],
    "DRY": [
        "Paneer",
        "Ghee",
        "Curd",
        "Butter",
        "Buttermilk",
        "Khoa",
        "Cheese Slices",
        "Lassi",
        "Toned Milk",
        "Shrikhand",
        "Cream",
        "Milk Powder",
    ],
}

SIZES = ["100g", "200g", "250g", "500g", "1kg", "100ml", "200ml", "500ml", "1L", "Pack of 6"]

BRANDS = ["Kaveri", "Himgiri", "Sunrise", "Desi Farm", "Nimbus", "Gully", "Tarang", "Mitti"]

SUPPLIER_NAMES = [
    "Kaveri Traders",
    "Deccan Wholesale",
    "Sahyadri Distributors",
    "Narmada Supply Co",
    "Coromandel Agencies",
    "Aravalli Stockists",
    "Ganga Bulk Mart",
    "Vindhya Merchants",
    "Konkan Trading House",
    "Satpura Supplies",
]

CITIES = [
    "Hyderabad",
    "Pune",
    "Chennai",
    "Indore",
    "Kochi",
    "Jaipur",
    "Lucknow",
    "Nagpur",
    "Vizag",
    "Mysuru",
    "Surat",
    "Bhopal",
]

CUSTOMERS = [
    "Asha Reddy",
    "Imran Qureshi",
    "Lakshmi Iyer",
    "Rohan Mehta",
    "Farah Khan",
    "Vikram Rao",
    "Neha Joshi",
    "Arjun Pillai",
    "Sana Sheikh",
    "Karthik Nair",
    "Meera Das",
    "Dev Malhotra",
]

MOQS = [1, 1, 5, 6, 10, 12, 24]


def build_world(seed: int) -> sqlite3.Connection:
    """Create a fresh in-memory database holding the world for ``seed``."""
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO meta VALUES ('clock_minutes', '0')")
    conn.execute("INSERT INTO meta VALUES ('world_seed', ?)", (str(seed),))
    conn.executemany("INSERT INTO categories VALUES (?, ?)", CATEGORIES.items())

    # Suppliers: 8 of the 10 names, each in a random city.
    supplier_ids = []
    for i, name in enumerate(rng.sample(SUPPLIER_NAMES, 8), start=1):
        sid = f"SUP-{i:02d}"
        supplier_ids.append(sid)
        conn.execute(
            "INSERT INTO suppliers VALUES (?, ?, ?, ?)",
            (sid, name, rng.choice(CITIES), rng.randint(2, 9)),
        )

    # Products: 12 per category, SKU numbers increase with random gaps.
    used_names: set[str] = set()
    for code in CATEGORIES:
        number = rng.randint(1, 400)
        for item in rng.sample(ITEMS[code], 12):
            number += rng.randint(1, 9)
            sku = f"{code}-{number:04d}"
            name = _unique_name(rng, item, used_names)
            price = rng.randrange(1500, 60000, 50)  # ₹15.00 to ₹599.50
            reorder = rng.randint(5, 30)
            # About a quarter of products sit below their reorder level.
            stock = (
                rng.randint(0, reorder - 1) if rng.random() < 0.27 else rng.randint(reorder, 150)
            )
            primary = rng.choice(supplier_ids)
            archived = 1 if rng.random() < 0.09 else 0
            conn.execute(
                "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (sku, name, CATEGORIES[code], primary, price, stock, reorder, archived),
            )
            # 2-3 quotes, always including the primary supplier. The cheapest
            # quote is often NOT the primary supplier (a deliberate quirk).
            others = rng.sample([s for s in supplier_ids if s != primary], rng.randint(1, 2))
            costs = rng.sample(range(int(price * 0.55), int(price * 0.85)), 1 + len(others))
            for sid, cost in zip([primary, *others], costs, strict=True):
                conn.execute(
                    "INSERT INTO quotes VALUES (?, ?, ?, ?)", (sku, sid, cost, rng.choice(MOQS))
                )

    _seed_history(conn, rng)
    conn.commit()
    return conn


def _unique_name(rng: random.Random, item: str, used: set[str]) -> str:
    while True:
        name = f"{rng.choice(BRANDS)} {item} {rng.choice(SIZES)}"
        if name not in used:
            used.add(name)
            return name


def _seed_history(conn: sqlite3.Connection, rng: random.Random) -> None:
    """Past sales and purchase orders, so the app does not look empty."""
    active = [
        r[0] for r in conn.execute("SELECT sku FROM products WHERE archived = 0 ORDER BY sku")
    ]
    prices = dict(conn.execute("SELECT sku, unit_price_paise FROM products"))

    # 30 past sales orders across September and October 2026, oldest first.
    day, minute = 1, 0
    for n in range(30):
        day += rng.randint(0, 2)
        minute = rng.randint(9 * 60, 20 * 60)
        month, dom = (9, day) if day <= 30 else (10, day - 30)
        created = f"2026-{month:02d}-{dom:02d}T{minute // 60:02d}:{minute % 60:02d}:00"
        oid = f"SO-{1001 + n}"
        status = "cancelled" if rng.random() < 0.1 else "placed"
        conn.execute(
            "INSERT INTO sales_orders VALUES (?, ?, ?, ?)",
            (oid, rng.choice(CUSTOMERS), status, created),
        )
        for sku in rng.sample(active, rng.randint(1, 3)):
            conn.execute(
                "INSERT INTO so_lines VALUES (?, ?, ?, ?)",
                (oid, sku, rng.randint(1, 4), prices[sku]),
            )

    # A few purchase orders in different states.
    quotes = conn.execute("SELECT sku, supplier_id, unit_cost_paise, moq FROM quotes").fetchall()
    for n, status in enumerate(["received", "received", "submitted", "draft"]):
        sku, sid, cost, moq = rng.choice(quotes)
        pid = f"PO-{5001 + n}"
        conn.execute("INSERT INTO purchase_orders VALUES (?, ?, ?)", (pid, sid, status))
        conn.execute("INSERT INTO po_lines VALUES (?, ?, ?, ?)", (pid, sku, max(moq, 12), cost))
