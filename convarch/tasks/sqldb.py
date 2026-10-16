"""Deterministic e-commerce SQLite database for the SQL tasks.

The data is generated from a fixed seed and has deliberate traps that make every
shard matter: cancelled/returned/shipped orders, line-item prices that differ from
the current list price, Electronics products sold only in cancelled or returned
orders, and a country with too few orders for a HAVING threshold.
"""

from __future__ import annotations

import datetime as dt
import random
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "shop.db"

SCHEMA = """\
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,          -- full name, unique
    email TEXT NOT NULL UNIQUE,
    country TEXT NOT NULL,       -- e.g. 'USA', 'Germany', 'France', 'UK', 'Canada', 'India', 'Brazil'
    signup_date TEXT NOT NULL    -- 'YYYY-MM-DD'
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,      -- 'Electronics', 'Books', 'Home', 'Clothing', 'Toys'
    price REAL NOT NULL          -- current list price
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date TEXT NOT NULL,    -- 'YYYY-MM-DD'
    status TEXT NOT NULL         -- 'delivered', 'shipped', 'cancelled', 'returned'
);
CREATE TABLE order_items (
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL     -- price actually charged per unit (after discounts)
);
"""

FIRST = ["Ava", "Liam", "Noah", "Emma", "Olivia", "Mia", "Lucas", "Sofia", "Mateo", "Hana",
         "Arjun", "Priya", "Jonas", "Lea", "Chloe", "Louis", "Oliver", "Amelia", "Ethan", "Zoe",
         "Rafael", "Ines", "Kenji", "Anya", "Felix", "Clara", "Omar", "Leila", "Diego", "Nora"]
LAST = ["Smith", "Muller", "Dubois", "Patel", "Brown", "Silva", "Kim", "Garcia", "Rossi", "Weber",
        "Martin", "Singh", "Taylor", "Costa", "Novak", "Lopez", "Wright", "Fischer", "Moreau", "Khan"]
COUNTRIES = [("USA", 18), ("Germany", 9), ("France", 8), ("UK", 8), ("Canada", 8), ("India", 6), ("Brazil", 3)]

PRODUCTS = {
    "Electronics": [("Wireless Earbuds", 79.0), ("USB-C Hub", 39.0), ("Mechanical Keyboard", 119.0),
                    ("4K Monitor", 329.0), ("Smart Plug", 24.0), ("Action Camera", 249.0),
                    ("E-Reader", 139.0), ("Bluetooth Speaker", 59.0)],
    "Books": [("Data Science Handbook", 45.0), ("The Silent Coast", 18.0), ("Cooking for Two", 27.0),
              ("History of Maps", 32.0), ("Learning SQL", 39.0), ("Poems of the North", 15.0)],
    "Home": [("Ceramic Vase", 34.0), ("Throw Blanket", 49.0), ("Desk Lamp", 42.0),
             ("Coffee Grinder", 69.0), ("Wall Clock", 29.0), ("Herb Planter", 22.0)],
    "Clothing": [("Rain Jacket", 89.0), ("Wool Socks", 14.0), ("Denim Jeans", 64.0),
                 ("Running Shoes", 119.0), ("Linen Shirt", 48.0)],
    "Toys": [("Puzzle 1000pc", 19.0), ("Building Blocks", 54.0), ("Plush Bear", 21.0),
             ("Kite", 26.0), ("Board Game Night", 36.0)],
}
# Electronics products with special sales histories (names are referenced by tasks).
ONLY_RETURNED = "Smart Plug"
ONLY_CANCELLED = "Action Camera"
NEVER_ORDERED = {"E-Reader", "Bluetooth Speaker", "Poems of the North"}


def _rand_date(rng: random.Random, start: dt.date, end: dt.date) -> str:
    return (start + dt.timedelta(days=rng.randrange((end - start).days + 1))).isoformat()


def build(path: Path = DB_PATH, seed: int = 2) -> Path:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)

    names: list[str] = []
    while len(names) < sum(w for _, w in COUNTRIES):
        n = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if n not in names:
            names.append(n)
    customers = []
    i = 0
    for country, count in COUNTRIES:
        for _ in range(count):
            name = names[i]
            i += 1
            email = name.lower().replace(" ", ".") + "@example.com"
            signup = _rand_date(rng, dt.date(2022, 1, 1), dt.date(2024, 6, 30))
            # Buying intensity: a few heavy buyers, Brazil kept small on purpose.
            weight = 0.35 if country == "Brazil" else rng.choice([0.5, 1, 1, 1.5, 2, 4])
            customers.append((i, name, email, country, signup, weight))
    con.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", [c[:5] for c in customers])

    products = []
    pid = 0
    for cat, items in PRODUCTS.items():
        for pname, price in items:
            pid += 1
            products.append((pid, pname, cat, price))
    con.executemany("INSERT INTO products VALUES (?,?,?,?)", products)
    by_name = {p[1]: p for p in products}
    sellable = [p for p in products if p[1] not in NEVER_ORDERED | {ONLY_RETURNED, ONLY_CANCELLED}]

    statuses = ["delivered"] * 68 + ["shipped"] * 8 + ["cancelled"] * 14 + ["returned"] * 10
    weights = [c[5] for c in customers]
    orders, items = [], []
    for oid in range(1, 481):
        cust = rng.choices(customers, weights=weights)[0]
        date = _rand_date(rng, dt.date(2023, 1, 1), dt.date(2025, 3, 31))
        status = rng.choice(statuses)
        if date >= "2025-03-15" and status == "delivered":
            status = "shipped"
        orders.append((oid, cust[0], date, status))
        for prod in rng.sample(sellable, rng.randint(1, 4)):
            qty = rng.randint(1, 4)
            unit = round(prod[3] * rng.uniform(0.75, 1.0), 2)
            items.append((oid, prod[0], qty, unit))

    def attach(product_name: str, status: str, n: int) -> None:
        cands = [o for o in orders if o[3] == status]
        for o in rng.sample(cands, n):
            p = by_name[product_name]
            items.append((o[0], p[0], 1, round(p[3] * 0.9, 2)))

    attach(ONLY_CANCELLED, "cancelled", 3)
    attach(ONLY_RETURNED, "returned", 2)

    con.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
    con.executemany("INSERT INTO order_items VALUES (?,?,?,?)", items)
    con.commit()
    con.close()
    return path


def ensure_db() -> Path:
    if not DB_PATH.exists():
        build()
    return DB_PATH


def run_query(sql: str, timeout_s: float = 5.0) -> list[tuple]:
    """Run a query read-only with a time limit. Raises on error."""
    import time

    con = sqlite3.connect(f"file:{ensure_db()}?mode=ro", uri=True)
    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()
