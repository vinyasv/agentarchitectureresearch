"""Text-to-SQL tasks graded by execution against shop.db.

Each task's gold query comes from a small builder function. `ablations` maps a shard
id to the query you get by ignoring that shard (one builder argument changed); the
validator checks that each ablated query is graded *wrong*, which proves the shard
is load-bearing on this data. `sort_cols` names the ORDER BY columns of the gold
result; the validator checks they have no ties, so row order is well defined.
"""

from .schema import Shard, Task
from .sqldb import SCHEMA

CONTEXT = (
    "You are a data assistant for an online shop. You answer questions by writing "
    "queries against the shop's SQLite database, which has this schema:\n\n"
    f"{SCHEMA}\n"
    "An order's line items are in order_items; revenue for a line is quantity * unit_price."
)

DELIVERABLE = (
    "Deliverable format: whenever you provide the query, give exactly one SQLite SELECT "
    "statement in a single ```sql fenced block."
)


def _in(values) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


# ---------------------------------------------------------------- builders

def top_customers(statuses=("delivered",), price="oi.unit_price", start="2024-04-01", end="2025-03-31",
                  exclude=("Germany",), limit=3, rnd=True):
    total = f"SUM(oi.quantity * {price})"
    total = f"ROUND({total}, 2)" if rnd else total
    where = [f"o.order_date BETWEEN '{start}' AND '{end}'"]
    if statuses:
        where.append(f"o.status IN {_in(statuses)}")
    if exclude:
        where.append(f"c.country NOT IN {_in(exclude)}")
    return (f"SELECT c.name, {total} AS total FROM customers c JOIN orders o ON o.customer_id = c.id "
            f"JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id "
            f"WHERE {' AND '.join(where)} GROUP BY c.id ORDER BY total DESC LIMIT {limit}")


def monthly_orders(countries=("Germany", "France", "UK"), skip_cancelled=True, start="2024-10-01",
                   min_products=2, fmt="%Y-%m"):
    where = [f"c.country IN {_in(countries)}", f"o.order_date >= '{start}'", "o.order_date < '2025-04-01'"]
    if skip_cancelled:
        where.append("o.status != 'cancelled'")
    if min_products:
        where.append(f"(SELECT COUNT(DISTINCT oi.product_id) FROM order_items oi WHERE oi.order_id = o.id) >= {min_products}")
    return (f"SELECT strftime('{fmt}', o.order_date) AS month, COUNT(*) AS order_count FROM orders o "
            f"JOIN customers c ON c.id = o.customer_id WHERE {' AND '.join(where)} GROUP BY month ORDER BY month")


def unsold(categories=("Electronics",), not_sales=("cancelled", "returned"), min_price=10, order="p.price DESC"):
    sold = "SELECT oi.product_id FROM order_items oi JOIN orders o ON o.id = oi.order_id"
    if not_sales:
        sold += f" WHERE o.status NOT IN {_in(not_sales)}"
    where = [f"p.id NOT IN ({sold})"]
    if categories:
        where.append(f"p.category IN {_in(categories)}")
    cols = "p.name"
    if min_price is not None:
        where.append(f"p.price > {min_price}")
        cols = "p.name, p.price"
    return f"SELECT {cols} FROM products p WHERE {' AND '.join(where)} ORDER BY {order}"


def repeat_customers(countries=("USA", "Canada", "UK"), since="2024-01-01", threshold=4,
                     order="n DESC, c.email", status="delivered"):
    where = [f"c.country IN {_in(countries)}"]
    if status:
        where.append(f"o.status = '{status}'")
    if since:
        where.append(f"o.order_date >= '{since}'")
    return (f"SELECT c.email, COUNT(*) AS n FROM customers c JOIN orders o ON o.customer_id = c.id "
            f"WHERE {' AND '.join(where)} GROUP BY c.id HAVING n >= {threshold} ORDER BY {order}")


def category_share(statuses=("delivered", "shipped"), start="2024-01-01", end="2024-12-31",
                   exclude=("Books",), as_pct=True, order="pct DESC"):
    where = [f"o.order_date BETWEEN '{start}' AND '{end}'"]
    if statuses:
        where.append(f"o.status IN {_in(statuses)}")
    if exclude:
        where.append(f"p.category NOT IN {_in(exclude)}")
    value = "ROUND(100.0 * rev / (SELECT SUM(rev) FROM r), 1)" if as_pct else "ROUND(rev, 2)"
    return (f"WITH r AS (SELECT p.category AS cat, SUM(oi.quantity * oi.unit_price) AS rev FROM order_items oi "
            f"JOIN orders o ON o.id = oi.order_id JOIN products p ON p.id = oi.product_id "
            f"WHERE {' AND '.join(where)} GROUP BY p.category) "
            f"SELECT cat, {value} AS pct FROM r ORDER BY {order}")


def avg_order_value(per_order=True, status="delivered", min_orders=3, signed_before="2024-01-01",
                    rnd=True, order="avg_value DESC"):
    where = []
    if status:
        where.append(f"o.status = '{status}'")
    if signed_before:
        where.append(f"c.signup_date < '{signed_before}'")
    w = f"WHERE {' AND '.join(where)}" if where else ""
    agg = "AVG(v)" if per_order else "AVG(line_v)"
    agg = f"ROUND({agg}, 2)" if rnd else agg
    if per_order:
        inner = (f"SELECT o.id, c.country, SUM(oi.quantity * oi.unit_price) AS v FROM orders o "
                 f"JOIN customers c ON c.id = o.customer_id JOIN order_items oi ON oi.order_id = o.id {w} GROUP BY o.id")
        return (f"WITH ov AS ({inner}) SELECT country, {agg} AS avg_value FROM ov GROUP BY country "
                f"HAVING COUNT(*) >= {min_orders} ORDER BY {order}")
    inner = (f"SELECT o.id AS oid, c.country, oi.quantity * oi.unit_price AS line_v FROM orders o "
             f"JOIN customers c ON c.id = o.customer_id JOIN order_items oi ON oi.order_id = o.id {w}")
    return (f"WITH li AS ({inner}) SELECT country, {agg} AS avg_value FROM li GROUP BY country "
            f"HAVING COUNT(DISTINCT oid) >= {min_orders} ORDER BY {order}")


TASKS = [
    Task(
        id="sql_top_customers",
        domain="sql",
        title="Top customers by spend",
        shards=[
            Shard(1, "who are our biggest spenders?"),
            Shard(2, "Only count orders that were actually delivered; ignore cancelled, returned, or still-shipping orders."),
            Shard(3, "Use what they actually paid on each line item (quantity times the unit price charged), not the products' current list price."),
            Shard(4, "Only look at orders placed in 2024."),
            Shard(5, "Give me the top 5, with the customer's name and their total spend rounded to 2 decimals, biggest spender first."),
            Shard(6, "Actually, use the last 12 months instead: orders from April 1, 2024 through March 31, 2025.", revises=4,
                  full_text="Only look at orders placed from April 1, 2024 through March 31, 2025."),
            Shard(7, "Leave out customers in Germany; that region reports separately."),
            Shard(8, "This is for the quarterly board deck.", distractor=True),
            Shard(9, "Make it the top 3 instead of the top 5.", revises=5,
                  full_text="Give me the top 3, with the customer's name and their total spend rounded to 2 decimals, biggest spender first."),
        ],
        gold=top_customers(),
        grading={"ordered": True, "sort_cols": [1]},
        ablations={2: top_customers(statuses=()), 3: top_customers(price="p.price"),
                   6: top_customers(start="2024-01-01", end="2024-12-31"), 7: top_customers(exclude=()),
                   9: top_customers(limit=5)},
    ),
    Task(
        id="sql_monthly_eu_orders",
        domain="sql",
        title="Monthly European order counts",
        shards=[
            Shard(1, "how many orders do we get per month?"),
            Shard(2, "Only for customers based in Germany or France."),
            Shard(3, "Don't count cancelled orders (returned ones still count)."),
            Shard(4, "Only from July 2024 through March 2025."),
            Shard(5, "Show the month as YYYY-MM next to the order count, oldest month first."),
            Shard(6, "Add the UK to that list as well.", revises=2,
                  full_text="Only for customers based in Germany, France, or the UK."),
            Shard(7, "Only count orders that contain at least 2 different products."),
            Shard(8, "We're planning a European marketing push.", distractor=True),
            Shard(9, "Actually, start from October 2024 instead of July, still through March 2025.", revises=4,
                  full_text="Only from October 2024 through March 2025."),
        ],
        gold=monthly_orders(),
        grading={"ordered": True, "sort_cols": [0]},
        ablations={3: monthly_orders(skip_cancelled=False), 5: monthly_orders(fmt="%m/%Y"),
                   6: monthly_orders(countries=("Germany", "France")), 7: monthly_orders(min_products=0),
                   9: monthly_orders(start="2024-07-01")},
    ),
    Task(
        id="sql_unsold_electronics",
        domain="sql",
        title="Products that never sold",
        shards=[
            Shard(1, "which of our products have never sold?"),
            Shard(2, "I only care about the Electronics category."),
            Shard(3, "Cancelled and returned orders don't count as sales."),
            Shard(4, "Just give me the product names, in alphabetical order."),
            Shard(5, "Also show each product's list price, and only include products priced over $10."),
            Shard(6, "Sort them by price, most expensive first, instead of alphabetically.", revises=4,
                  full_text="Give me the product names, sorted by list price, most expensive first."),
            Shard(7, "We're deciding what to discontinue.", distractor=True),
        ],
        gold=unsold(),
        grading={"ordered": True, "sort_cols": [1]},
        ablations={2: unsold(categories=()), 3: unsold(not_sales=()), 5: unsold(min_price=None),
                   6: unsold(order="p.name")},
    ),
    Task(
        id="sql_repeat_customers",
        domain="sql",
        title="Repeat customers",
        shards=[
            Shard(1, "I want a list of our repeat customers."),
            Shard(2, "A repeat customer is someone with at least 3 delivered orders."),
            Shard(3, "Only customers in the USA or Canada."),
            Shard(4, "Show their email and how many delivered orders they have."),
            Shard(5, "Sort by number of orders, highest first, then by email alphabetically."),
            Shard(6, "Oh, actually include customers in the UK as well.", revises=3,
                  full_text="Only customers in the USA, Canada, or the UK."),
            Shard(7, "Only count delivered orders placed in 2024 or later."),
            Shard(8, "Actually, make it at least 4 delivered orders to count as a repeat customer.", revises=2,
                  full_text="A repeat customer is someone with at least 4 delivered orders."),
            Shard(9, "We want to send them a loyalty discount.", distractor=True),
        ],
        gold=repeat_customers(),
        grading={"ordered": True, "sort_cols": [1, 0]},
        ablations={4: repeat_customers(status=None), 5: repeat_customers(order="c.email"),
                   6: repeat_customers(countries=("USA", "Canada")), 7: repeat_customers(since=None),
                   8: repeat_customers(threshold=3)},
    ),
    Task(
        id="sql_category_share",
        domain="sql",
        title="Revenue share by category",
        shards=[
            Shard(1, "what share of our revenue does each product category bring in?"),
            Shard(2, "Count revenue from line items of delivered or shipped orders only."),
            Shard(3, "Restrict it to Q4 2024 (October through December)."),
            Shard(4, "Express each share as a percentage of total revenue, rounded to 1 decimal place, next to the category name."),
            Shard(5, "Sort with the biggest category first."),
            Shard(6, "Actually, use all of 2024 instead of just Q4.", revises=3,
                  full_text="Restrict it to orders placed in 2024."),
            Shard(7, "Leave the Books category out entirely, both from the list and from the total."),
            Shard(8, "It's for a slide in the investor update.", distractor=True),
        ],
        gold=category_share(),
        grading={"ordered": True, "sort_cols": [1]},
        ablations={2: category_share(statuses=()), 4: category_share(as_pct=False, order="pct DESC"),
                   5: category_share(order="cat"), 6: category_share(start="2024-10-01"),
                   7: category_share(exclude=())},
    ),
    Task(
        id="sql_avg_order_value",
        domain="sql",
        title="Average order value by country",
        shards=[
            Shard(1, "what's the average order value in each country?"),
            Shard(2, "An order's value is the sum of quantity times unit price across all of its items, so average over orders, not over individual line items."),
            Shard(3, "Only delivered orders."),
            Shard(4, "Only include countries with at least 10 such orders."),
            Shard(5, "Round to 2 decimals, and list country and average with the highest average first."),
            Shard(6, "Lower that cutoff to 3 orders.", revises=4,
                  full_text="Only include countries with at least 3 such orders."),
            Shard(7, "Only count orders from customers who signed up before 2024."),
            Shard(8, "Finance asked for this.", distractor=True),
        ],
        gold=avg_order_value(),
        grading={"ordered": True, "sort_cols": [1]},
        ablations={2: avg_order_value(per_order=False), 3: avg_order_value(status=None),
                   5: avg_order_value(rnd=False, order="country"), 6: avg_order_value(min_orders=10),
                   7: avg_order_value(signed_before=None)},
    ),
]
