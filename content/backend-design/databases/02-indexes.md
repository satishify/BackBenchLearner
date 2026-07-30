---
title: "Indexes and how they speed up queries"
description: "Understand database indexes and how they improve query performance and trade-offs."
---

Your `users` table hits a million rows. `SELECT * FROM users WHERE email = ?` goes from instant to multi-second. The query is correct; the database is **scanning** every row. An **index** is a side structure (usually a B-tree) that lets the engine jump to matching rows — at the cost of slower writes and extra disk.

## Intuition

A book’s index does not copy the whole book; it maps keywords → page numbers. A database index maps column values → row locations (or primary keys). Without it, the planner may choose a **sequential scan**. With a selective index, it does an **index lookup** (and maybe a heap/table fetch).

Indexes help `WHERE`, `JOIN`, `ORDER BY`, and `UNIQUE` constraints. They hurt `INSERT`/`UPDATE`/`DELETE` because each write updates the index too. Too many indexes is a classic “fast reads, dead writes” failure mode.

## How it works

```mermaid
flowchart LR
    Q["WHERE email = 'a@x.com'"] --> I[Email B-tree index]
    I --> R[Row pointers]
    R --> T[(users heap/table)]
```

Types you will meet:

- **B-tree** — default for equality and ranges (`price > 10`)
- **Hash** — equality only (engine-specific)
- **Composite** `(a, b)` — helps `WHERE a=? AND b=?`; leftmost prefix matters
- **Covering** — index contains all columns needed so table fetch is skipped
- **Partial** — index subset (`WHERE deleted_at IS NULL`)

`EXPLAIN QUERY PLAN` (SQLite) or `EXPLAIN ANALYZE` (Postgres) shows whether your index is used. Guessing is not a strategy.

## In code

Create indexes, contrast plans, and expose a FastAPI lookup that depends on the email index.

```python
import sqlite3
import time
from fastapi import FastAPI, HTTPException

app = FastAPI()
DB = "users.db"

def conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

def setup(n: int = 50_000):
    with conn() as db:
        db.execute("DROP TABLE IF EXISTS users")
        db.execute(
            "CREATE TABLE users ("
            "id INTEGER PRIMARY KEY, email TEXT NOT NULL, name TEXT)"
        )
        db.executemany(
            "INSERT INTO users (email, name) VALUES (?, ?)",
            [(f"user{i}@example.com", f"User {i}") for i in range(n)],
        )
        db.commit()

def bench(email: str) -> float:
    with conn() as db:
        t0 = time.perf_counter()
        db.execute("SELECT id, name FROM users WHERE email=?", (email,)).fetchone()
        return time.perf_counter() - t0

@app.on_event("startup")
def startup():
    setup()
    # Before index
    slow = bench("user49999@example.com")
    with conn() as db:
        print(db.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM users WHERE email=?",
            ("user49999@example.com",),
        ).fetchall())
        db.execute("CREATE UNIQUE INDEX idx_users_email ON users(email)")
        db.commit()
        print(db.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM users WHERE email=?",
            ("user49999@example.com",),
        ).fetchall())
    fast = bench("user49999@example.com")
    print(f"before={slow:.4f}s after={fast:.4f}s")

@app.get("/users/by-email/{email}")
def by_email(email: str):
    with conn() as db:
        row = db.execute(
            "SELECT id, email, name FROM users WHERE email=?", (email,)
        ).fetchone()
    if not row:
        raise HTTPException(404)
    return dict(row)

# Composite index example (orders by customer + created)
"""
CREATE INDEX idx_orders_customer_created
  ON orders (customer_id, created_at DESC);

-- Uses index:
SELECT * FROM orders WHERE customer_id = 9 ORDER BY created_at DESC LIMIT 20;

-- May NOT use it well:
SELECT * FROM orders WHERE created_at > '2026-01-01';
"""
```

Rule of thumb: index columns used in high-frequency equality filters and foreign keys used in joins. Measure with production-like data volumes — 100-row demos lie.

Composite order matters. An index on `(customer_id, created_at)` supports `WHERE customer_id = ?` and `WHERE customer_id = ? AND created_at > ?`, but usually **not** a filter on `created_at` alone. Put equality columns first, then range columns. If the API always filters by tenant, lead with `tenant_id` so every lookup stays in one slice of the tree.

Also remember unique indexes are constraints: `CREATE UNIQUE INDEX` both speeds lookups and prevents duplicates. Primary keys already have an index; you rarely need a second one on the same column.

## What goes wrong

- **Index on low-cardinality columns alone** — boolean `is_active` rarely helps; combine with selective columns or partial indexes.
- **Functions on columns** — `WHERE LOWER(email)=?` may skip an index on `email`; use functional indexes or store normalized values.
- **Leading-wildcard LIKE** — `LIKE '%foo'` cannot use a normal B-tree well.
- **Over-indexing** — every write updates ten indexes; deploy slows to a crawl.
- **Unused indexes** — still cost writes; audit periodically.

## One-line summary

Indexes trade extra storage and slower writes for fast lookups by turning scans into seeks — create them for real query predicates and verify with `EXPLAIN`.

## Key terms

- **Index:** auxiliary structure mapping keys to rows.
- **B-tree:** balanced tree structure used by most DB indexes.
- **Sequential scan:** read most/all rows to find matches.
- **Composite index:** index on multiple columns.
- **Selectivity:** how well a predicate narrows rows.
- **Covering index:** satisfies a query without touching the table heap.
- **EXPLAIN:** command showing the planner’s chosen strategy.
