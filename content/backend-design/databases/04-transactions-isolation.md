---
title: "Transactions & isolation levels"
description: "Learn database transactions and isolation levels to manage consistency and concurrency."
---

Two checkout requests read inventory = 1, both decide they can sell, both write sold = 1 — you oversold. Transactions without the right **isolation** still race. Isolation levels are the dial between “see fewer anomalies” and “hold more locks / abort more often.”

## Intuition

Inside a transaction you run several reads and writes. Concurrently, other transactions do the same. **Isolation** defines which of their changes you are allowed to observe before they commit, and whether your reads are repeatable.

Classic anomalies:

- **Dirty read** — see uncommitted data from another txn
- **Non-repeatable read** — same row, two values in one txn
- **Phantom read** — new rows appear matching a prior query
- **Lost update** — two writers overwrite each other

SQL defines levels: Read Uncommitted, Read Committed, Repeatable Read, Serializable. Postgres defaults to **Read Committed**; SQLite is typically serializable-ish with locking. Higher isolation ≠ always better for every endpoint — know your anomaly budget.

## How it works

```mermaid
sequenceDiagram
    participant T1
    participant T2
    participant DB

    T1->>DB: BEGIN; SELECT stock WHERE id=1  -- sees 1
    T2->>DB: BEGIN; SELECT stock WHERE id=1  -- sees 1
    T1->>DB: UPDATE stock=0; COMMIT
    T2->>DB: UPDATE stock=0; COMMIT
    Note over DB: Lost update / oversell without version check
```

| Level | Dirty read | Non-repeatable | Phantom |
| --- | --- | --- | --- |
| Read Uncommitted | Possible | Possible | Possible |
| Read Committed | No | Possible | Possible |
| Repeatable Read | No | No | Possible* |
| Serializable | No | No | No |

\*Engines differ; Postgres RR prevents many phantoms via snapshots but true serializability needs `SERIALIZABLE`.

Patterns that help even at Read Committed:

- `SELECT … FOR UPDATE` to lock rows you will change
- Optimistic `UPDATE … WHERE version=?`
- Single SQL statement that encodes the rule (`UPDATE … WHERE stock >= 1`)

## In code

Demonstrate a race, then fix with a transactional conditional update (works under concurrent FastAPI workers).

```python
import sqlite3
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
DB = "inventory.db"

def connect():
    c = sqlite3.connect(DB, timeout=5)
    c.row_factory = sqlite3.Row
    c.isolation_level = None  # manage transactions manually
    return c

@app.on_event("startup")
def init():
    with connect() as db:
        db.execute("BEGIN")
        db.execute("DROP TABLE IF EXISTS products")
        db.execute(
            "CREATE TABLE products ("
            "id INTEGER PRIMARY KEY, name TEXT, stock INTEGER NOT NULL)"
        )
        db.execute("INSERT INTO products VALUES (1, 'Mug', 1)")
        db.execute("COMMIT")

class Buy(BaseModel):
    product_id: int
    qty: int = 1

@app.post("/buy")
def buy(body: Buy):
    db = connect()
    try:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT stock FROM products WHERE id=?", (body.product_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404)
        # Conditional write: atomic check+update in one statement is even better
        cur = db.execute(
            "UPDATE products SET stock = stock - ? "
            "WHERE id=? AND stock >= ?",
            (body.qty, body.product_id, body.qty),
        )
        if cur.rowcount != 1:
            db.execute("ROLLBACK")
            raise HTTPException(409, "Out of stock")
        db.execute("COMMIT")
    except HTTPException:
        raise
    except Exception:
        db.execute("ROLLBACK")
        raise
    finally:
        db.close()
    return {"ok": True}

# Postgres-style illustration (psycopg):
"""
with conn.transaction():
    cur.execute(
        "SELECT stock FROM products WHERE id=%s FOR UPDATE",
        (product_id,),
    )
    stock = cur.fetchone()[0]
    if stock < qty:
        raise OutOfStock()
    cur.execute(
        "UPDATE products SET stock = stock - %s WHERE id=%s",
        (qty, product_id),
    )
"""

def _race_demo():
    """Two threads both try to buy the last item."""
    errors = []

    def attempt():
        try:
            buy(Buy(product_id=1, qty=1))
        except Exception as e:
            errors.append(e)

    t1, t2 = threading.Thread(target=attempt), threading.Thread(target=attempt)
    t1.start(); t2.start(); t1.join(); t2.join()
    with connect() as db:
        stock = db.execute("SELECT stock FROM products WHERE id=1").fetchone()[0]
    print("final stock", stock, "errors", len(errors))
```

`FOR UPDATE` (Postgres) serializes writers on that row. Serializable isolation can abort one transaction with a conflict error — your app must **retry** the whole business transaction.

## What goes wrong

- **Read then write without locks/conditions** — classic oversell/lost update.
- **Long transactions** — hold locks, amplify deadlocks, stall the fleet.
- **Retrying only the failed statement** — must retry the full logical txn with fresh reads.
- **Assuming Read Committed = safe inventory** — it is not, without extra patterns.
- **Silent catch of serialization failures** — user thinks order placed; it rolled back.

## One-line summary

Transactions group work; isolation levels and locking/conditional updates determine which concurrent anomalies you still must handle in application code.

## Key terms

- **Isolation level:** how much concurrent interference a transaction can see.
- **Dirty read:** reading uncommitted data.
- **Lost update:** two transactions overwrite one field; one change disappears.
- **Phantom:** new rows appear for a repeated range query.
- **SELECT FOR UPDATE:** lock selected rows until commit.
- **Serializable:** strongest isolation; outcomes match some serial order.
- **Serialization failure:** abort that clients should retry.
