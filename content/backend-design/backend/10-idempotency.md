---
title: "Idempotency (especially for payments)"
description: "Learn idempotency patterns to make backend operations safe for retries and duplicates."
---

The user double-clicks Pay. The mobile app times out at 30s and retries. Your server charged the card twice. Finance escalates. **Idempotency** is the property that makes “at least once” delivery on flaky networks safe: doing the same logical operation one time or ten times yields one business effect.

## Intuition

HTTP already treats GET, PUT, and DELETE as idempotent. `POST /charges` is not — each call may create a new charge. The fix used by Stripe and many payment APIs: the client sends an **Idempotency-Key** (UUID) with the first attempt. The server stores the key → outcome. Retries with the same key return the **same result** without re-charging.

Idempotency is not only payments. Order creation, seat booking, “send invite email,” and wallet transfers all need it whenever clients retry or users mash buttons.

## How it works

```mermaid
sequenceDiagram
    participant C as Client
    participant API as API
    participant DB as DB

    C->>API: POST /charges Idempotency-Key: k1
    API->>DB: BEGIN; lock/insert key k1
    alt first time
        API->>API: charge card
        API->>DB: store response for k1; COMMIT
        API-->>C: 201 charge_id=9
    else replay
        API->>DB: key k1 exists
        API-->>C: 200 same body as first
    end
```

Implementation checklist:

1. Client generates a new key **per logical user action** (not per HTTP retry of that action).
2. Server inserts the key in a unique index **before** side effects, or uses a transaction/lease so two racing requests cannot both proceed.
3. Persist final status + response body (or resource id).
4. On conflict in-flight, wait or return `409` until the first finishes.
5. Expire keys (e.g. 24h) but keep payment records forever.

Hash the request body with the key: if the same key arrives with a **different** body, return `422` — do not silently ignore.

## In code

FastAPI + SQLite idempotent charge endpoint with a uniqueness constraint.

```python
import json
import hashlib
import sqlite3
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

def db():
    c = sqlite3.connect("payments.db")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    return c

@app.on_event("startup")
def init():
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS idempotency_keys (
              key TEXT PRIMARY KEY,
              request_hash TEXT NOT NULL,
              status TEXT NOT NULL,          -- processing | completed
              response_code INTEGER,
              response_body TEXT,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS charges (
              id INTEGER PRIMARY KEY,
              amount_cents INTEGER NOT NULL,
              user_id INTEGER NOT NULL
            );
            """
        )

class ChargeIn(BaseModel):
    user_id: int
    amount_cents: int = Field(gt=0)

def body_hash(body: ChargeIn) -> str:
    raw = json.dumps(body.model_dump(), sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()

@app.post("/charges")
def create_charge(
    body: ChargeIn,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    if not idempotency_key:
        raise HTTPException(400, "Idempotency-Key required")
    rh = body_hash(body)

    with db() as conn:
        existing = conn.execute(
            "SELECT * FROM idempotency_keys WHERE key=?", (idempotency_key,)
        ).fetchone()
        if existing:
            if existing["request_hash"] != rh:
                raise HTTPException(422, "Idempotency-Key reused with different body")
            if existing["status"] != "completed":
                raise HTTPException(409, "Request in progress; retry later")
            return JSON_safe(existing["response_code"], existing["response_body"])

        try:
            conn.execute(
                "INSERT INTO idempotency_keys (key, request_hash, status) "
                "VALUES (?, ?, 'processing')",
                (idempotency_key, rh),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            # raced with another request
            raise HTTPException(409, "Request in progress; retry later")

        # Side effect once
        cur = conn.execute(
            "INSERT INTO charges (amount_cents, user_id) VALUES (?, ?)",
            (body.amount_cents, body.user_id),
        )
        charge_id = cur.lastrowid
        payload = {"charge_id": charge_id, "amount_cents": body.amount_cents}
        conn.execute(
            "UPDATE idempotency_keys SET status='completed', "
            "response_code=201, response_body=? WHERE key=?",
            (json.dumps(payload), idempotency_key),
        )
        conn.commit()
    return payload

def JSON_safe(code: int, body: str):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=code, content=json.loads(body))
```

Natural SQL uniqueness helps too: `UNIQUE(user_id, client_order_id)` so a duplicate insert fails closed even if the idempotency table is wrong.

## What goes wrong

- **Key per retry library attempt** — regenerating UUIDs each HTTP call defeats the feature; key must be stable for the user action.
- **Check-then-act without uniqueness** — two parallel POSTs both see “missing” and both charge; use unique insert or `SELECT FOR UPDATE`.
- **Storing only “success”** — if you crash after charge but before save, retries double-charge; write the key row first or use an outbox carefully.
- **Ignoring body mismatch** — same key, different amount, returns old charge; always hash the body.
- **Idempotent HTTP but non-idempotent emails** — side effects outside the transaction still duplicate; gate them on the same key.

## One-line summary

Idempotency keys (plus unique constraints) make retried POSTs safe by recording each logical operation once and replaying the stored result.

## Key terms

- **Idempotent:** repeating an operation does not change the result beyond the first application.
- **Idempotency-Key:** client-supplied unique id for one logical request.
- **At-least-once delivery:** networks/clients may retry; assume duplicates.
- **Exactly-once effect:** business outcome achieved once despite duplicate requests.
- **Replay:** returning the original response for a seen key.
- **Request hash:** digest of body bound to the key to detect misuse.
- **Unique constraint:** database-level guard against duplicate rows.
