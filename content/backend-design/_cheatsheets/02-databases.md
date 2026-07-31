---
title: "Databases revision"
slug: databases
module: "Databases chapter"
minutes: 30
description: "SQL/NoSQL, indexes, ACID, isolation, pagination, sharding, replicas, locking."
---

Revision for **Databases & Data Handling**.

## SQL vs NoSQL
- Relational: strong schema, joins, transactions — great for money and invariants.
- Document / KV / wide-column / graph: scale-out or flexible shapes; often weaker multi-doc transactions.
- Choose for access patterns + consistency needs, not fashion.

## Indexes
- B-tree (default): equality + range. Hash: equality. Covering indexes avoid table lookups.
- Cost: slower writes, storage. Too many indexes = write hell.
- Composite: leftmost prefix rule. Selective columns first (usually).

## ACID
- Atomicity, Consistency (constraints), Isolation, Durability.
- “Consistency” here != distributed CAP consistency — don’t mix vocab in interviews.

## Transactions & isolation
- Phenomena: dirty read, non-repeatable read, phantom.
- Levels (rough): Read Uncommitted -> Read Committed -> Repeatable Read -> Serializable.
- Default often Read Committed. Serializable / SSI when correctness > throughput.

## Normalization vs denormalization
- Normalize to reduce update anomalies. Denormalize / materialize for read latency and simple queries.
- Every denorm needs an update strategy (triggers, jobs, dual writes carefully).

## Pagination
- Offset: simple, drifts under inserts, expensive deep pages.
- Keyset / cursor: `WHERE (created_at, id) < (?, ?) ORDER BY ... LIMIT n` — stable and fast with the right index.

## Sharding & partitioning
- Partition = split data (by key/range/hash). Shard = often distributed partition.
- Hot keys kill you. Resharding is expensive — plan key and routing early.
- Cross-shard joins/transactions are the tax you pay.

## Read replicas & write scaling
- Replicas scale reads; replication lag -> stale reads. Route accordingly.
- Writes still hit primary (or a carefully designed multi-primary — hard).
- Cache before you prematurely shard.

## Duplicates & locking
- Unique constraints + idempotency keys prevent duplicates at the source of truth.
- **Optimistic**: version/ETag; retry on conflict — good for low contention.
- **Pessimistic**: `SELECT ... FOR UPDATE` — good for high contention critical sections; watch deadlocks.

## 30-minute drill
1. Pick isolation level for “transfer money between two rows.”
2. Explain why offset page 5000 is slow.
3. Hot-key example on a hash shard and one mitigation.
