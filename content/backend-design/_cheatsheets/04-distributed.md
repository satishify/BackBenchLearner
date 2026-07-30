---
title: "Distributed systems revision"
slug: distributed
module: "Distributed chapter"
minutes: 25
description: "Load balancing, scale-out, monolith vs microservices, sync/async, queues."
---

Revision for **Distributed Systems & Scaling**.

## Load balancing
- Spread traffic: L4 (TCP) vs L7 (HTTP, path/host). Health checks + connection draining.
- Algorithms: round-robin, least-conn, consistent hashing (sticky-ish without sticky hell).
- LB is not magic HA — backends + data layer still fail.

## Horizontal vs vertical
- Vertical: bigger box — simple ceiling. Horizontal: more boxes — needs statelessness or shared state design.
- Scale the bottleneck you measured, not the one you fear.

## Monolith vs microservices
- Monolith: one deploy, simple transactions, easy refactor inside process.
- Microservices: independent deploy/scale; pay with network, dual writes, distributed tracing, org overhead.
- Start modular monolith; extract when a team/boundary/scalability pressure is real.

## Service communication
- Sync HTTP/gRPC: simple request path; cascading failures if unbounded.
- Async events/queues: temporal decoupling; eventual consistency; harder mental model.
- Timeouts, retries with jitter, bulkheads, circuit breakers — default when calling others.

## Message queues
- Producer → broker → consumers. At-least-once delivery is common → **idempotent consumers**.
- Competing consumers scale throughput. Poison messages → DLQ.
- Ordering: per-partition / per-key, not global free lunch.
- Use for: emails, webhooks, fan-out, smoothing spikes — not as a database.

## 25-minute drill
1. Sketch request path: client → CDN → LB → 3 app nodes → primary+replica DB.
2. Argue monolith vs services for a 3-person team.
3. Why “exactly once” processing is usually “at-least-once + idempotency.”
