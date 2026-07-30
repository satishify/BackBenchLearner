---
title: "Reliability revision"
slug: reliability
module: "Reliability chapter"
minutes: 25
description: "Delivery semantics, retries, locks, races, sagas, degradation, observability, deploys, spikes."
---

Revision for **Reliability & Real-World Problems**.

## Delivery semantics
- **At-most-once**: may lose. **At-least-once**: may duplicate. **Exactly-once**: end-to-end myth without careful idempotency + dedupe + transactional outbox patterns.
- Design for at-least-once + idempotent handlers.

## Retries, timeouts, circuit breakers
- Always set timeouts. Retry only transient failures; exponential backoff + jitter.
- Circuit breaker: stop calling a sick dependency; fail fast; half-open probe.
- Don’t retry non-idempotent POSTs without keys.

## Race conditions & locking
- Two requests, one invariant → lost updates. Fix with transactions, constraints, optimistic versions, or distributed locks.
- Distributed locks (Redis etc.): understand fencing tokens / TTL expiry; locks are not consensus magic.

## Event-driven & sagas
- Events decouple writers from many readers. Outbox pattern avoids dual-write loss.
- **Saga**: break distributed transaction into steps + compensations. Not ACID — design compensations.
- Choreography vs orchestration: who owns the flowchart?

## Graceful degradation
- Shed noncritical features; serve stale cache; read-only mode; queue writes.
- Define SLOs: what “good enough” looks like under fire.

## Observability
- Logs (events), metrics (aggregates), traces (request path). Correlate with request ids.
- Alert on symptoms (error rate, latency, saturation) not only causes.

## Deployments
- Blue/green, canary, rolling. Migrations: expand → migrate → contract (backward compatible).
- Feature flags for risky behavior. Rollback plan before you need it.

## Traffic spikes
- Autoscale + queue + cache + rate limits. Pre-warm. Know your DB connection ceiling.
- Load test the path that will melt (checkout, login), not the homepage alone.

## 25-minute drill
1. Write an idempotent webhook handler outline.
2. Pick saga vs 2PC for “book hotel + flight” and why.
3. Name three golden signals you’d page on.
