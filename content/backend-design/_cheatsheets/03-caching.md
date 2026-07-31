---
title: "Caching revision"
slug: caching
module: "Caching chapter"
minutes: 20
description: "Why cache, eviction, consistency patterns, CDN/edge, when cache lies."
---

Revision for **Caching & Performance**.

## Why cache
- Trade freshness for latency and cheaper origin load.
- Cache the expensive + hot + somewhat stale-tolerant. Don’t cache what must be exact without a plan.

## Common patterns
- **Cache-aside**: app reads cache -> miss -> DB -> fill. App writes DB then invalidates/updates cache.
- **Write-through / write-behind**: cache participates in writes (more complexity).
- Stampede: many miss together -> lock / singleflight / probabilistic early expire.

## Eviction
- LRU / LFU / TTL / size caps. Memory is finite — pick policy matching access skew.
- Explicit invalidation on write often beats hoping TTL is “good enough” for mutable data.

## Consistency
- Invalidate on write (delete key) is simpler than updating every shape of cached object.
- Read-your-writes: sometimes sticky to primary or short bypass after mutate.
- Multi-layer: browser -> CDN -> app cache -> DB — each layer needs a story.

## CDN & edge
- Static assets and public pages shine at the edge. Personalized HTML is harder (vary cookies carefully).
- Cache-Control / ETag / Surrogate keys. Purge strategy for deploys and content edits.

## When cache makes systems wrong
- Stale permissions, prices, inventory -> real money bugs.
- Partial invalidation (forgot secondary key) -> zombie data.
- Caching errors (5xx) or empty results without short TTL -> outages amplify.
- Debug: “Is this layer serving old bytes?” before rewriting business logic.

## 20-minute drill
1. Design cache-aside for `GET /users/:id` including invalidation on PATCH.
2. Name one stampede defense.
3. When must you skip CDN caching?
