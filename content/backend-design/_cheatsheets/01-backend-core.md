---
title: "Backend core concepts"
slug: backend-core
module: "Backend chapter"
minutes: 25
description: "APIs, HTTP, auth, sessions/JWT, OAuth2, rate limits, idempotency."
---

Revision for the **Backend Core Concepts** chapter.

## APIs & styles
- **API** = contract for clients to trigger behavior / fetch data without knowing internals.
- **REST**: resources + HTTP verbs + status codes; cache-friendly GETs; evolve carefully.
- **GraphQL**: client-shaped queries; great for nested UIs; watch N+1, authz per field, caching complexity.
- Prefer boring REST until product needs GraphQL’s flexibility.

## HTTP methods & status codes
- GET safe/idempotent. POST create/non-idempotent by default. PUT replace. PATCH partial. DELETE remove.
- **2xx** success * **3xx** redirect * **4xx** client fault * **5xx** server fault.
- Memorable: 200 OK, 201 Created, 204 No Content, 400 Bad Request, 401 Unauthorized (auth missing/bad), 403 Forbidden (authz), 404, 409 Conflict, 429 Too Many Requests, 500, 502/503.

## Stateless vs stateful
- Stateless request carries all needed context (tokens, ids) -> easy horizontal scale.
- Stateful server sessions need sticky sessions or shared session store.
- Pick stateful only when you must (WebSockets, shopping cart edge cases) and design the store.

## Authn vs authz
- **Authentication** — who are you? **Authorization** — what may you do?
- Never conflate “logged in” with “allowed to mutate this resource.” Check ownership / roles / policies on every sensitive path.

## Session vs JWT
- **Server session**: opaque id in cookie; revoke easy; needs store.
- **JWT**: signed claims; scalable validation; revocation harder (short TTL + denylist / version).
- Cookies: `HttpOnly`, `Secure`, `SameSite`. Don’t store long-lived secrets in localStorage if you can avoid XSS blast radius.

## OAuth2 (mental model)
- Client never sees user’s password at the resource server.
- Authorization code (+ PKCE for public clients) is the default web/mobile pattern.
- Access token short-lived; refresh token carefully stored; scopes = coarse authz.

## Rate limiting vs throttling
- Cap abuse and protect capacity: token bucket / sliding window / fixed window.
- Return 429 + `Retry-After`. Limit by user, IP, API key — not only global.
- Throttling smooths bursts; limiting hard-stops over quota.

## Idempotency
- Retries must not double-charge. Clients send `Idempotency-Key` on POST payments/orders.
- Server stores key -> response (or in-flight lock) for a TTL. Same key + same body -> same outcome.

## 25-minute drill
1. Map 401 vs 403 with one example each.
2. Design idempotent “create payment” in 5 bullets.
3. When would you refuse JWT and pick sessions?
