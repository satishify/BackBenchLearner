---
title: "Guardrails and Security"
description: "Guardrails and Security: concepts, examples, and practical notes for learners on BackbenchLearner."
---

Guardrails are controls that keep AI behavior within allowed boundaries.

## Guardrail layers

1. **Input guardrails:** block unsafe/malicious prompts.
2. **Processing guardrails:** restrict tool/data access.
3. **Output guardrails:** moderate and validate response.

## Security basics

- Use least privilege for tools and APIs.
- Never expose secrets in prompts or logs.
- Mask PII and sensitive enterprise data.
- Audit all critical tool calls.
