---
title: "Monitoring and Observability"
description: "Monitoring and Observability: concepts, examples, and practical notes for learners on BackbenchLearner."
---

Observability gives visibility into AI system behavior so teams can detect quality, safety, and reliability issues early.

```mermaid
flowchart LR
    A[Requests] --> B[Agent service]
    B --> C[Logs]
    B --> D[Metrics]
    B --> E[Traces]
    C --> F[Dashboards]
    D --> F
    E --> F
    F --> G[Alerts]
```

## What to monitor

- Accuracy and hallucination rate.
- Latency (P50/P95).
- Token and tool cost.
- Safety violation counts.
- Tool/API failure rates.
