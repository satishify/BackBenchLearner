---
title: "Reranking and Query Rewriting"
description: "Improve what enters the prompt by rewriting messy questions and reranking first-stage candidates with a stronger model."
---

First-stage retrieval is built for **recall at speed**: grab many maybe-relevant chunks cheaply. **Reranking** is built for **precision**: score those candidates carefully and keep the best few for the LLM. **Query rewriting** fixes the input so the first stage has a fighting chance.

## Intuition

| Stage | Plain-English idea | Analogy |
| --- | --- | --- |
| **First-stage retrieve** | Cast a wide net cheaply | Throw fish on the deck |
| **Rerank** | Pick the keepers carefully | Sort the catch |
| **Query rewrite** | Turn chat into search language | Translator between chat and search |

Users ask like humans: "hey can u explain that leave thing?" Search indexes prefer "casual leave policy carry over."

```mermaid
flowchart LR
  U[User question] --> W[Query rewrite]
  W --> R1[First-stage retrieve top-N]
  R1 --> RR[Rerank to top-k]
  RR --> L[LLM context]
```

## How it works

### Pre-retrieval and post-retrieval

| Phase | Plain-English idea |
| --- | --- |
| **Pre-retrieval** | Rewrite the query so the retriever sees better keywords or context |
| **Post-retrieval** | Rerank or filter chunks after retrieval, before the LLM sees them |

The raw user question is not always the best search query. The first chunks returned are not always the best evidence.

### Query rewriting patterns

| Method | What it does | Example |
| --- | --- | --- |
| **Rewriting** | Expand vague prompts into precise search questions | "OOMKilled" → "What causes OOMKilled container exit code in Kubernetes?" |
| **Follow-up question** | Turn context-dependent question into standalone | "Does it apply to contractors?" → "Does parental leave apply to contract employees?" |
| **Multi-query** | Generate several related queries in parallel | Split a comparison into one query per system |
| **Step-back prompting** | Ask a broader question first for foundational context | Specific case → broader system-level question |
| **HyDE** | Hypothetical Document Embeddings—LLM writes a pretend answer/doc, embed that for search | Risky if the draft invents facts |

Keep rewrites **instrumented**: log original vs rewritten and measure hit rates.

### Rerankers

| Type | Plain-English idea | Speed |
| --- | --- | --- |
| **Cross-encoder** | Encodes query and document together for a relevance score | Slow but accurate—for N ≤ 100 |
| **LLM reranker** | Prompt a model to order passages | Flexible, costlier |
| **Feature reranker** | Boost by recency, clicks, metadata match | Fast add-on |

### Typical cascade

Hybrid retrieve N=50 → cross-encode → keep k=5 → generate. Most quality gains per dollar sit here, not in doubling LLM size.

## In code

Toy rewrite plus a stand-in cross-encoder (token overlap).

```python
import re

GLOSSARY = {"pto": "paid time off", "hpa": "horizontal pod autoscaling"}

def rewrite(query: str) -> str:
    q = query.lower()
    q = re.sub(r"\b(hey|please|can you|explain)\b", " ", q)
    for src, dst in GLOSSARY.items():
        q = re.sub(rf"\b{src}\b", dst, q)
    return " ".join(q.split())

candidates = [
    "Employees receive 12 casual leaves; unused leaves may carry over up to 5.",
    "Horizontal pod autoscaling adds replicas when CPU is high.",
]

def fake_cross_encoder(query: str, doc: str) -> float:
    q_toks = set(query.split())
    d_toks = doc.lower().split()
    overlap = len(q_toks & set(d_toks))
    return overlap / (1.0 + 0.01 * len(d_toks))

original = "hey can u explain PTO carry over?"
q = rewrite(original)
print("rewritten:", q)
for doc, score in sorted(
    [(d, fake_cross_encoder(q, d)) for d in candidates],
    key=lambda x: x[1], reverse=True,
):
    print(f"{score:.3f} | {doc[:64]}...")
```

## What goes wrong

- **Rewrite hallucination** — HyDE invents a wrong API name; retrieval locks onto fiction.
- **Reranking too few** — If stage-one never retrieved the truth, rerankers cannot resurrect it.
- **Reranking too many** — Cross-encoding thousands of docs blows latency budgets.
- **Chatty context pollution** — Feeding the whole conversation as the search query dilutes keywords.
- **Ignoring freshness** — Relevance-only rerankers promote obsolete policies.

:::key
Change one stage at a time in A/B tests: rewrite on/off, then rerank on/off, then hybrid weights. Simultaneous changes make wins impossible to attribute.
:::

## One-line summary

Rewrite user questions into searchable forms, retrieve broadly, then rerank with a stronger model so only the best evidence reaches the generator.

## Key terms

- **Query rewriting:** transforming user text into better search queries.
- **Multi-query retrieval:** paraphrases fused for higher recall.
- **HyDE (Hypothetical Document Embeddings):** embed a synthetic answer/doc to improve dense search.
- **Reranker / cross-encoder:** joint query–document scorer for precision.
- **Cascade retrieval:** cheap wide retrieval followed by expensive precise ranking.
- **Step-back prompting:** broader question first to recover foundational context.
