---
title: "Module 2 - Prompting, RAG & Agents revision"
slug: module-2-prompting-rag
module: "Module 2"
minutes: 35
description: "APIs, evals, prompting, RAG indexes, production retrieval, agentic loops — beginner-friendly revision."
---

Chapters **2.1–2.9**. Skim headings; drill weak spots in full lessons.

## 2.1–2.2 LLMs, decoding, APIs, structured outputs

- **Pretrain** (predict next word on huge text) → **finetune / align** (shape behavior) → **infer** (call the model at answer time).
- **Context window** = hard budget for system rules + chat history + retrieved docs + output.
- **Decoding knobs:** temperature, top-p, max tokens, stop sequences. Even with temperature 0, exact repeats are not guaranteed (provider quirks).
- **Structured outputs:** JSON schema, tool schemas, constrained decoding. Always validate—never trust raw model text as typed data.
- **Task prompts:** summarize / Q&A / classify—specify format, audience, constraints, and examples.

## 2.3–2.5 Evals, safety, advanced prompting, security

### Evals

- **Offline:** gold question sets, rubrics, LLM-as-judge (can be biased—calibrate). **Online:** thumbs up/down, task success, latency, cost.
- **Regression:** freeze a prompt suite; fail the release if quality drops.

### Prompt patterns

- **System role** = durable policy. **User message** = this task. **Few-shot** = show the pattern, not a novel.
- **Chain-of-thought (CoT)** for multi-step reasoning; ask for a final answer in a fixed format for grading.
- **Contracts beat vibes:** name fields, allowed values, length limits, "if unknown say UNKNOWN."

### Security

- **Prompt injection:** untrusted content (web, PDF, email) steers the model. Separate trust boundaries; do not run tools on attacker text blindly.
- **Least privilege** for tools. Log tool calls. Human-in-the-loop for irreversible actions.

## 2.6–2.8 RAG fundamentals to production

### What RAG is

**RAG** (retrieval-augmented generation) = find relevant text first, add it to the prompt, then let the LLM answer from that evidence.

```
query → embed → retrieve top-k → (rerank) → pack into prompt → generate (+ cite)
```

### Chunking

| Problem | Plain-English idea |
| --- | --- |
| **Too big** | Noisy context; one chunk mixes many topics |
| **Too small** | Loses meaning; answer spans get cut off |
| **Overlap (10–20%)** | Facts near chunk boundaries appear in two chunks |
| **Metadata** | source, section, date, tenant—enables filters and citations |

### Search types

| Type | Plain-English idea | Good for |
| --- | --- | --- |
| **BM25 (keyword)** | Match exact words with smart scoring | Error codes, SKUs, names |
| **Dense (semantic)** | Match meaning via embeddings | Paraphrases, synonyms |
| **Hybrid** | Run both; merge with RRF or weighted fusion | Production default |

### Indexes (know the tradeoffs)

| Index | Plain-English idea | Trade-off |
| --- | --- | --- |
| **Flat / brute force (KNN)** | Compare query to every vector | Exact; slow at scale |
| **IVF (inverted file index)** | Cluster vectors; search **nprobe** buckets | Faster; approximate—tune recall vs latency |
| **PQ (product quantization)** | Compress vectors into short codes | Huge memory win; more approximation error |
| **HNSW (hierarchical navigable small world)** | Layered graph of neighbors | Strong recall/speed; RAM-heavy |

**Knobs:** IVF **nprobe** and HNSW **efSearch** both spend more query time for better recall.

### Production RAG

- **Two-stage ranking:** bi-encoder retrieve cheaply → cross-encoder rerank top-N → generate.
- **Retrieval gateway:** auth before search, query shaping, semantic cache (high threshold—false hits hurt), routing to the right index or tool.
- **Citations & grounding:** require chunk IDs; verify claims against cited text; refuse when retrieval is weak.
- **Eval (RAGAS-style):** faithfulness, answer relevancy, context precision/recall—not only "sounds good."
- **Ops:** version embeddings + index; watch empty retrieval, p95 latency, cost per query.

## 2.9 Agentic AI & multi-agent

- **Chatbot** = reply in thread. **Agent** = plan + tools + memory + stop condition.
- **Loop:** observe → think/plan → act (tool) → observe → … → answer or escalate.
- **Tools:** typed functions; validate args; timeouts; idempotency for side effects.
- **Memory:** short-term (scratchpad / messages) vs long-term (store + retrieve). Do not dump everything into context.
- **Multi-agent:** specialize roles; orchestrate (router, sequential, hierarchical). More agents ≠ more quality—more failure modes.
- **Reflection / self-critique:** second pass that checks against tools or rubric; cap retries.

## 30-minute drill

1. Draw IVF vs HNSW in one sentence each + one failure mode.
2. Write a 5-line system prompt that blocks tool use on untrusted pasted text.
3. List three RAG metrics you'd put on a dashboard before launch.
