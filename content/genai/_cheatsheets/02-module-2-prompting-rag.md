---
title: "Module 2 - Prompting, RAG & Agents revision"
slug: module-2-prompting-rag
module: "Module 2"
minutes: 35
description: "APIs, evals, prompting, RAG indexes, production retrieval, agentic loops — exam-dense."
---

Chapters **2.1–2.9**. Skim headings; drill weak spots in full lessons.

## 2.1–2.2 LLMs, decoding, APIs, structured outputs

- **Pretrain** (next-token on huge text) -> **finetune / align** -> **infer** (API call).
- Context window = hard budget for system + history + retrieved docs + output.
- Decoding knobs: temperature, top-p, max tokens, stop sequences. Determinism is not guaranteed without seed + temp 0 (still provider quirks).
- **Structured outputs**: JSON schema / tool schemas / constrained decoding. Always validate; never trust raw model text as typed data.
- Task prompts: summarize / QA / classify — specify format, audience, constraints, examples.

## 2.3–2.5 Evals, safety, advanced prompting, security

### Evals
- Offline: gold sets, rubrics, LLM-as-judge (biased — calibrate). Online: thumbs, task success, latency, cost.
- Regression: freeze a prompt suite; fail the PR if quality drops.

### Prompt patterns
- System role = durable policy. User = task instance. Few-shot = show pattern, not novels.
- CoT for multi-step reasoning; ask for final answer in a fixed format for grading.
- Contracts beat vibes: fields, enums, length, “if unknown say UNKNOWN”.

### Security
- **Prompt injection**: untrusted content (web, PDF, email) steers the model. Separate trust boundaries; don’t let tools run on attacker text blindly.
- Least privilege for tools. Log tool calls. Human-in-the-loop for irreversible actions.

## 2.6–2.8 RAG fundamentals to production

### Pipeline
`query -> embed -> retrieve top-k -> (rerank) -> stuff into prompt -> generate (+ cite)`

### Chunking
- Too big -> noisy context. Too small -> lose meaning. Overlap helps boundaries. Metadata (source, section) enables filters.

### Indexes (know the tradeoffs)
- **Flat / brute force** — exact, slow at scale.
- **IVF** — cluster centroids; search `nprobe` lists. Faster, approximate; tune recall vs latency.
- **PQ (product quantization)** — compress vectors into codes; huge memory win, more approx error.
- **HNSW** — graph ANN; strong recall/latency; RAM heavy. Common default in vector DBs.

### Production RAG
- Hybrid search (BM25 + dense). Rerankers when top-k is cheap noise.
- Citations & grounding: force quotes / IDs; refuse when retrieval weak.
- Eval: faithfulness, answer relevance, context precision/recall — not only “sounds good”.
- Ops: version embeddings + index; watch empty retrieval, p95 latency, cost/query.

## 2.9 Agentic AI & multi-agent

- **Chatbot** = reply in thread. **Agent** = plan + tools + memory + stop condition.
- Loop: observe -> think/plan -> act (tool) -> observe -> ... -> answer or escalate.
- Tools: typed functions; validate args; timeouts; idempotency for side effects.
- Memory: short-term (scratchpad / messages) vs long-term (store + retrieve). Don’t dump everything into context.
- Multi-agent: specialize roles; orchestrate (router, sequential, hierarchical). More agents is not more quality — more failure modes.
- Reflection / self-critique: second pass that checks against tools or rubric; cap retries.

## 30-minute drill
1. Draw IVF vs HNSW in one sentence each + one failure mode.
2. Write a 5-line system prompt that blocks tool use on untrusted pasted text.
3. List three RAG metrics you’d put on a dashboard before launch.
