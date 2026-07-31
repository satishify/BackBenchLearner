---
title: "LangChain, LlamaIndex, AutoGen, CrewAI"
description: "Compare common agent/RAG frameworks by strengths — and choose based on workflow shape, not hype."
---

Frameworks speed up GenAI apps by packaging prompts, tools, indexes, and agent loops. They do not remove the need for evals, guardrails, or clear ownership of side effects. Pick the tool that matches your workflow shape — then keep your domain logic portable.

## Intuition

Four names show up constantly:

| Framework | Strength | Typical use |
| --- | --- | --- |
| LangChain | Chains, tools, prompt pipelines | General LLM apps |
| LlamaIndex | Data connectors + indexing | Knowledge / RAG assistants |
| AutoGen | Agent conversation patterns | Multi-agent message passing |
| CrewAI | Role-based orchestration | Task teams with shared goals |

Think “batteries” not “brain.” Your product still needs schemas, threat models, and CI gates.

## How it works

### LangChain

Broad ecosystem: prompt templates, retrievers, tool wrappers, LCEL/runnables, many vendor integrations. Good when you want one toolkit from prototype to agents. Risk: deep abstraction stacks that are hard to debug — keep traces on and avoid nesting magic.

### LlamaIndex

Centered on connecting data -> indexes -> query engines. Strong default for RAG-heavy assistants (files, Notion, DBs). You still own chunking strategy, evaluation, and citation policy; the library accelerates ingestion and retrieval patterns.

### AutoGen

Patterns for multiple agents exchanging messages (assistant/user proxy, group chat). Useful when the problem naturally looks like a conversation among roles. Watch cost: chatty agents amplify tokens; impose turn limits and termination conditions.

### CrewAI

Role/goal/backstory style crews that divide labor on a task. Friendly mental model for “researcher + writer + reviewer.” Same multi-agent cautions: typed artifacts beat free-form chatter; measure whether the crew beats a single agent.

### How to choose

- **RAG-heavy knowledge app:** LlamaIndex or LangChain retrievers.
- **Tool-using single agent:** LangChain (or a thin custom loop).
- **Explicit multi-agent collaboration:** evaluate AutoGen and CrewAI on your eval suite.
- **Team constraints:** observability, deploy story, license, and who already knows the stack often dominate micro-benchmarks.

### Portable core (keep yours)

Regardless of framework, own these in your code:

1. Tool allowlists and argument validation.
2. Golden evals and release gates.
3. Logging/trace IDs around every tool call.
4. HITL hooks for risky actions.
5. Memory retention and PII rules.

If a framework makes those hard, wrap it thinly or skip it.

## In code

A framework-agnostic skeleton — swap the `complete` and `retrieve` adapters for LangChain/LlamaIndex later.

```python
from dataclasses import dataclass

@dataclass
class FrameworkAdapters:
    retrieve: callable  # query -> list[str]
    complete: callable  # prompt -> str

def answer_with_rag(q: str, fx: FrameworkAdapters) -> str:
    chunks = fx.retrieve(q)[:4]
    context = "\n---\n".join(chunks)
    prompt = (
        "Answer using only CONTEXT. If missing, say you do not know.\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {q}"
    )
    return fx.complete(prompt)

# Fake adapters standing in for LlamaIndex/LangChain wiring
fx = FrameworkAdapters(
    retrieve=lambda q: [f"Policy: refunds within 30 days for '{q}'"],
    complete=lambda p: "Refunds are available within 30 days."
        if "30 days" in p else "I do not know.",
)
print(answer_with_rag("refund window?", fx))
```

Learn the framework’s tracing first; demos without traces become production mysteries.

## What goes wrong

- **Framework shopping.** Rewrites every quarter; no eval lift.
- **Abstraction blindness.** Cannot see the exact prompt or tool args in prod.
- **Multi-agent default.** Crews for problems a scripted chain would solve.
- **Version churn.** APIs move; pin versions and lock eval baselines.
- **Skipping fundamentals.** Fancy agents on bad chunking still hallucinate.
- **Vendor lock-in of business rules.** Refund policy living only inside a proprietary chain object.

## Putting it into practice

Spike two candidates for 48 hours on the same golden set and the same tools. Score quality, p95 latency, cost, and “time for a new engineer to find the prompt.” The last score predicts maintenance pain better than GitHub stars. Keep business rules (refund limits, allowlists) in your repo modules, not only inside notebook cells or proprietary chain blobs.

Pin versions, snapshot prompts used in CI, and treat framework upgrades like model upgrades: run the full suite before merging. Frameworks earn their keep when they shorten the path to a traced, evaluated path to production — not when they hide it.

## Build vs buy the loop

For a single tool-using agent, a 100-line host loop plus your vendor’s tool-calling API is often enough. Reach for LangChain/LlamaIndex when connectors and retrievers would otherwise dominate your calendar. Reach for AutoGen/CrewAI when you have measured a multi-agent win. The embarrassing outcome is a heavy framework wrapping one prompt and one HTTP GET — complexity with no leverage.

## Eval portability

Keep golden cases and graders outside the framework’s preferred storage format when you can — plain JSON/YAML in git. That way a migration from CrewAI to a custom loop does not rewrite your quality bar. Frameworks should be replaceable adapters under a stable evaluation harness; if uninstalling the library deletes your suite, the suite was never yours.

## One-line summary

Use LangChain, LlamaIndex, AutoGen, or CrewAI as accelerators for chains, RAG, or multi-agent patterns — but keep validation, evals, and safety in your own thin control layer.

## Key terms

- **Chain / runnable:** composed steps from input to model output.
- **Index / retriever:** data structures that fetch context for RAG.
- **Multi-agent framework:** library that coordinates multiple LLM roles.
- **Adapter:** thin wrapper that isolates your app from framework APIs.
- **Tracing:** structured logs of prompts, tools, and timings.
- **Pinning:** locking library/model versions for reproducible evals.
