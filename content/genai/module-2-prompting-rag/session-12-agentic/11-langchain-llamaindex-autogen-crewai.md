---
title: "LangChain, LlamaIndex, AutoGen, CrewAI"
description: "Compare common agent and RAG (retrieval-augmented generation) frameworks—and choose based on workflow shape, not hype."
---

**What is this for?** To compare popular frameworks for building agent and RAG apps—and explain when **LangGraph** fits vs plain **LangChain**.

**Why does it exist?** Frameworks speed up prototyping, but they do not remove the need for evals, guardrails, or clear ownership of side effects. Pick the tool that matches your **workflow shape**.

## Intuition

Four names show up constantly:

| Framework | Plain-English strength | Typical use |
| --- | --- | --- |
| **LangChain** | Chains, tools, prompt pipelines | General LLM apps |
| **LangGraph** | Cyclic, stateful graph control flow | Reasoning loops, multi-agent graphs |
| **LlamaIndex** | Data connectors + indexing | Knowledge / RAG assistants |
| **AutoGen** | Agent conversation patterns | Multi-agent message passing |
| **CrewAI** | Role-based orchestration | Task teams with shared goals |

Think "batteries," not "brain." Your product still needs schemas, threat models, and CI gates.

### LangChain vs LangGraph

| | LangChain | LangGraph |
| --- | --- | --- |
| **Plain-English idea** | Component framework | Graph that can loop and remember |
| **Flow shape** | Mostly linear chain | Cyclic graph with state |
| **Good for** | Sequential tasks | Reasoning loops, branches, HITL pauses |
| **Analogy** | Straight line | Flowchart |

:::key
LangChain helps you assemble prompts, tools, and models. LangGraph wires those pieces into a graph that can loop, branch, pause, and resume.
:::

## How it works

### LangChain

Broad ecosystem: prompt templates, retrievers, tool wrappers, runnables, many vendor integrations. Good when you want one toolkit from prototype to agents.

Risk: deep abstraction stacks that are hard to debug—keep traces on and avoid nesting magic.

### LangGraph

Graph-based framework for **looping, stateful agent workflows**:

- **Nodes** = agent steps or functions.
- **Edges** = fixed routes between nodes.
- **Conditional edges** = router functions that inspect state and pick the next node.
- **`compile()`** = turns the graph definition into a runnable, **checkpointable** application.

Example shape:

```python
graph = StateGraph(AgentState)
graph.add_node('triage_agent', triage_node)
graph.add_node('infra_agent', infra_node)
graph.add_conditional_edges('triage_agent', router_function)
graph.add_edge('infra_agent', 'triage_agent')
app = graph.compile()
```

Use LangGraph when the workflow needs loops, branches, human approval checkpoints, or recursion limits.

### LlamaIndex

Centered on connecting data → indexes → query engines. Strong default for RAG-heavy assistants (files, Notion, databases). You still own chunking strategy, evaluation, and citation policy.

### AutoGen

Patterns for multiple agents exchanging messages (assistant/user proxy, group chat). Useful when the problem naturally looks like a conversation among roles. Watch cost: chatty agents amplify tokens; impose turn limits and termination conditions.

### CrewAI

Role/goal/backstory style crews that divide labor on a task. Friendly mental model for "researcher + writer + reviewer." Same multi-agent cautions: typed artifacts beat free-form chatter; measure whether the crew beats a single agent.

### How to choose

| If your problem looks like... | Lean toward |
| --- | --- |
| RAG-heavy knowledge app | LlamaIndex or LangChain retrievers |
| Tool-using single agent | LangChain or a thin custom loop |
| Cyclic loops, HITL, multi-agent graphs | LangGraph |
| Explicit multi-agent chat | AutoGen or CrewAI (on your eval suite) |

Team constraints—observability, deploy story, license, who already knows the stack—often dominate micro-benchmarks.

### Portable core (keep yours)

Regardless of framework, own these in your code:

1. Tool allowlists and argument validation.
2. Golden evals and release gates.
3. Logging/trace IDs around every tool call.
4. HITL hooks for risky actions.
5. Memory retention and PII rules.
6. Provenance: trace claims back to tool-call history.

If a framework makes those hard, wrap it thinly or skip it.

## In code

A framework-agnostic skeleton—swap the `complete` and `retrieve` adapters for LangChain/LlamaIndex later.

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

fx = FrameworkAdapters(
    retrieve=lambda q: [f"Policy: refunds within 30 days for '{q}'"],
    complete=lambda p: "Refunds are available within 30 days."
        if "30 days" in p else "I do not know.",
)
print(answer_with_rag("refund window?", fx))
```

Learn the framework's tracing first; demos without traces become production mysteries.

## What goes wrong

- **Framework shopping.** Rewrites every quarter; no eval lift.
- **LangChain for cyclic loops.** Fighting the framework instead of using LangGraph.
- **Multi-agent default.** Crews for problems a scripted chain would solve.
- **Abstraction blindness.** Cannot see the exact prompt or tool args in prod.
- **Skipping fundamentals.** Fancy agents on bad chunking still hallucinate.

## Putting it into practice

Spike two candidates for 48 hours on the same golden set and the same tools. Score quality, p95 latency, cost, and "time for a new engineer to find the prompt."

Pin versions, snapshot prompts used in CI, and treat framework upgrades like model upgrades: run the full suite before merging.

## Build vs buy the loop

For a single tool-using agent, a 100-line host loop plus your vendor's tool-calling API is often enough. Reach for LangChain/LlamaIndex when connectors and retrievers would otherwise dominate your calendar. Reach for LangGraph when you need loops, branches, or HITL. Reach for AutoGen/CrewAI when you have measured a multi-agent win.

## Eval portability

Keep golden cases and graders outside the framework's preferred storage format when you can—plain JSON/YAML in git. Frameworks should be replaceable adapters under a stable evaluation harness.

## One-line summary

Use LangChain, LlamaIndex, AutoGen, or CrewAI as accelerators—but use LangGraph when you need cyclic stateful control; keep validation, evals, and safety in your own thin layer.

## Key terms

- **LangChain:** framework for assembling LLM components and tools.
- **LangGraph:** graph-based framework for looping, stateful agent workflows.
- **Node:** one function or agent step in a graph.
- **Conditional edge:** route chosen by inspecting current state.
- **Checkpointing:** saving state so a graph can pause and resume later.
- **Chain / runnable:** composed steps from input to model output.
- **Adapter:** thin wrapper that isolates your app from framework APIs.
