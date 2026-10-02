---
title: "Module 4 - VLM, multimodal RAG, and agentic AI revision"
slug: module-4-multimodal-agentic
module: "Module 4"
minutes: 40
description: "Revision for VLM architectures, applications, multimodal retrieval, agents, and multi-agent coordination."
---

Chapter **4.1** follows this path: Foundations → CLIP → LLaVA → Qwen-VL → SAM → Synthesis.

## Foundations

- **VLM** = pixels → vision encoder → connector → consumer → output (score, text, box, or mask).
- **ViT**: patchify image; `(H/P) × (W/P)` patch tokens; add position info. Example: 224×224, 16×16 patches → **196** tokens.
- Four paradigms: **contrastive** (CLIP), **generative** (LLaVA), **grounded generative** (Qwen-VL), **promptable segmentation** (SAM).

## CLIP

- Dual encoders; **symmetric contrastive** loss; diagonal of batch similarity matrix = correct pairs.
- **Zero-shot**: encode image once; compare to text templates like `a photo of a {label}`.
- Pick CLIP for **matching/retrieval**; not for long explanations.

## LLaVA

- Frozen CLIP ViT → **projector** → visual tokens in LLM sequence.
- Stage 1: train projector only. Stage 2: instruction tune projector + LLM.
- **LLaVA-NeXT / AnyRes**: high-res tiles + global view for OCR/charts (more tokens).

## Qwen-VL

- **256 learned queries** cross-attend over variable patches; **2D position** on keys.
- Grounding via `<ref>`, `<box>` tokens; coordinates **0–1000** normalized, generated as text.
- Fixed token budget = predictable cost; may lose fine detail.

## SAM

- **Image encoder once** (cache) → **prompt encoder** + **mask decoder** per click/box.
- **Three candidate masks** for ambiguous points; **IoU** scores quality.
- Composes with grounding: box from VLM → SAM mask. SAM does not name the object.

## Synthesis

| Need | Start with |
| --- | --- |
| Tag / retrieve | CLIP |
| Chat about image | LLaVA |
| OCR + boxes + multi-image | Qwen-VL |
| Pixel mask | SAM |
| Grasp / rotoscope pipeline | VLM + SAM |

## 4.2 VLM applications

- **Image understanding** — CLIP for tags; generative VLM for sentences. Use visible-only, length-limited prompts.
- **VQA** — question decides the evidence. Allow `not present`; list objects before counting.
- **Document intelligence** — OCR + layout + relationships. Request named JSON and use `null` for missing fields.
- **Visual reasoning** — list evidence, reason, then verify. A detailed chain can still be wrong.
- **Chart QA** — extract labels and values first; calculate with code; state uncertainty.
- **SAM recap** — heavy image encoding once, cheap mask decoding for each prompt.

| Application failure | Planned control |
| --- | --- |
| Caption hallucination | Visible-only prompt + negative examples |
| VQA false assumption | Verify object exists |
| Guessed document field | `null` + validation + human review |
| Compounding reasoning error | Independent verifier |
| Tiny/thin segmentation miss | Domain testing and prompt refinement |

## 4.3 Multimodal RAG

- **RAG** separates retrieval from generation: find evidence first, then answer from it.
- **Text search**: BM25 protects exact words; dense embeddings match meaning; hybrid search combines both.
- **Two-stage retrieval**: fast ANN search over the corpus, then careful reranking over a small candidate set.
- **Image search**: CLIP-style shared space supports text-to-image and image-to-image retrieval.
- **Document RAG**: parse then embed for clean text, or search complete page images when layout and charts matter.
- **Late interaction**: each query token matches its best document token or image patch; sum the best scores.
- **ColPali**: VLM-based page-image retrieval without OCR in its core representation.
- **Trade-off**: patch vectors preserve detail but need a larger, more expensive index.
- **Grounded generation**: answer only from retrieved pages, cite the exact evidence, and allow `not found`.
- **Evaluate separately**: Recall@K/nDCG for retrieval; correctness, faithfulness, citations, and abstention for generation.

| Data | Good starting point |
| --- | --- |
| Plain text | Text or hybrid RAG |
| Products/photos | CLIP-style image search |
| Reports/charts/forms | Multimodal document RAG |
| High-stakes answers | Retrieval + claim citations + validation |

## 4.4 RAG to assistants and agents

### Capability ladder

| Stage | What is added | Travel example |
| --- | --- | --- |
| **LLM** | Language generation | Writes a plausible itinerary |
| **RAG** | External knowledge | Retrieves company policy |
| **Assistant** | Conversation context | Remembers dates and budget |
| **Tool-using assistant** | External capabilities | Calls flight and hotel APIs |
| **Agent** | Runtime decisions | Replans after a failed policy check |
| **Multi-agent** | Delegation | Splits flights, hotels, meetings, expenses |

- **RAG answers; agents finish.** Use RAG for grounded questions and an agent for a goal whose steps depend on results.
- A model only **requests** a tool call. Application code validates and executes it.
- **MCP** standardises tool/resource discovery and calls; it does not replace permissions or safety checks.

### Workflow vs agent

- **Workflow** — developer fixes the order at build time.
- **Agent** — model chooses the next step at runtime.
- Main test: if the full flowchart can be drawn before the request arrives, start with a workflow.
- Practical systems combine both: fixed validation and approval around one genuinely dynamic agent step.

### Agent loop

`think → act → observe → remember → repeat`

- Must be **bounded** by steps, time, and cost.
- Must be **observable** through replayable traces.
- Must be **recoverable** when a tool returns an error.
- A failed tool or policy check is an observation that may trigger replanning.

### Agent patterns

| Pattern | Shape | Good fit |
| --- | --- | --- |
| **ReAct** | Think, act, observe, repeat | Short tasks driven by latest result |
| **Plan-and-execute** | Plan first, execute, replan failures | Long tasks and parallel substeps |
| **Reflection** | Draft, critique, revise | Reports, code, and final write-ups |

Context cost grows because each model call often resends the goal, tools, and earlier history. Truncate observations, summarise old decisions, and keep structured task state outside the transcript.

### Retrieval, state, and multiple agents

- Inside an agent, retrieval is a **tool** that can be called, skipped, repeated, or given a rewritten query.
- **Conversation state** lives for one thread.
- **Task state** lives for one run and tracks what is done or pending.
- **Long-term memory** survives across sessions and should contain only durable facts.
- Start with one agent. Split into specialists only when tool overload, context crowding, or separate failure domains are measured problems.
- Handoffs should be typed artifacts with amounts, dates, status, and source references.

### Framework rule

Choose by your technology stack and operational needs, not hype. Keep goals, tool policies, state, traces, approvals, and evaluation portable outside any one SDK.

## 4.5 Multi-agent systems

A team is not a bigger model. It is specialists plus a coordinator that owns the goal.

### When to split

- One agent hides a trade-off (cheap vs comfortable vs policy).
- Good local choices clash (23:50 arrival vs 23:00 check-in).
- Specialisation, parallelism, smaller context, or a checker would help.

Several tools can still belong to one agent. Start with one loop.

### Roles, contracts, handoffs

An agent = **role + model + tools + state + contract**.

If you cannot say "this agent owns X and returns Y," do not delegate yet.

Handoffs stay small: goal, constraints, expected format. Not the whole chat.

### Coordination cycle

`decompose → execute → attribute → reconcile` (repeat)

Bound it with a **round limit**, a **token budget**, and a **deadlock check**. Escalate a repeated conflict instead of looping forever.

Search flights and hotels together when they do not need each other. Run budget and policy after a pair is chosen.

### Supervisor, state, validation

- Default pattern: **supervisor–worker**. One hub owns the goal and the audit trail.
- Peer-to-peer is for debate or review, not for spending money.
- Validate schema, money, and times before a result enters shared state.
- Send each specialist a **slice**, not the full state.
- Record failures as structured errors, not empty text.

### Permissions, trust, logs

| Work | Rule |
| --- | --- |
| Search / policy / budget | Automatic, read-only |
| Book / send | Human confirms first |

Treat another agent's text as **untrusted data**. Wrap it, check it, reject "ignore the budget," and still require approval for writes.

Log delegation, tool call, raw result, validation, latency/retries, and who approved the write.

### Cost, eval, standards

One round of 4 specialists ≈ 8 model-facing messages plus their tool calls.

Score **selection, arguments, task completion, groundedness, conflicts caught, rounds/cost**. A fluent itinerary is not a pass.

| Standard | Connects |
| --- | --- |
| **A2A** | Agent ↔ agent |
| **MCP** | Agent ↔ tools and data |
| **ANP** | Agents on the open web (proposed) |
| **AGNTCY** | Directory, identity, messaging |

Neither A2A nor MCP makes a booking safe.

### Five rules

1. Specialise only when it helps.
2. Contract every agent.
3. Coordinate in a bounded loop.
4. Keep handoffs small and checked.
5. Treat every extra agent as extra risk.

## 20-minute drill

1. Walk through ViT patch count for 224×224 and 16×16 patches.
2. Explain why LLaVA needs a projector even when dimensions match.
3. Convert one normalized `<box>` to pixel coordinates for a given image size.
4. Sketch a two-step pipeline: Qwen-VL finds box → SAM segments.
5. Write a caption prompt with scope and length constraints.
6. Write an invoice JSON schema that uses `null` for missing values.
7. Compare BM25, dense, and hybrid search for an exact policy number.
8. Sketch text query → page retrieval → VLM answer → page citation.
9. Explain late interaction without using the formula.
10. Diagnose separately: the right page was retrieved, but the answer invented a number.
11. Explain the six stages from LLM to multi-agent using the Paris request.
12. Decide workflow or agent for monthly expense reporting and justify the choice.
13. Trace the Paris loop through the failed seven-day booking check.
14. Compare ReAct, plan-and-execute, and reflection using one sentence each.
15. Classify a fact as conversation state, task state, or long-term memory.
16. Explain why MCP discovery does not make a tool safe.
17. Give one reason to keep a single agent and one reason to split into specialists.
18. Rewrite `travel_agent: handle the trip` into a contract with inputs, outputs, and limits.
19. Sketch the coordination cycle and name one stop condition.
20. Decide supervisor vs peer-to-peer for a booking desk and justify it.
21. Treat "ignore the budget" from hotel_agent as data and say what gets logged.
22. Split A2A from MCP in one sentence each.
