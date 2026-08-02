---
title: "Prompt vs RAG vs Fine-Tuning"
description: "A simple decision framework: prompting changes the instruction, RAG changes the context, fine-tuning changes the model."
---

You have three pathways to get useful work from an LLM. The big mental model:

- **Prompting** changes the instruction
- **RAG** changes the context (retrieved documents)
- **Fine-tuning** changes the model weights

Pick the lever that matches the bug.

## Intuition

| Method | What changes | Best when | Main trade-off |
| --- | --- | --- | --- |
| **Prompting** | Only the input instruction | Fast experiments, low cost, simple tasks | Can be inconsistent; wording-sensitive |
| **RAG** | Model gets retrieved external knowledge at answer time | Facts change often, or answers must come from private/current docs | Quality depends on retrieval and chunking |
| **Fine-tuning** | Model weights update on task data | You want a stable style, format, or domain habit | Training cost, data quality, forgetting risk |

:::key
If knowledge changes every week, start with RAG. If behavior must stay stable across many future prompts, consider fine-tuning. If a clear prompt already works, stay with prompting.
:::

## How it works

### A practical decision checklist

1. Does the knowledge change often? → Prefer **RAG** first.
2. Do you need a stable, reusable behavior (tone, schema, domain phrasing)? → **Fine-tuning** becomes attractive.
3. How much labeled data do you have? Small data usually favors prompting, RAG, or light PEFT before full fine-tuning.
4. Is this one task or many related tasks? Related tasks can share one multi-task fine-tune later.
5. Is the base model already almost right? If yes, train less (freeze more / lighter methods).

### Split workflows are normal

One product can use **both**:

- Fresh policy facts → RAG (and cite the retrieved text)
- Permanent email tone → fine-tuning

You do not have to force one tool for every workflow.

### Where knowledge should live

A useful framing is “fine-tuning or retrieval?”:

- Put **changing facts** in an external store and retrieve them.
- Put **stable habits** into weights when you truly need them to stick.

## What goes wrong

- Full fine-tuning a model on policies that change monthly, then shipping stale answers.
- Endless prompt rewriting for a tone that never stabilizes — when labeled examples exist for fine-tuning.
- Fine-tuning with almost no data, then blaming the optimizer.

## One-line summary

Prompt for control, RAG for fresh/private facts, fine-tune for stable reusable behavior — choose by what must change.

## Key terms

- **Prompting** — Steering the model with instructions only.
- **RAG** — Retrieve documents at answer time and feed them to the model.
- **Fine-tuning** — Updating weights on task-specific data.
- **Knowledge injection** — Getting task or domain knowledge into the system’s behavior (via prompt, retrieval, or weights).
