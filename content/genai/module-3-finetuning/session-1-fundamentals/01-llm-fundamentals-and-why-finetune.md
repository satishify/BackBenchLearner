---
title: "LLM Fundamentals and Why Fine-Tune"
description: "What language models and LLMs are, the three common architectures, and why fine-tuning is often needed after pretraining."
---

A **language model** is a model of text. Given the words so far, it estimates what comes next. A **large language model (LLM)** is that idea scaled up — many parameters, huge training data, usually built on transformers.

## Intuition

Core job in one line: predict the next token.

Example: “The cat sat on the ___”

The model assigns probabilities to possible next words (`mat`, `sofa`, `roof`, …). That next-token engine powers autocomplete, chat, translation-style systems, and more.

:::key
Pretraining teaches general language skill. Fine-tuning shapes that skill for your task, domain, or style.
:::

## How it works

### What “large” adds

Scale (parameters + data + compute) is why modern LLMs can:

- Write fluent text and dialogue
- Pick up a new task from a few examples in the prompt (in-context learning)
- Handle more reasoning, knowledge, code, and math than small models

Well-known families include GPT, Claude, Gemini, and Llama-style open-weight models.

### Three common LLM shapes

| Type | Plain-English idea | Typical uses |
| --- | --- | --- |
| **Encoder-only** | Looks both left and right (bidirectional). Great at understanding. | Classification, NER, similarity, search embeddings (BERT family) |
| **Decoder-only** | Reads left-to-right; generates the next token. | Chat and free-form generation (GPT-style) |
| **Encoder–decoder** | Encoder reads input; decoder writes output. | Translation-style and many sequence-to-sequence tasks |

### The usual training stages

| Stage | What happens |
| --- | --- |
| **Pretraining** | Learn general language from huge unlabeled text |
| **Fine-tuning** | Adapt to a task, domain, or instruction style with more focused data |
| **Safety / alignment** | Extra training so the model follows policies and behaves more helpfully/safely |

### Why fine-tuning is often necessary

A general pretrained model is a strong starting point, but many products need:

- Domain specialists (medicine, finance, legal drafting style)
- Stable formats and habits that prompting alone does not lock in
- Behavior that should stick across many future prompts, not only one clever instruction

Fine-tuning updates the model on task-specific data so those habits become part of the weights.

## What goes wrong

- Treating a general chat model as already “done” for a specialist job.
- Fine-tuning to store facts that change every week (retrieval is usually better for that).
- Skipping the question “do we need weight updates at all?”

## One-line summary

LLMs learn next-token prediction at huge scale; fine-tuning is how you turn that general skill into stable, task-specific behavior.

## Key terms

- **Language model (LM)** — Model that predicts the next piece of text.
- **LLM** — A very large language model, usually transformer-based.
- **Pretraining** — Broad training before your task-specific work.
- **Fine-tuning** — Further training on task-specific data.
