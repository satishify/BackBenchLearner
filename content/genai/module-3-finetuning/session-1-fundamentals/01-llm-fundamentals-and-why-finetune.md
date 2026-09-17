---
title: "LLM Fundamentals and Why Fine-Tune"
description: "What language models and LLMs are, the three common architectures, and why fine-tuning is often needed after pretraining."
---

A **language model** is a model of text. Given the words so far, it estimates what comes next. A **large language model (LLM)** is that idea scaled up — many parameters, huge training data, usually built on transformers.

## Intuition

Core job in one line: predict the next token.

Example: “The cat sat on the ___”

The model assigns probabilities to possible next words:

| Next word | Probability |
| --- | --- |
| `mat` | 0.41 |
| `sofa` | 0.22 |
| `floor` | 0.18 |
| `roof` | 0.06 |
| `refrigerator` | 0.001 |

It then picks one, adds it to the sentence, and repeats the whole process for the following word. Everything an LLM does — chat, summarising, writing code — is this one step run over and over.

:::note Analogy
Think of someone who has read a huge share of the internet and has become extremely good at finishing your sentences. Ask a question and they keep completing text until the answer is written. They are not looking anything up; they are predicting what usually comes next.
:::

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

An easy way to remember the difference:

- **Encoder-only** is a *reader*. It sees the whole sentence at once, like you reading a paragraph before answering a comprehension question. Perfect for “is this review positive or negative?”
- **Decoder-only** is a *writer*. It can only look backwards at what it has already written, like a person speaking a sentence out loud one word at a time. Perfect for chat.
- **Encoder–decoder** is a *translator*. One half reads the whole input, the other half writes a fresh output. Perfect for “turn this English paragraph into French.”

Almost every chat assistant you use today is decoder-only.

### The usual training stages

| Stage | What happens |
| --- | --- |
| **Pretraining** | Learn general language from huge unlabeled text |
| **Fine-tuning** | Adapt to a task, domain, or instruction style with more focused data |
| **Safety / alignment** | Extra training so the model follows policies and behaves more helpfully/safely |

:::note Analogy
This is the same path a new doctor takes. School and college give broad knowledge (**pretraining**). Medical residency turns that into a specific profession (**fine-tuning**). Hospital rules and ethics training decide how they are allowed to behave with patients (**alignment**).

You cannot skip to residency without school, and school alone does not make someone ready to treat patients.
:::

### Why fine-tuning is often necessary

A general pretrained model is a strong starting point, but many products need:

- Domain specialists (medicine, finance, legal drafting style)
- Stable formats and habits that prompting alone does not lock in
- Behavior that should stick across many future prompts, not only one clever instruction

Fine-tuning updates the model on task-specific data so those habits become part of the weights.

### A concrete example

An insurance company wants every claim note summarised in exactly this shape:

```text
CLAIM: <id>
LOSS TYPE: <fire | flood | theft | other>
AMOUNT: <number or unknown>
NEXT ACTION: <one sentence>
```

With prompting, the model gets it right most of the time, but on maybe 1 note in 12 it adds a friendly sentence at the top, or writes “Fire damage” instead of `fire`. At 4,000 notes a day, that is hundreds of broken records.

After fine-tuning on 2,000 correctly formatted examples, the format becomes a habit instead of a request. The prompt can then be short, because the model has already learned what the answer should look like.

That is the real test for fine-tuning: **do you need this behaviour every single time, forever?**

## What goes wrong

- Treating a general chat model as already “done” for a specialist job.
- Fine-tuning to store facts that change every week (retrieval is usually better for that). Training last month’s price list into the weights means retraining when the prices move.
- Skipping the question “do we need weight updates at all?” Many teams spend a week fine-tuning a problem that a five-line prompt fix would have solved.
- Assuming the model “understands” instead of predicting. It has no idea whether the sentence it just produced is true — which is exactly why grounding and evaluation matter.

## One-line summary

LLMs learn next-token prediction at huge scale; fine-tuning is how you turn that general skill into stable, task-specific behavior.

## Key terms

- **Language model (LM)** — Model that predicts the next piece of text.
- **LLM** — A very large language model, usually transformer-based.
- **Pretraining** — Broad training before your task-specific work.
- **Fine-tuning** — Further training on task-specific data.
