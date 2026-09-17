---
title: "Supervised and Instruction Fine-Tuning"
description: "SFT versus instruction fine-tuning (IFT): labeled pairs, chat-style data format, and what changes after instruction tuning."
---

**Supervised fine-tuning (SFT)** means: show the model many examples of “given this input, produce that output,” and update weights so it copies the desired pattern. **Instruction fine-tuning (IFT)** is the chat-friendly version of that idea — teach the model to follow natural-language instructions.

## Intuition

Simple analogy: SFT is like tutoring with answer keys. Each row says what good looks like. After enough examples, the model internalizes the habit — not only for that exact wording, but for similar asks.

:::note Analogy
A base model that has only been pretrained behaves like a brilliant person who never learned conversation etiquette. Ask “what is the capital of France?” and instead of answering, it may continue your text:

> What is the capital of France? What is the capital of Spain? What is the capital of Italy?

It is not being difficult — it is doing its job, predicting plausible continuations. Instruction fine-tuning is what teaches it that a question is meant to be **answered**, not extended.
:::

:::key
SFT teaches behavior from labeled pairs. Instruction fine-tuning specializes that idea for “user asks → assistant replies” chat format.
:::

## How it works

### Supervised fine-tuning (SFT)

- You need clear inputs and target outputs.
- Format must be consistent (same schema, same role tags if you use chat).
- Loss usually focuses on predicting the **target** tokens (the answer), not forcing the model to memorize the user’s question as if it wrote it.

### Instruction fine-tuning (IFT)

Instruction data often looks like:

- A user instruction (and maybe context)
- An ideal assistant response

Training teaches the model: when someone asks in natural language, answer helpfully in the expected style.

One training row, written out in full:

```json
{
  "messages": [
    {"role": "system",    "content": "You are a support agent for a bank."},
    {"role": "user",      "content": "I lost my debit card. What should I do?"},
    {"role": "assistant", "content": "I'm sorry to hear that. Please block the card immediately from the app under Cards > Block, then request a replacement. Your new card arrives in 3-5 working days."}
  ]
}
```

Two details matter here:

- The **same** role structure must be used during training and when the model is later served. If training used `system/user/assistant` and production sends a plain string, the model sees an unfamiliar shape and quality drops.
- The loss is calculated on the **assistant** text only. The model is being taught to produce that reply, not to invent the customer's question.

### SFT vs IFT (practical view)

| | **SFT (broad)** | **IFT (instruction-focused)** |
| --- | --- | --- |
| Data | Any labeled input→output pairs | Instruction / chat-style pairs |
| Goal | Task skill (classify, extract, draft…) | Follow instructions in assistant form |
| Overlap | IFT is a common modern form of SFT for chat models | Same training idea; different data shape |

Many people say “SFT” when they mean instruction tuning a chat model. That is fine — just know which data format you are using.

### Effect of instruction fine-tuning

After IFT, models usually become better at:

- Following directions
- Producing useful assistant-style answers
- Sticking to requested formats

They still need good data. Garbage instructions teach garbage habits.

### How much data do you need?

There is no magic number, but a useful starting frame:

| Goal | Rough scale of examples |
| --- | --- |
| Lock in one output format | Hundreds |
| Teach a task with real variety | A few thousand |
| Broad instruction-following from a base model | Tens of thousands and up |

Quality beats quantity almost every time. 500 carefully checked examples usually beat 5,000 scraped ones, because the model copies whatever it is shown — including the mistakes.

## What goes wrong

- Mixing random formats in one JSONL file so the model never sees a stable pattern.
- Training on answers only, with unclear instructions.
- Expecting IFT to install fresh facts that belong in RAG.
- Letting a few sloppy examples slip in. If 5% of your answers are rude or truncated, the model learns that rudeness and truncation are sometimes correct.
- Using one chat template during training and a different one in production, then wondering why the model got worse after deployment.

## One-line summary

Supervised fine-tuning teaches from labeled examples; instruction fine-tuning is that idea applied to chat-style “follow my request” data.

## Key terms

- **SFT** — Supervised fine-tuning on labeled pairs.
- **Instruction fine-tuning (IFT)** — SFT focused on instruction→response behavior.
- **Chat template** — The role/format wrapper used at train and serve time.
