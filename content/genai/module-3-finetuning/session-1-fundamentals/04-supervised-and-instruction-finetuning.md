---
title: "Supervised and Instruction Fine-Tuning"
description: "SFT versus instruction fine-tuning (IFT): labeled pairs, chat-style data format, and what changes after instruction tuning."
---

**Supervised fine-tuning (SFT)** means: show the model many examples of “given this input, produce that output,” and update weights so it copies the desired pattern. **Instruction fine-tuning (IFT)** is the chat-friendly version of that idea — teach the model to follow natural-language instructions.

## Intuition

Simple analogy: SFT is like tutoring with answer keys. Each row says what good looks like. After enough examples, the model internalizes the habit — not only for that exact wording, but for similar asks.

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

## What goes wrong

- Mixing random formats in one JSONL file so the model never sees a stable pattern.
- Training on answers only, with unclear instructions.
- Expecting IFT to install fresh facts that belong in RAG.

## One-line summary

Supervised fine-tuning teaches from labeled examples; instruction fine-tuning is that idea applied to chat-style “follow my request” data.

## Key terms

- **SFT** — Supervised fine-tuning on labeled pairs.
- **Instruction fine-tuning (IFT)** — SFT focused on instruction→response behavior.
- **Chat template** — The role/format wrapper used at train and serve time.
