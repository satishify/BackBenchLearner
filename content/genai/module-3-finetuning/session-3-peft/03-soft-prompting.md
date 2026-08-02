---
title: "Soft Prompting"
description: "Adapt in token space with learnable virtual prompts instead of adding new layers — discrete vs continuous prompts in plain English."
---

**Soft prompting** moves adaptation out of the architecture and into **token space**. Instead of only writing a hand-crafted text prompt, we learn virtual prompt tokens whose embeddings are trained — while the backbone model can stay frozen.

## Intuition

| Approach | Where adaptation happens | Plain-English idea |
| --- | --- | --- |
| **Adapters** | Architecture space | Add small modules into the network |
| **Soft prompts** | Token space | Add and learn task-specific virtual tokens / context |

The big idea: **good context can steer the language model without changing its weights.**

:::key
Discrete prompt = real words you type. Continuous (soft) prompt = trainable “virtual tokens” the model learns.
:::

## How it works

### Discrete vs continuous prompts

| Kind | Meaning | Catch |
| --- | --- | --- |
| **Discrete prompt** | A normal string of real tokens written by a person | Small wording changes can swing quality a lot |
| **Continuous / soft prompt** | A sequence of trainable virtual token embeddings | Not real vocabulary words — learned vectors that act like extra context |

Why continuous prefix-style tuning exists: manual prompts are brittle. Learning a soft prompt can be more stable and compact than rewriting English by hand.

### What you gain

- Much smaller trainable footprint than full fine-tuning
- Easy to swap tasks if the base model stays frozen
- Adaptation without redesigning the network

### What to watch

Soft prompts can be sensitive to:

- How long the prompt is
- Which layers see the prompt
- Whether one fixed prompt is too blunt for every input

Longer soft prompts also mean more tokens to process — so cost can grow with prompt length even when the backbone is frozen.

## What goes wrong

- Treating soft prompts as “free” — long prompts still cost compute at train and serve time.
- Using one generic soft prompt for every weird input case.
- Confusing soft prompts with adapters (different place of adaptation).

## One-line summary

Soft prompting steers a frozen model by learning virtual prompt tokens in embedding space, instead of rewriting the network.

## Key terms

- **Soft prompt** — Learnable virtual tokens used as trainable context.
- **Virtual token** — A learned embedding that is not a fixed vocabulary word.
- **Discrete prompt** — Ordinary human-written text tokens.
