---
title: "Fine-Tuning Approaches"
description: "Unsupervised, supervised, and safety/alignment fine-tuning — plus full fine-tuning versus parameter-efficient options in plain English."
---

“Fine-tuning” is not one single recipe. This chapter maps the main **approaches** so you can name what you are doing and why.

## Intuition

| Approach | Plain-English idea |
| --- | --- |
| **Unsupervised fine-tuning** | Continue training on domain text without instruction labels (domain language soak) |
| **Supervised fine-tuning (SFT)** | Train on input → desired output pairs |
| **Safety / alignment fine-tuning** | Extra training so the model follows policies and preferred behavior |
| **Full fine-tuning** | Almost every weight can update |
| **PEFT** | Train only a small part (adapters, LoRA, soft prompts, …) |

:::key
Full fine-tuning is the most powerful and most expensive. PEFT is often the middle ground between prompting and updating everything.
:::

## How it works

### Unsupervised (continued) fine-tuning

Feed lots of domain text (legal corpus, codebase, medical notes) so the model absorbs domain language. There may be no “instruction → answer” labels. Useful for domain familiarity; not the same as teaching a chat format.

### Supervised fine-tuning

You provide clear examples: given this input, produce that output. This is the workhorse for task adaptation and instruction-style models (next chapter goes deeper).

### Safety / alignment fine-tuning

After (or alongside) capability training, you further shape the model so it is more helpful, honest, and policy-compliant. Methods vary (preference data, RL-style loops, and related recipes). For this fundamentals session, remember the **goal**: safer, more aligned behavior — not every algorithm detail.

### Full fine-tuning

Every (or almost every) weight can move.

- **Good for:** strong domain shift, high-value tasks, consistent formats when you have enough data
- **Risk:** cost, overfitting on small data, forgetting general skills
- **Example:** a legal drafting assistant that must internalize a house style

### PEFT (parameter-efficient fine-tuning)

Update only a small number of parameters (or add tiny modules / soft prompts).

PEFT families you will meet later in Module 3:

| Family | Idea in one line |
| --- | --- |
| **Additive** | Add small modules (adapters) |
| **Selective** | Train only some existing weights |
| **Re-parameterization** | Cheap update forms (LoRA / QLoRA) |
| **Soft prompting** | Learn virtual prompt tokens |

Use PEFT when you want most of the benefit of adaptation without paying full fine-tune cost.

## What goes wrong

- Calling every training run “SFT” when it was unlabeled domain text.
- Jumping to full fine-tuning when PEFT or RAG would have been enough.
- Ignoring alignment/safety until after a risky model is already in production.

## One-line summary

Fine-tuning comes in flavors — unsupervised, supervised, alignment, full, or PEFT — pick by data type, cost, and how much of the model must change.

## Key terms

- **SFT** — Supervised fine-tuning on labeled pairs.
- **Full fine-tuning** — Updating nearly all weights.
- **PEFT** — Training only a small part of the model.
- **Alignment** — Shaping the model toward preferred / safer behavior.
