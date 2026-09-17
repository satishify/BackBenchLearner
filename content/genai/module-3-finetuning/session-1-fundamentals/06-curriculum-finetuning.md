---
title: "Curriculum Fine-Tuning"
description: "Train from easier examples to harder ones in a planned order — like teaching a person, not shuffling chaos from day one."
---

**Curriculum learning** means you do not show the model everything at random from the start. You begin with easier examples, then move to harder ones once the basics are stable.

## Intuition

Same idea as teaching a person: confidence and pattern recognition improve when difficulty rises gradually.

Example: sentiment classification —

1. First: obvious positive and negative reviews
2. Later: sarcasm, mixed sentiment, very short texts

:::note Analogy
Nobody teaches a child to swim by dropping them in the deep end on day one. They start in shallow water, then a float, then the shallow lane, and only later the deep end. Each stage is only safe because the previous one worked.

Training a model on your hardest adversarial examples from step one is the deep end. It may thrash around, learn something unstable, and never build the basics it needed first.
:::

The difference in practice, on review data:

| Stage | Example | Why it belongs here |
| --- | --- | --- |
| Easy | “Absolutely loved it. Best purchase this year.” | Clear words, clear label |
| Medium | “Good product, slow delivery.” | Two sentiments in one sentence |
| Hard | “Well, that was *exactly* what I needed. 🙄” | Sarcasm — the words say positive, the meaning is negative |

If the model has not yet learned that “loved it” is positive, the sarcastic example teaches it nothing useful. It just adds noise.

:::key
Curriculum = planned easy → hard order, not “whatever the dataloader shuffles.”
:::

## How it works

### Why curriculum can help LLMs

- Early training is fragile; easy wins stabilize the run
- Hard edge cases make more sense after the model has the basics
- Can reduce wasted steps on examples the model is not ready for

### Designing a curriculum: three key decisions

| Decision | Plain-English question |
| --- | --- |
| **Difficulty signal** | What makes an example “easy” or “hard” for this task? |
| **Pacing** | How fast do you introduce harder buckets? |
| **Mixing** | Do you drop easy data later, or keep a little rehearsal of easy cases? |

### Scheduling strategies (simple view)

- **Stages** — Train on easy bucket, then medium, then hard
- **Blend** — Slowly raise the fraction of hard examples over time
- **Competence-based** — Move on when validation on the current bucket looks healthy

There is no single magic schedule. Start simple: two or three difficulty buckets and a clear promotion rule.

A blended schedule is often the safest, because the easy examples never fully disappear:

| Training phase | Easy | Medium | Hard |
| --- | --- | --- | --- |
| Phase 1 | 80% | 20% | 0% |
| Phase 2 | 40% | 40% | 20% |
| Phase 3 | 20% | 40% | 40% |

That small share of easy data in the last phase is **rehearsal** — it keeps the basics fresh while the hard cases are being learned, much like a musician still playing scales while preparing a difficult piece.

### How do you decide what is "hard"?

You rarely have a difficulty label, so teams use a stand-in signal:

- **Length** — longer documents are usually harder
- **Model loss** — examples the current model gets most wrong
- **Human disagreement** — if two annotators disagreed, it is genuinely hard
- **Rule of thumb for the task** — sarcasm, negation, mixed sentiment, rare formats

The signal does not need to be perfect. Even a rough easy/hard split is better than pure shuffle when training is unstable.

## What goes wrong

- Calling random shuffle a “curriculum.”
- Jumping to the hardest adversarial set on day one with tiny data.
- Never revisiting easy cases, then forgetting the basics (a little rehearsal helps).

## One-line summary

Curriculum fine-tuning raises difficulty on purpose — easy first, harder later — so the model builds skill in a stable order.

## Key terms

- **Curriculum learning** — Training from easy examples to harder ones in a planned sequence.
- **Difficulty bucket** — A group of examples labeled easy / medium / hard.
- **Pacing** — How quickly harder data enters training.
