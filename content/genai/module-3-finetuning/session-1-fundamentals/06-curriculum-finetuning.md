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
