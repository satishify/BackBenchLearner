---
title: "Catastrophic Forgetting"
description: "Why narrow fine-tuning can erase broad skills, how to measure it, and plain-English ways to reduce it."
---

**Catastrophic forgetting** happens when you fine-tune a pretrained model on a narrow task and the updates overwrite the weights that held broader skills. The new task gets better. Everyday abilities — following instructions, coding, other languages, simple reasoning — can quietly get worse.

## Intuition

Picture a generalist who studies only medical Q&A for weeks and somehow “forgets” how to write Python or speak French. The model did not delete a folder named “French.” The same shared weights store many skills. Push those weights hard toward one narrow dataset, and other skills can fade.

Why it is scary:

- **Silent failure** — Task metrics go up while general benchmarks fall, unless you measure both.
- **Worst with full fine-tuning** — Updating all weights at a high learning rate on small, focused data erases prior knowledge fastest.

:::key
Always score a fixed general “canary” suite before and after every fine-tune — not only the target task.
:::

## How it works

### Why fine-tuning erases knowledge

| Cause | Plain-English idea |
| --- | --- |
| Shared weights | Many skills live tangled in the same parameters. |
| Narrow data | Fine-tune examples cover a tiny slice of what pretraining saw. |
| Weight drift | The farther you wander from the starting model, the more old skills suffer. |
| No memory of old tasks | A normal optimizer does not “remember” earlier abilities unless you re-show old data or add a penalty for drifting. |

### Measure before vs after

Imagine a 7B model fully fine-tuned for a few epochs on legal contracts: contract accuracy jumps, while general benchmarks drop. That pattern is the warning light. Keep one fixed general eval set and compare base vs fine-tune every time.

### Mitigation strategies

| Strategy | Plain-English idea | When to use it |
| --- | --- | --- |
| **Conservative hyperparameters** | Smaller learning rate, fewer epochs, early stop on canary evals | Always — your default baseline (almost free) |
| **Rehearsal / data mixing** | Mix a little general instruction data (often about 1–10%) into training so old skills keep getting practice | When you have (or can proxy) general data |
| **Regularize toward the start** | Penalize big drift from the pretrained weights (L2-SP or EWC-style ideas) | Full fine-tune required and little general replay data |
| **Weight averaging** | After training, blend fine-tuned weights with the original model to recover generality | Model already fine-tuned and general quality dropped |
| **Freeze layers** | Lock lower layers; tune only the top part closest to the task | Tight compute or very small datasets |

#### A bit more detail (still plain)

- **Rehearsal** — Interleave domain data with diverse general examples (instructions, code, math, other languages — whatever you fear losing). Target learning may slow a little; that is the trade.
- **Regularize toward pretrained weights** — Gently pull weights back toward the starting point. Too strong a pull underfits the new task; too weak and you still forget. Tune against canary scores.
- **Conservative knobs** — Full fine-tune learning rates around 1×10⁻⁵–2×10⁻⁵; often 1–2 epochs is enough. Train less than “feels safe” if canaries dip.
- **Weight averaging** — Mix “base” and “fine-tuned” with a blend factor; sweep a few blends and pick the best trade-off between task score and general score.
- **Freeze layers** — Keep embeddings and early blocks fixed; open only the top blocks. Tuning roughly the top quarter to half of blocks often captures most task gains with less forgetting.

### Choosing a stack

Start with conservative hyperparameters. Add rehearsal if you can. Use freezing when data or compute is tight. Reach for regularization or averaging when you must fully fine-tune and general skills are slipping.

## What goes wrong

- Celebrating a rising task metric while ignoring fallen general scores.
- Five epochs on a tiny niche dataset at a hot learning rate, then wondering why the assistant “got dumb.”
- No canary set — forgetting stays invisible until users complain.

## One-line summary

Narrow fine-tuning can overwrite broad skills; measure general canaries, then reduce forgetting with smaller updates, rehearsal, freezing, regularization, or blending back toward the base model.

## Key terms

- **Catastrophic forgetting** — Losing older abilities after narrow new training.
- **Canary / anchor eval** — Fixed general prompts you always re-score.
- **Rehearsal** — Mixing general data into fine-tuning.
- **Weight averaging** — Blending fine-tuned and base weights after training.
