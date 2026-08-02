---
title: "Additive PEFT: Adapters"
description: "Small trainable modules added to a frozen model — sequential vs residual adapters, and why they beat tuning only the last layers."
---

**Adapters** are small trainable blocks inserted into the network. The big pretrained model stays mostly frozen. Each task gets a tiny side path that learns how to tweak the hidden states for that task.

## Intuition

Think of the base model as a shared engine. An adapter is a small bolt-on box for one job (support tickets, summarization, QA). You keep one engine and swap boxes.

Two practical requirements for good adapters:

1. **Few new parameters** — so total size grows slowly as you add tasks.
2. **Near-identity start** — the adapter begins almost as “do nothing,” so training starts gently from the pretrained behavior.

:::key
Freeze the original weights; update only the adapter. Often you touch roughly ~1–2% of the original parameter count.
:::

## How it works

### Sequential adapters

The adapter sits **in the main path** (in the residual stream). Information flows through the adapter as part of the layer.

- Can reach similar accuracy while updating only about **1%** of original parameters.
- Trade-offs to remember: some **inference latency**, cost that scales with depth, harder to parallelize with existing compute, and multi-task serving can get expensive if you are not careful.

### Residual (parallel) adapters

The adapter sits **beside** the main layer, then **adds** its output back to the original hidden state.

Plain picture of a common bottleneck adapter:

1. Shrink the hidden vector (down-project)
2. Apply a simple non-linearity (for example ReLU)
3. Expand back up
4. Add that correction to the original hidden state

In words: `new_hidden = old_hidden + up(relu(down(old_hidden)))`

Why residual adapters are appealing for generation: they keep the original stream intact and inject task behavior more gently.

### Task structure helps

Adapters often work better when the input clearly marks the task — for example special tokens for QA segments (`document`, `question`, `answer`), or similar markers for dialogue and summarization. That structure gives the small module clearer patterns to learn.

### Other ideas in the same family

| Idea | Plain-English idea |
| --- | --- |
| **AdapterFusion** | Combine several trained adapters so one task can borrow skills from others |
| **Tiny-Attention Adapter** | Keep the add-on small, but let it use a little attention to steer the base model |
| **Multi-task / routing setups** | Share or route among adapters instead of always training one giant fully tuned model |

### Why not just tune the last few layers?

Tuning only the top layers can work, but adapters are more **modular**:

- One frozen backbone for many tasks
- Swap only the tiny task module
- Cleaner maintenance and often better parameter efficiency

## What goes wrong

- Expecting zero latency cost with sequential adapters in every layer.
- Forgetting near-identity initialization and shocking the model at step one.
- Building one adapter per task with no serving plan — modular is good only if you can load/swap them cleanly.

## One-line summary

Adapters add a small trainable module to a frozen backbone so each task gets its own cheap specialty without rewriting the whole model.

## Key terms

- **Adapter** — Small trainable module that adjusts hidden states.
- **Sequential adapter** — Adapter placed in the main computation path.
- **Residual adapter** — Adapter in parallel that adds a correction back to the stream.
- **AdapterFusion** — Combining multiple adapters for reuse.
