---
title: "Choosing Batch Size"
description: "Micro-batch, gradient accumulation, effective batch size, and how batch size trades off with learning rate and stability."
---

**Batch size** is not only “how much fits in GPU memory.” It changes how noisy each update is, how fast you finish an epoch, and how the model generalizes. What usually matters is the **effective batch size** — how many examples influence one optimizer step.

## Intuition

| Term | Plain-English idea |
| --- | --- |
| **Micro-batch** | Examples processed in one forward/backward pass on one GPU |
| **Gradient accumulation** | How many micro-batches you combine before one weight update |
| **Effective batch** | micro-batch × accumulation × number of GPUs |

Example: micro-batch 2, accumulate 16, 4 GPUs → effective batch 128. You still fit only 2 examples at a time on each GPU, but each update “sees” 128 examples.

:::key
If the batch is too small, the loss jitters. If it is too large, you may get too few updates and underfit. When you raise batch size a lot, you often need to rethink the learning rate too.
:::

## How it works

### How people choose a batch size

1. Start from what memory allows for the micro-batch.
2. Use gradient accumulation to reach a sensible effective batch if one micro-batch is tiny.
3. Watch the loss: noisy and never settling usually means “effective batch too small or learning rate too high.”
4. If training is oddly flat despite a healthy learning rate, you may be taking too few optimizer steps (effective batch too large for the data size).

### Trade-offs

| Direction | What tends to happen |
| --- | --- |
| Smaller effective batch | Noisier gradients, more jitter, more steps per epoch |
| Larger effective batch | Smoother gradients, fewer updates, risk of under-training if the dataset is small |
| Raise batch a lot | Often need a carefully retuned learning rate (not a random guess) |

### Let the loss curves judge

- Heavy step-to-step jitter that never settles → try a larger effective batch **or** a lower learning rate.
- Smooth curves that barely move → you may need more steps (smaller effective batch, more epochs, or a slightly higher learning rate — change one thing at a time).

### Practical recommendation

- Pick the largest micro-batch that fits safely.
- Accumulate to an effective batch that keeps the loss reasonably smooth.
- Keep the learning-rate schedule (warmup + decay) in place.
- Re-check validation — batch size is a tool for stability and speed, not a trophy number.

## What goes wrong

- Confusing “batch size on the GPU” with “examples per update.”
- Huge accumulation on a tiny dataset so the model barely gets any updates.
- Changing batch size and learning rate at the same time, then not knowing which change helped.

## One-line summary

Care about effective batch size (micro-batch × accumulation × GPUs): large enough for stable updates, small enough that you still take enough learning steps.

## Key terms

- **Micro-batch** — Per-GPU chunk in one forward/backward pass.
- **Gradient accumulation** — Combining several micro-batches before one update.
- **Effective batch size** — True number of examples per optimizer step.
