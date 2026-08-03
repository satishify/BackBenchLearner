---
title: "Gradient Checkpointing and Paged Optimizer"
description: "Two memory tricks that complete the QLoRA stack: recompute activations, and smooth optimizer memory spikes."
---

QLoRA is not only about 4-bit weights. It works because several memory-saving methods are stacked. Two big ones are **gradient checkpointing** and a **paged optimizer**.

## Intuition

GPU memory fails for different reasons:

- Weights too large → quantization / LoRA help
- Activations too large → checkpointing helps
- Optimizer state spikes → paging helps

:::key
Each trick attacks a different part of the memory budget. Together they make large-model training realistic.
:::

## How it works

### Gradient checkpointing

During normal backpropagation, the model stores many **activations** so it can compute gradients later. That storage can be huge.

**Gradient checkpointing** discards some activations on purpose and **recomputes** them in the backward pass.

| Benefit | Cost |
| --- | --- |
| Less activation memory | Extra compute during backprop |

A common placement intuition: put checkpoints roughly every **√n** layers (for **n** layers). Then the backward pass can restart from a nearby checkpoint instead of replaying the whole network from layer 1 every time.

### Paged optimizer

Optimizer state (the extra numbers Adam-style methods keep) can cause sudden **out-of-memory** spikes — especially with long sequences or uneven batch sizes.

A **paged optimizer** keeps that state more flexibly, often by paging data between GPU and CPU memory, so a spike is less likely to crash the run.

It does not remove memory cost entirely. It **smooths the spikes** that cause failures.

### Where these fit in QLoRA

| Piece | Memory pain it targets |
| --- | --- |
| 4-bit / NF4 weights | Base weight storage |
| LoRA adapters | Trainable parameter count |
| Double quantization | Scale / metadata overhead |
| Gradient checkpointing | Activation memory |
| Paged optimizer | Optimizer spikes |

## What goes wrong

- Enabling checkpointing and then being surprised that steps get slower (that is the trade-off).
- Ignoring optimizer spikes and only shrinking weights — then still OOMing mid-update.
- Placing no checkpoints (or too few) so recomputation becomes painfully long.

## One-line summary

Checkpointing saves activation memory by recomputing; a paged optimizer softens optimizer spikes — both help QLoRA fit on fewer GPUs.

## Key terms

- **Gradient checkpointing** — Discard and recompute activations to save memory.
- **Paged optimizer** — Optimizer strategy that reduces sudden GPU memory spikes.
- **Out-of-memory (OOM)** — Crash when the GPU budget is exceeded.
- **Activation memory** — Memory used to store intermediate layer outputs during training.
