---
title: "LoRA: Low-Rank Adaptation"
description: "Freeze the base model and learn two small matrices that approximate the weight update."
---

**LoRA** (Low-Rank Adaptation) freezes the base model and learns two small matrices that together approximate the weight update. Instead of one huge dense update, you learn a **down-projection** and an **up-projection**.

## Intuition

Think of the frozen weight matrix **W** as the pretrained skill. Fine-tuning wants a correction **ΔW**. LoRA says: do not learn every cell of ΔW. Learn a thin path through a small rank **r**.

A common form:

`W' = W + (α / r) · B · A`

- **W** stays frozen (keeps pretrained knowledge)
- **B · A** is the low-rank correction
- **α / r** scales the update so its size stays stable when you change rank

:::key
LoRA = frozen backbone + tiny low-rank update. Higher rank = more flexibility and more parameters.
:::

## How it works

### Symbols in plain English

| Symbol | Meaning |
| --- | --- |
| **W** | Frozen base weight matrix |
| **ΔW** | Weight update that should adapt the model |
| **A** | Down-projection matrix with rank **r** |
| **B** | Up-projection matrix that maps back to model size |
| **r** | Low rank used by the adapter |
| **α** | Scaling factor that controls LoRA strength |

### Size of a LoRA adapter

For a square **d × d** weight:

| Scenario | Trainable parameters |
| --- | --- |
| Full square weight | **d²** |
| LoRA on that matrix | **2 · d · r** |
| Example: d = 4096, r = 8 | **2 × 4096 × 8 = 65,536** |

That example is roughly **256× smaller** than training the full 4096×4096 matrix. That is why LoRA is attractive for task-specific or client-specific adapters.

### Rank and scaling choices

- **Higher rank** → more flexibility, more trainable parameters
- **Lower rank** → cheaper, but may underfit a hard adaptation
- **α / r scaling** → keeps update magnitude better behaved when rank changes

### LoRA as a generalization of full fine-tuning

If the rank is large enough, a low-rank update can approximate a dense update. In that sense, LoRA is a flexible middle ground: small rank for cheap PEFT, larger rank when you need more capacity — without opening every weight by default.

## What goes wrong

- Picking a tiny rank for a big domain shift, then blaming LoRA.
- Forgetting the scaling term and wondering why rank changes feel unstable.
- Training LoRA but also unfreezing the whole backbone “just in case,” which defeats the memory win.

## One-line summary

LoRA learns a small low-rank update while keeping the backbone frozen — most of the adaptation benefit, far fewer trainable parameters.

## Key terms

- **LoRA** — Low-Rank Adaptation with frozen base weights.
- **Rank (r)** — Size of the low-rank bottleneck.
- **α (alpha)** — Strength / scaling of the LoRA update.
- **Adapter size** — Roughly **2dr** trainable parameters for a square **d × d** matrix.
