---
title: "Re-Parameterization and Intrinsic Dimension"
description: "Why useful weight updates often live in a smaller space — the idea that opens the door to LoRA."
---

**Re-parameterization** means: do not ask the model to learn one giant dense update directly. Instead, express that update through a smaller set of parameters.

## Intuition

Imagine you need to nudge a huge weight matrix. You could learn every entry. Or you could learn a compressed version that still captures the useful change.

Research and practice both point to this: many learning problems can be solved in a much **lower-dimensional subspace** than the raw parameter count suggests. That smaller effective size is called the **intrinsic dimension**.

:::key
If the useful weight change has low intrinsic rank, a low-rank factorization can capture most of the adaptation.
:::

## How it works

### Two linked ideas

1. **Over-parameterized models often live on a low intrinsic dimension** — lots of weights, but the useful learning direction is smaller.
2. **The change during adaptation also tends to have low intrinsic rank** — so the *update* can be compressed too.

That is the conceptual doorway into LoRA:

- Full update: learn a big dense ΔW
- Re-parameterized update: learn a small factored form that approximates ΔW

### Why this helps PEFT design

| Idea | Plain-English meaning |
| --- | --- |
| **Intrinsic dimension** | The smaller space that still solves the learning problem well |
| **Low-rank update** | Approximate the weight change with a thin factorization |
| **Frozen backbone** | Keep pretrained knowledge; only learn the small correction |

You are not claiming the whole model is tiny. You are claiming the *task-specific change* often is.

## What goes wrong

- Assuming every task has a tiny intrinsic dimension — some hard shifts need a richer update (higher rank or another method).
- Confusing “fewer parameters” with “no adaptation power.” Compression helps only if the useful signal fits the smaller space.
- Jumping to LoRA settings without understanding why low-rank updates can work at all.

## One-line summary

Re-parameterization PEFT works because useful adaptations often live in a smaller space than the full weight matrix — so we learn that smaller update on purpose.

## Key terms

- **Re-parameterization** — Representing the adaptation with fewer or smarter parameters.
- **Intrinsic dimension** — The smaller effective dimension that can still solve the learning problem.
- **Low-rank update** — A compressed weight change used by methods like LoRA.
