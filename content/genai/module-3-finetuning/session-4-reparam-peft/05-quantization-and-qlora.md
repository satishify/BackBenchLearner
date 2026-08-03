---
title: "Quantization and QLoRA"
description: "Store the frozen base model in fewer bits — NF4, chunking, double quantization, and what QLoRA adds beyond LoRA."
---

**Quantization** stores numbers with fewer bits. **QLoRA** combines a 4-bit frozen backbone with LoRA adapters so you keep most of LoRA’s adaptation power while the base model takes much less memory.

## Intuition

Plain LoRA still needs the base weights in higher precision in memory. Quantized PEFT asks: can the *frozen* backbone live in 4-bit form while the small adapter still learns the task?

That is QLoRA’s core move:

- Base model → compressed (often **NF4** 4-bit)
- Task change → still learned with **LoRA**

:::key
QLoRA = 4-bit frozen backbone + LoRA adapters. Quantization shrinks storage; LoRA still does the learning.
:::

## How it works

### Why quantization helps

Quantization converts a fat number type (for example 32-bit float) into a thinner one (for example 8-bit or 4-bit). Goals:

- Less memory
- Better chance of fitting large models on fewer GPUs

### Uniform quantization (simple baseline)

Uniform quantization uses evenly spaced bins and a scale constant. It is easy to understand, but not ideal when weights are roughly **bell-shaped** (normal-like): most values sit near zero, and evenly spaced bins waste precision in the wrong places.

### Chunking-based quantization

Instead of quantizing a whole giant tensor with one global scale, quantize **smaller blocks** independently. That reduces damage from outliers (one extreme value no longer ruins the scale for everything).

### NF4 in simple language

**NF4** means **4-bit NormalFloat**. It is designed for weights that are roughly normally distributed:

- Finer precision near zero (where most weights live)
- Coarser precision in the tails

Typical workflow:

1. Fetch the NF4 levels
2. Take the weight tensor
3. Normalize it (often by absmax)
4. Map normalized values to 4-bit codes
5. Pack two 4-bit codes into one byte
6. Later dequantize by reversing the scale when needed for compute

### Double quantization

Block-wise quantization needs **scale constants** for each block. Those constants also take space. **Double quantization** compresses those constants too — quantize the quantization metadata.

This matters more when blocks are small, because more blocks mean more scale metadata.

### QLoRA ingredients (first look)

| Ingredient | What it saves | Why it helps |
| --- | --- | --- |
| **NF4** | Base weight storage | Cuts model footprint sharply |
| **Double quantization** | Quantization constants | Reduces metadata overhead |
| **Gradient checkpointing** | Activation memory | Trade a bit of compute for memory (next chapter) |
| **Paged optimizer** | Optimizer memory spikes | Avoids sudden out-of-memory crashes (next chapter) |

## What goes wrong

- Treating quantization as “free accuracy” — it saves memory; quality still needs checking.
- Using naive uniform quantization on heavy-tailed weights and blaming the model.
- Forgetting that QLoRA still needs a sensible LoRA rank and clean data.

## One-line summary

Quantization compresses frozen weights; QLoRA pairs 4-bit storage with LoRA so large models become trainable under tight memory.

## Key terms

- **Quantization** — Storing values with fewer bits.
- **QLoRA** — LoRA plus a 4-bit quantized frozen backbone.
- **NF4** — 4-bit NormalFloat, tuned for roughly normal weight distributions.
- **Double quantization** — Compressing the quantization scale metadata itself.
