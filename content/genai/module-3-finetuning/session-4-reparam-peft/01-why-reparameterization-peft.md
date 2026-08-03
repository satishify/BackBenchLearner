---
title: "Why Re-Parameterization PEFT Exists"
description: "Why full fine-tuning becomes impractical at scale, and how re-parameterization PEFT cuts trainable cost."
---

Full fine-tuning updates almost every weight in a giant model. The GPU must hold the weights, the gradients, and the optimizer state at once. For very large models — or for many client-specific copies — that cost grows too fast.

## Intuition

A simple picture:

- One 13B model in 16-bit form can take about **26 GB** just for weights.
- If you need **200** client-specific full copies, storage alone can explode into **terabytes**.
- Training and serving those full copies means huge GPU bills.

**PEFT** (parameter-efficient fine-tuning) keeps the base model frozen and changes only a small part. **Re-parameterization PEFT** goes one step further: it rewrites the *update itself* into a cheaper form (LoRA and QLoRA are the main examples in this lesson).

:::key
Full fine-tuning touches almost everything. Re-parameterization PEFT learns a small update instead of a huge dense one.
:::

## How it works

### What each method changes

| Method | What changes | Typical footprint |
| --- | --- | --- |
| **Full fine-tuning** | Almost all weights | Huge weights + gradients + optimizer state |
| **LoRA** | Small low-rank adapters only | A few million trainable parameters instead of billions |
| **QLoRA** | 4-bit frozen base + LoRA adapters | Much smaller training memory; often one GPU |

### Why this lesson matters

Lesson 3.3 covered adapters and soft prompts. This lesson focuses on the **re-parameterization** branch of PEFT:

1. Can we fine-tune large models with limited resources?
2. Why does a low-rank update often work (intrinsic dimension → LoRA)?
3. How does storing weights in fewer bits help (QLoRA)?
4. Can quantization + checkpointing + paged optimizers fit a big model on one GPU?

## What goes wrong

- Treating full fine-tuning as the default when GPU memory is already tight.
- Making one full model copy per customer with no plan for storage or serving.
- Calling a method “PEFT” when you are still updating nearly every weight.

## One-line summary

Re-parameterization PEFT exists because full fine-tuning is too heavy for many real systems — so we learn a smaller update instead of rewriting the whole model.

## Key terms

- **Full fine-tuning** — Updating nearly all model weights.
- **PEFT** — Training only a small part of the system.
- **Re-parameterization** — Expressing the weight update with fewer, smarter parameters.
- **LoRA / QLoRA** — The main re-parameterization methods in this lesson.
