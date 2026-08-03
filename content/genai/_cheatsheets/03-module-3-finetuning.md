---
title: "Module 3 - Fine-tuning revision"
slug: module-3-finetuning
module: "Module 3"
minutes: 25
description: "When to fine-tune, SFT vs preference, PEFT/LoRA recipes — interview-ready."
---

Chapters **3.1–3.4**. Focus on when and why—not memorizing every hyperparameter.

## 3.1 Fine-tuning fundamentals

### Decision framework
- **Prompting** changes the instruction. **RAG** changes the context. **Fine-tuning** changes the model.
- Changing facts → RAG. Stable style / format / domain habit → fine-tuning. Clear prompt already works → stay with prompting.

### Approaches
- **Unsupervised / continued** — Domain text soak, few or no instruction labels.
- **SFT** — Input → desired output pairs. **IFT** — Chat-style instruction → response.
- **Safety / alignment** — Extra shaping for preferred, safer behavior.
- **Full fine-tuning** vs **PEFT** — Update almost everything, or only a small part (adapters, LoRA, soft prompts).

### Training patterns
- **Multi-task** — Related tasks share one backbone and regularize each other.
- **Curriculum** — Easy examples first, harder later.
- **Efficient full FT** — Freeze most layers; gradual unfreezing / block-wise opening. Small data + close base model → freeze more.

## 3.2 Data prep & training mechanisms

- **How LLMs learn** — Tokenize → mini-batch → next-token guess → softmax → cross-entropy → small update. Stop when validation flattens.
- **Instability** — Spikes / exploding gradients / divergence. Clip gradients, warmup, decay learning rate, prefer bf16, save checkpoints.
- **Catastrophic forgetting** — Narrow fine-tune improves the task but hurts general skills. Measure canaries; use smaller updates, rehearsal, freeze layers, regularize, or weight averaging.
- **Learning-rate schedule** — Warmup + peak + decay (cosine is a common default). Peak learning rate is the big knob.
- **Batch size** — Care about effective batch = micro-batch × accumulation × GPUs. Too small → jitter; too large → too few updates.
- **Data prep** — Fix noise, bad labels, duplicates, messy formatting, stale facts; split honestly; watch domain shift.
- **Loss curves** — Train and val falling together with a small gap = healthy. Stop at the validation minimum.

## 3.3 PEFT: additive and soft prompting

- **Why PEFT** — Full fine-tuning is costly and can forget old skills. Train a tiny fraction of parameters (or a tiny prompt) instead. Families: selective, additive, re-parameterization (LoRA/QLoRA), soft prompting.
- **Adapters (additive)** — Small modules on a frozen backbone. Sequential (in the path) vs residual/parallel (add a correction). Freeze original weights; update only adapters. Modular multi-task swap.
- **Soft prompting** — Adapt in token space with learnable virtual tokens. Discrete prompt = real words; continuous prompt = trainable embeddings.
- **Prefix vs prompt tuning** — Both soft-prompt methods; prefix tuning emphasizes learned prefix context; prompt tuning learns prompt embeddings with a frozen model.
- **Smarter prompts** — SMoP (sparse mixture), APT (prefix length by layer), IDPG (prompt from the input), SPT (prompts only where needed).
- **Choose by fit** — Adapters for modular architecture-side adaptation; soft prompts for smallest frozen-backbone footprint; smarter variants when one blunt prompt is not enough.

## 3.4 Re-parameterization PEFT (LoRA & QLoRA)

- **Why it exists** — Full fine-tuning stores weights + gradients + optimizer for every parameter. Too heavy for many clients or one tight GPU.
- **Intrinsic dimension** — Useful adaptations often live in a smaller space than the full weight matrix.
- **LoRA** — Freeze W; learn low-rank update `(α / r)BA`. For square d×d, about **2dr** trainable params (example: d=4096, r=8 → 65,536).
- **Multi-tenant pattern** — One shared base + small adapter per client; hot-swap adapters at serve time.
- **QLoRA** — 4-bit frozen backbone (often **NF4**) + LoRA. Add **double quantization** for scale metadata.
- **Memory stack** — Checkpointing saves activations (extra compute); paged optimizer smooths optimizer spikes. Together with 4-bit + LoRA, large models become more realistic on fewer GPUs.

## Decision cheat

| Plain-English idea | When to use it |
| --- | --- |
| **RAG** | Facts change often or must come from private docs. |
| **SFT / LoRA** | Stable tone, schema, or domain phrasing. |
| **Preference data** | Safety, tone alignment, ranked choices. |
| **QLoRA** | Tight GPU budget; validate quality after. |
| **Per-tenant adapters** | Many customers, one shared base model. |

## 25-minute drill
1. Argue for RAG vs LoRA on a "company FAQ bot."
2. Name two evals that catch forgetting.
3. Explain LoRA rank in one sentence (how big the two small matrices are).
4. When would you merge vs keep adapters separate?
