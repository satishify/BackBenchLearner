---
title: "Module 3 - Fine-tuning revision"
slug: module-3-finetuning
module: "Module 3"
minutes: 25
description: "When to fine-tune, SFT vs preference, PEFT/LoRA recipes — interview-ready."
---

Chapters **3.1–3.3**. Focus on when and why—not memorizing every hyperparameter.

## 3.1 Fine-tuning fundamentals

### When to fine-tune
- Try prompting and RAG first. Fine-tune when you need a **stable habit**—tone, schema, domain phrasing—that clear instructions cannot fix.
- Do not fine-tune to store facts. Use retrieval for knowledge that changes.

### Flavors
- **SFT (supervised fine-tuning)** — Train on instruction → ideal response pairs.
- **Preference / RLHF / DPO-style** — Learn from ranked pairs (chosen vs rejected). Aligns tone and policy; needs careful data.
- **Continued pretraining** — Domain corpus without instruction format (legal text, codebases).

### Data
- Quality beats quantity. Deduplicate, strip PII, balance tasks, fix label noise.
- Instruction format must match your serving chat template.
- Hold out eval that mirrors production; watch **catastrophic forgetting** (losing old skills after narrow training).

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
