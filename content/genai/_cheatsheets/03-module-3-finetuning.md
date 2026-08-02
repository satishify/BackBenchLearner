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

## 3.2 Data prep & training mechanisms (lecture map)

- **How LLMs learn** — Tokenize → mini-batch → next-token guess → softmax → cross-entropy → small update. Stop when validation flattens.
- **Instability** — Spikes / exploding gradients / divergence. Clip gradients, warmup, decay learning rate, prefer bf16, save checkpoints.
- **Catastrophic forgetting** — Narrow fine-tune improves the task but hurts general skills. Measure canaries; use smaller updates, rehearsal, freeze layers, regularize, or weight averaging.
- **Learning-rate schedule** — Warmup + peak + decay (cosine is a common default). Peak learning rate is the big knob.
- **Batch size** — Care about effective batch = micro-batch × accumulation × GPUs. Too small → jitter; too large → too few updates.
- **Data prep** — Fix noise, bad labels, duplicates, messy formatting, stale facts; split honestly; watch domain shift.
- **Loss curves** — Train and val falling together with a small gap = healthy. Stop at the validation minimum.

## 3.3 PEFT, adapters, soft prompts

### Why PEFT exists
- Full fine-tuning updates every weight → expensive, huge artifacts, harder multi-tenant serving.
- **PEFT (parameter-efficient fine-tuning)** freezes the backbone and trains a tiny add-on instead.

### Adapters and LoRA
- **Adapters** — Small trainable blocks inserted into the network; learn a task-specific tweak around frozen features.
- **LoRA (Low-Rank Adaptation)** — Express the weight update as two small matrices: weight update ~= B times A (low rank). Higher rank = more capacity and VRAM.
- **QLoRA (Quantized LoRA)** — Quantize the frozen base (e.g. 4-bit) + train LoRA adapters. Fits big models on smaller GPUs; always check quality.

### Soft prompts
- **Prompt tuning** — Learn virtual token embeddings prepended to the input; backbone stays frozen.
- **Prefix tuning** — Learn prefixes injected into layer attention; more expressive than prompt tuning alone.
- Smallest footprint, but may underfit hard generative tasks—escalate to LoRA when needed.

### Training recipe (LoRA defaults)
- Learning rate ~1e-4 to 3e-4 (higher than full FT).
- Start rank 16, alpha 32 on attention projections.
- One to two epochs on clean data; change one knob at a time.

### Serving
- **Merge** LoRA into base → one simple checkpoint per skill.
- **Keep separate** → many small adapters on one base; hot-swap and cheap rollback.
- Version base + adapter + tokenizer + chat template together.

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
