---
title: "Module 3 - Fine-tuning revision"
slug: module-3-finetuning
module: "Module 3"
minutes: 25
description: "When to fine-tune, SFT vs preference, PEFT/LoRA recipes — interview-ready."
---

Chapters **3.1–3.3**. Prefer PEFT mental models over memorizing every hyperparameter.

## 3.1 Fine-tuning fundamentals

### When to fine-tune
- Prompting / RAG already fail after real effort: style, domain jargon, structured formats, latency (smaller specialist), or offline/on-prem needs.
- Don’t fine-tune to “store facts” — use retrieval. Fine-tune for *behavior and format*.

### Flavors
- **SFT (supervised fine-tuning)** — train on instruction -> ideal response pairs.
- **Preference / RLHF / DPO-style** — learn from ranked pairs (chosen vs rejected). Aligns tone and policy; needs careful data.
- **Continued pretraining** — domain corpus without instruction format (legal text, codebases).

### Data
- Quality matters more than quantity. Deduplicate, strip PII, balance tasks, fix label noise.
- Instruction format consistency (roles, special tokens) must match serving chat template.
- Hold out eval that mirrors production; watch **catastrophic forgetting** of general skills.

## 3.2 Data prep & training mechanisms

- Packing / packing efficiency; max length truncation strategy.
- Loss usually on assistant tokens only (mask user/system).
- Overfit signs: train loss drops a lot, val / prod quality drops, or the model regurgitates train examples.
- Checkpoints, early stopping on a task metric (not only loss).

## 3.3 PEFT, adapters, soft prompts

### Why PEFT
- Full finetune updates all weights -> expensive, huge artifacts, hard multi-tenant.
- **Adapters / LoRA**: train small low-rank updates; freeze base. Serve many adapters on one base.
- **LoRA**: approximates the weight update as two small matrices B and A multiplied together (low rank). Higher rank means more capacity and VRAM. Common start: modest rank, then tune learning rate.
- **QLoRA**: quantize base (e.g. 4-bit) + LoRA — consumer GPU friendly; watch quality/latency tradeoffs.
- **Prompt / prefix tuning**: learn soft tokens; even smaller footprint; sometimes weaker than LoRA for complex tasks.

### Serving
- Merge LoRA into base for single-model deploy, or load adapters dynamically.
- Version base + adapter + tokenizer + chat template together.

## Decision cheat
| Need | Prefer |
|------|--------|
| Fresh facts | RAG |
| Tone / schema / domain phrasing | SFT / LoRA |
| Safety / preference | Preference data + policy |
| Tiny GPU | QLoRA / smaller base |
| Many customers’ styles | Per-tenant adapters |

## 30-minute drill
1. Argue for RAG vs LoRA on a “company FAQ bot”.
2. Name two evals that catch forgetting.
3. Explain LoRA rank in one sentence (how big the two small matrices are).
