---
title: "Why PEFT Exists"
description: "Why full fine-tuning gets expensive at scale, what PEFT means in plain English, and the main PEFT families."
---

When a model already has billions of parameters, updating **every** weight for every new task is often wasteful. **PEFT** (parameter-efficient fine-tuning) keeps most of the pretrained model frozen and trains only a small add-on — or a small change — instead.

## Intuition

Full fine-tuning a large model can need huge GPU memory and many cards. PEFT asks a simpler question:

> Why update 7 billion parameters when a few million will do?

In practice, PEFT often trains **well under 1%** of the parameters, while still adapting the model to a new task.

It helps with:

- Lower GPU memory and compute cost
- Faster training and cheaper multi-task serving
- Keeping more of the general knowledge already in the base model
- Reducing **catastrophic forgetting** when tasks are narrow or many

:::key
PEFT = adapt a big pretrained model by changing only a small part of it.
:::

## How it works

### Where scale becomes a problem

Bigger models can be more capable — but full fine-tuning them gets “astronomically costly.” You may need many high-end GPUs just to update all weights. PEFT is the practical escape hatch: keep the big brain, train a small skill module.

### Multi-task fine-tuning pain

If you fully fine-tune one shared model for task after task, later tasks can wipe earlier skills (forgetting). PEFT lets many tasks **share one frozen backbone** and keep only tiny task-specific pieces.

### PEFT taxonomy

| Family | Plain-English idea | Examples |
| --- | --- | --- |
| **Selective** | Train only some existing weights (or sparse differences) | BitFit, Diff Pruning |
| **Additive** | Add small new modules into the network | Adapters, AdapterFusion |
| **Re-parameterization** | Rewrite the weight update in a cheaper form | LoRA, QLoRA |
| **Soft prompting** | Learn virtual prompt tokens instead of editing the backbone | Prefix tuning, prompt tuning, SMoP, APT, IDPG |

The next lessons go deep on two tracks: **additive PEFT (adapters)** and **soft prompting**. **LoRA / QLoRA** (re-parameterization PEFT) get their own deep dive in lesson 3.4.

## What goes wrong

- Jumping straight to full fine-tuning “because we have a GPU,” then running out of memory or forgetting old skills.
- Treating PEFT as magic that always improves every metric — it is about **cheaper, safer adaptation**, not a free lunch.

## One-line summary

PEFT adapts huge language models by training a tiny fraction of parameters (or a tiny prompt), so you save cost and protect general skills.

## Key terms

- **PEFT** — Fine-tuning by changing only a small part of a pretrained model.
- **Full fine-tuning** — Updating (almost) all model weights for a task.
- **Catastrophic forgetting** — Losing older skills after narrow new training.
- **Post-training** — Training done after the base model was pretrained.
