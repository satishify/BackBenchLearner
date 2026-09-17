---
title: "Fine-Tuning Approaches"
description: "Unsupervised, supervised, and safety/alignment fine-tuning — plus full fine-tuning versus parameter-efficient options in plain English."
---

“Fine-tuning” is not one single recipe. This chapter maps the main **approaches** so you can name what you are doing and why.

## Intuition

| Approach | Plain-English idea |
| --- | --- |
| **Unsupervised fine-tuning** | Continue training on domain text without instruction labels (domain language soak) |
| **Supervised fine-tuning (SFT)** | Train on input → desired output pairs |
| **Safety / alignment fine-tuning** | Extra training so the model follows policies and preferred behavior |
| **Full fine-tuning** | Almost every weight can update |
| **PEFT** | Train only a small part (adapters, LoRA, soft prompts, …) |

:::note Analogy
Think about how a person learns a new job.

- **Unsupervised fine-tuning** is sitting in the office for a month, reading old case files and absorbing the vocabulary. Nobody grades you, but you start to *sound* like the team.
- **Supervised fine-tuning** is working through solved examples: here is the customer's question, here is the approved answer. You copy the pattern until it becomes natural.
- **Alignment** is the code of conduct: what you must refuse, what you must escalate, how you behave when you are unsure.

The first two build ability. The third decides how that ability is used.
:::

:::key
Full fine-tuning is the most powerful and most expensive. PEFT is often the middle ground between prompting and updating everything.
:::

## How it works

### Unsupervised (continued) fine-tuning

Feed lots of domain text (legal corpus, codebase, medical notes) so the model absorbs domain language. There may be no “instruction → answer” labels. Useful for domain familiarity; not the same as teaching a chat format.

The data is just raw text, with no question attached:

```text
The insured party shall indemnify the underwriter against any
loss arising from misrepresentation of material fact...
```

After enough of this, the model stops being surprised by words like *indemnify* and *underwriter*, and predicts legal phrasing more naturally. What it has **not** learned is how to answer your questions — that needs the next approach.

### Supervised fine-tuning

You provide clear examples: given this input, produce that output. This is the workhorse for task adaptation and instruction-style models (next chapter goes deeper).

Here the data always comes in pairs:

```json
{"input": "Customer says the parcel never arrived. Draft a reply.",
 "output": "Hi Sam, I'm sorry your parcel hasn't arrived..."}
```

The model is graded on how close its answer is to the approved one, so it learns the task, not just the vocabulary.

### Safety / alignment fine-tuning

After (or alongside) capability training, you further shape the model so it is more helpful, honest, and policy-compliant. Methods vary (preference data, RL-style loops, and related recipes). For now, remember the **goal**: safer, more aligned behavior. Lesson **3.5** goes deep on RLHF and DPO.

### Full fine-tuning

Every (or almost every) weight can move.

- **Good for:** strong domain shift, high-value tasks, consistent formats when you have enough data
- **Risk:** cost, overfitting on small data, forgetting general skills
- **Example:** a legal drafting assistant that must internalize a house style

### PEFT (parameter-efficient fine-tuning)

Update only a small number of parameters (or add tiny modules / soft prompts).

PEFT families you will meet later in Module 3:

| Family | Idea in one line |
| --- | --- |
| **Additive** | Add small modules (adapters) |
| **Selective** | Train only some existing weights |
| **Re-parameterization** | Cheap update forms (LoRA / QLoRA) |
| **Soft prompting** | Learn virtual prompt tokens |

Use PEFT when you want most of the benefit of adaptation without paying full fine-tune cost.

:::note Analogy
Full fine-tuning is renovating the whole house — every room, every wire. PEFT is fitting a new set of light switches: the building is untouched, but the behaviour you notice every day changes.

And because the house itself is unchanged, you can keep several sets of switches and swap them per customer. That is exactly how one base model serves many fine-tuned variants.
:::

### Which one, in practice

| Your situation | Sensible approach |
| --- | --- |
| The model does not know your domain's *language* | Unsupervised / continued pretraining |
| You have labelled input→output examples | SFT |
| The model is capable but occasionally unsafe or unhelpful | Alignment |
| You have lots of data, budget, and a big behaviour change | Full fine-tuning |
| You have modest data and want low cost and easy rollback | PEFT |

Most teams in practice land on **SFT with PEFT** — enough to change behaviour, cheap enough to repeat when the requirements change.

## What goes wrong

- Calling every training run “SFT” when it was unlabeled domain text.
- Jumping to full fine-tuning when PEFT or RAG would have been enough.
- Ignoring alignment/safety until after a risky model is already in production.

## One-line summary

Fine-tuning comes in flavors — unsupervised, supervised, alignment, full, or PEFT — pick by data type, cost, and how much of the model must change.

## Key terms

- **SFT** — Supervised fine-tuning on labeled pairs.
- **Full fine-tuning** — Updating nearly all weights.
- **PEFT** — Training only a small part of the model.
- **Alignment** — Shaping the model toward preferred / safer behavior.
