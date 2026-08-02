---
title: "Choosing a PEFT Method"
description: "A plain comparison of adapters vs soft prompts vs smarter prompt variants — pick by cost, task shape, and how much structure you need."
---

The practical takeaway: **choose the method that matches your cost, data shape, and the kind of adaptation you need** — not the method with the most jargon.

## Intuition

| Method | What is trained | Best for | Main trade-off |
| --- | --- | --- | --- |
| **Adapters** | Tiny modules inside the network | Task-specific behavior with modular reuse | Adds a bit of architecture, not only tokens |
| **Prefix tuning** | Virtual prefix tokens / context | Generation tasks; compact adaptation | Sensitive to prefix design |
| **Prompt tuning** | Prompt embeddings only | Simple PEFT baseline | Needs careful prompt length; can be brittle |
| **SMoP** | Sparse prompt mixture | Smarter prompt selection | More complexity than plain prompt tuning |
| **APT** | Layer-wise prefix allocation | Tasks with different layer needs | More design choices |
| **IDPG** | Prompt generator + prompt vectors | Instance-dependent settings | Adds another learned module |

## How it works

### When to choose what

| Goal | Prefer |
| --- | --- |
| Modular, architecture-aware tuning; swap task modules on one backbone | **Adapters** |
| Smallest possible trainable footprint; keep backbone frozen | **Soft prompting** (prompt / prefix tuning) |
| One prompt feels too generic for the data | **Smarter soft prompts** (SMoP, APT, IDPG, SPT) |
| Full fine-tuning is too expensive, brittle, or forgetful | **PEFT in general** |

### Clean conceptual split

- **Adapters** add architectural elements (adaptation in network space).
- **Soft prompts** adapt in token / embedding space.

Both are PEFT. They solve the same big problem (cheap adaptation) in different places.

### Quick reminder of the PEFT efficiency principle

Trainable parameters should stay **much smaller** than total parameters. If your “PEFT” setup is nearly as heavy as full fine-tuning, you have lost the plot.

## What goes wrong

- Picking the trendiest variant before you need it.
- Full fine-tuning when an adapter or soft prompt would have been enough.
- Soft prompts when you really needed modular multi-task adapters (or the reverse).

## One-line summary

Pick adapters for modular network-side adaptation, soft prompts for tiny frozen-backbone adaptation, and smarter prompt methods when one blunt prompt is not enough.

## Key terms

- **Modular reuse** — Share one backbone; swap small task pieces.
- **Prompt-space adaptation** — Change learnable prompts, not full weights.
- **Architecture-space adaptation** — Add small modules inside the network.
