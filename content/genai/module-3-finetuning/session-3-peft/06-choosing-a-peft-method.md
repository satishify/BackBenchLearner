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

:::note Analogy
You want a car to handle a mountain road better.

**Soft prompting** is changing how you drive it — the same vehicle, guided differently. Cheap, instant, reversible, and it works well as long as the car is fundamentally capable of the job.

**Adapters** are fitting different suspension. Now the machine itself behaves differently, which reaches things driving style never could — but it costs more and it is a real modification.

Ask which one your problem is. "The model can do this but needs steering" is a prompting problem. "The model does not behave the way this domain requires" is an adapter problem.
:::

### A practical order to try things

Most teams get there fastest by escalating only when the cheaper option genuinely fails:

1. **Prompting** — free, instant. Surprisingly often enough.
2. **Prompt or prefix tuning** — smallest trainable footprint; good when the base model is already capable.
3. **LoRA / adapters** — the reliable workhorse when behaviour must really change. This is where most production fine-tunes land.
4. **Full fine-tuning** — only with a large dataset, a big domain shift, and budget for the forgetting risk.

The rule of thumb: move down the list only when you have evidence the step above it is not working, not because the next one sounds more serious.

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
