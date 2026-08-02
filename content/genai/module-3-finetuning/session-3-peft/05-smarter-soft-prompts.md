---
title: "Smarter Soft Prompts"
description: "SMoP, APT, IDPG, and SPT — making soft prompts selective instead of one blunt prompt for every layer and every input."
---

A single soft prompt for every input and every layer can be too blunt. These methods make soft prompts **smarter and more selective**.

## Intuition

Different layers do different jobs. Different inputs need different hints. So we should not always:

- Use one prompt for everything
- Give every layer the same prefix length
- Ignore the specific input instance
- Force prompts into every layer

:::key
Prompt tuning should be selective, not wasteful.
:::

## How it works

### SMoP — Sparse Mixture of Prompts

**Plain-English idea:** keep several prompt candidates, and **activate only the useful ones** for a given input (sparse mixture), instead of forcing one mega-prompt to do all the work.

- Good when the prompt budget is limited
- Uses a routing idea so not every prompt component fires every time

### APT — Adaptive Prefix Tuning

**Plain-English idea:** do **not** treat all layers equally. Change prefix length (prompt capacity) across layers.

Why that matters:

- **Bottom layers** tend to capture shallow, phrase-level features → may need **longer** prefixes
- **Higher layers** deal more with abstract meaning → often need **shorter** prefixes on average

| Task flavor | Layer emphasis | Why |
| --- | --- | --- |
| Entity / relation extraction | Bottom layers | Local phrase structure matters a lot |
| Commonsense / causal reasoning | Higher layers | Semantic abstraction matters more |

APT can also use gating so a layer only “keeps” the prefix tokens it needs.

### IDPG — Instance-Dependent Prompt Generation

**Plain-English idea:** the right soft prompt should depend on **this input**, not only on the task name.

A small prompt-generator network builds prompt vectors from the input. Useful when one task has many subcases and a static prompt is too generic.

### SPT — Selective Prompt Tuning

**Plain-English idea:** ask whether **every** layer needs soft prompting at all. Usually no. Learn to insert prompts **only where they help**.

## What goes wrong

- Jumping to SMoP/APT/IDPG before a simple prompt or adapter baseline is working.
- Assuming “more prompt tokens everywhere” is always better.
- Using one static prompt when inputs vary wildly (IDPG’s motivation).

## One-line summary

SMoP, APT, IDPG, and SPT make soft prompts selective — by mixture, by layer, by input instance, or by choosing where prompts are even needed.

## Key terms

- **SMoP** — Sparse mixture of prompts; activate only useful prompts.
- **APT** — Adaptive prefix tuning; vary prompt length by layer.
- **IDPG** — Instance-dependent prompt generation from the input.
- **SPT** — Selective prompt tuning; insert prompts only where needed.
