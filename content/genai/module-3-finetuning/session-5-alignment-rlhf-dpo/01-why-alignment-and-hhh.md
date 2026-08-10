---
title: "Why Alignment Matters and the HHH Target"
description: "A base model predicts tokens well, but alignment teaches it to answer like a good assistant: helpful, honest, and harmless."
---

A base language model is good at predicting the next token. That does **not** automatically make it safe, truthful, or useful in the way people want. **Alignment** is the post-training work that shapes behavior after the model already knows language.

## Intuition

Think of two stages:

1. **Pretraining / SFT** — learn language and basic task habits
2. **Alignment** — learn *how to respond* the way people prefer

The common target is called **HHH**:

| Goal | Plain meaning |
| --- | --- |
| **Helpful** | Actually solve the user’s problem, not only sound polished |
| **Honest** | Stay truthful; do not invent facts or fake certainty |
| **Harmless** | Refuse or redirect requests that could cause real harm |

:::key
Alignment shapes behavior toward human preference and safety — not just next-token fluency.
:::

## How it works

### Tiny examples

- **Helpful:** User shows `KeyError: 'user_id'`. A helpful answer points to the missing key and suggests a fix.
- **Honest:** User repeats a common myth. An honest answer corrects it instead of agreeing to please them.
- **Harmless:** User asks for dangerous instructions. A harmless answer declines and stays safe.

### The trade-off

Optimizing only one HHH goal can go wrong:

| If you over-optimize… | Risk |
| --- | --- |
| Only helpfulness | The model may comply with bad requests |
| Only harmlessness | It may refuse too much, even safe asks |
| Only honesty | It may dump harsh facts in an unhelpful way |

Good alignment balances all three.

## What goes wrong

- Treating a fluent chatbot as already “aligned.”
- Pushing one HHH knob so hard that the other two break.
- Confusing alignment with storing fresh facts (that is still RAG / tools territory).

## One-line summary

Alignment exists because next-token skill is not the same as being a helpful, honest, harmless assistant.

## Key terms

- **Alignment** — Post-training that shapes behavior toward preferred / safer answers.
- **HHH** — Helpful, Honest, Harmless.
- **Post-training** — Extra training after the model already knows language.
