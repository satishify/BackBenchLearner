---
title: "Reading the Loss Curves"
description: "How training vs validation loss tells you underfitting, overfitting, no learning, or a healthy fine-tune — and when to stop."
---

Your **ultimate guide** at the end of this lecture is the pair of curves: training loss and validation loss over steps. Read them together. They tell you which failure mode you are in — and when to stop.

## Intuition

| Pattern | What it means | What to try |
| --- | --- | --- |
| **Underfitting** | Both losses stay high and plateau early | Train longer, improve data, or give the model more room to learn |
| **Overfitting** | Train loss keeps falling while validation loss turns back up | Stop earlier, simplify updates (freeze / smaller learning rate), add better or more diverse data |
| **No learning** | Both curves flat near the starting value | Check data pipeline, labels, learning rate, and that gradients are actually flowing |
| **Healthy fine-tune** | Both losses fall together with a small, stable gap | Stop near the validation-loss minimum |

:::key
Stop where **validation** looks best — not where training loss is tiniest.
:::

## How it works

### What good learning looks like

Hallmarks of a healthy run:

- **Curves move together** — Validation follows training downward; the model is generalizing, not only memorizing.
- **Small, stable gap** — A modest train–val gap that stops widening is normal.
- **Diminishing returns** — Loss flattens smoothly; you do not see wild oscillation, spikes, or divergence.
- **Stop at the val minimum** — Checkpoint there. Training longer often buys nothing (or buys overfitting).

### Tie-back to earlier chapters

- Spikes and divergence → revisit the **training instability** checklist.
- Task up, general skills down → revisit **catastrophic forgetting** canaries (loss curves alone will not show that).
- Curves look sick despite a sane schedule → revisit **data preparation**.

## What goes wrong

- Shipping the checkpoint with the lowest training loss.
- Ignoring validation until the end of a long run.
- Reading only one curve and calling it “done.”

## One-line summary

Watch training and validation loss together: fall together with a small gap means healthy learning; stop at the validation minimum.

## Key terms

- **Underfitting** — Model never gets good on train or validation.
- **Overfitting** — Train looks great; validation gets worse.
- **Early stopping** — Halt when validation stops improving.
- **Validation loss** — Loss on held-out data used to pick checkpoints.
