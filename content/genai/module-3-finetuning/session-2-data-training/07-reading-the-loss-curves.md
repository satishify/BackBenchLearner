---
title: "Reading the Loss Curves"
description: "How training vs validation loss tells you underfitting, overfitting, no learning, or a healthy fine-tune — and when to stop."
---

Your best guide is the pair of curves: training loss and validation loss over steps. Read them together. They tell you which failure mode you are in — and when to stop.

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

:::note Analogy
Training loss is your score on the practice questions you have already seen the answers to. Validation loss is your score on a mock exam you have never seen.

A student who memorises the practice set gets a perfect practice score and a poor mock score. That gap is the whole story of overfitting: the practice score kept improving, but the actual ability stopped improving a while ago.

When you choose which checkpoint to ship, you are choosing the student with the best mock score — not the one who memorised the practice book.
:::

Here is what those four patterns actually look like as numbers:

```text
Overfitting                      Healthy
step  train   val                step  train   val
100   1.80   1.85                100   1.80   1.86
300   1.20   1.35                300   1.21   1.30
500   0.70   1.28                500   0.88   1.05
700   0.35   1.41  <- val rose   700   0.71   0.94
900   0.12   1.66  <- worse      900   0.66   0.91  <- flattening
```

On the left, everything after step 500 made the model worse for real users while looking better on the training chart. The checkpoint worth keeping is step 500.

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

### When the curve looks fine but something is wrong

Two situations deserve suspicion even when the chart is pretty:

- **Validation loss falls unusually fast and low.** Check whether validation examples leaked into the training file. Duplicated rows across the two sets make the model look brilliant and tell you nothing.
- **Loss looks great, outputs look worse.** Loss measures next-token probability, not usefulness. A model can get better at predicting your training phrasing while becoming more repetitive or less helpful. Read a handful of real generated outputs at each checkpoint, not just the numbers.

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
