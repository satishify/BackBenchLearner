---
title: "Preparing the Data"
description: "Data quality problems that break fine-tunes — noise, bad labels, duplicates, formatting, outdated facts — plus small-data and split pitfalls."
---

Clever schedules cannot fully rescue bad data. This chapter is about the data problems that quietly break fine-tunes, and the habits that keep training honest.

## Intuition

Slide-level agenda for today still holds: **important characteristics of data**, then **how to prepare it**. Five quality issues show up again and again.

| Problem | What goes wrong | Typical fix |
| --- | --- | --- |
| **Noisy data** | Menus, ads, OCR junk, spam, or AI-generated slop pollute the corpus | Filter hard; remove obvious garbage |
| **Incorrect labels / “truth”** | Wrong answers, buggy code, mis-tagged language taught as fact | Audit labels; drop or fix bad rows |
| **Duplicates** | Memorization rises; evals get contaminated | Remove exact and near duplicates |
| **Inconsistent formatting** | HTML, Markdown, PDF text, and LaTeX mixed badly | Normalize structure before training |
| **Outdated information** | Old APIs, old policies, old officeholders stated as current | Refresh corpus; track cutoffs |

:::key
Good fine-tuning starts with good data. Fix the rows before you chase fancy training tricks.
:::

## How it works

### More data risks (beyond the big five)

| Risk | Plain-English idea | What to do |
| --- | --- | --- |
| **Small datasets** | Too few examples → easy overfitting, high variance | Prefer transfer / light tuning; augment carefully; do not train forever |
| **Missing edge cases** | Rare but important failures never appear in train | Actively collect those cases; mine production misses |
| **Lack of diversity** | Model works for one group and fails for others | Set coverage targets; measure per subgroup |
| **Imbalanced classes** | Always predicting the majority looks “accurate” | Use precision/recall/F1; rebalance or weight classes |
| **Bad splits** | Test scores look great because of leakage | Split first; never fit transforms on the full dataset |
| **Domain / distribution shift** | Train on web docs, deploy on live chats and tools | Align training mix to deployment; use retrieval for fresh facts |

### Splitting train / validation / test

A common honest split is roughly **70% train / 15% validation / 15% test** (adjust to your size).

**Pitfalls (leakage):**

- Scaling or cleaning using the whole dataset before splitting.
- Near-duplicates sitting in both train and test.
- Random splits on time-series data (the model “sees the future”).
- Same user or patient in both train and test.

**Better habits:**

- Split first; fit any transforms on train only.
- Stratify so class balance matches across splits.
- For time data, train on the past and test on the future.
- Keep groups (user / patient / device) entirely in one split.
- On small data, use cross-validation — but keep a final test set sacred.

### Domain and usage shift

| Shift | Example |
| --- | --- |
| **Usage shift** | Trained on documents; deployed on chat and tools |
| **Temporal shift** | World and APIs moved on after the training cutoff |
| **Domain shift** | Web-trained model dropped into clinical notes or contracts |

Mitigations in practice: instruction-style post-training, more deployment-like data late in training, domain refresh with replay, retrieval for current facts, and monitoring live quality.

## What goes wrong

- Training on crawl sludge and blaming the optimizer.
- Reporting accuracy on fraud detection when “always not fraud” scores 99.8%.
- A beautiful validation score that was contaminated by duplicates or future leakage.

## One-line summary

Clean, deduplicated, well-split, deployment-like data beats clever training tricks; treat noise, labels, duplicates, format, and staleness as first-class bugs.

## Key terms

- **Deduplication** — Removing repeated or near-repeated samples.
- **Leakage** — Test information sneaking into training.
- **Stratified split** — Keeping class proportions similar across splits.
- **Domain shift** — Train world and deploy world do not match.
