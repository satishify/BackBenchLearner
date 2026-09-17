---
title: "Multi-Task Fine-Tuning"
description: "Train one shared model on several related tasks so they share features and help regularize each other."
---

**Multi-task fine-tuning** trains one model on several related tasks at the same time. A shared backbone learns reusable language patterns; the tasks can support each other instead of fighting in separate silos.

## Intuition

Example for a support assistant: train **intent classification**, **slot filling**, and **FAQ matching** together. All three need similar customer-language understanding, so sharing a backbone often helps.

:::note Analogy
Think of a medical student who studies anatomy, pharmacology, and diagnosis together rather than one subject in total isolation. Understanding how the heart works makes the heart medicines easier to remember, and both make chest-pain cases easier to diagnose.

The subjects are different, but they lean on the same foundation — so learning them together is more efficient than learning each from scratch.

Now imagine that same student studying medicine and classical music simultaneously. There is no shared foundation, so the hours simply compete. That is what unrelated multi-task training feels like to a model.
:::

Here is what those three related tasks look like on one customer message:

```text
Message: "My card was charged twice for order 5567 on Monday."

Intent classification -> billing_dispute
Slot filling          -> {order_id: 5567, issue: duplicate_charge, date: Monday}
FAQ matching          -> "What to do about duplicate charges"
```

All three need the model to understand the same sentence. Training them together means that understanding is learned once and reused three times.

:::key
Related tasks can share features and regularize each other — that is the main attraction of multi-task training.
:::

## How it works

### Why it can help

- Shared representations for shared language
- Extra signal when each single task has limited data
- One deployed backbone instead of many fully separate models (depending on your setup)

### When it fits

| Prefer multi-task when… | Be careful when… |
| --- | --- |
| Tasks are clearly related | Tasks pull the model in opposite directions |
| You want shared language features | One task dominates the data mix |
| Labels exist for each task | You have no way to balance sampling |

### Practical habits

- Mix batches so no single task always wins
- Watch metrics **per task**, not only one average score
- If one task collapses, rebalance data or separate that task

### Why balance matters so much

Imagine this data mix:

| Task | Examples | Share of training |
| --- | --- | --- |
| Intent classification | 90,000 | 90% |
| Slot filling | 8,000 | 8% |
| FAQ matching | 2,000 | 2% |

The model quickly learns that getting intent right pays off far more than anything else, so it optimises for intent and lets FAQ matching drift. The average score still looks respectable — which is exactly the trap.

A common fix is to **sample** rather than simply concatenate: draw roughly equal numbers of examples from each task per batch, even when the underlying datasets are very different sizes.

Watching per-task scores makes the problem visible immediately:

```text
Epoch 3
  intent_accuracy   0.94  (up)
  slot_f1           0.81  (flat)
  faq_accuracy      0.42  (down)  <- this task is being crushed
```

## What goes wrong

- Calling unrelated tasks “multi-task” and hoping they magically help.
- Averaging metrics so a failing task hides behind a strong one.
- Ignoring catastrophic forgetting when many tasks are trained in a harsh schedule (lesson 3.2 covers defenses).

## One-line summary

Multi-task fine-tuning shares one backbone across related tasks so they can borrow useful features from each other.

## Key terms

- **Multi-task learning** — Training on several tasks together with shared parameters.
- **Shared backbone** — The common model body reused across tasks.
- **Task head** — A small output piece specialized for one task (when used).
