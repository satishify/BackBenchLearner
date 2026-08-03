---
title: "Multi-Task Fine-Tuning"
description: "Train one shared model on several related tasks so they share features and help regularize each other."
---

**Multi-task fine-tuning** trains one model on several related tasks at the same time. A shared backbone learns reusable language patterns; the tasks can support each other instead of fighting in separate silos.

## Intuition

Example for a support assistant: train **intent classification**, **slot filling**, and **FAQ matching** together. All three need similar customer-language understanding, so sharing a backbone often helps.

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
