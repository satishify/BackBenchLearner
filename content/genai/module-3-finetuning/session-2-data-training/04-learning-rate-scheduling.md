---
title: "Learning Rate Scheduling"
description: "Why one fixed learning rate is rarely ideal, what warmup and decay do, and when to pick constant, linear, cosine, step, restarts, or WSD."
---

The **learning rate** is how big each training step is. One fixed value for the whole run is rarely ideal. Early training often needs larger steps to make progress; late training needs smaller steps so the model can settle safely. A **schedule** changes the learning rate over time.

## Intuition

The dilemma in one sentence:

- **High learning rate** — learns fast early, but later bounces around and can spike.
- **Low learning rate** — very stable, but you may run out of budget before the model converges.

You want both stories in one run: bigger steps early, smaller steps later.

Training also has phases:

1. **Fragile start** — New data shocks the pretrained weights. Full-size steps here cause instant spikes → use **warmup** (rise from near zero).
2. **Rapid learning** — Gradients are useful; most of the task is learned → spend time near the **peak** learning rate.
3. **Convergence** — The “valley” gets narrower → **decay** the learning rate.

:::key
Peak learning rate is often the single most important knob in fine-tuning. The schedule decides what happens after that peak.
:::

## How it works

### Four numbers that define a schedule

| Knob | Plain-English idea |
| --- | --- |
| **Warmup length** | How long the learning rate ramps from ~0 to the peak (often about 3% of steps; raise to 5–10% if the run is short or unstable) |
| **Peak learning rate** | The maximum. Rough guide: about 1×10⁻⁵–2×10⁻⁵ for full fine-tuning; about 1×10⁻⁴–2×10⁻⁴ for LoRA |
| **Decay shape** | How you go down from the peak (linear, cosine, steps, restarts, …) |
| **Floor (min learning rate)** | Where decay ends — zero, or a small floor if you want learning to stay slightly alive |

### Common schedules

| Schedule | Plain-English idea | When it fits |
| --- | --- | --- |
| **Constant (+ warmup)** | After warmup, keep one learning rate forever | Short, controlled runs (for example short LoRA); more spike risk on long runs |
| **Linear decay** | After warmup, fall in a straight line toward zero | Clean, simple baseline |
| **Cosine decay** | Stay productive near the peak longer, then glide gently down | Modern default for many LLM fine-tunes |
| **Step decay** | Stay flat, then drop by a factor at milestones (or when validation stalls) | When you want sharp, reactive drops |
| **Cosine with restarts** | Decay, jump back up, decay again | Exploring more than one “basin” in one budget |
| **Warmup–Stable–Decay (WSD)** | Warm up, hold a plateau, then decay at the end | Longer or open-ended budgets |

### Matching the schedule to the run

- Prefer **cosine + warmup** as a default for full fine-tunes.
- Use **constant + warmup** only when the run is short and you are watching carefully.
- If late spikes appear on a flat learning rate, try cosine with the **same peak** — often the late spikes disappear.

## What goes wrong

- Skipping warmup on a hot peak learning rate.
- Using a huge constant learning rate for many epochs.
- Tuning ten schedule shapes before you have checked that the peak learning rate itself is sane.

## One-line summary

Schedules give you large steps when learning is easy and small steps when the model needs to settle — warmup plus a decaying shape (often cosine) is the usual safe pattern.

## Key terms

- **Warmup** — Gradual rise of the learning rate at the start.
- **Peak learning rate** — Maximum step size in the run.
- **Cosine decay** — Smooth curve that spends time near the peak, then eases down.
- **WSD** — Warmup, then a stable plateau, then a final decay.
