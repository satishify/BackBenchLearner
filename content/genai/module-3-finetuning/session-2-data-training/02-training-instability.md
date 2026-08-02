---
title: "Training Instability"
description: "Why fine-tuning spikes or diverges, and a simple checklist (clipping, warmup, schedule, precision) to keep runs healthy."
---

Sometimes a fine-tune looks fine for a while, then the loss jumps, the run fills with nonsense numbers, or the model starts outputting gibberish. That is **training instability**: one oversized update damages the weights, which makes predictions worse, which makes the next update even wilder.

## Intuition

Think of walking down a narrow mountain path in the dark. One giant step can throw you off the trail. Fine-tuning sits in a similar place: the good pretrained solution lives in a fairly sharp valley, so huge steps are dangerous.

Common reasons instability shows up:

- The loss landscape around a pretrained model is sharp (easy to overshoot).
- Low-precision math (especially older fp16 setups) can overflow into Inf / NaN.
- One weird batch (huge length, corrupted text, strange tokens) can produce a gradient far bigger than normal.
- Deep networks multiply small problems layer by layer.

## How it works

### Three failure modes (easy to confuse)

| Failure | What you see | What it means |
| --- | --- | --- |
| **Loss spike** | Loss jumps once, then often recovers | One batch made a huge gradient; later steps may repair the damage, but you wasted progress |
| **Gradient explosion** | Gradient sizes grow quickly over many steps | Bad update → worse loss → bigger gradient → worse update (a feedback loop) |
| **Divergence** | Loss keeps rising or sticks near “random guess”; outputs become junk | The model left the useful region; it will not self-heal |

Spikes often cluster where the learning rate is highest (just after warmup).

### Fix 1: Gradient clipping

Clipping says: “If the gradient is bigger than this cap, shrink it. Keep the direction, limit the size.”

- A common default for LLM fine-tuning is `max_grad_norm = 1.0`.
- Clipping is a seatbelt, not a steering wheel. If more than about 20% of steps hit the clip, your learning rate is probably too high.

### Fix 2: Learning-rate schedule (preview)

A constant high learning rate for a long run often causes late spikes. Warmup (start small) plus a decaying schedule (for example cosine) keeps early steps gentle and late steps careful. The next lesson goes deep on schedules; for stability, remember: **warmup + decay beats “flat and aggressive.”**

### A stability checklist that compounds

| Layer | Simple setting | What it helps prevent |
| --- | --- | --- |
| Clipping | cap gradient norm around 1.0 | Spikes; feeds the explosion loop |
| Schedule | cosine (or similar) decay | Late-run spikes |
| Warmup | about 3% of steps | Blow-ups at step zero |
| Precision | prefer bf16 over fragile fp16 when available | Inf / NaN cascades |
| Peak learning rate | about 1e-5–2e-5 full fine-tune; about 1e-4–2e-4 for LoRA | Overshooting the valley |
| Monitoring | watch gradient size; alert if it is many times the usual | Catch explosion early |
| Checkpoints | save often; keep the last few | Rescue the run if it diverges |

:::tip
If one legal-domain run spiked late on a constant learning rate, switching to cosine with the **same peak** learning rate often removes those late spikes. Schedule shape matters.
:::

## What goes wrong

- Raising the learning rate because “training feels slow,” then wondering why the run exploded.
- Treating clipping as a license for huge learning rates.
- Not saving checkpoints — one bad stretch can wipe hours of work.

## One-line summary

Instability is usually “a step that was too big”; clip gradients, warm up, decay the learning rate, and watch gradient size so one bad batch does not ruin the run.

## Key terms

- **Loss spike** — Sudden jump that may later recover.
- **Gradient explosion** — Gradients growing out of control across steps.
- **Divergence** — Training leaves useful solutions; loss does not heal.
- **Gradient clipping** — Cap how large an update signal can be.
- **Warmup** — Start with a tiny learning rate, then rise to the peak.
