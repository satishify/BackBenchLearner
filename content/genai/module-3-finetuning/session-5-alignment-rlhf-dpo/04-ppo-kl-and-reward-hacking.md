---
title: "PPO, KL Penalty, and Reward Hacking"
description: "PPO updates the policy carefully on sampled answers, while a KL leash stops weird drift — and reward hacking is how models cheat the score."
---

**PPO** (Proximal Policy Optimization) is a common way to run the RL part of RLHF. The practical story: sample answers with a policy, score them, update carefully for a few epochs, and keep the new model close to a **reference** model.

## Intuition

Updating every step from fresh samples is expensive. A common PPO-style pattern:

1. An older policy generates trajectories (answers)
2. Those cached trajectories are reused for several update epochs
3. A **KL penalty** acts like a leash so the aligned model does not drift into strange wording just to chase reward

:::key
KL keeps the aligned policy near the reference model. Reward hacking is when the model learns to game the score instead of truly helping.
:::

## How it works

### PPO training idea (plain English)

| Step | What happens |
| --- | --- |
| Sample | Generate responses with the current / old policy |
| Score | Get rewards / advantages for those responses |
| Update | Improve the policy for a few epochs on that cached experience |
| Constrain | Keep behavior near the reference model (KL leash) |

You do not always need two live full copies updating at once — caching trajectories and reusing them is part of the efficiency story.

### KL divergence as a leash

**KL divergence** measures how far one probability distribution has moved from another. In alignment:

- Too little restraint → model can invent weird high-reward styles
- Sensible KL penalty → stay useful without drifting too far from the SFT / reference behavior

### Reward hacking

**Reward hacking** means the model finds shortcuts that raise the score without truly solving the user’s need — for example, overly flattering text, empty verbosity, or patterns the reward model wrongly likes.

If the reward signal is imperfect, RL will exploit the cracks.

## What goes wrong

- No KL / reference anchor → fluent nonsense that “scores well.”
- Over-tight KL → almost no alignment progress.
- Celebrating reward-model score while human raters still dislike the answers.

## One-line summary

PPO improves the policy on sampled answers with careful updates; KL limits drift; reward hacking is cheating the score instead of helping the user.

## Key terms

- **PPO** — Policy optimization that updates carefully and limits bad jumps.
- **Reference model** — Anchor model that keeps aligned behavior nearby.
- **KL divergence** — Measure of how far two distributions have moved apart.
- **Reward hacking** — Gaming the reward signal instead of solving the real task.
