---
title: "The RLHF Pipeline and InstructGPT Story"
description: "RLHF path: preference data → reward model → policy optimization. InstructGPT and ChatGPT popularized instruction tuning plus RLHF."
---

**RLHF** means **Reinforcement Learning from Human Feedback**. The big pipeline is:

1. Collect preference data (chosen vs rejected)
2. Train a **reward model**
3. Optimize the **policy** with RL (often PPO-style) under a reference / KL leash

## Intuition

First teach demos with supervised fine-tuning. Then use human preferences to further shape *how* the model answers — not only what facts it can recite.

:::key
RLHF = preferences → reward model → policy optimization. The model learns to answer more like a useful assistant.
:::

## How it works

### End-to-end path

| Stage | What you do |
| --- | --- |
| **SFT** | Train on high-quality instruction/demo answers |
| **Reward modeling** | Learn a critic from preference pairs |
| **RL optimization** | Update the policy so high-reward answers become more likely |
| **Reference / KL** | Keep the policy from drifting too far |

### InstructGPT / ChatGPT story (plain version)

- **InstructGPT** is the classic example: instruction fine-tuning, then preference-based alignment with RLHF.
- **ChatGPT-style** dialogue systems follow a similar idea: instruction-style training plus RLHF for conversational behavior.

The lesson is not “memorize product names.” The lesson is the training story: demos first, preferences second.

### RLHF vs DPO at a glance

| Dimension | **RLHF** | **DPO** (next lesson) |
| --- | --- | --- |
| Main path | Preferences → reward model → RL | Preferences → direct policy update |
| Complexity | More moving parts | Fewer moving parts |
| Stability | Can be sensitive | Often simpler to train |
| What stays central | Reward model + RL loop | Pairwise prefs + reference policy |

## What goes wrong

- Skipping SFT quality and hoping RLHF fixes everything.
- Weak preference data → a reward model that teaches the wrong taste.
- Measuring only automatic reward while humans still dislike the answers.

## One-line summary

RLHF turns human preference pairs into a reward model, then uses RL to push the policy toward answers people prefer.

## Key terms

- **RLHF** — Reinforcement Learning from Human Feedback.
- **SFT** — Supervised fine-tuning on demonstrations before preference optimization.
- **Policy model** — The generator being aligned.
- **Reference model** — Anchor that limits drift during RL updates.
