---
title: "Policy Gradient and Advantage"
description: "The policy is the next-token distribution. Policy gradients raise the chance of token choices that led to better final answers."
---

The **policy** is the model’s probability distribution over the next token given the current context. In alignment with RL, we want to raise the chance of actions that lead to higher reward.

## Intuition

Core idea in one sentence:

> If a token choice helped produce a better final response, make that choice more likely next time.

That family of methods is called **policy gradient**. A classic sample-based version is **REINFORCE**: sample full answers (trajectories), score them, and update from those samples.

:::key
Positive advantage means “better than expected for this state.” Negative means “worse than expected.”
:::

## How it works

### What we optimize

We care about **expected total reward** for a whole response, not only one token in isolation.

In words: train the policy so good full answers become more probable.

### Why the raw signal is noisy

REINFORCE can be unbiased but **high variance** — the training signal jumps around. Helpers:

| Idea | Plain meaning |
| --- | --- |
| **Baseline** | Subtract a reference level so you measure relative quality, not raw score alone |
| **Advantage** | How much better (or worse) an action was than expected for that state |

Advantage intuition:

- **Positive advantage** → that action was better than average for the situation → increase its probability
- **Negative advantage** → worse than expected → decrease its probability

### Keep the math light

You do not need to memorize the formula to use the idea. The practical message is:

1. Sample responses
2. Score them
3. Push probability up for better-than-expected choices
4. Use advantage/baseline so the push is less noisy

## What goes wrong

- Updating from a few lucky samples and calling it “aligned.”
- Ignoring variance — training looks random and unstable.
- Optimizing reward of one flashy phrase instead of the whole answer’s quality.

## One-line summary

Policy gradient methods increase the probability of token choices that earned positive advantage on the full response.

## Key terms

- **Policy** — Next-token probability distribution given context.
- **Policy gradient** — Update rule that follows reward-weighted log-probabilities.
- **REINFORCE** — Sample trajectories and estimate the gradient from them.
- **Advantage** — How much better an action was than expected.
