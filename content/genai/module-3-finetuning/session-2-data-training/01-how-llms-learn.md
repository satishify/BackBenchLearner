---
title: "How LLMs Learn"
description: "The training loop in plain English: tokens, mini-batches, next-token prediction, loss, and small weight updates."
---

Before we talk about data cleaning or learning-rate tricks, it helps to see what fine-tuning actually *does*. Training is one loop, run many times. Each time, the model looks at a small pack of examples, guesses the next piece of text, checks how wrong it was, and nudges its weights a tiny bit to do better next time.

## Intuition

Imagine practicing a skill with flashcards:

1. You see a few cards (a mini-batch).
2. You guess the answers.
3. You see how far off you were (the loss).
4. You adjust a little (an update).
5. You shuffle and repeat.

That is basically how a large language model (LLM) learns during fine-tuning.

:::key
One training step = one mini-batch → compute loss → update weights. An epoch = one full pass through the training set.
:::

## How it works

### Text becomes numbers (tokenization)

The model never reads letters the way we do. A tokenizer turns text into a list of token IDs. Training packs those IDs into fixed-length windows (the context length) so the GPU can process many examples efficiently.

### Mini-batches (the sweet spot)

| Approach | Plain-English idea | Trade-off |
| --- | --- | --- |
| **Full batch** | Use the whole dataset for one update. | Most accurate direction, but far too slow for big data. |
| **One example at a time** | Update after every single row. | Cheap, but the signal is very noisy. |
| **Mini-batch** | Update after a small group of examples. | Stable enough, and GPUs like this pattern. |

### Predict the next token everywhere

During training, the model predicts the next token at **every** position in the window, not only once at the end. That gives many teaching signals from each sequence.

### From raw scores to “how wrong am I?”

1. The model outputs raw scores (logits).
2. Softmax turns those scores into probabilities that sum to 1.
3. Cross-entropy asks: “How much probability did you put on the correct next token?” Averaging that over the batch gives one loss number.

Lower loss means the model is assigning higher probability to the real text.

### Backpropagation and the optimizer step

Backpropagation figures out, for each trainable weight, which way to push it to reduce the loss. Then the optimizer takes a step:

`new_weights = old_weights - learning_rate * gradient`

- **Learning rate** = how big the step is. Too big and training blows up; too small and it crawls.
- In practice people use adaptive optimizers like **Adam** (not plain old SGD) for LLM fine-tuning.
- Then load the next mini-batch and repeat.

### Steps, epochs, and when to stop

| Term | Meaning |
| --- | --- |
| **Step** | One mini-batch → one weight update |
| **Epoch** | One full pass over the training set |
| **When to stop** | When loss on held-out validation data flattens — more training often means memorizing |

The wiggle in the training curve is normal. Each mini-batch is a random sample, so its loss is a noisy estimate of the “true” average loss.

## What goes wrong

- Thinking the model “reads” text directly — it only sees token IDs.
- Stopping only when training loss is tiny — validation may already be getting worse.
- Treating every loss wiggle as a bug — some noise is expected.

## One-line summary

Fine-tuning repeatedly shows the model mini-batches of tokens, scores next-token guesses with cross-entropy, and takes small optimizer steps downhill on that loss.

## Key terms

- **Tokenization** — Turning text into model-readable token IDs.
- **Mini-batch** — A small group of examples used for one update.
- **Logits** — Raw scores before they become probabilities.
- **Softmax** — Turns scores into a probability distribution.
- **Cross-entropy** — Loss that punishes low probability on the correct token.
- **Learning rate** — Size of each optimizer step.
- **Epoch** — One full pass over the training data.
