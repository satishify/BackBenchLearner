---
title: "Activation Functions and MLP"
description: "Why nonlinear activations matter, sigmoid/tanh/ReLU tradeoffs, and how multilayer perceptrons stack — with a tiny NumPy forward pass."
---

A stack of linear layers without nonlinearity collapses into one big matrix multiply — no matter how deep you make it. **Activation functions** are the ingredient that makes depth useful: they bend the decision surface so networks can approximate curves, XOR-like regions, and the rich features inside modern models. The **multilayer perceptron (MLP)** is the canonical stack of those nonlinear units, and it still appears inside transformers as the feed-forward block after attention.

## Intuition

Imagine drawing a map of land vs water with only straight cuts of a knife. One cut gives a half-plane. Two cuts, still polygons with straight edges — but if each cut can be *warped* before the next, you can carve islands, rings, and wiggly coasts. Nonlinearity is that warp.

Without a nonlinear activation, stacking linear layers collapses:

```
W2 * (W1 * x + b1) + b2  =  (W2 * W1) * x + (something)
```

That is still just one big linear map. Depth buys nothing. With a nonlinearity between layers, compositions can approximate extremely flexible functions (universal approximation in theory; practical power with enough width, depth, and data).

Different activations bend differently. **Sigmoid** squashes to a value between 0 and 1 like a soft probability. **Tanh** centers around zero between -1 and 1. **ReLU** passes positives unchanged and zeros negatives — cheap, sparse, and the default in many hidden layers.

Why practitioners care: choosing an activation is choosing how gradients flow and how representations look. A saturated sigmoid hidden stack is hard to train deeply; a ReLU stack trains fast but needs care around dead units. Output activations also encode the *meaning* of the prediction (a probability vs an unbounded score), which is why they are paired with specific losses.

## How it works

**Common activations.**

| Activation | Formula (plain English) | Range | Typical use |
| --- | --- | --- | --- |
| Sigmoid | `1 / (1 + e^(-z))` | between 0 and 1 | Binary output, gates |
| Tanh | `tanh(z)` | between -1 and 1 | Older hidden layers |
| ReLU | `max(0, z)` — keep positives, zero out negatives | 0 and up | Modern hidden layers |

**Tradeoffs.** Sigmoid and tanh **saturate**: for large `|z|`, the derivative flattens near zero, so gradients vanish in deep stacks (vanishing gradient problem). Tanh often trains a bit better than sigmoid in hidden layers because it is zero-centered. ReLU avoids saturation on the positive side and is fast to compute, but units can “die” if they always receive negative pre-activations (gradient forever zero). Variants (Leaky ReLU, GELU, Swish) soften that edge; GELU is common in transformers.

**MLP stacking.** An MLP is a stack of linear layers with a nonlinear activation between them:

```
h1 = activation(W1 * x  + b1)
h2 = activation(W2 * h1 + b2)
...
y_hat = WL * h(L-1) + bL
```

Hidden layers use a nonlinear activation (often ReLU). The output activation depends on the task: identity for regression, sigmoid for binary classification, softmax for multi-class. Pair the output with a matching loss (MSE, BCE, cross-entropy).

```mermaid
flowchart LR
  X[Input x] --> H1["Hidden: W1x+b1 → ReLU"]
  H1 --> H2["Hidden: W2h1+b2 → ReLU"]
  H2 --> O["Output: W3h2+b3 → linear/sigmoid/softmax"]
```

**Why this beats one perceptron.** Two hidden units with ReLU can create piecewise-linear regions; more units and layers tile space into richer polyhedral decision regions. XOR becomes solvable: one hidden layer can remap the four corners into a linearly separable feature space, then the output layer finishes the job.

**Width vs depth.** A very wide shallow MLP can approximate many functions, but depth often reuses features hierarchically with fewer total parameters for structured data (images, language). In practice you choose both: enough width to express combinations, enough depth to compose them. Transformers still rely on MLP blocks (usually two linear maps with a GELU/SwiGLU in between) after each attention layer — the same “linear → nonlinear → linear” motif at massive scale.

**Output pairing (quick table).** Regression → linear output + MSE. Binary classification → sigmoid + binary cross-entropy. Multi-class → softmax + categorical cross-entropy. Softmax turns logits into a probability distribution that sums to 1; the loss then punishes confident wrong classes especially hard.

## In code

Print activation values for a few inputs, then run a tiny 2-layer MLP forward pass with ReLU.

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0.0, z)

zs = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
print("z      ", zs)
print("sigmoid", np.round(sigmoid(zs), 3))
print("tanh   ", np.round(tanh(zs), 3))
print("relu   ", relu(zs))

# MLP: 2 → 3 (ReLU) → 1 (linear)
rng = np.random.default_rng(0)
x = np.array([0.5, -1.0])          # one example, 2 features
W1 = rng.normal(scale=0.8, size=(2, 3))
b1 = np.array([0.1, -0.2, 0.0])
W2 = rng.normal(scale=0.8, size=(3, 1))
b2 = np.array([0.0])

h = relu(x @ W1 + b1)              # hidden activations
y_hat = float(h @ W2 + b2)         # scalar output
print("hidden h:", np.round(h, 3))
print("y_hat   :", round(y_hat, 3))
```

Notice how ReLU zeros some hidden units (sparsity) while sigmoid/tanh always emit a smooth value in a bounded range. That difference shows up in gradient flow during training.

## What goes wrong

- **Linear hidden layers.** Removing ReLU/tanh between layers makes a “deep” net equivalent to a shallow linear model — XOR and curved boundaries stay impossible.
- **Sigmoid everywhere in deep nets.** Saturating activations in every layer starve early layers of gradient; prefer ReLU-family (or GELU) in hidden stacks, and reserve sigmoid for binary outputs or gates.
- **Wrong output head.** Softmax + MSE, or linear output + cross-entropy, fights the math. Match activation to loss and task.
- **Exploding pre-activations.** Huge weights push sigmoid/tanh into flat regions or send ReLU into huge positive values; careful initialization and normalization matter as models deepen.
- **Dead ReLUs.** If a unit’s input is always negative, it never updates. Monitor activation histograms if training stalls.

## One-line summary

Activations inject nonlinearity so stacked layers (an MLP) can learn complex boundaries; ReLU dominates hidden layers while sigmoid/tanh still matter at outputs and in gated designs.

## Key terms

- **Nonlinearity / activation:** elementwise function after a linear map that prevents layer collapse.
- **Sigmoid:** smooth map to a value between 0 and 1; saturates for large `|z|`.
- **Tanh:** zero-centered map between -1 and 1; also saturates.
- **ReLU:** `max(0, z)`; sparse, efficient, can die.
- **MLP (multilayer perceptron):** feed-forward stack of linear layers interleaved with activations.
- **Hidden layer:** intermediate nonlinear representation between input and output.
- **Saturation / vanishing gradients:** near-zero derivatives in flat regions of sigmoid/tanh that slow learning in deep stacks.
