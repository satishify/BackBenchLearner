---
title: "The Training Loop — Learning Rate and Epochs"
description: "Walk through forward, loss, backward, and update; choose learning rates and epochs; monitor train vs validation loss on a toy gradient-descent loop."
---

**What is the training loop?** It is the four-step ritual repeated thousands of times: predict, score the error, compute gradients, nudge weights. Every deep learning framework hides this behind `model.fit` or a training step.

**Why do learning rate and epochs matter?** They are the two knobs beginners feel first. Learning rate sets step size; epochs set how many times the model sees the full dataset. Get either wrong and training either crawls, explodes, or memorizes instead of learning.

:::key
Forward → loss → backward → update, repeated over batches and epochs. Learning rate = step size. Train vs validation curves tell you whether you are learning or memorizing.
:::

## Intuition

Picture the hiker in fog descending a valley (the loss surface). Each step of the loop:

1. **Forward** — Stand at current parameters; compute the prediction for a batch of data.
2. **Loss** — Measure how bad that prediction is.
3. **Backward** — Feel the local slope (gradients of loss with respect to each weight).
4. **Update** — Take a step downhill: `weight = weight - learning_rate * gradient`.

The **learning rate** (often written as eta or "eta") is step length:

- Too small → endless tiny shuffles, painfully slow progress.
- Too large → leaps past the valley, loss spikes or becomes NaN.

An **epoch** is one full pass through the training set. Real datasets are processed in **mini-batches** so you get many updates per epoch without loading everything at once.

**Underfitting vs overfitting — the studying trap:**

| Problem | Plain-English idea | Symptom |
| --- | --- | --- |
| **Underfitting (high bias)** | Model is too simple — like a student who only learns "all four-legged animals are dogs" | Poor on both training and test data |
| **Overfitting (high variance)** | Model memorizes training noise word-for-word instead of learning patterns | 100% on practice, fails on new data |

Healthy training: both train and validation loss drop, then flatten. Overfitting: train keeps falling while validation rises. Underfitting: both stay high.

## How it works

**Batch vs epoch.** Suppose 10,000 examples and batch size 100. Each epoch has 100 update steps. After 20 epochs you have taken 2,000 gradient steps — but each example has been seen about 20 times.

**Why shuffle.** Randomizing order each epoch reduces the chance that a weird contiguous slice of data biases every update the same way.

**Learning rate schedules (awareness).** Many runs start with a constant learning rate, then decay it (step decay, cosine). Warmup starts small and ramps up for large-batch transformer training. For this lesson, a carefully chosen constant learning rate is enough.

| Learning rate | Typical symptom |
| --- | --- |
| Much too high | Loss → NaN or oscillates upward |
| Slightly high | Fast early drop, then unstable plateaus |
| Good | Smooth decrease of train (and val) loss |
| Too low | Tiny slope; needs huge epoch count |

```mermaid
flowchart TD
    A[Init parameters] --> B[Shuffle / next batch]
    B --> C[Forward: predict]
    C --> D[Compute loss]
    D --> E[Backward: gradients]
    E --> F["Update: weights = weights - lr * gradients"]
    F --> G{More batches in epoch?}
    G -->|yes| B
    G -->|no| H[End of epoch: log train and val loss]
    H --> I{More epochs?}
    I -->|yes| B
    I -->|no| J[Done / checkpoint]
```

**Monitoring.** Log train loss every N steps and validation loss every epoch. Plot both. Early stopping reads this plot automatically. Watch for exploding loss after a learning rate change — that is your first debugging signal.

## In code

Toy linear regression with mini-batch gradient descent. We fit `y ~= w * x + b` on noisy data and watch loss fall.

```python
import numpy as np

rng = np.random.default_rng(42)

n = 200
x = rng.normal(size=n)
y = 3.0 * x - 1.0 + rng.normal(scale=0.3, size=n)

idx = rng.permutation(n)
train_idx, val_idx = idx[:160], idx[160:]
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def forward(x, w, b):
    return w * x + b

w, b = 0.0, 0.0
lr = 0.05
epochs = 40
batch_size = 32

history = []
for epoch in range(epochs):
    perm = rng.permutation(len(x_train))
    x_epoch, y_epoch = x_train[perm], y_train[perm]

    for start in range(0, len(x_train), batch_size):
        xb = x_epoch[start:start + batch_size]
        yb = y_epoch[start:start + batch_size]

        pred = forward(xb, w, b)
        err = pred - yb
        dw = 2.0 * np.mean(err * xb)
        db = 2.0 * np.mean(err)
        w -= lr * dw
        b -= lr * db

    train_loss = mse(y_train, forward(x_train, w, b))
    val_loss = mse(y_val, forward(x_val, w, b))
    history.append((epoch, train_loss, val_loss))

print(f"learned w={w:.3f}, b={b:.3f} (true ~ 3, -1)")
print("epoch | train_mse | val_mse")
for epoch, tr, va in history[::8]:
    print(f"{epoch:5d} | {tr:9.4f} | {va:7.4f}")
```

Try `lr = 1.0` and watch loss explode. Try `lr = 1×10⁻⁵` and notice almost no movement after many epochs.

## What goes wrong

- **Silent overfitting** — Train loss looks great because you only plotted it. Always reserve a validation set and compare curves.
- **Epoch worship** — "Train for 100 epochs" is meaningless without looking at validation loss. Sometimes 15 epochs with early stopping beats 100.
- **Batch size side effects** — Larger batches give stabler gradients but fewer steps per epoch; the same learning rate may need retuning.
- **Forgetting to zero gradients** — In autograd frameworks, gradients accumulate by default. Skip `optimizer.zero_grad()` and you train on summed ghosts of past batches.
- **Data leakage in validation** — Preprocessing fit on train+val (e.g. scaling) makes validation loss optimistic. Fit scalers on train only.
- **Shuffling validation** — Optional for metrics; never required. Do shuffle training.

## One-line summary

The training loop is forward → loss → backward → update, repeated over batches and epochs — with learning rate setting step size and train/val curves telling you whether you are learning or memorizing.

## Key terms

- **Training loop** — Repeated forward, loss, backward, and parameter update.
- **Learning rate** — Step size in gradient descent updates.
- **Epoch** — One full pass over the training dataset.
- **Batch / mini-batch** — Subset of examples used for one update.
- **Gradient** — Vector of partial derivatives of loss with respect to parameters.
- **Validation loss** — Loss on held-out data used to monitor generalization.
- **Convergence** — Loss settling near a minimum rather than diverging or wandering.
