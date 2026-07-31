---
title: "What is Machine Learning (ML)"
description: "Machine learning basics: learn from data, supervised/unsupervised/RL, regression vs classification, train/test split, and a tiny closed-form line fit."
---

Machine learning (ML) is how most modern AI systems learn: instead of writing every rule by hand, you show the model examples and it picks up the pattern. This shift exists because many real-world problems are too messy to describe with manual rules alone.

- You give the model **inputs** (features) and often **labels** (correct answers).
- The model adjusts its **parameters** so predictions match reality.
- You check it on **held-out data** it has never seen — not just training examples.
- The same basic loop covers simple lines and billion-parameter neural networks.

## Intuition

You already do a form of machine learning every day. After enough rainy mornings, you bring an umbrella without deriving meteorology from first principles. Your brain updates from **experience**. In software, experience is data: features (inputs) and, often, labels (desired outputs).

Three questions organize almost every ML project:

1. What am I predicting or discovering?
2. What examples do I have?
3. How will I know the model works on data it has never seen?

If you cannot answer (3), you are not doing ML engineering — you are fitting a curve and hoping.

## How it works

**Learning from data.** A model is a function with adjustable parameters. Training searches for parameter values that make predictions match reality — usually by minimizing a **loss** (squared error, cross-entropy, etc.). After training, you run **inference** on new inputs. The same skeleton covers linear regression and large neural nets; scale and architecture change, the loop does not.

**Three broad paradigms:**

| Plain-English idea | When to use it |
| --- | --- |
| **Supervised learning** — inputs plus labels; predict labels for new inputs | Spam filters, credit scoring, image triage |
| **Unsupervised learning** — inputs only; find hidden structure | Customer segments, anomaly hints, compression |
| **Reinforcement learning (RL)** — actions plus rewards over time; learn a policy | Game-playing agents, some robotics and recommendation loops |

Semi-supervised and self-supervised methods blur the lines: they use lots of unlabeled data plus a pretext task (predict the next token, fill in a masked image) to learn useful representations.

**Regression vs classification (supervised).**

| Plain-English idea | When to use it |
| --- | --- |
| **Regression** — predict a continuous number (price, temperature, latency) | Forecasting quantities |
| **Binary classification** — predict one of two labels (spam/ham, disease/no disease) | Yes/no decisions |
| **Multi-class classification** — predict one of many labels (which digit, which topic) | Picking among several categories |

Same idea — map input to output — different output type and usually different loss.

**Train / test split.** If you grade a student only on questions they memorized, you overestimate skill. Likewise, if you evaluate a model on the same rows used to fit it, you overestimate generalization. Hold out a **test set** (and often a **validation set** for tuning) so metrics reflect performance on unseen data. Touch the test set once for final reporting; use validation for model selection.

```mermaid
flowchart LR
  D[Full dataset] --> TR[Train set]
  D --> VA[Validation set]
  D --> TE[Test set]
  TR --> FIT[Fit parameters]
  VA --> TUNE[Tune / select model]
  FIT --> TUNE
  TUNE --> EVAL[Final evaluate]
  TE --> EVAL
```

## In code

Fit a line with the closed-form least-squares solution (stdlib only), then compare to a naive average baseline. No scikit-learn.

```python
# Predict y from x: y ~= a + b*x  (ordinary least squares, 1-D)
xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [2.1, 3.9, 6.2, 7.8, 10.1]

n = len(xs)
mean_x = sum(xs) / n
mean_y = sum(ys) / n

# Closed form: b = Cov(x,y) / Var(x); a = mean_y - b * mean_x
var_x = sum((x - mean_x) ** 2 for x in xs)
cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
b = cov_xy / var_x
a = mean_y - b * mean_x

def predict(x: float) -> float:
    return a + b * x


# Average baseline: always predict mean_y (ignores x)
baseline = mean_y

# Hold out last point as a tiny "test" example
x_test, y_test = xs[-1], ys[-1]
err_model = (predict(x_test) - y_test) ** 2
err_base = (baseline - y_test) ** 2

print(f"fit: y = {a:.3f} + {b:.3f}*x")
print(f"test MSE model={err_model:.3f} baseline={err_base:.3f}")
```

Even this toy fit shows the ML loop: choose a hypothesis class (a line), fit on train data, compare to a baseline, evaluate on held-out points. Real projects add more features, regularization, cross-validation, and careful splits — the skeleton stays the same. Always ship a baseline first; if a fancy model cannot beat "predict the mean," stop and fix the data or the problem statement.

## What goes wrong

- **Data leakage.** Using future information or the label itself as a feature makes test scores look amazing and production fail.
- **Wrong split.** Random splits on time-ordered or grouped data (same user in train and test) inflate metrics.
- **Imbalanced classes.** 99% accuracy can hide a useless minority-class detector; pick metrics that match the decision (precision, recall, F1).
- **Overfitting.** A model that memorizes noise on the train set will not travel. Prefer simpler models, more data, or regularization when train >> test performance.
- **Paradigm mismatch.** Supervised labels may be expensive; unsupervised clusters may not match business labels; RL needs careful reward design or it will game the metric.
- **Silent distribution shift.** The world changes after launch; without monitoring, yesterday's good model becomes tomorrow's quiet failure.

## One-line summary

Machine learning fits models from data so systems generalize to new inputs — via supervised, unsupervised, or reinforcement paradigms, evaluated on held-out data.

## Key terms

- **Machine learning (ML)** — improving task performance by learning patterns from data or feedback
- **Supervised learning** — training with labeled examples
- **Unsupervised learning** — discovering structure without labels
- **Reinforcement learning (RL)** — learning actions from reward signals over time
- **Regression** — predicting continuous targets
- **Classification** — predicting discrete class labels
- **Train / validation / test split** — partitions for fitting, tuning, and unbiased evaluation
- **Baseline** — simple reference model used to judge whether learning is worthwhile
- **Generalization** — performance on data not seen during training
- **Loss function** — scalar objective that training tries to minimize
