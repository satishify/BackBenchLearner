---
title: "Self-Attention: Queries, Keys, and Values"
description: "What query, key, and value (Q/K/V) mean, how scaled dot-product attention works, and a tiny NumPy walkthrough."
---

**What this is:** When an LLM resolves "it" in a sentence or latches onto a constraint in your system prompt, it is running **self-attention**. Each token builds a query ("what do I need?"), compares it to other tokens' keys ("what do I offer?"), and mixes their values ("what content should flow?").

**Why it matters:** This block is the core compute inside almost every modern GenAI model. Master it and the rest of the Transformer stack becomes much clearer.

## Intuition

### The YouTube search analogy

| Role | Plain-English idea | YouTube example |
| --- | --- | --- |
| **Query (Q)** | "What am I searching for?" | What you type in the search bar |
| **Key (K)** | "What is my identity label?" | Video titles and tags the server checks |
| **Value (V)** | "What core features do I hold?" | The actual video content you watch |

The platform matches your query against all keys, turns match scores into a ranking (softmax), and returns a blend of video values weighted by relevance.

### A sentence example

Sentence: *"The chef cooked the food because it was delicious."*

To figure out what **"it"** refers to:

- **Query for "it":** "I am a pronoun. What noun in this sentence do I describe?"
- **Keys for all words:** "chef" → animate actor; "cooked" → action; "food" → edible noun
- **Values for all words:** The actual semantic content of each word

The query for "it" multiplies against all keys. "chef" gets a weak match (chefs aren't usually "delicious"). "food" gets a strong match. Softmax puts ~85% focus on "food". Multiplying that weight distribution by the values pulls "food" semantics into the updated representation of "it".

:::key
**Self-attention** means Q, K, and V all come from the **same** sequence (after learned linear projections). Every position can refine itself using the whole context — the reason transformers handle long-range agreement better than local word-pattern heuristics.
:::

## How it works

Start with token embeddings `X` of shape `(n, d_model)` — `n` tokens, each a `d_model`-vector. Three learned weight matrices produce:

```
Q = X * W_Q    # (n, d_k)  — "what am I looking for?"
K = X * W_K    # (n, d_k)  — "what do I offer for matching?"
V = X * W_V    # (n, d_v)  — "what content do I carry?"
```

Often `d_v = d_k`. Pairwise scores are dot products between queries and keys:

```
scores_ij = (Q_i * K_j) / sqrt(d_k)
```

Divide by `sqrt(d_k)` so that as dimension grows, scores do not explode into huge magnitudes that push softmax into tiny gradients (**softmax saturation**).

Then the full attention formula:

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

Row `i` of the softmax matrix is a probability distribution over positions that token `i` will read from. Multipying by `V` yields a new vector for each token: a soft retrieval of information from the sequence.

| Step | What happens |
| --- | --- |
| **1. Project** | Each token becomes Q, K, and V via learned matrices |
| **2. Score** | Every query dot-products with every key → compatibility scores |
| **3. Scale** | Divide by `sqrt(d_k)` to keep scores stable |
| **4. Softmax** | Turn scores into weights that sum to 1 per row |
| **5. Mix** | Weighted sum of values → updated token representation |

**Complexity:** Computing `Q * K^T` is O(n² * d_k). That quadratic term is why context-window length dominates cost — "128k context" is an engineering story, not a free lunch.

```mermaid
flowchart LR
  X[Token embeddings X] --> Q[Q = X * W_Q]
  X --> K[K = X * W_K]
  X --> V[V = X * W_V]
  Q --> S["Scores = Q * K^T / sqrt(d_k)"]
  K --> S
  S --> W[Softmax weights]
  W --> O[Weighted sum of V]
  V --> O
```

In stacked layers, the input to attention is already a contextualized residual stream. Early layers often mix local syntax; deeper layers mix longer semantic links. Frameworks add batch and head dimensions — the math per sequence stays the same; GPUs shine because every query–key score is a dense matrix multiply.

## In code

A complete single-head attention forward pass on a toy sequence:

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

rng = np.random.default_rng(1)
n, d_model, d_k = 4, 8, 8
X = rng.normal(size=(n, d_model))
W_Q = rng.normal(size=(d_model, d_k)) * 0.2
W_K = rng.normal(size=(d_model, d_k)) * 0.2
W_V = rng.normal(size=(d_model, d_k)) * 0.2

Q, K, V = X @ W_Q, X @ W_K, X @ W_V
scores = (Q @ K.T) / np.sqrt(d_k)
weights = softmax(scores, axis=-1)
out = weights @ V

print("weights row 0 (sum~1):", np.round(weights[0], 3), "sum", weights[0].sum())
print("output shape", out.shape)  # (n, d_k)
```

Row sums near 1.0 confirm a valid attention distribution. Change `X[0]` drastically and watch row-0 weights shift — that is content-based routing in miniature.

## What goes wrong

- **Skipping the scale.** Without `/ sqrt(d_k)`, high-dimensional dot products saturate softmax; gradients shrink and training stalls.
- **Confusing Q/K/V roles.** Q is "what I ask"; K is "how I am indexed"; V is "what I contribute." Swapping them in an explanation is a common interview miss.
- **Thinking softmax picks one token.** Softmax is soft: many positions can share weight. A hard argmax would destroy gradient flow and nuance.
- **Numerical instability.** Softmax should subtract the row max (as in the code) before `exp` to avoid overflow.
- **Assuming one head sees everything.** A single head has limited subspace; multi-head attention (next lesson) fixes that by running several projections in parallel.

## One-line summary

Self-attention updates each token by softmax-weighted retrieval over values, with weights from scaled query–key dot products on the same sequence.

## Key terms

- **Query (Q)** — Projection used to ask for relevant context.
- **Key (K)** — Projection used to be matched against queries.
- **Value (V)** — Projection mixed into the output when attended.
- **Scaled dot-product attention** — `softmax(Q * K^T / sqrt(d_k)) * V`.
- **Attention weights** — Row-wise softmax distribution over positions.
- **d_k** — Key/query dimension; sets the scale factor.
- **Self-attention** — Q/K/V derived from the same token sequence.
- **Softmax saturation** — When large score values flatten gradients near zero, stalling learning.
