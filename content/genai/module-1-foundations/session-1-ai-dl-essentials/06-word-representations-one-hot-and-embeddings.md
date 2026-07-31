---
title: "Word Representations — One-Hot and Embeddings"
description: "Why one-hot vectors cannot share meaning, how learned embeddings place similar words nearby, and a tiny NumPy lookup that foreshadows every LLM token table."
---

Machine learning models cannot read raw text — they need numbers. **One-hot** vectors and **embeddings** are the two classic ways to turn words into numbers. This topic exists because models need a numeric form they can actually compute on.

- **One-hot** gives each word a long vector that is all zeros except one 1 — simple but no notion of similarity.
- **Embeddings** are short, dense vectors where similar words sit close together.
- In one-hot, "cat" and "dog" look as unrelated as "cat" and "quantum."
- In embeddings, similar words end up nearby — so king - man + woman is approximately queen.

## Intuition

Imagine a dictionary with 10,000 entries. A **one-hot** encoding gives each word a 10,000-dimensional vector that is all zeros except a single `1` at that word's index. "Dog" and "puppy" look as unrelated as "dog" and "quantum" — their dot product is zero. The model must rediscover every synonym from scratch through later weights.

An **embedding** is a short dense vector (say 64 or 768 numbers) looked up by word ID. Training nudges those vectors so words that appear in similar contexts drift together. Distance and direction start to mean something: analogies, clustering of sentiment words, and smoother generalization to rare synonyms. You are no longer storing a flag; you are storing a **point in meaning-space**.

Think of one-hot as a locker number and embedding as a GPS coordinate for the idea behind the word. Lockers don't know which neighbors are related; GPS does.

## How it works

**Vocabulary and IDs.** Map each unique token to an integer `0 ... V-1`. Unknown words often share an `<unk>` id.

**One-hot.** For token id `i` and vocab size `V`:

```
one_hot[j] = 1 if j == i else 0
```

Length is always `V`. Sparse, high-dimensional, no learned parameters in the encoding itself.

**Embedding lookup.** Maintain a matrix `E` of shape `(V, d)` where `d` is the embedding dimension. The representation of token `i` is simply row `i`:

```
x = E[i]          # shape (d,)
```

During training, gradients update `E[i]` (and only rows that appear in the batch). Downstream layers see dense vectors; the one-hot is never materialized in practice — lookup *is* "multiply by a one-hot" in matrix form.

**One-hot vs embeddings:**

| Plain-English idea | When to use it |
| --- | --- |
| **One-hot** — sparse code; every word is an orthogonal spike with no shared meaning | Baseline encoding, tiny vocabularies, teaching the concept |
| **Embeddings** — dense code; similar words end up close in vector space | Semantic tasks, language models, any modern NLP pipeline |

**Why geometry helps.** If two rows of `E` are close (high **cosine similarity**), linear layers and attention treat those tokens more alike. Cosine similarity measures how aligned two vectors are:

```
cos(theta) = dot(A, B) / (||A|| * ||B||)
```

That is the seed of transfer: learning about *run* helps with *running*.

## In code

```python
import numpy as np

vocab = {"the": 0, "cat": 1, "sat": 2, "mat": 3, "<unk>": 4}
V, d = len(vocab), 8
rng = np.random.default_rng(0)
E = rng.normal(0, 0.1, size=(V, d))  # embedding table

def one_hot(token_id, V):
    v = np.zeros(V)
    v[token_id] = 1.0
    return v

def embed(token, E, vocab):
    tid = vocab.get(token, vocab["<unk>"])
    return E[tid]  # same as one_hot(tid) @ E

print("one-hot 'cat':", one_hot(vocab["cat"], V))
print("embed 'cat' shape:", embed("cat", E, vocab).shape)
print("dot(one-hot cat, one-hot sat):",
      one_hot(1, V) @ one_hot(2, V))  # always 0 for distinct words
```

In frameworks this is `nn.Embedding(V, d)`. LLMs use the same idea at token (subword) level, not whole words.

## What goes wrong

**Huge one-hots in the forward pass.** Materializing `V`-length vectors for every position wastes memory; always use embedding lookup.

**Frozen random embeddings.** If you never train `E` (and have no pretrained table), later layers get arbitrary coordinates — learning is much harder.

**Vocab mismatch.** A token id from a different tokenizer indexes the wrong row. Always keep tokenizer and embedding table paired.

**Thinking embeddings "are" meaning.** They encode distributional similarity from data, including bias and corpus quirks — not a clean dictionary definition.

## One-line summary

One-hot marks which word you mean; embeddings learn where that word lives in a dense space so similar words can share structure.

## Key terms

- **Vocabulary** — Ordered set of tokens the model knows, each with an integer id.
- **One-hot** — Sparse vector with a single 1 at the token's index.
- **Embedding** — Dense learned vector for a token, usually a row in a lookup table.
- **Embedding dimension (`d`)** — Length of each dense vector.
- **Lookup table / embedding matrix** — Parameter tensor of shape `(V, d)` indexed by token id.
- **Distributional hypothesis** — Words in similar contexts tend to get similar embeddings.
- **Cosine similarity** — Measure of how aligned two vectors are, used to compare word meanings.
