---
title: "Module 1 - Foundations revision"
slug: module-1-foundations
module: "Module 1"
minutes: 30
description: "AI to DL to transformers in one sitting. Re-read before Module 1 mock."
---

Dense bullets for chapters **1.1–1.3**. Open full lessons when a bullet feels fuzzy.

## 1.1 AI & Deep Learning Essentials

### Stack of ideas
- **AI** — goal-directed behavior under uncertainty (rational agent: observe -> act -> succeed).
- **ML** — learn parameters from data instead of hand-coding every rule.
- **Deep learning** — ML with multi-layer neural nets that learn hierarchical features.
- **GenAI** — models that *sample* new content (text, images, code) from a learned distribution.

### Classical vs learned
- Rules / search / expert systems scale poorly in noisy high-dimensional worlds (vision, speech, open language).
- Supervised learning: labeled `(x, y)`. Unsupervised: structure in `x`. RL: actions + reward.

### Representation
- **One-hot** — sparse, no similarity. **Embeddings** — dense vectors; nearby = similar meaning/use.
- Word2Vec / GloVe intuition: “you shall know a word by the company it keeps.”

### Neuron to MLP
- Linear: `z = w*x + b`. Nonlinearity (ReLU, sigmoid, tanh) makes depth useful.
- **MLP** = stacked linear + activation. Universal approximator in theory; practice needs data, regularization, right capacity.
- **Backprop** = chain rule through the graph. **Gradient descent**: update weights by subtracting `learning_rate * gradient_of_loss`.
- Overfit vs underfit; bias-variance tradeoff. Fix overfit: more data, dropout, weight decay, early stop. Fix underfit: capacity / train longer / better features.

### Quick self-check
- Can you place ChatGPT inside AI > ML > DL > GenAI?
- Why does ReLU help vanishing gradients vs deep sigmoids?

## 1.2 Deep Learning Essentials

### Training loop
- Batch -> forward -> loss -> backward -> optimizer step. Epoch = one pass over data.
- **Learning rate** too high -> diverge; too low -> crawl. Schedules and warmups matter for big models.
- Train / val / test split. Never tune on test.

### CNN (vision intuition)
- Convolution shares filters spatially; translation equivariance; local receptive fields.
- Pooling / strides reduce spatial size. Depth stacks simple edges -> parts -> objects.

### RNN / LSTM (sequence intuition)
- Hidden state carries past tokens. Vanilla RNNs struggle with long dependencies (vanishing/exploding gradients).
- **LSTM/GRU** gates control what to keep/forget — still sequential -> hard to parallelize -> transformers win for most NLP.

## 1.3 Transformer Architecture

### Why transformers
- Parallel over sequence (vs RNN step-by-step). Attention = soft lookup of relevant context.
- Scales with data + compute; foundation of modern LLMs.

### Self-attention (Q, K, V)
- For each position: query asks “what do I need?”, keys answer “what do I contain?”, values carry content.
- Scores: `softmax(Q * K^T / sqrt(d_k)) * V`. Divide by `sqrt(d_k)` so softmax stays well-behaved as dimension grows.
- **Multi-head**: several subspaces in parallel (syntax vs semantics, etc.).

### Positional encoding
- Attention is permutation-invariant without positions. Add / inject position (sinusoidal, learned, RoPE in modern LLMs).

### Decoder stack (GPT-style)
- **Causal mask** — position `i` cannot attend to later positions (autoregressive next-token training).
- Cross-attention (encoder-decoder MT models): decoder queries encoder keys/values. Decoder-only LLMs skip a separate encoder.

### Autoregressive decoding
- Train: predict next token given left context. Infer: sample / greedy / beam / nucleus (`top-p`) / temperature.
- Higher temperature -> flatter distribution (more random). `top-p` keeps the smallest set of tokens whose probabilities sum to at least `p`.

### Must-remember diagram in your head
`tokens -> embed+pos -> [N x (MHA + FFN + residual + LN)] -> logits -> next token`

## 30-minute drill
1. Explain attention to a rubber duck without jargon for 2 minutes.
2. Sketch one transformer block (residual + LN placement).
3. Name three failure modes of deep nets (overfit, vanishing grad, poor LR) and one fix each.
