---
title: "Decoding Parameters and Output Control"
description: "Master temperature, top-k, top-p, penalties, max tokens, and stop sequences so LLM outputs stay useful under production constraints."
---

Sampling knobs are how you turn a probability distribution into a product behavior. The same prompt can yield a crisp JSON object or a meandering poem depending on decoding. Engineers who treat these as afterthoughts spend their weeks chasing “flaky” models that were never configured for the job.

## Intuition

At each step the model scores the whole vocabulary. Decoding is the policy that picks one token:

- **Greedy / near-greedy** — take the best (or nearly best) option -> stable, repetitive risk.
- **Truncated sampling** — only consider a shortlist (top-k or nucleus / top-p) -> controlled variety.
- **Penalties and stops** — reshape scores or force an exit -> fight loops and enforce boundaries.

Temperature changes the shape of the distribution; top-k / top-p change which slice you sample from; max tokens and stop sequences decide when the loop ends. Latency and cost scale with how long you let that loop run.

:::key
Decode for the contract: factual extraction wants a short leash; ideation wants room. Defaults are rarely optimal for either.
:::

## How it works

### Core parameters

| Parameter | Controls | Notes |
| --- | --- | --- |
| temperature | Sharpness of softmax | Primary diversity dial |
| top_k | Keep only k highest-prob tokens | Hard shortlist |
| top_p | Keep nucleus with mass >= p | Adaptive shortlist |
| frequency / presence penalty | Discourage reused tokens | Helps long prose; can hurt code |
| max_tokens / max_output | Hard length cap | Cost + latency ceiling |
| stop sequences | End when a string appears | Great for delimiters and turn ends |

### How the pieces compose

1. Model emits logits `z`.
2. Optional penalties adjust logits for tokens already used.
3. Temperature scales: `z' = z / T`.
4. Softmax -> probabilities.
5. Top-k and/or top-p mask the allowed set.
6. Sample (or argmax) one token; append; repeat until EOS, stop string, or max length.

```mermaid
flowchart LR
    L[Logits] --> P[Penalties]
    P --> T[Temperature]
    T --> S[Softmax]
    S --> K[Top-k / Top-p mask]
    K --> C[Choose token]
    C --> L
```

### Top-k vs top-p

- **Top-k = 40** always keeps forty tokens even if thirty-nine are near-zero junk when the model is confident — or truncates useful mass when the distribution is flat.
- **Top-p = 0.9** grows/shrinks with uncertainty — usually the better default when you want one adaptive knob.

Using both is fine if you understand the order your API applies them; setting both extremely tight (tiny k and tiny p) can leave an empty or brittle candidate set.

### Penalties

- **Presence penalty** — punish tokens that appeared at least once -> pushes topic diversity.
- **Frequency penalty** — punish in proportion to count -> fights “the the the” loops.

Overdoing penalties on code or legal text makes identifiers and defined terms drift. Prefer clearer prompts and lower temperature before cranking penalties.

### Stops and max tokens

Stop sequences are surgical: end at `\n\nUser:`, ` ``` `, or `</json>`. Max tokens is a blunt instrument: it prevents runaway cost but can cut mid-JSON. Pair them: stop on a closer when possible; set max as a safety net.

### Presets that ship

- **Factual Q&A / extraction:** T 0.0–0.3, top_p 0.2–0.7, modest max_tokens, optional stop after schema close.
- **Support tone:** T 0.3–0.5, top_p ~0.8, presence penalty light if replies repeat.
- **Code:** T 0.1–0.3, avoid heavy frequency penalty, stop on fences if you wrap prompts that way.
- **Brainstorm:** T 0.7–1.0, top_p 0.9–0.95, larger max_tokens, accept more variance.

## In code

Toy decode step with temperature + top-k:

```python
import math
import random


def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    z = sum(exps)
    return [e / z for e in exps]


def apply_temperature(logits, T: float):
    return [x / T for x in logits]


def top_k_mask(probs, k: int):
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
    allowed = set(order)
    masked = [probs[i] if i in allowed else 0.0 for i in range(len(probs))]
    s = sum(masked) or 1.0
    return [p / s for p in masked]


def sample(logits, T=0.8, k=3):
    probs = softmax(apply_temperature(logits, T))
    probs = top_k_mask(probs, k)
    return random.choices(range(len(probs)), weights=probs, k=1)[0]


random.seed(1)
vocab = ["yes", "no", "maybe", "unknown"]
logits = [2.2, 1.1, 0.4, -0.5]
print(vocab[sample(logits, T=0.2, k=2)])
```

Stop-sequence trimming for structured replies:

```python
def apply_stops(text: str, stops: list[str]) -> str:
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


raw = '{"ok": true}\n\nUser: ignore'
print(apply_stops(raw, ["\n\nUser:", "\n\n"]))
```

Frequency penalty sketch:

```python
def penalize(logits, token_counts: list[int], penalty: float):
    # subtract penalty * count from each logit
    return [z - penalty * c for z, c in zip(logits, token_counts)]
```

Config object you can version in experiments:

```python
PRESETS = {
    "extract": {"temperature": 0.1, "top_p": 0.3, "max_tokens": 256},
    "ideate": {"temperature": 0.9, "top_p": 0.95, "max_tokens": 800},
}
```

## What goes wrong

- **Creative defaults on extraction jobs** — enums become synonyms; parsers die.
- **Max tokens too low for JSON** — truncated objects look like model stupidity.
- **Stop sequence that appears inside legitimate content** — early cut on code comments or URLs.
- **Penalty wars** — code variables renamed mid-answer; legal terms avoided.
- **Tuning k and p blind** — no golden eval set, so “improvements” are vibes.
- **Ignoring provider-specific defaults** — a new SDK version changes temperature default and your regression suite was empty.

:::tip
Store decoding presets next to prompts in source control. When quality drifts, diff the config before you rewrite the entire prompt.
:::

## One-line summary

Decoding parameters are the runtime policy over next-token probabilities — tune temperature, truncation, penalties, and stops to match stability, cost, and format needs.

## Key terms

- **Decoding** — Algorithm that turns model scores into chosen tokens.
- **Top-k** — Restrict sampling to the k most likely tokens.
- **Top-p** — Restrict sampling to a probability nucleus.
- **Repetition / frequency penalty** — Score adjustment against reused tokens.
- **Presence penalty** — Penalty for any token that already appeared.
- **Max tokens** — Upper bound on generated length.
- **Stop sequence** — String that forces generation to end when emitted or matched.
- **Greedy decoding** — Always pick the highest-probability next token.
