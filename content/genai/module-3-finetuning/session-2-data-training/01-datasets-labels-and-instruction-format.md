---
title: "Datasets, Labels, and Instruction Format"
description: "How to shape SFT data: instruction/response pairs, chat templates, labels, quality filters, and leakage traps."
---

Fine-tuning quality is mostly dataset quality. A mediocre base model with clean, consistent examples often beats a strong model trained on noisy Slack dumps. This lesson covers how engineers structure labels and instruction formats so training teaches the behavior you intend.

## Intuition

Supervised fine-tuning (SFT) teaches: "given this instruction (and context), produce that completion." Every row is a tiny contract:

- **Input side** — user request, system rules, optional retrieved context, tools results.
- **Output side** — the ideal assistant reply (and sometimes tool calls).
- **Format** — how roles and special tokens wrap those fields so the tokenizer and chat template agree.

:::key
The model learns the distribution of your targets. Inconsistent labels train inconsistency.
:::

If half your rows say "return bare JSON" and half wrap answers in prose, the model will randomly pick a style. Labels are not just "correct answers"—they are the style guide baked into weights.

## How it works

### Common record shapes

**Instruction / response (Alpaca-style):**

```text
instruction: Summarize the ticket in one sentence.
input: User cannot reset MFA after phone change.
output: User blocked on MFA reset after updating their phone number.
```

**Chat messages (preferred for chat models):**

```json
{
  "messages": [
    {"role": "system", "content": "You are a support triage bot. Reply in JSON."},
    {"role": "user", "content": "I can't log in after the MFA change."},
    {"role": "assistant", "content": "{\"intent\":\"auth\",\"priority\":\"P2\"}"}
  ]
}
```

Chat format matters because production inference uses the same **chat template** (special tokens for roles). Train the way you serve.

### What "label" means here

For causal LMs, the usual SFT loss is next-token prediction on the assistant tokens. You often **mask** system/user tokens so the loss focuses on what the assistant should say. Conceptually:

```text
loss = -mean(log p(token_t | tokens_<t))  for t in assistant_span
```

Bad labels (wrong JSON, leaked PII, contradictory policies) pull probability mass toward those mistakes.

### Quality bar before training

| Check | Why |
| --- | --- |
| Schema validity | Invalid JSON teaches invalid JSON |
| Single style guide | Mixed tone/format increases entropy |
| Deduplication | Near-duplicates inflate overfit |
| PII / secrets scrub | Models can memorize and regurgitate |
| Train / eval split by user or ticket | Random split leaks near-duplicates |
| Difficulty mix | Only easy rows → brittle on live mess |

Aim for **consistency over volume** early. A few hundred excellent rows beat ten thousand noisy ones for many format/style tasks. Scale data once the format is locked.

### Mixing sources

- **Gold human writes** — highest trust; expensive.
- **Model-assisted drafts + human edit** — fast, but watch for inherited quirks.
- **Production logs with human fixes** — realistic; scrub PII and filter low-effort replies.
- **Synthetic expansion** — useful for coverage; always human-spot-check a sample.

:::tip
Freeze a `SYSTEM_PROMPT` and output schema in a shared file. Generate training rows and prod prompts from the same source of truth.
:::

## In code

Validate and lightly normalize JSONL before any trainer sees it.

```python
import json
from pathlib import Path


REQUIRED_ROLES = {"system", "user", "assistant"}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def validate_chat_row(row: dict) -> list[str]:
    errs = []
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return ["messages must be a list with >= 2 turns"]
    roles = [m.get("role") for m in msgs]
    if roles[-1] != "assistant":
        errs.append("last message must be assistant (the label)")
    for m in msgs:
        if m.get("role") not in REQUIRED_ROLES:
            errs.append(f"bad role: {m.get('role')}")
        if not str(m.get("content", "")).strip():
            errs.append("empty content")
    # Example task rule: assistant must be JSON object
    try:
        json.loads(msgs[-1]["content"])
    except Exception:
        errs.append("assistant content is not valid JSON")
    return errs


# Illustrative rows in memory (same shape as a .jsonl file)
demo = [
    {
        "messages": [
            {"role": "system", "content": "Reply with JSON only."},
            {"role": "user", "content": "Password reset loop"},
            {"role": "assistant", "content": '{"intent":"auth","priority":"P2"}'},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},  # fails JSON rule
        ]
    },
]

for i, row in enumerate(demo):
    print(i, validate_chat_row(row) or "ok")
```

HuggingFace-style packaging (pseudocode):

```python
# from datasets import Dataset
# ds = Dataset.from_list(clean_rows)
# def to_text(example, tokenizer):
#     return tokenizer.apply_chat_template(
#         example["messages"], tokenize=False, add_generation_prompt=False
#     )
# texts = [to_text(r, tokenizer) for r in ds]
```

## What goes wrong

- **Train/serve mismatch** — Training on plain `"### Instruction"` text then serving with a Llama/ChatML template undoes the lesson.
- **Label leakage in eval** — Near-duplicate tickets in train and test inflate metrics; split by conversation id or time.
- **Hidden instructions in user text** — Unfiltered user content that contradicts system policy teaches the model to ignore the system role.
- **Over-cleaning** — Removing all hard cases makes train loss look great and production look worse.
- **Unlabeled multi-task soup** — Mixing summarization, coding, and SQL without task tags confuses the model unless volume is huge and balanced.

:::warn
Never put API keys, customer PII, or internal secrets into SFT corpora. Memorization is a feature of gradient descent, not a bug you can ignore.
:::

## One-line summary

Structure SFT data as consistent instruction or chat turns, put the quality bar on assistant labels, and train with the same template you will use at serving time.

## Key terms

- **SFT dataset** — Labeled instruction/response or chat examples for supervised fine-tuning.
- **Chat template** — Tokenizer-specific wrapping of roles into a single training string.
- **Label / target** — The assistant tokens the loss is computed on.
- **Loss masking** — Ignoring user/system tokens so gradients focus on assistant outputs.
- **JSONL** — Line-delimited JSON, common for training corpora.
- **Gold label** — Human-approved target response.
- **Data leakage** — Eval contamination or secrets appearing in training text.
