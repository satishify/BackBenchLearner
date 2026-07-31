---
title: "LLM Evolution and Model Families"
description: "How foundation models evolved from BERT and GPT through instruction tuning, open weights, multimodality, and reasoning — and how to choose a family for a product job."
---

If you only remember one story about modern GenAI, make it this: we went from "finetune a classifier per task" to "prompt a general sequence model," then to "align it with humans," then to "wire it to tools and other modalities." Model names change every quarter; the family shapes below stay useful for years.

## Intuition

**What changed over time?** Three different jobs for language models:

1. **Understand** text you already have (classify, extract, embed for search).
2. **Generate** new text conditioned on a prompt (chat, code, summaries).
3. **Transform** input into a different output sequence (translate, rewrite under a schema).

| Architecture family | Plain-English idea | Best at |
| --- | --- | --- |
| **Encoder-only (BERT-like)** | Reads the whole input at once, both directions | Classification, extraction, embeddings |
| **Decoder-only (GPT-like)** | Reads left-to-right, generates one token at a time | Chat, code, agents, summarization |
| **Encoder–decoder (T5/BART-like)** | Reads full input, writes full output | Translation, structured rewrite |

Everything else — RLHF (Reinforcement Learning from Human Feedback), multimodal towers, "reasoning" post-training — is a refinement on top of one of these skeletons.

:::key
Pick the architecture family for the job shape first; pick a vendor or checkpoint second. Marketing names change; encoder / decoder / encoder–decoder do not.
:::

## How it works

### Pretrain, then specialize

Modern foundation models share a two-phase life:

- **Pretraining** — optimize a self-supervised objective on massive corpora (masked tokens, next tokens, or denoising). This builds general representations and fluency.
- **Post-training** — supervised instruction data, preference optimization (RLHF / DPO-style methods), safety filters, and sometimes tool-use or chain-of-thought traces. This turns a raw predictor into a product-shaped assistant.

### Milestone map (engineer view)

| Era | Example shape | What changed for builders |
| --- | --- | --- |
| Encoder boom | BERT-like | Pretrain once, finetune small heads for NLP tasks |
| Generative scale | GPT-like decoders | In-context learning; less per-task finetuning |
| Text-to-text | T5 / BART-like | One model, many tasks via prefixes and denoising |
| Alignment | Instruct / chat models | Helpful defaults; chat APIs become the product surface |
| Open weights | LLaMA-class, Mistral-class | Self-host, finetune, and compete on cost/latency |
| Multimodal + tools | Vision-language, audio | Same chat loop, richer inputs and function calls |
| Reasoning-heavy | Long CoT / test-time compute | Spend more tokens/latency for harder problems |

```mermaid
flowchart LR
    P[Pretrain on unlabeled text] --> B[Base model]
    B --> S[Supervised instruction tuning]
    S --> A[Preference / RLHF-style alignment]
    A --> C[Chat / API product]
    C --> T[Tools + RAG + multimodal]
```

### Architecture families in detail

**Encoder-only (BERT-like).** Bidirectional attention over the input. Great for classification, named entity recognition (NER), and dense embeddings. Not a chat generator by default — you usually attach a task head or use the embedding tower.

**Decoder-only (GPT-like).** Causal (left-to-right) attention. Autoregressive generation is native. Almost every modern chat and coding assistant sits here.

**Encoder–decoder (T5/BART-like).** Encode the full source, then decode the target. Natural for translation, structured rewrite, and some summarization setups.

### Alignment: InstructGPT and RLHF

A raw language model is trained to continue text, not necessarily to help a user. **Instruction tuning** and **RLHF** shift behavior toward answers people prefer:

1. **Supervised fine-tuning (SFT)** — train on human-written demonstrations of good responses.
2. **Reward model** — collect human comparisons between candidate answers and train a model to predict preference.
3. **Policy optimization** — update the assistant model to get higher reward while staying close to the supervised model.

Why this matters: a smaller aligned model can feel more useful than a larger raw model because it follows instructions, refuses unsafe tasks more often, and formats answers the way users expect.

### Open-weight, multimodal, and reasoning models

The recent ecosystem expanded in three directions:

| Direction | Plain-English idea | When it helps |
| --- | --- | --- |
| **Open weights** | Released parameters you can host or finetune | Privacy, local deployment, cost control |
| **Multimodal models** | Text + images + audio in one interface | OCR, chart understanding, visual Q&A |
| **Reasoning models** | Spend more compute at answer time on hard tasks | Math, coding, planning — at higher latency/cost |

### Choosing a family in practice

- Need search embeddings or a cheap classifier → encoder / embedding models.
- Need chat, agents, code → decoder chat models.
- Need tight X→Y transforms with clear input/output → consider encoder–decoder or a decoder with strict structured output.
- Need private deployment or deep finetuning → open-weights decoder families.
- Need images/PDFs in the same loop → multimodal chat models.

## In code

You do not need a GPU to practice the *decision* layer. Model a tiny registry that maps product jobs to family recommendations:

```python
FAMILIES = {
    "encoder": {
        "jobs": ["classify", "ner", "embed"],
        "notes": "Bidirectional; add a head or use embeddings.",
    },
    "decoder": {
        "jobs": ["chat", "code", "agents", "summarize"],
        "notes": "Causal LM; default for GenAI products.",
    },
    "encoder_decoder": {
        "jobs": ["translate", "rewrite", "text_to_text"],
        "notes": "Strong when input and output are both sequences.",
    },
}


def recommend(job: str) -> list[str]:
    return [name for name, meta in FAMILIES.items() if job in meta["jobs"]]


print(recommend("embed"))   # ['encoder']
print(recommend("agents"))  # ['decoder']
```

Track a fake "release lineage" so product notes stay honest about base vs chat vs reasoning variants:

```python
from dataclasses import dataclass


@dataclass
class Checkpoint:
    name: str
    family: str
    stage: str  # base | instruct | preference | reasoning


lineage = [
    Checkpoint("corp-7b-base", "decoder", "base"),
    Checkpoint("corp-7b-instruct", "decoder", "instruct"),
    Checkpoint("corp-7b-chat", "decoder", "preference"),
]

assert lineage[-1].stage == "preference"
# Chat APIs almost always expose a post-trained checkpoint, not the raw base.
```

## What goes wrong

- **Using a base model as a product chatbot** — without instruction/preference training, the model continues text instead of following user intent.
- **Treating every decoder as interchangeable** — tokenizers, refusal styles, and tool formats break prompts that "worked on Model A."
- **Forcing chat models into pure embedding jobs** — possible with tricks, usually worse than a dedicated encoder/embedding model.
- **Ignoring post-training stage** — "7B" is not a quality label; base vs instruct vs reasoning variants behave differently.
- **Assuming open weights equal free lunch** — you inherit hosting, eval, safety, and update burden.

:::warn
Never equate parameter count with fitness. A smaller, well-aligned model with tools and retrieval often beats a larger raw generator on real product metrics.
:::

## One-line summary

Foundation-model history is a shift from task-specific encoders to aligned, tool-using generative families — choose encoder, decoder, or encoder–decoder by job shape, then validate the specific checkpoint.

## Key terms

- **Foundation model** — Large pretrained model adapted to many downstream tasks.
- **Encoder-only** — Bidirectional model optimized for understanding / embeddings.
- **Decoder-only** — Causal language model optimized for autoregressive generation and chat.
- **Encoder–decoder** — Sequence-to-sequence family for mapping an input sequence to an output sequence.
- **Pretraining** — Self-supervised training on broad data before task specialization.
- **Post-training / alignment** — Instruction and preference stages that shape assistant behavior.
- **In-context learning** — Steering a model with examples and instructions inside the prompt.
- **Open weights** — Released parameters you can host or finetune yourself.
- **Multimodal model** — Model that accepts more than text (e.g. images) in one interface.
