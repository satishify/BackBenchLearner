---
title: "Vision-Language Models (VLMs)"
description: "Short intro to VLMs: image understanding, VQA, and how they differ from text-only LLMs."
---

A **vision-language model (VLM)** accepts images (and often text prompts) and produces text — captions, answers, or structured descriptions. Think of it as an LLM that can also **look**.

## Intuition

You point a phone at a whiteboard and ask “summarize the action items.” A VLM encodes the photo, conditions a language model on that encoding, and writes the summary. Classic CV classifiers only output class labels; VLMs speak in natural language.

## Common tasks

- **Captioning** — describe an image.
- **VQA** — answer a question about an image.
- **Document / UI understanding** — read screenshots, forms, slides.
- **Grounding** — point to regions that match a phrase (advanced).

## How it works (high level)

Many VLMs use a **vision encoder** (e.g. ViT-style) whose outputs are projected into the token space of a language model. Training mixes caption data, interleaved image-text documents, and instruction-tuning with visual Q&A.

:::key
This lesson is a stub. Add architecture diagrams, API examples, and failure cases when your notes arrive.
:::

## What goes wrong

- Hallucinated objects that are not in the image.
- Tiny text / OCR misses.
- Over-trusting the model for medical or legal visual decisions without human review.

## One-line summary

VLMs let language models consume images so you can ask questions about what is on the screen or in a photo.

## Key terms

- **VLM** — model that jointly handles vision and language.
- **VQA** — visual question answering.
- **Vision encoder** — network that turns pixels into tokens or embeddings.
