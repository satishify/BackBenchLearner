---
title: "Multimodal AI Overview"
description: "What multimodal models are and why text-only LLMs are not enough for images, audio, and video."
---

Most of Modules 1–3 treated **text** as the main modality. **Multimodal AI** means one system that can take in (and sometimes produce) more than one kind of signal — text, images, audio, video, or sensor data — in a shared representation space.

## Intuition

A support bot that only reads tickets misses the screenshot the user attached. A design tool that only outputs text cannot sketch the UI. Multimodal models close that gap: they learn to map pixels and words into related vectors so “a red mug on a wooden table” and a photo of that scene land near each other.

## How it works (sketch)

1. **Encoders** turn each modality into embeddings (vision encoder for images, text encoder for captions).
2. **Fusion / alignment** trains those spaces so related pairs match (contrastive learning is a common recipe).
3. **Decoders or heads** produce answers: captions, answers to visual questions, or generated images.

:::key
Placeholder chapter — expand with diagrams and labs once your Module 4 notes are ready.
:::

## What to remember

- Multimodal ≠ “many models glued with if-else”; the point is a **shared** representation.
- Product risk often sits at the boundary: OCR errors, wrong crop, or unsafe image generation.

## One-line summary

Multimodal AI joins text with vision (and other signals) so products can see and describe the world, not only chat about it.

## Key terms

- **Modality** — a data type (text, image, audio…).
- **Embedding** — vector representation used for similarity and fusion.
- **Alignment** — training so related items across modalities sit close in vector space.
