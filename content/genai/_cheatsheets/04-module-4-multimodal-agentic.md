---
title: "Module 4 - VLM architectures revision"
slug: module-4-multimodal-agentic
module: "Module 4"
minutes: 20
description: "Revision for CLIP, LLaVA, Qwen-VL, SAM, and how to compose them."
---

Chapter **4.1** follows the lecture map: Foundations → CLIP → LLaVA → Qwen-VL → SAM → Synthesis.

## Foundations

- **VLM** = pixels → vision encoder → connector → consumer → output (score, text, box, or mask).
- **ViT**: patchify image; `(H/P) × (W/P)` patch tokens; add position info. Example: 224×224, 16×16 patches → **196** tokens.
- Four paradigms: **contrastive** (CLIP), **generative** (LLaVA), **grounded generative** (Qwen-VL), **promptable segmentation** (SAM).

## CLIP

- Dual encoders; **symmetric contrastive** loss; diagonal of batch similarity matrix = correct pairs.
- **Zero-shot**: encode image once; compare to text templates like `a photo of a {label}`.
- Pick CLIP for **matching/retrieval**; not for long explanations.

## LLaVA

- Frozen CLIP ViT → **projector** → visual tokens in LLM sequence.
- Stage 1: train projector only. Stage 2: instruction tune projector + LLM.
- **LLaVA-NeXT / AnyRes**: high-res tiles + global view for OCR/charts (more tokens).

## Qwen-VL

- **256 learned queries** cross-attend over variable patches; **2D position** on keys.
- Grounding via `<ref>`, `<box>` tokens; coordinates **0–1000** normalized, generated as text.
- Fixed token budget = predictable cost; may lose fine detail.

## SAM

- **Image encoder once** (cache) → **prompt encoder** + **mask decoder** per click/box.
- **Three candidate masks** for ambiguous points; **IoU** scores quality.
- Composes with grounding: box from VLM → SAM mask. SAM does not name the object.

## Synthesis

| Need | Start with |
| --- | --- |
| Tag / retrieve | CLIP |
| Chat about image | LLaVA |
| OCR + boxes + multi-image | Qwen-VL |
| Pixel mask | SAM |
| Grasp / rotoscope pipeline | VLM + SAM |

## 15-minute drill

1. Walk through ViT patch count for 224×224 and 16×16 patches.
2. Explain why LLaVA needs a projector even when dimensions match.
3. Convert one normalized `<box>` to pixel coordinates for a given image size.
4. Sketch a two-step pipeline: Qwen-VL finds box → SAM segments.
