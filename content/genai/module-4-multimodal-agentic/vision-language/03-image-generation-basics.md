---
title: "Image Generation Basics"
description: "Brief overview of text-to-image generation, diffusion intuition, and product controls."
---

**Image generation** turns a text prompt (and sometimes a reference image) into new pixels. Modern systems are usually **diffusion** or related generative models, wrapped in product UIs with safety filters and style controls.

## Intuition

You describe “a watercolor fox under streetlights.” The model starts from noise (or a latent) and iteratively denoise toward an image that matches the prompt embedding. Editing tools (“inpaint this region”) constrain which pixels may change.

## Practical knobs (product view)

- **Prompt / negative prompt** — what to include and avoid.
- **Seed** — reproducibility.
- **Guidance / CFG** — how strongly to follow the prompt vs stay natural.
- **Resolution / steps** — quality vs latency/cost.

:::key
Stub only — expand with diffusion math, ControlNet-style conditioning, and eval when you have Module 4 session notes.
:::

## What goes wrong

- Brand / likeness misuse and copyright risk.
- Unsafe or biased imagery if filters are weak.
- Prompt injection via images in multimodal apps (treat untrusted images carefully).

## One-line summary

Text-to-image models synthesize pictures from prompts; treat them as creative tools that still need safety and rights review.

## Key terms

- **Diffusion** — generate by iterative denoising from noise.
- **Latent** — compressed space where many generators operate for speed.
- **Guidance** — strength of prompt adherence during sampling.
