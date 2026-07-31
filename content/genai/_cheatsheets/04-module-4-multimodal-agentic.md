---
title: "Module 4 - Multimodal & agents revision"
slug: module-4-multimodal-agentic
module: "Module 4"
minutes: 15
description: "Stub-friendly revision for VLMs and agentic systems; deepen with Module 2.9 for agents."
---

Chapters **4.1–4.2** are shorter stubs. Use this for vocabulary; dive into **2.9** for agent depth.

## 4.1 Vision-language & image generation

- **VLM**: joint model that maps images (and often text) into a shared space — caption, VQA, document understanding.
- Typical pattern: vision encoder -> projector -> LLM decoder. Tokenize image patches / features like “soft tokens”.
- Failure modes: OCR weakness, counting, spatial relations, hallucinated objects — evaluate on *your* image types.
- **Image generation** (diffusion / related): iterative denoising from noise conditioned on text. Prompting = subject + style + constraints; safety filters matter in prod.
- Multimodal RAG: retrieve images + text; cite visual evidence carefully.

## 4.2 Agentic systems (recap)

- **Agent** = large language model (LLM) wrapped with tools, memory, and planning—not just a chat user interface (UI).
- **Loop:** observe → think → act (tool) → observe → … until done, budget hit, or human-in-the-loop (HITL) escalation.
- **ReAct (Reason + Act):** think one step, run one tool, read the observation, then think again—ground reasoning in real data.
- **Memory tiers:** episodic (this session’s thread) vs semantic (long-term facts in a vector database). Do not dump all history into every prompt.
- **Task decomposition:** break goals into named, checkable steps; planner decides order, executor runs one step.
- **Orchestration:** state machines / LangGraph when you need loops, branches, or approval checkpoints. Single agent with tools often beats premature multi-agent swarms.
- **Provenance:** trace claims back to tool-call history—do not trust inline citations without verifying source IDs.
- For full depth (HITL, reflection, multi-agent roles, LangChain vs LangGraph), revise **Cheat sheet - Module 2** section 2.9 and Session 12 lessons.

## 15-minute drill
1. Contrast captioning vs VQA vs “open-world chat about an image”.
2. List three agent failure modes (looping, tool misuse, context overflow).
