---
title: "Multimodal Agents (Teaser)"
description: "How vision + tools combine: agents that can see screenshots and act."
---

Module 4 joins **vision and multimodal retrieval** (4.1–4.3) with **agents** (4.4). A multimodal agent might: read a screenshot -> extract an error code -> call a runbook API -> propose a fix — with a human approval step.

## Building blocks

- VLM or vision tool for “what’s on screen.”
- Text/tools for search, tickets, code.
- Policies: what the agent may do without asking.

:::key
Teaser only — flesh out architectures and labs when your notes are ready.
:::

## One-line summary

Multimodal agents use eyes (vision) plus hands (tools) to complete goals, not only chat.

## Key terms

- **Multimodal agent** — agent that consumes non-text inputs.
- **Grounding** — tying actions to observed evidence (image, doc, API).
