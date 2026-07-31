---
title: "Agents vs Chatbots (Recap)"
description: "Short contrast: one-shot chat vs agent loops that use tools and memory."
---

A **chatbot** typically: user message -> model reply -> done.  
An **agent** typically: user goal -> plan -> tool call -> observe -> maybe plan again -> final answer.

## When you need an agent

- Multi-step workflows (search, then calculate, then file a ticket).
- Fresh data behind APIs or databases.
- Actions with side effects that need confirmation.

## When a chatbot is enough

- FAQ and drafting inside a fixed context.
- Single-turn summarization with no tools.

:::key
Stub. Expand with diagrams and your Module 4 examples later. See also Module 2.9 for full lessons.
:::

## One-line summary

Agents add loops, tools, and state; chatbots mainly generate the next reply.

## Key terms

- **One-shot** — single generate step.
- **Agent loop** — repeated reason/act/observe until stop.
