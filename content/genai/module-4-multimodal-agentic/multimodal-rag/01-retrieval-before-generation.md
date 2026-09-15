---
title: "Multimodal RAG: Retrieve Before You Answer"
description: "See the common pattern behind text, image, and document retrieval, and why good evidence must come before generation."
---

Imagine asking an AI:

> What was our Q3 revenue growth?

The answer may be inside a paragraph, a table, or a bar chart. The model should not guess from memory. It should first **find the right evidence**, then answer from it.

That is the idea behind **Retrieval-Augmented Generation (RAG)**.

## Intuition

RAG has two separate jobs:

1. **Retrieval** finds useful existing material.
2. **Generation** reads that material and writes an answer.

```mermaid
flowchart LR
    Q[User question] --> R[Retriever<br/>find evidence]
    R --> E[Top matching passages<br/>images or pages]
    E --> G[Generator<br/>read evidence]
    Q --> G
    G --> A[Grounded answer]
```

:::key
A fluent model cannot repair missing evidence. If retrieval returns the wrong page, even a strong generator is likely to give the wrong answer.
:::

## Text RAG and multimodal RAG

| System | What it can retrieve |
| --- | --- |
| Text-only RAG | Text passages |
| Multimodal RAG | Text, images, tables, charts, audio, video, or complete pages |

Real documents are visual. A report may explain a result in a chart without repeating it in a sentence. A manual may show a connection only in a diagram. Multimodal RAG keeps this evidence available.

## The common retrieval pattern

Text search, image search, and page search all follow a similar pattern:

```mermaid
flowchart TB
    CORPUS[Searchable collection] --> ENC1[Encode items as vectors]
    ENC1 --> IDX[Store vectors in an index]
    QUERY[New query] --> ENC2[Encode the query]
    ENC2 --> SEARCH[Find nearby vectors]
    IDX --> SEARCH
    SEARCH --> TOP[Return top matches]
```

A **vector** is simply a list of numbers that represents useful meaning or visual information. Similar items should get nearby vectors.

- The **encoder** turns raw content into vectors.
- The **index** stores those vectors for fast search.
- The **retriever** returns existing evidence.
- The **generator** is optional; it turns retrieved evidence into a response.

## One pattern, different content

| Task | Query | Search collection | Useful technique |
| --- | --- | --- | --- |
| Text search | Text | Text passages | Dense text embeddings |
| Image search | Text or image | Images | CLIP-style shared space |
| Document search | Text | Complete page images | Vision-space retrieval |
| Multimodal QA | Question | Mixed evidence | Retrieval + VLM generation |

The machinery is similar. What changes is **what gets encoded** and **which model encodes it**.

## Worked example

Suppose a company has 50,000 policy pages.

1. Before users ask anything, encode and index the pages.
2. A user asks, “Can I cancel during the trial?”
3. Encode that question.
4. Find the nearest policy passages or pages.
5. Give only those results and the question to the model.
6. Ask the model to answer with a page reference.

This is faster and easier to update than asking a model to memorise every policy.

## Why retrieval helps

- **Freshness** — update the searchable collection without retraining the generator.
- **Grounding** — provide evidence instead of relying only on model memory.
- **Verification** — show where an answer came from.
- **Scale** — search a large collection without reading every item during every request.

Retrieval reduces hallucination, but it does not remove it. The model may still ignore evidence or make an unsupported claim.

## One-line summary

Multimodal RAG first retrieves the most useful text or visual evidence, then asks a model to answer from that evidence.

## Key terms

- **RAG** — Retrieval-Augmented Generation.
- **Corpus** — The collection being searched.
- **Embedding** — A vector representation of content.
- **Retriever** — Component that selects evidence.
- **Generator** — Model that writes the final answer.
- **Grounding** — Connecting an answer to supplied evidence.
