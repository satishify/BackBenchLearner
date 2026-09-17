---
title: "Image Search with a Shared Vector Space"
description: "Understand text-to-image and image-to-image search, how CLIP enables both, and where fine-grained queries fail."
---

**Image search** finds matching images from a description or another image.

It is useful for product catalogs, stock-photo libraries, reverse image search, and content moderation.

## Intuition

### The obvious approach, and why it is not used

Suppose you have 200,000 product photos and a shopper types "black waterproof hiking backpack."

The instinctive solution is to write a caption for every photo, then run ordinary text search over the captions. That does work, and for years it was how image search was built. It has two stubborn problems.

First, the caption is written **before anyone asks anything**. Whoever wrote it decided which details mattered. If the caption says "black backpack" and the shopper cares about side pockets, that information was thrown away at indexing time and no amount of clever searching recovers it.

Second, it puts a translation step between the question and the evidence. Every mistake in the caption becomes a permanent, invisible error in the index.

### The better idea: skip the words

Modern image search does not translate pictures into text at all. It converts **both** the photo and the query into vectors, and places them in the same space — so they can be compared directly, without either becoming words first.

That is the entire trick, and everything else in this lesson follows from it. A CLIP-style model is what makes it possible, because it was trained specifically so that a picture and its caption land in the same region.

:::note Analogy
Think of a shared vector space as a giant map where everything is placed by meaning rather than by type.

Photographs of mountains at sunset land in one region of the map. The *phrase* "snow-capped mountains at sunset" lands in that same region, because CLIP was trained to put matching pictures and words in the same place.

Searching then becomes a geography question: drop a pin where the query lands, and return whatever is nearby. The system never has to translate the picture into words — both were already placed on the same map.
:::

## Two directions

### Text-to-image

The user types:

> Snow-capped mountains at sunset

The system returns matching images, even when nobody added that exact caption to them.

### Image-to-image

The user supplies a reference image. The system returns visually similar images.

Both directions can use the same image index. Only the query encoder changes.

## Why CLIP fits

CLIP learns with matched image-text pairs. Its image encoder and text encoder place both kinds of content in a **shared vector space**.

```mermaid
flowchart LR
    T[Text query] --> TE[Text encoder]
    IQ[Image query] --> IE1[Image encoder]
    DB[Image collection] --> IE2[Image encoder]
    TE --> S[Shared vector space]
    IE1 --> S
    IE2 --> S
    S --> TOP[Nearest images]
```

Because text and image vectors are comparable, a text description can search image vectors directly. The system does not need to write a caption for every image first.

:::key
Text search and image search can reuse the same ANN indexing pattern. The important difference is which encoders create the vectors.
:::

## A simple flow

Before search:

1. Encode every catalog image.
2. Save each vector with the image ID.
3. Build the ANN index.

For each user query:

```python
query_vector = clip.encode_text(
    "black waterproof hiking backpack with side pockets"
)
image_ids = image_index.search(query_vector, top_k=10)
return catalog.images(image_ids)
```

For image-to-image search, replace `encode_text` with the image encoder.

## Exact filters can still matter

A shared embedding is good at broad visual meaning. It may not reliably preserve an exact SKU, size, brand, or every spatial detail.

For product search, combine signals:

```mermaid
flowchart LR
    Q[User query] --> F[Metadata filters<br/>brand, SKU, size]
    Q --> V[CLIP visual similarity]
    F --> M[Merge or rerank]
    V --> M
    M --> R[Final results]
```

For example, filter by `brand = Acme` and then rank the remaining images by similarity to “red waterproof backpack.”

## How image retrieval is measured

Common benchmarks include **Flickr30K** and **MS COCO**.

- **Recall@1** — Is the correct result first?
- **Recall@5** — Is it anywhere in the first five?
- **Recall@10** — Is it anywhere in the first ten?

If 80 of 100 queries find their correct image in the first five, Recall@5 is 80%.

Benchmark scores depend on the model, data, direction, and test rules. Compare systems under the same protocol.

## What goes wrong

### Fine detail

“A red car” is easier than “the small red sedan facing left in the third row.”

### Compositional queries

A request with colour, object type, direction, position, and relation may lose one or more constraints in a single vector.

### Long descriptions

Very detailed queries can exceed what the text encoder learned to represent well.

### Bias

The training data and searchable collection affect which images appear “most relevant.” Test different cultures, environments, and long-tail examples.

### Similar is not identical

Image-to-image retrieval may focus on background, colour, or style instead of the specific object feature the user intended.

Controls include metadata filters, reranking, user feedback, and task-specific evaluation.

## One-line summary

CLIP-style image search compares text and images in one vector space, while filters and reranking protect exact or fine-grained requirements.

## Key terms

- **Text-to-image retrieval** — Find images using words.
- **Image-to-image retrieval** — Find images using a reference image.
- **Shared vector space** — Comparable representations for different content types.
- **Cross-modal retrieval** — Search one modality with another.
- **Recall@K** — Whether the correct result appears in the top K.
