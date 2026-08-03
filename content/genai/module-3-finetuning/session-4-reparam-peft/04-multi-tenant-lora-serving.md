---
title: "Multi-Tenant LoRA Serving"
description: "One frozen base model plus many small client adapters — the SaaS pattern that makes LoRA practical at scale."
---

A strong real-world use case for LoRA and QLoRA is **multi-tenant** customer support: one SaaS platform, many enterprise clients, each wanting its own tone and domain vocabulary.

## Intuition

Example:

- A fintech client needs careful, compliance-aware answers
- A gaming client wants a casual tone

Full fine-tuning a separate 13B or 70B model for each client is a non-starter:

- **200 clients × ~26 GB** for LLaMA-13B-style full copies ≈ **5.2 TB** of model storage
- Each full fine-tune costs heavy GPU hours
- Serving 200 separate large models means many GPUs sitting warm

:::key
Keep one shared base model. Train a small LoRA adapter per client. Hot-swap the adapter per request.
:::

## How it works

### Training side (often with QLoRA)

1. Load the base model in **4-bit** (for example 13B drops from about 26 GB toward about 7 GB)
2. Fine-tune a **separate LoRA adapter** on each client’s data
3. Each adapter can be small (on the order of tens of MB in a typical example)
4. With 4-bit loading and careful optimizers, this can fit on a single consumer-class GPU instead of a big multi-GPU cluster

### Serving side (LoRA hot-swap)

1. Keep **one base model** resident in GPU memory
2. For each request, load the correct client adapter (hot-swap)
3. **200 × ~50 MB adapters ≈ 10 GB** total — versus terabytes of full model copies

Systems designed for this pattern (for example LoRAX-style serving) exist so you can swap adapters without reloading a full model every time.

### Approach comparison

| Approach | Storage / compute story |
| --- | --- |
| Full model per client | Huge storage; separate heavy GPU copy per client |
| LoRA adapter per client | Small adapter files swapped on demand |
| One 4-bit base + many adapters | Best fit for multi-tenant SaaS serving |

## What goes wrong

- Training one giant shared model for all clients and hoping tone differences vanish.
- Shipping adapters with no routing — if you cannot pick the right adapter per tenant, modularity does not help.
- Ignoring eval per client: one adapter can look fine while another regresses.

## One-line summary

For many clients, one frozen base model plus many small LoRA adapters beats storing and serving a full fine-tuned copy for each tenant.

## Key terms

- **Multi-tenant serving** — One base system serving many clients with different behavior.
- **Hot-swap** — Load the right small adapter for the current request without reloading the whole model.
- **Adapter footprint** — The small storage size of a client-specific LoRA file.
