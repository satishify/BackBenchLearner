---
title: "Putting QLoRA Together"
description: "How 4-bit weights, LoRA, checkpointing, and a paged optimizer combine so large models can train on one GPU."
---

Single-GPU fine-tuning becomes realistic when you stack the right ingredients. Each piece shrinks a different part of the memory bill.

## Intuition

Rough memory story for a **70B** model:

| Setup | Bits / memory story | GPU picture |
| --- | --- | --- |
| Naive full fine-tuning | Heavy weights + gradients + optimizer | Many data-center GPUs |
| LoRA (higher-precision base) | Smaller trainable set, base still bulky | Fewer GPUs, still heavy |
| QLoRA-style stack | About **5.2 bits/param** class budget in this sketch | Often **1×** data-center GPU class |

The exact numbers depend on hardware and settings. The lesson is the composition: **4-bit weights + small adapters + controlled activations + controlled optimizer spikes**.

:::key
Use full fine-tuning only when you can afford it. Prefer LoRA for tiny trainable budgets. Prefer QLoRA when memory is tight.
:::

## How it works

### Stack roles

| Component | Role in the memory budget |
| --- | --- |
| **4-bit weights (NF4)** | Shrink base model storage |
| **LoRA adapters** | Keep trainable parameters tiny |
| **Double quantization** | Reduce metadata overhead |
| **Gradient checkpointing** | Save activation memory |
| **Paged optimization** | Reduce memory spikes |

### Tiny code sketch (concept only)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model, quantization_config=bnb, device_map="auto"
)
model.gradient_checkpointing_enable()

lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora)

args = TrainingArguments(
    optim="paged_adamw_8bit",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
)
```

Read this as a map of ideas, not a copy-paste production recipe:

- `nf4` + double quant → compress the frozen backbone
- `LoraConfig` → tiny trainable update
- `gradient_checkpointing_enable` → save activation memory
- `paged_adamw_8bit` → soften optimizer spikes

### Quick revision rules

1. Use **full fine-tuning** only when you can afford the compute and really need every weight to move.
2. Use **LoRA** when you want most of the benefit with a tiny trainable parameter budget.
3. Use **QLoRA** when memory is tight and you want the frozen backbone in 4-bit form.
4. Use **gradient checkpointing** when activations, not just weights, are hurting GPU memory.
5. Use **paged optimization** when optimizer spikes are what push you over the limit.

## What goes wrong

- Enabling every flag at once with no eval — you cannot tell which knob helped.
- Assuming “fits on one GPU” means “quality is automatic.” Always measure the task.
- Serving many tenants without an adapter swap plan after training QLoRA successfully.

## One-line summary

QLoRA works as a stack: compress the frozen base, learn a tiny LoRA update, and control activation and optimizer memory so large models become trainable.

## Key terms

- **QLoRA stack** — NF4 + double quant + LoRA + checkpointing + paged optimizer.
- **Bits per parameter** — Rough storage budget after compression tricks.
- **Effective batch** — micro-batch × accumulation × GPUs (how many samples feed one update).
