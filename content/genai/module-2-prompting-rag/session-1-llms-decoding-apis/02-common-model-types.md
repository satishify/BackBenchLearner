---
title: "Common Model Types: Text, Image, Audio, Video, Multimodal"
description: "Unimodal vs multimodal generative models across text, image, audio, and video—and a practical guide to choosing a modality pipeline."
---

Not every problem wants a chat box. Speech notes, product photos, surveillance clips, and support tickets each live in a different **modality** — a type of signal the model consumes or produces. Choosing the wrong modality pipeline wastes compute and creates awkward UX (forcing users to describe a screenshot in words when vision would suffice).

## Intuition

A **modality** is a channel of information: text, images, audio, video, tabular fields, sensor streams. Models are often **unimodal** (one channel in, one out) or **multimodal** (multiple channels, jointly).

Think of product shapes, not brand names:

- **Text models** map text → text (or embeddings).
- **Image models** map noise/text → pixels, or pixels → labels/captions.
- **Audio models** map waveforms/spectrograms ↔ text or other audio.
- **Video models** treat time + frames (often heavy cousins of image models).
- **Multimodal models** share a representation space so, for example, an image and a question can jointly produce an answer.

:::key
Pick the modality that matches the *native* form of the user’s problem. Translation into text is a workaround, not always an architecture.
:::

## How it works

### Unimodal families (high level)

| Family | Typical I/O | What they are good at | Common failure mode |
| --- | --- | --- | --- |
| **Text / LLM** | text → text | Reasoning over language, code, structured drafts | Weak on pixels/sound without tools |
| **Image generation** | text/noise → image | Concepts, layouts, variants | Text spelling, precise counts, brand fidelity |
| **Image understanding** | image → labels/captions/embeddings | Search, moderation, OCR assist | Fine print, rare objects without fine-tuning |
| **Speech / audio** | audio ↔ text or audio | Transcription, voice interfaces, SFX | Accents, overlap, domain jargon |
| **Video** | text/video → video or understanding | Short clips, summarization, detection over time | Cost, temporal consistency, long duration |

### Multimodal in practice

Multimodal systems usually:

1. Encode each modality into vectors (encoders).
2. Fuse or align those vectors (shared space, cross-attention, adapters).
3. Decode into the target modality (text answer, caption, edited image).

```mermaid
flowchart LR
    T[Text encoder] --> F[Shared / fused representation]
    I[Image encoder] --> F
    A[Audio encoder] --> F
    F --> O[Decoder: text / image / action]
```

You do not need one giant model for every demo. Many production stacks are **pipelines**: speech-to-text → LLM → text-to-speech, or image embedder → vector search → LLM for explanations. That is multimodal *as a system* even if each step is unimodal.

### Model families without the vendor brochure

Keep architecture shapes in mind, not slogans:

- **Autoregressive text models** — generate left-to-right tokens (most chat LLMs).
- **Embedding / dual-encoder models** — map text or images to vectors for search and clustering.
- **Diffusion-style image (and some video) models** — iteratively denoise toward a sample guided by text or other conditions.
- **Speech recognition / synthesis stacks** — often specialized encoders and decoders rather than “one LLM that hears.”
- **Vision-language models** — image encoder + language model (or joint training) for captioning and visual Q&A.

Exact training recipes change yearly; the *job* each family does changes slowly. Design against the job.

### When to pick which pipeline

Use this decision sketch:

1. **Source of truth is text** (tickets, policies, code) → text LLM ± retrieval.
2. **User shows something visual** (receipt, UI bug, shelf photo) → vision or multimodal understanding; do not force them to type a description if avoidable.
3. **Hands-busy / voice-native** (driving, warehouse) → speech in/out; keep text as the internal reasoning layer if helpful.
4. **Creative assets** (ads, storyboards) → image/video generation with human review; lock brand kits separately.
5. **Compliance or exact numbers** → prefer structured extractors + databases; use generative models for narrative around verified fields.

Also separate **understanding** from **generation**. Classifying whether a photo shows a damaged package is different from generating a marketing image of a package. Same modality, different risk, latency, and evaluation.

:::tip
Start with the cheapest modality that preserves information. Transcribing audio then prompting an LLM is often enough; jump to native audio-in models when latency, prosody, or overlapping speakers matter.
:::

## Worked example

A field-support app receives: a photo of an error screen, a voice note, and a short typed symptom.

| Approach | Pipeline | Pros | Cons |
| --- | --- | --- | --- |
| A. Text-only | User must type everything | Simple | Loses visual detail; high user effort |
| B. Stitched unimodal | ASR → OCR/caption → LLM | Clear ownership per step | Error compounds across stages |
| C. Multimodal model | Image + transcript → answer | Fewer handoffs | Harder to debug; vendor lock-in risk |

```python
from dataclasses import dataclass


@dataclass
class Ticket:
    text: str
    image_bytes: bytes | None = None
    audio_bytes: bytes | None = None


def choose_pipeline(ticket: Ticket) -> str:
    has_image = ticket.image_bytes is not None
    has_audio = ticket.audio_bytes is not None

    if has_image and has_audio:
        return "multimodal_or_asr_plus_vision_then_llm"
    if has_image:
        return "vision_caption_or_vqa_then_llm"
    if has_audio:
        return "asr_then_llm"
    return "text_llm_with_optional_rag"


print(choose_pipeline(Ticket("app crashes on save")))
# text_llm_with_optional_rag

print(choose_pipeline(Ticket("see photo", image_bytes=b"...png")))
# vision_caption_or_vqa_then_llm
```

The function is deliberately boring: modality choice is a product decision before it is a model-zoo decision.

Extend the same idea into a tiny routing table you could paste into a design doc:

```python
ROUTES = {
    ("text",): "llm_rag",
    ("image",): "vision_then_llm",
    ("audio",): "asr_then_llm",
    ("image", "audio"): "parallel_asr_vision_then_llm",
    ("video",): "keyframe_or_video_model_then_llm",
}


def route(*modalities: str) -> str:
    key = tuple(sorted(set(modalities)))
    return ROUTES.get(key, "clarify_inputs_with_user")


print(route("text"))
print(route("image", "audio"))
print(route("video"))
```

If `clarify_inputs_with_user` appears often in real traffic, your intake UX — not your model — is the bottleneck.

## What goes wrong

- **Modality mismatch** — Feeding screenshots into a text-only bot via “describe the image” creates lossy telephone games.
- **Pipeline latency stacking** — ASR + OCR + LLM + TTS can exceed interactive budgets; measure end-to-end, not per model.
- **Silent failure at boundaries** — Bad transcription becomes confident LLM advice; always surface low-confidence ASR or OCR.
- **Over-buying multimodal** — A full video model for “extract the total from this invoice PDF” is overkill; OCR + rules/LLM may win.
- **Evaluation theater** — Pretty demos on clean studio audio/images hide production noise (glare, accents, compression).
- **Privacy** — Images and audio often contain PII; modality upgrades expand your threat surface.

:::warn
Multimodal does not mean “smarter about facts.” It means “can see or hear more.” Grounding and permissions still apply.
:::

## One-line summary

Match models to modalities — text, image, audio, video, or multimodal fusion — and prefer the simplest pipeline that keeps the user’s native signal intact.

## Key terms

- **Modality** — A type of input/output signal such as text, image, audio, or video.
- **Unimodal model** — Operates primarily on a single modality.
- **Multimodal model** — Jointly handles two or more modalities.
- **Encoder / decoder** — Modules that map a modality into vectors or back out to raw signals/tokens.
- **ASR (Automatic Speech Recognition)** — Speech → text.
- **TTS (Text-to-Speech)** — Text → audio.
- **VQA (Visual Question Answering)** — Answer questions about an image.
- **Embedding** — Dense vector representation used for search, clustering, or fusion across modalities.
