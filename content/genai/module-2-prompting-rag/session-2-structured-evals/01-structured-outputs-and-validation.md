---
title: "Structured Outputs, Validation, and Reliability"
description: "Get machine-readable LLM outputs with schemas, JSON mode, validators, and repair loops that hold up in production."
---

Free-form prose is for humans. Downstream code wants **objects**: enums, IDs, date ranges, confidence scores. **Structured outputs** turn the model from a storyteller into a component you can wire into queues, UI forms, and agents — but only if you validate every response like it came from an untrusted client.

## Intuition

**What is a structured output?** A model response constrained to a machine-readable shape (often JSON) that your code can parse and use.

**Why is asking for "JSON" in a prompt not enough?** You will get JSON... until you do not. Trailing commas, markdown fences, renamed keys, and partially truncated objects appear under load.

Reliability is a stack:

1. Tell the model the schema.
2. Prefer API features that constrain decoding to valid JSON / schema when available.
3. Validate in your process with a typed model.
4. Repair or fail closed — never "best-effort parse" into a money path.

:::key
Schema in the prompt is documentation. Schema in the validator is enforcement. You need both.
:::

## How it works

### The reliability stack

```mermaid
flowchart TB
    S[Define schema] --> M[Model call with JSON / strict mode]
    M --> V[Validate types and invariants]
    V -->|ok| U[Use object]
    V -->|fail| R[Retry with error feedback]
    R --> M
    R -->|budget exceeded| F[Fail closed / fallback]
```

### Model-side options

| Technique | Plain-English idea | Typical failure it prevents |
| --- | --- | --- |
| **Plain JSON prompt** | Ask for JSON in text | Markdown fences, prose, or malformed JSON |
| **JSON mode / response_format** | Forces JSON syntax | May not enforce your specific fields |
| **Strict structured outputs** | Binds generation to a schema | Wrong types and silent downstream bugs |
| **Tool calling** | Structured function arguments from the model | Free-form text where code expects a schema |

Always assume vendor features differ. Your app-side validator is the portable guarantee.

### App-side validation

Libraries in the **Pydantic** family (or dataclasses + hand checks) give you:

- Required fields and types
- Enums and ranges
- Nested objects and lists
- Clear error messages you can feed back on retry

**Why Pydantic helps:** it validates and can coerce values before they spread through your pipeline. That is especially important when the model says `"42"` as a string but your code needs an integer.

Example conceptual schema:

```
SearchQuery
  rewritten_query: string
  published_daterange: { start: date, end: date }
  domains_allow_list: list[string]
```

### Retry with feedback

On validation failure, do not only "try again colder." Send the error:

`Your previous output failed validation: published_daterange.end is before start. Reply with corrected JSON only.`

Cap retries (e.g. 2). Log failure rates per prompt version — a spike means schema or model drift.

### Designing schemas models can hit

- Prefer flat or shallow nesting; deep trees invite missing braces under token limits.
- Use enums for closed sets (`"Positive" | "Neutral" | "Negative"`) instead of free strings.
- Make rarely needed fields optional; required fields should be ones you always have evidence for.
- Include a first-class abstain path: `status: "ok" | "unknown"` beats forcing a fake value.

### Constrained generation note

Constraints work on **tokens**, not characters. A single token may include a leading space or several characters, so a regex or allowlist must account for the tokenizer, not just the spelling of the output.

Use token masking for simple surface constraints (digits-only, yes/no). Use structured decoding when you need grammar correctness, nested JSON, or something that must never become invalid mid-sequence.

## In code

Stdlib validation without external deps:

```python
from dataclasses import dataclass
from datetime import date
import json


@dataclass
class DateRange:
    start: date
    end: date

    def __post_init__(self):
        if self.end < self.start:
            raise ValueError("end before start")


@dataclass
class SearchQuery:
    rewritten_query: str
    published_daterange: DateRange
    domains_allow_list: list[str]


def parse_search_query(raw: str) -> SearchQuery:
    data = json.loads(raw)
    dr = data["published_daterange"]
    return SearchQuery(
        rewritten_query=str(data["rewritten_query"]).strip(),
        published_daterange=DateRange(
            start=date.fromisoformat(dr["start"]),
            end=date.fromisoformat(dr["end"]),
        ),
        domains_allow_list=[str(d) for d in data.get("domains_allow_list", [])],
    )


good = '''
{"rewritten_query": "llm evaluation",
 "published_daterange": {"start": "2024-01-01", "end": "2024-12-31"},
 "domains_allow_list": ["arxiv.org"]}
'''
print(parse_search_query(good))
```

Pydantic validation with automatic type coercion:

```python
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int


p = Person(name="Sam", age="10")
print(p.age)  # 10, as an int
```

Strip accidental markdown fences before parse:

```python
def strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
        if t.lower().startswith("json"):
            t = t[4:].lstrip()
    return t
```

Retry loop sketch:

```python
def complete_structured(call_model, schema_errors_max=2):
    feedback = None
    for _ in range(schema_errors_max + 1):
        raw = call_model(feedback)
        try:
            return parse_search_query(strip_fences(raw))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            feedback = f"Validation error: {e}. Return corrected JSON only."
    raise RuntimeError("structured_output_failed")
```

With `model_json_schema()`, Pydantic can produce the schema for a tool definition automatically, which removes a lot of brittle manual JSON writing.

## What goes wrong

- **Parsing with regex hope** — works in demos; breaks on nested quotes.
- **Trusting JSON mode alone** — missing keys still pass a `json.loads`.
- **Silent defaults** — wrong type coerced to empty string; bug ships.
- **Retry storms** — no budget; one bad schema burns the rate limit.
- **Schema too clever** — deep nesting and optional everything; model flails; simplify.
- **Valid JSON, invalid business rule** — types pass but `user_id` belongs to another tenant — validators must include authorization invariants.

:::tip
Version your schemas (`SearchQueryV2`) and keep golden fixtures in CI. Prompt edits that break fixtures should fail the build before they fail production.
:::

## One-line summary

Structured GenAI outputs need a schema, constrained generation when available, strict validation, and bounded repair — never trust free text as a typed API.

## Key terms

- **Structured output** — Model response constrained to a machine-readable shape (often JSON).
- **JSON mode** — API setting that requests/requires JSON syntax.
- **Schema** — Formal field/type contract for an object.
- **Validation** — Checking a payload against types and business rules.
- **Repair / retry loop** — Re-prompting with validator errors to fix output.
- **Fail closed** — Refuse the action when validation fails.
- **Grammar / constrained decoding** — Generation restricted to strings that match a schema or grammar.
- **Pydantic** — Python library for data validation and schema generation.
