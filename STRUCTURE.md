# BackbenchLearner – Content structure

## Source of truth

Lessons are authored as **Markdown** under `content/`. The generator writes the HTML that GitHub Pages serves.

```
content/
├── topics.yml                 ← topic order in the header
├── redirects.yml              ← old URL → new URL (emits redirect stubs)
├── backend-design/
│   ├── _topic.yml             ← topic manifest (chapters, labels)
│   └── backend/
│       ├── 01-what-is-api.md
│       ├── _quiz.yml          ← optional chapter quiz
│       └── …
└── genai/
    ├── _topic.yml             ← modules + chapters
    └── module-1-foundations/
        └── transformers/
            └── 01-….md
```

Published HTML lands in `backend-design/` and `genai/`. Redirect stubs keep the old `Backend & System Design/` folder paths and `Gen AI & Agentic AI/` URLs working.

## Authoring a lesson

1. Add a Markdown file next to its siblings, numbered so order is clear: `03-http-methods.md`.
2. Front matter is required:

```yaml
---
title: "HTTP methods"
description: "What GET, POST, PUT, PATCH, and DELETE mean in practice."
---
```

3. Body supports headings, lists, tables, ` ```python ` / ` ```mermaid `, and callouts:

```markdown
:::key
Idempotency means retrying is safe.
:::
```

4. Regenerate:

```bash
python3 tools/build_site.py
```

Do **not** hand-edit the generated `.html` files — they are overwritten on the next build.

## Authoring a quiz

Put `_quiz.yml` in the chapter folder:

```yaml
title: Chapter title
questions:
  - q: Question text?
    options:
      - Wrong
      - Right
      - Also wrong
    answer: 1          # 0-based index
    why: Short explanation shown after the learner answers.
```

Rebuild; the quiz appears as the last sidebar row for that chapter and as “Take the chapter quiz” after the last lesson.

### Module mock exams

Timed mocks live under `content/<topic>/_mocks/*.yml` (e.g. `content/genai/_mocks/module-1.yml`). Schema:

```yaml
id: genai/mock/module-1
title: "Module 1 Mock Exam"
minutes: 90
questions:
  - q: "..."
    section: "Foundations"
    options: ["A", "B", "C", "D"]
    answer: 0
    why: "..."
```

Practice opens them via `#practice/genai/module-1` with a countdown timer, change-until-submit, and section breakdown.

## Reading time

`tools/build_site.py` estimates minutes from word count + diagrams + tables + code lines, and writes them into `scripts/curriculum.js`. Re-run the generator after substantial edits.

## Private notes

Licensed study PDFs and `TOC.pages` must stay **outside** this repo (see `_bbl-private-notes/` beside the project). `.gitignore` blocks `*.pages` and `Notes-Summary*`.
