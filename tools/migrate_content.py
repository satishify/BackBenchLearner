#!/usr/bin/env python3
"""One-off migration: hand-written lesson HTML into the content/ Markdown tree.

Reads the legacy ``TOPICS`` registry still embedded in index.html so lesson
order and labels come from the site itself rather than being retyped, converts
each page to Markdown, and writes it to its new home along with the topic
manifests and the redirect map for the old URLs.

    python3 tools/migrate_content.py --dry-run
    python3 tools/migrate_content.py
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import html_to_md  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"

LEGACY_BASE = {
    "backend-design": "Backend & System Design",
    "gen-ai-agentic-ai": "Gen AI & Agentic AI",
}

# Where each legacy topic's lessons end up. Backend keeps its chapters and only
# the topic folder is renamed; Gen AI is reorganised onto the five-module TOC.
TOPIC_TARGET = {"backend-design": "backend-design", "gen-ai-agentic-ai": "genai"}

BACKEND_MANIFEST = {
    "id": "backend-design",
    "basePath": "backend-design",
    "navLabel": "Backend & System Design",
    "title": "Backend & System Design Notes",
    "welcomeTitle": "Backend & System Design Notes",
    "welcomeTagline": (
        "Core concepts, databases, caching, distributed systems, and reliability "
        "— with diagrams, worked examples, and Python."
    ),
}

GENAI_MANIFEST = {
    "id": "genai",
    "basePath": "genai",
    "navLabel": "Gen AI & Agentic AI",
    "title": "Gen AI & Agentic AI",
    "welcomeTitle": "Gen AI & Agentic AI",
    "welcomeTagline": (
        "A five-module path from neural network foundations to fine-tuned, "
        "production-ready GenAI systems."
    ),
}

GENAI_MODULES = [
    {
        "id": "module-1-foundations",
        "title": "Module 1 · Foundations of GenAI & LLMs",
        "blurb": "Understand the machinery before you use it.",
        "chapters": [
            ("ai-deep-learning-essentials", "1.1 AI & Deep Learning Essentials"),
            ("transformers", "1.2 The Transformer Architecture"),
            ("foundation-models", "1.3 Working with Foundation Models"),
        ],
    },
    {
        "id": "module-2-prompting-rag",
        "title": "Module 2 · Advanced Prompting & RAG Systems",
        "blurb": "Retrieval that holds up outside a demo.",
        "chapters": [
            ("advanced-prompting", "2.1 Advanced Prompt Engineering"),
            ("rag-fundamentals", "2.2 RAG Fundamentals"),
            ("advanced-rag", "2.3 Advanced RAG Techniques"),
        ],
    },
    {
        "id": "module-3-finetuning",
        "title": "Module 3 · LLM Fine-tuning & Alignment",
        "blurb": "When your own data is the advantage.",
        "chapters": [
            ("finetuning-fundamentals", "3.1 Fine-Tuning Fundamentals"),
            ("peft", "3.2 Parameter-Efficient Fine-Tuning"),
            ("lab-sprint", "3.3 Lab Sprint & Review"),
        ],
    },
    {
        "id": "module-4-multimodal-agentic",
        "title": "Module 4 · Multimodal & Agentic AI",
        "blurb": "Systems that see, reason, and act.",
        "chapters": [
            ("vision-language", "4.1 Vision-Language Models & Image Generation"),
            ("agentic-systems", "4.2 Agentic AI Systems"),
        ],
    },
    {
        "id": "module-5-deployment-safety",
        "title": "Module 5 · Deployment, Optimisation & AI Safety",
        "blurb": "Turning a prototype into something you can operate.",
        "chapters": [
            ("production-rag-agents", "5.1 Production-Grade RAG & Agent Orchestration"),
            ("serving-and-apis", "5.2 Model Serving & API Development"),
            ("responsible-genai", "5.3 Responsible and Trusted GenAI"),
            ("capstone", "5.4 Industry Capstone Project"),
        ],
    },
]

# Ordered destination for every legacy Gen AI lesson, keyed by new chapter path.
GENAI_PLACEMENT: dict[str, list[str]] = {
    "module-1-foundations/ai-deep-learning-essentials": [
        "gen-ai-basics/what-is-ai",
        "gen-ai-basics/what-is-machine-learning",
        "gen-ai-basics/what-is-deep-learning",
        "gen-ai-basics/what-is-generative-ai",
        "gen-ai-basics/ai-ml-deep-learning-genai",
        "gen-ai-basics/what-are-llms-and-text-generation",
        "gen-ai-basics/common-model-types",
        "gen-ai-basics/pretraining-finetuning-inference",
        "deep-learning-essentials/artificial-neuron-and-perceptron",
        "deep-learning-essentials/activation-functions-and-mlp",
        "deep-learning-essentials/backpropagation-and-gradient-descent",
        "deep-learning-essentials/overfitting-underfitting-bias-variance",
        "deep-learning-essentials/regularization-techniques",
    ],
    "module-1-foundations/transformers": [
        "transformer-architecture/why-transformers",
        "transformer-architecture/self-attention-qkv",
        "transformer-architecture/positional-encoding-and-mha",
        "transformer-architecture/decoder-masking-cross-attention",
        "transformer-architecture/autoregressive-decoding",
    ],
    "module-1-foundations/foundation-models": [
        "llm-models-and-apis/llm-evolution-and-model-families",
        "core-llm-concepts/tokens-context-window-temperature-top-p",
        "llm-models-and-apis/prompting-fundamentals-and-roles",
        "llm-models-and-apis/decoding-parameters-and-control",
        "llm-models-and-apis/apis-and-tool-calling",
        "llm-models-and-apis/structured-outputs-and-validation",
        "core-llm-concepts/hallucinations-and-limitations",
    ],
    "module-2-prompting-rag/advanced-prompting": [
        "core-llm-concepts/prompt-engineering",
    ],
    "module-2-prompting-rag/rag-fundamentals": [
        "core-llm-concepts/embeddings-semantic-search",
        "core-llm-concepts/rag-retrieval-augmented-generation",
        "build-skills/vector-databases",
    ],
    "module-4-multimodal-agentic/agentic-systems": [
        "agentic-ai-basics/what-makes-an-agent",
        "agentic-ai-basics/reasoning-planning-tools-memory",
        "agentic-ai-basics/chatbot-vs-agent",
        "agentic-ai-basics/single-vs-multi-agent",
        "agentic-ai-basics/human-in-the-loop-workflows",
        "agent-design-topics/task-decomposition",
        "agent-design-topics/tool-calling-function-calling",
        "agent-design-topics/short-term-vs-long-term-memory",
        "agent-design-topics/reflection-self-correction-loops",
        "agent-design-topics/orchestration-workflow-control",
        "build-skills/langchain-llamaindex-autogen-crewai",
    ],
    "module-5-deployment-safety/serving-and-apis": [
        "build-skills/apis-for-models",
        "build-skills/basic-python-for-experimentation",
        "build-skills/workflow-automation-concepts",
    ],
    "module-5-deployment-safety/responsible-genai": [
        "practical-topics/use-cases",
        "practical-topics/evaluation-metrics",
        "practical-topics/guardrails-and-security",
        "practical-topics/prompt-injection-and-tool-misuse-risks",
        "practical-topics/monitoring-and-observability",
    ],
    "module-5-deployment-safety/capstone": [
        "capstone/end-to-end-ai-agent-use-case",
    ],
}


@dataclass
class LegacyLesson:
    topic_id: str
    section_title: str
    hash: str
    path: str
    label: str

    @property
    def source(self) -> Path:
        return ROOT / LEGACY_BASE[self.topic_id] / self.path


# --------------------------------------------------------------------------
# Reading the legacy registry out of index.html
# --------------------------------------------------------------------------

_SECTION = re.compile(r"title: '([^']*)',\s*\n\s*links: \[")
_LESSON = re.compile(r"\{ hash: '([^']*)', path: '([^']*)', label: '([^']*)' \}")


def read_legacy_registry() -> list[LegacyLesson]:
    text = (ROOT / "index.html").read_text(encoding="utf-8")
    starts = {}
    for topic_id in LEGACY_BASE:
        match = re.search(rf"'{re.escape(topic_id)}': \{{", text)
        if not match:
            raise SystemExit(f"could not find legacy topic {topic_id} in index.html")
        starts[topic_id] = match.start()

    ordered = sorted(starts.items(), key=lambda kv: kv[1])
    bounds = []
    for index, (topic_id, start) in enumerate(ordered):
        end = ordered[index + 1][1] if index + 1 < len(ordered) else len(text)
        bounds.append((topic_id, start, end))

    lessons: list[LegacyLesson] = []
    for topic_id, start, end in bounds:
        chunk = text[start:end]
        sections = [(m.start(), m.group(1)) for m in _SECTION.finditer(chunk)]
        for match in _LESSON.finditer(chunk):
            title = ""
            for pos, name in sections:
                if pos < match.start():
                    title = name
                else:
                    break
            lessons.append(
                LegacyLesson(
                    topic_id=topic_id,
                    section_title=title,
                    hash=match.group(1),
                    path=match.group(2),
                    label=match.group(3),
                )
            )
    return lessons


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def yaml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def front_matter(title: str, description: str) -> str:
    return "---\n" f"title: {yaml_quote(title)}\n" f"description: {yaml_quote(description)}\n" "---\n\n"


def write(path: Path, content: str, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_manifest_yaml(manifest: dict, chapters: list[tuple[str, str]] | None, modules: list[dict] | None) -> str:
    lines = [
        "# Topic manifest. Chapter order here drives sidebar order.",
        f"id: {manifest['id']}",
        f"basePath: {yaml_quote(manifest['basePath'])}",
        f"navLabel: {yaml_quote(manifest['navLabel'])}",
        f"title: {yaml_quote(manifest['title'])}",
        f"welcomeTitle: {yaml_quote(manifest['welcomeTitle'])}",
        f"welcomeTagline: {yaml_quote(manifest['welcomeTagline'])}",
    ]
    if modules:
        lines.append("modules:")
        for module in modules:
            lines.append(f"  - id: {module['id']}")
            lines.append(f"    title: {yaml_quote(module['title'])}")
            lines.append(f"    blurb: {yaml_quote(module['blurb'])}")
            lines.append("    chapters:")
            for chapter_id, chapter_title in module["chapters"]:
                lines.append(f"      - id: {chapter_id}")
                lines.append(f"        title: {yaml_quote(chapter_title)}")
    if chapters:
        lines.append("chapters:")
        for chapter_id, chapter_title in chapters:
            lines.append(f"  - id: {chapter_id}")
            lines.append(f"    title: {yaml_quote(chapter_title)}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    legacy = read_legacy_registry()
    by_hash = {f"{lesson.topic_id}:{lesson.hash}": lesson for lesson in legacy}
    print(f"legacy registry: {len(legacy)} lessons")

    # Backend chapters keep their identity; capture order and titles as found.
    backend_chapters: list[tuple[str, str]] = []
    backend_lessons: dict[str, list[LegacyLesson]] = {}
    for lesson in legacy:
        if lesson.topic_id != "backend-design":
            continue
        chapter_id = lesson.hash.split("/")[0]
        if chapter_id not in backend_lessons:
            backend_lessons[chapter_id] = []
            backend_chapters.append((chapter_id, lesson.section_title))
        backend_lessons[chapter_id].append(lesson)

    placements: list[tuple[LegacyLesson, str, str, int]] = []  # lesson, topic, chapter, order
    for chapter_id, lessons in backend_lessons.items():
        for order, lesson in enumerate(lessons, start=1):
            placements.append((lesson, "backend-design", chapter_id, order))

    placed_genai: set[str] = set()
    for chapter_path, hashes in GENAI_PLACEMENT.items():
        for order, lesson_hash in enumerate(hashes, start=1):
            key = f"gen-ai-agentic-ai:{lesson_hash}"
            lesson = by_hash.get(key)
            if lesson is None:
                raise SystemExit(f"placement refers to unknown lesson: {lesson_hash}")
            placed_genai.add(lesson_hash)
            placements.append((lesson, "genai", chapter_path, order))

    missing = [
        lesson.hash
        for lesson in legacy
        if lesson.topic_id == "gen-ai-agentic-ai" and lesson.hash not in placed_genai
    ]
    if missing:
        raise SystemExit("Gen AI lessons with no destination: " + ", ".join(missing))

    redirects: list[tuple[str, str]] = []
    warnings: list[str] = []
    for lesson, topic, chapter_path, order in placements:
        if not lesson.source.exists():
            raise SystemExit(f"missing source file: {lesson.source}")
        converted = html_to_md.convert(lesson.source.read_text(encoding="utf-8"))
        title = converted.title or re.sub(r"^\d+\.\s*", "", lesson.label)
        description = converted.description or title
        slug = lesson.hash.split("/")[-1]

        md_path = CONTENT / topic / chapter_path / f"{order:02d}-{slug}.md"
        write(md_path, front_matter(title, description) + converted.markdown, args.dry_run)

        old_rel = f"{LEGACY_BASE[lesson.topic_id]}/{lesson.path}"
        new_rel = f"{TOPIC_TARGET[lesson.topic_id]}/{chapter_path}/{slug}.html"
        redirects.append((old_rel, new_rel))

        for warning in converted.warnings:
            warnings.append(f"{lesson.path}: {warning}")

    write(
        CONTENT / "backend-design" / "_topic.yml",
        build_manifest_yaml(BACKEND_MANIFEST, backend_chapters, None),
        args.dry_run,
    )
    write(
        CONTENT / "genai" / "_topic.yml",
        build_manifest_yaml(GENAI_MANIFEST, None, GENAI_MODULES),
        args.dry_run,
    )
    write(CONTENT / "topics.yml", "- backend-design\n- genai\n", args.dry_run)

    redirect_lines = ["# Old URL -> new URL. Consumed by tools/build_site.py to emit stubs."]
    for old, new in redirects:
        redirect_lines.append(f"- from: {yaml_quote(old)}")
        redirect_lines.append(f"  to: {yaml_quote(new)}")
    write(CONTENT / "redirects.yml", "\n".join(redirect_lines) + "\n", args.dry_run)

    print(f"placed {len(placements)} lessons")
    print(f"  backend-design: {sum(1 for p in placements if p[1] == 'backend-design')}")
    print(f"  genai:          {sum(1 for p in placements if p[1] == 'genai')}")
    print(f"redirects: {len(redirects)}")
    if warnings:
        print(f"\n{len(warnings)} conversion warnings:")
        for warning in warnings[:40]:
            print(f"  - {warning}")
    else:
        print("\nno conversion warnings")
    if args.dry_run:
        print("\n(dry run — nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
