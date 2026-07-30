#!/usr/bin/env python3
"""Generate the BackbenchLearner static site from Markdown sources.

    python3 tools/build_site.py            # build everything
    python3 tools/build_site.py --check    # report what would change, write nothing

Reads ``content/<topic>/_topic.yml`` manifests plus the lesson Markdown beside
them, and writes:

    <basePath>/<chapter>/<slug>.html      lesson pages
    scripts/curriculum.js                 registry the shell and lessons share
    sitemap.xml                           every live lesson URL
    <old path>.html                       redirect stubs from content/redirects.yml

The generated HTML is committed, so GitHub Pages still serves plain static
files. This script is a local authoring convenience, not a deploy step.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from html import escape as html_escape
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bblmd  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required: python3 -m pip install -r tools/requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"
TEMPLATES = ROOT / "tools" / "templates"
SITE_URL = "https://backbenchlearner.com"

# Reading time model. Prose is counted at WORDS_PER_MINUTE; diagrams, tables and
# code cost extra because they are studied, not skimmed.
WORDS_PER_MINUTE = 200
DIAGRAM_WORDS = 100
TABLE_WORDS = 60
CODE_LINE_WORDS = 4


@dataclass
class Lesson:
    slug: str
    title: str
    description: str
    source: Path
    chapter: "Chapter"
    order: int
    body_html: str
    minutes: int

    @property
    def rel_path(self) -> str:
        """Path of the generated file, relative to the topic's basePath."""
        return f"{self.chapter.rel_path}/{self.slug}.html"

    @property
    def hash(self) -> str:
        return f"{self.chapter.rel_path}/{self.slug}"

    @property
    def out_path(self) -> Path:
        return ROOT / self.chapter.topic.base_path / self.rel_path

    @property
    def canonical(self) -> str:
        return f"{SITE_URL}/{quote(self.chapter.topic.base_path)}/{quote(self.rel_path)}"


@dataclass
class Chapter:
    id: str
    title: str
    topic: "Topic"
    module_id: str | None
    order: int
    lessons: list[Lesson] = field(default_factory=list)
    quiz: dict | None = None

    @property
    def rel_path(self) -> str:
        return f"{self.module_id}/{self.id}" if self.module_id else self.id

    @property
    def quiz_id(self) -> str:
        return f"{self.topic.id}/{self.rel_path}"

    @property
    def minutes(self) -> int:
        return sum(lesson.minutes for lesson in self.lessons)


@dataclass
class Module:
    id: str
    title: str
    blurb: str = ""


@dataclass
class Topic:
    id: str
    base_path: str
    nav_label: str
    title: str
    welcome_title: str
    welcome_tagline: str
    modules: list[Module] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def split_front_matter(text: str, source: Path) -> tuple[dict, str]:
    match = FRONT_MATTER.match(text)
    if not match:
        raise SystemExit(f"{source}: missing front matter block")
    meta = yaml.safe_load(match.group(1)) or {}
    if not isinstance(meta, dict):
        raise SystemExit(f"{source}: front matter must be a mapping")
    return meta, text[match.end() :]


def reading_minutes(stats: bblmd.RenderStats) -> int:
    weighted = (
        stats.words
        + stats.diagrams * DIAGRAM_WORDS
        + stats.tables * TABLE_WORDS
        + stats.code_lines * CODE_LINE_WORDS
    )
    return max(1, math.ceil(weighted / WORDS_PER_MINUTE))


def load_lesson(path: Path, chapter: Chapter, order: int) -> Lesson:
    meta, body = split_front_matter(path.read_text(encoding="utf-8"), path)
    title = meta.get("title")
    if not title:
        raise SystemExit(f"{path}: front matter needs a title")
    description = meta.get("description", "").strip()
    if not description:
        raise SystemExit(f"{path}: front matter needs a description (used for SEO)")

    slug = meta.get("slug") or re.sub(r"^\d+[-_]", "", path.stem)
    body_html, stats = bblmd.render(body, indent=4)
    return Lesson(
        slug=slug,
        title=str(title).strip(),
        description=description,
        source=path,
        chapter=chapter,
        order=order,
        body_html=body_html,
        minutes=reading_minutes(stats),
    )


def load_quiz(path: Path, chapter: Chapter) -> dict | None:
    """Load a chapter quiz bank, validating it against the questions it claims."""
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    questions = data.get("questions") or []
    if not questions:
        raise SystemExit(f"{path}: quiz has no questions")

    cleaned = []
    for index, question in enumerate(questions, start=1):
        where = f"{path} question {index}"
        options = question.get("options") or []
        if len(options) < 2:
            raise SystemExit(f"{where}: needs at least two options")
        answer = question.get("answer")
        if not isinstance(answer, int) or not 0 <= answer < len(options):
            raise SystemExit(f"{where}: answer must be a 0-based index into options")
        if not question.get("q"):
            raise SystemExit(f"{where}: missing question text")
        cleaned.append(
            {
                "q": str(question["q"]).strip(),
                "options": [str(option).strip() for option in options],
                "answer": answer,
                "why": str(question.get("why", "")).strip(),
            }
        )
    return {
        "id": chapter.quiz_id,
        "title": data.get("title", chapter.title),
        "questions": cleaned,
    }


def load_topic(manifest_path: Path) -> Topic:
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    topic = Topic(
        id=data["id"],
        base_path=data.get("basePath", data["id"]),
        nav_label=data.get("navLabel", data.get("title", data["id"])),
        title=data.get("title", data["id"]),
        welcome_title=data.get("welcomeTitle", data.get("title", data["id"])),
        welcome_tagline=data.get("welcomeTagline", ""),
    )

    topic_dir = manifest_path.parent
    chapter_specs: list[tuple[str | None, dict]] = []
    for module_spec in data.get("modules", []) or []:
        topic.modules.append(
            Module(
                id=module_spec["id"],
                title=module_spec["title"],
                blurb=module_spec.get("blurb", ""),
            )
        )
        for chapter_spec in module_spec.get("chapters", []) or []:
            chapter_specs.append((module_spec["id"], chapter_spec))
    for chapter_spec in data.get("chapters", []) or []:
        chapter_specs.append((None, chapter_spec))

    for index, (module_id, spec) in enumerate(chapter_specs):
        chapter = Chapter(
            id=spec["id"],
            title=spec["title"],
            topic=topic,
            module_id=module_id,
            order=index,
        )
        chapter_dir = topic_dir / chapter.rel_path
        if not chapter_dir.is_dir():
            print(f"  ! {topic.id}/{chapter.rel_path}: no content directory yet, skipping")
            continue
        sources = sorted(p for p in chapter_dir.glob("*.md") if not p.name.startswith("_"))
        for order, source in enumerate(sources, start=1):
            chapter.lessons.append(load_lesson(source, chapter, order))
        if not chapter.lessons:
            print(f"  ! {topic.id}/{chapter.rel_path}: no lessons yet, skipping")
            continue
        chapter.quiz = load_quiz(chapter_dir / "_quiz.yml", chapter)
        topic.chapters.append(chapter)

    return topic


def load_topics() -> list[Topic]:
    manifests = sorted(CONTENT.glob("*/_topic.yml"))
    if not manifests:
        raise SystemExit(f"no topic manifests found under {CONTENT}")
    order_file = CONTENT / "topics.yml"
    topics = [load_topic(path) for path in manifests]
    if order_file.exists():
        wanted = yaml.safe_load(order_file.read_text(encoding="utf-8")) or []
        by_id = {topic.id: topic for topic in topics}
        ordered = [by_id[tid] for tid in wanted if tid in by_id]
        ordered += [topic for topic in topics if topic.id not in wanted]
        return ordered
    return topics


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


class Writer:
    """Writes files, skipping no-op writes so mtimes and diffs stay quiet."""

    def __init__(self, check_only: bool) -> None:
        self.check_only = check_only
        self.written: list[Path] = []
        self.unchanged = 0

    def write(self, path: Path, content: str) -> None:
        if path.exists() and path.read_text(encoding="utf-8") == content:
            self.unchanged += 1
            return
        self.written.append(path)
        if self.check_only:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def relative_root(depth: int) -> str:
    return "/".join([".."] * depth) if depth else "."


def escape_attr(value: str) -> str:
    return html_escape(value, quote=True)


def render_lesson(lesson: Lesson, template: str) -> str:
    topic = lesson.chapter.topic
    depth = 1 + lesson.rel_path.count("/")
    # Titles and labels land in both attributes and text, so escape quotes too.
    replacements = {
        "{{TITLE}}": escape_attr(lesson.title),
        "{{DESCRIPTION}}": escape_attr(lesson.description),
        "{{CANONICAL}}": lesson.canonical,
        "{{REL}}": relative_root(depth),
        "{{BODY}}": lesson.body_html,
        "{{TOPIC_ID}}": topic.id,
        "{{TOPIC_LABEL}}": escape_attr(topic.nav_label),
        "{{LESSON_HASH}}": lesson.hash,
        "{{SOURCE}}": str(lesson.source.relative_to(ROOT)),
    }
    out = template
    for token, value in replacements.items():
        out = out.replace(token, value)
    return out


def build_curriculum_js(topics: list[Topic]) -> str:
    registry: dict[str, object] = {}
    for topic in topics:
        registry[topic.id] = {
            "id": topic.id,
            "basePath": topic.base_path,
            "navLabel": topic.nav_label,
            "title": topic.title,
            "welcomeTitle": topic.welcome_title,
            "welcomeTagline": topic.welcome_tagline,
            "modules": [
                {"id": module.id, "title": module.title, "blurb": module.blurb}
                for module in topic.modules
            ],
            "chapters": [
                {
                    "id": chapter.rel_path,
                    "module": chapter.module_id,
                    "title": chapter.title,
                    "quizId": chapter.quiz_id,
                    "hasQuiz": chapter.quiz is not None,
                    "minutes": chapter.minutes,
                    "lessons": [
                        {
                            "slug": lesson.slug,
                            "hash": lesson.hash,
                            "path": lesson.rel_path,
                            "label": lesson.title,
                            "minutes": lesson.minutes,
                        }
                        for lesson in chapter.lessons
                    ],
                }
                for chapter in topic.chapters
            ],
        }

    payload = json.dumps(registry, indent=2, ensure_ascii=False)
    order = json.dumps([topic.id for topic in topics], ensure_ascii=False)
    return (
        "/* Generated by tools/build_site.py — do not edit by hand. */\n"
        "window.BBL = window.BBL || {};\n"
        f"window.BBL.TOPIC_ORDER = {order};\n"
        f"window.BBL.CURRICULUM = {payload};\n"
    )


def load_mock_banks() -> dict[str, dict]:
    """Load module/topic mock exams from content/<topic>/_mocks/*.yml."""
    banks: dict[str, dict] = {}
    for path in sorted(CONTENT.glob("*/_mocks/*.yml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        mock_id = str(data.get("id") or f"{path.parent.parent.name}/mock/{path.stem}")
        questions = data.get("questions") or []
        cleaned = []
        for index, item in enumerate(questions, start=1):
            options = item.get("options") or []
            answer = item.get("answer")
            if not item.get("q") or len(options) < 2 or answer is None:
                raise SystemExit(f"{path}: question {index} needs q, options, answer")
            if not isinstance(answer, int) or answer < 0 or answer >= len(options):
                raise SystemExit(f"{path}: question {index} has invalid answer index")
            cleaned.append(
                {
                    "q": item["q"],
                    "options": [str(opt) for opt in options],
                    "answer": answer,
                    "why": str(item.get("why", "")),
                    "section": str(item.get("section", "General")),
                }
            )
        banks[mock_id] = {
            "id": mock_id,
            "title": data.get("title", path.stem),
            "minutes": int(data.get("minutes", 90)),
            "kind": "mock",
            "questions": cleaned,
        }
    return banks


@dataclass
class CheatSheet:
    topic_id: str
    topic_label: str
    slug: str
    title: str
    description: str
    minutes: int
    module: str
    order: int
    body_html: str
    source: Path

    @property
    def rel_path(self) -> str:
        return f"cheatsheets/{self.topic_id}/{self.slug}.html"

    @property
    def sheet_id(self) -> str:
        return f"{self.topic_id}/{self.slug}"

    @property
    def out_path(self) -> Path:
        return ROOT / self.rel_path

    @property
    def canonical(self) -> str:
        return f"{SITE_URL}/{quote(self.rel_path)}"


def load_cheatsheets(topics: list[Topic]) -> list[CheatSheet]:
    """Load revision guides from content/<topic>/_cheatsheets/*.md."""
    by_id = {topic.id: topic for topic in topics}
    sheets: list[CheatSheet] = []
    for topic_dir in sorted(CONTENT.glob("*/_cheatsheets")):
        topic_id = topic_dir.parent.name
        topic = by_id.get(topic_id)
        if not topic:
            print(f"  ! cheatsheets for unknown topic {topic_id}, skipping")
            continue
        for order, path in enumerate(sorted(topic_dir.glob("*.md")), start=1):
            meta, body = split_front_matter(path.read_text(encoding="utf-8"), path)
            title = meta.get("title")
            if not title:
                raise SystemExit(f"{path}: cheatsheet needs a title")
            description = str(meta.get("description", "")).strip() or f"Revision guide: {title}"
            slug = str(meta.get("slug") or re.sub(r"^\d+[-_]", "", path.stem))
            minutes = int(meta.get("minutes", 30))
            module = str(meta.get("module", ""))
            body_html, _stats = bblmd.render(body, indent=4)
            sheets.append(
                CheatSheet(
                    topic_id=topic_id,
                    topic_label=topic.nav_label,
                    slug=slug,
                    title=str(title).strip(),
                    description=description,
                    minutes=minutes,
                    module=module,
                    order=order,
                    body_html=body_html,
                    source=path,
                )
            )
    return sheets


def render_cheatsheet(sheet: CheatSheet, template: str) -> str:
    depth = sheet.rel_path.count("/")
    replacements = {
        "{{TITLE}}": escape_attr(sheet.title),
        "{{DESCRIPTION}}": escape_attr(sheet.description),
        "{{CANONICAL}}": sheet.canonical,
        "{{REL}}": relative_root(depth),
        "{{BODY}}": sheet.body_html,
        "{{TOPIC_ID}}": sheet.topic_id,
        "{{TOPIC_LABEL}}": escape_attr(sheet.topic_label),
        "{{SHEET_ID}}": sheet.sheet_id,
        "{{MINUTES}}": str(sheet.minutes),
        "{{SOURCE}}": str(sheet.source.relative_to(ROOT)),
    }
    out = template
    for token, value in replacements.items():
        out = out.replace(token, value)
    return out


def build_cheatsheets_js(sheets: list[CheatSheet]) -> str:
    registry: dict[str, list[dict]] = {}
    for sheet in sheets:
        registry.setdefault(sheet.topic_id, []).append(
            {
                "id": sheet.sheet_id,
                "slug": sheet.slug,
                "title": sheet.title,
                "description": sheet.description,
                "minutes": sheet.minutes,
                "module": sheet.module,
                "path": sheet.rel_path,
            }
        )
    payload = json.dumps(registry, indent=2, ensure_ascii=False)
    return (
        "/* Generated by tools/build_site.py from content/_cheatsheets — do not edit by hand. */\n"
        "window.BBL = window.BBL || {};\n"
        f"window.BBL.CHEATSHEETS = {payload};\n"
    )


def build_quizzes_js(topics: list[Topic]) -> str:
    banks: dict[str, dict] = {}
    for topic in topics:
        for chapter in topic.chapters:
            if chapter.quiz:
                banks[chapter.quiz_id] = chapter.quiz
    banks.update(load_mock_banks())
    payload = json.dumps(banks, indent=2, ensure_ascii=False)
    return (
        "/* Generated by tools/build_site.py from content/_quiz.yml and _mocks — do not edit by hand. */\n"
        "window.BBL = window.BBL || {};\n"
        f"window.BBL.QUIZZES = {payload};\n"
    )


def build_sitemap(topics: list[Topic], redirects: list[dict], sheets: list[CheatSheet] | None = None) -> str:
    urls = [f"{SITE_URL}/"]
    for topic in topics:
        for chapter in topic.chapters:
            for lesson in chapter.lessons:
                urls.append(lesson.canonical)
            if chapter.quiz:
                urls.append(f"{SITE_URL}/quiz.html?t={quote(topic.id)}&amp;c={quote(chapter.rel_path)}")
    for sheet in sheets or []:
        urls.append(sheet.canonical)

    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for url in urls:
        lines.append(f"  <url><loc>{url}</loc></url>")
    lines.append("</urlset>")
    return "\n".join(lines) + "\n"


def build_redirects(writer: Writer, topics: list[Topic]) -> int:
    spec_path = CONTENT / "redirects.yml"
    if not spec_path.exists():
        return 0
    entries = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or []
    template = (TEMPLATES / "redirect.html").read_text(encoding="utf-8")

    lessons_by_rel: dict[str, Lesson] = {}
    for topic in topics:
        for chapter in topic.chapters:
            for lesson in chapter.lessons:
                lessons_by_rel[f"{topic.base_path}/{lesson.rel_path}"] = lesson

    count = 0
    for entry in entries:
        old = entry["from"]
        new = entry["to"]
        lesson = lessons_by_rel.get(new)
        if lesson is None:
            print(f"  ! redirect target missing, skipped: {new}")
            continue
        old_path = ROOT / old
        depth = old.count("/")
        target = "/".join([".."] * depth) + "/" + quote(new) if depth else quote(new)
        html = template
        for token, value in {
            "{{TITLE}}": escape_attr(lesson.title),
            "{{TARGET}}": target,
            "{{CANONICAL}}": lesson.canonical,
            "{{REL}}": relative_root(depth),
        }.items():
            html = html.replace(token, value)
        writer.write(old_path, html)
        count += 1
    return count


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report changes without writing")
    args = parser.parse_args()

    topics = load_topics()
    template = (TEMPLATES / "lesson.html").read_text(encoding="utf-8")
    cheat_template = (TEMPLATES / "cheatsheet.html").read_text(encoding="utf-8")
    writer = Writer(check_only=args.check)

    lesson_count = 0
    for topic in topics:
        for chapter in topic.chapters:
            for lesson in chapter.lessons:
                writer.write(lesson.out_path, render_lesson(lesson, template))
                lesson_count += 1

    sheets = load_cheatsheets(topics)
    for sheet in sheets:
        writer.write(sheet.out_path, render_cheatsheet(sheet, cheat_template))

    writer.write(ROOT / "scripts" / "curriculum.js", build_curriculum_js(topics))
    writer.write(ROOT / "scripts" / "quizzes.js", build_quizzes_js(topics))
    writer.write(ROOT / "scripts" / "cheatsheets.js", build_cheatsheets_js(sheets))
    writer.write(ROOT / "sitemap.xml", build_sitemap(topics, [], sheets))
    redirect_count = build_redirects(writer, topics)

    quiz_count = sum(1 for topic in topics for chapter in topic.chapters if chapter.quiz)
    question_count = sum(
        len(chapter.quiz["questions"])
        for topic in topics
        for chapter in topic.chapters
        if chapter.quiz
    )
    mock_banks = load_mock_banks()
    mock_q = sum(len(bank["questions"]) for bank in mock_banks.values())

    verb = "would write" if args.check else "wrote"
    print(f"\n{lesson_count} lessons across {len(topics)} topics")
    for topic in topics:
        total = sum(len(chapter.lessons) for chapter in topic.chapters)
        minutes = sum(chapter.minutes for chapter in topic.chapters)
        print(
            f"  {topic.id:14s} {total:3d} lessons  {len(topic.chapters):2d} chapters  "
            f"{minutes:4d} min  ({minutes // 60}h {minutes % 60:02d}m)"
        )
    print(f"{quiz_count} chapter quizzes, {question_count} chapter questions")
    print(f"{len(mock_banks)} module mocks, {mock_q} mock questions")
    print(f"{len(sheets)} cheat sheets ({sum(s.minutes for s in sheets)} min total)")
    print(f"{redirect_count} redirect stubs")
    print(f"{verb} {len(writer.written)} files, {writer.unchanged} unchanged")
    if args.check and writer.written:
        for path in writer.written[:20]:
            print(f"    ~ {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
