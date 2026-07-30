"""Markdown renderer for BackbenchLearner lessons.

Deliberately small and purpose-built: it supports exactly the constructs the
lesson template uses, and emits the class names the site's CSS expects
(``.diagram-wrap`` for Mermaid, ``.callout`` for asides, ``language-*`` on code).

Supported block syntax
    ## Heading / ### Heading / #### Heading
    Paragraphs separated by blank lines
    - unordered lists (nest with two spaces)
    1. ordered lists (nest with two spaces)
    ```python fenced code (```mermaid becomes a diagram, ```html is raw passthrough)
    | GFM | tables |
    > blockquotes
    :::note Optional title ... ::: callouts (note / tip / warn / key)
    --- horizontal rule

Supported inline syntax
    **bold**  *italic*  `code`  [text](url)  \\* escapes
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

# Lines carrying this prefix are preformatted: the indenter must leave their
# leading whitespace exactly as authored, or Python and Mermaid both break.
RAW = "\x00raw\x00"

CALLOUT_TYPES = {"note", "tip", "warn", "key"}
CALLOUT_DEFAULT_TITLES = {
    "note": "Note",
    "tip": "Tip",
    "warn": "Watch out",
    "key": "Key idea",
}


@dataclass
class RenderStats:
    """Counts used to estimate reading time."""

    words: int = 0
    diagrams: int = 0
    tables: int = 0
    code_lines: int = 0
    headings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Inline
# --------------------------------------------------------------------------

_CODE_SPAN = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", re.DOTALL)
_ITALIC = re.compile(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])", re.DOTALL)
_ESCAPE = re.compile(r"\\([\\`*_\[\]<>&|])")

_SENTINEL = "\x00BBL{}\x00"


def render_inline(text: str) -> str:
    """Render inline markup, escaping everything that is not a construct."""
    stash: list[str] = []

    def keep(fragment: str) -> str:
        stash.append(fragment)
        return _SENTINEL.format(len(stash) - 1)

    # Backslash escapes are stashed first so they survive every later pass.
    text = _ESCAPE.sub(lambda m: keep(html.escape(m.group(1))), text)
    text = _CODE_SPAN.sub(lambda m: keep(f"<code>{html.escape(m.group(1))}</code>"), text)
    text = _LINK.sub(
        lambda m: keep(
            '<a href="{}">{}</a>'.format(html.escape(m.group(2), quote=True), html.escape(m.group(1)))
        ),
        text,
    )

    text = html.escape(text)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    text = _ITALIC.sub(r"<em>\1</em>", text)

    for index, fragment in enumerate(stash):
        text = text.replace(_SENTINEL.format(index), fragment)
    return text


def plain_text(text: str) -> str:
    """Strip markup so word counts are not inflated by syntax characters."""
    text = _CODE_SPAN.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = re.sub(r"[*_>#|`\-]", " ", text)
    return text


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------

_HEADING = re.compile(r"^(#{2,4})\s+(.*)$")
_FENCE = re.compile(r"^```([\w+-]*)\s*$")
_CALLOUT_OPEN = re.compile(r"^:::(\w+)\s*(.*)$")
_LIST_ITEM = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
_TABLE_DIVIDER = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")


def slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_]+", "-", slug).strip("-")


class _Renderer:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.pos = 0
        self.out: list[str] = []
        self.stats = RenderStats()

    # -- helpers ---------------------------------------------------------
    def peek(self) -> str | None:
        return self.lines[self.pos] if self.pos < len(self.lines) else None

    def count_words(self, text: str) -> None:
        self.stats.words += len(re.findall(r"\w+", plain_text(text)))

    # -- top level -------------------------------------------------------
    def run(self) -> tuple[str, RenderStats]:
        while self.pos < len(self.lines):
            line = self.lines[self.pos]
            if not line.strip():
                self.pos += 1
                continue
            if _FENCE.match(line):
                self.fenced_block()
            elif _CALLOUT_OPEN.match(line) and _CALLOUT_OPEN.match(line).group(1) in CALLOUT_TYPES:
                self.callout()
            elif _HEADING.match(line):
                self.heading()
            elif line.strip() in {"---", "***", "___"}:
                self.out.append("<hr />")
                self.pos += 1
            elif line.lstrip().startswith(">"):
                self.blockquote()
            elif self.is_table_start():
                self.table()
            elif _LIST_ITEM.match(line):
                self.list_block(indent=0)
            else:
                self.paragraph()
        return "\n".join(self.out), self.stats

    # -- block handlers --------------------------------------------------
    def heading(self) -> None:
        match = _HEADING.match(self.lines[self.pos])
        assert match
        level = len(match.group(1))
        text = match.group(2).strip()
        self.count_words(text)
        if level == 2:
            self.stats.headings.append(text)
            self.out.append(f'<h2 id="{slugify(text)}">{render_inline(text)}</h2>')
        else:
            self.out.append(f"<h{level}>{render_inline(text)}</h{level}>")
        self.pos += 1

    def fenced_block(self) -> None:
        lang = _FENCE.match(self.lines[self.pos]).group(1).lower()
        self.pos += 1
        body: list[str] = []
        while self.pos < len(self.lines) and not _FENCE.match(self.lines[self.pos]):
            body.append(self.lines[self.pos])
            self.pos += 1
        self.pos += 1  # closing fence
        content = "\n".join(body)

        if lang == "mermaid":
            self.stats.diagrams += 1
            self.out.append('<div class="diagram-wrap">')
            self.out.append('  <div class="mermaid">')
            self.out.extend(RAW + line for line in content.split("\n"))
            self.out.append("  </div>")
            self.out.append("</div>")
            return

        if lang == "html":
            self.out.extend(RAW + line for line in content.split("\n"))
            return

        self.stats.code_lines += len([ln for ln in body if ln.strip()])
        label = html.escape(lang or "text", quote=True)
        escaped = html.escape(content).split("\n")
        self.out.append(f'<div class="code-block" data-lang="{label}">')
        self.out.append('  <button class="code-copy" type="button" aria-label="Copy code">Copy</button>')
        # The first code line stays glued to <code> so browsers do not render a
        # leading blank line, and the closing tags stay glued to the last line.
        opening = f'  <pre><code class="language-{label}">'
        if len(escaped) == 1:
            self.out.append(f"{opening}{escaped[0]}</code></pre>")
        else:
            self.out.append(opening + escaped[0])
            self.out.extend(RAW + line for line in escaped[1:-1])
            self.out.append(RAW + escaped[-1] + "</code></pre>")
        self.out.append("</div>")

    def callout(self) -> None:
        match = _CALLOUT_OPEN.match(self.lines[self.pos])
        assert match
        kind = match.group(1)
        title = match.group(2).strip() or CALLOUT_DEFAULT_TITLES.get(kind, kind.title())
        self.pos += 1
        body: list[str] = []
        while self.pos < len(self.lines) and self.lines[self.pos].strip() != ":::":
            body.append(self.lines[self.pos])
            self.pos += 1
        self.pos += 1  # closing :::

        inner, inner_stats = _Renderer(body).run()
        self.stats.words += inner_stats.words + len(re.findall(r"\w+", title))
        self.stats.diagrams += inner_stats.diagrams
        self.stats.tables += inner_stats.tables
        self.stats.code_lines += inner_stats.code_lines
        self.out.append(f'<aside class="callout callout-{kind}">')
        self.out.append(f'  <p class="callout-title">{render_inline(title)}</p>')
        self.out.append(inner)
        self.out.append("</aside>")

    def blockquote(self) -> None:
        body: list[str] = []
        while self.pos < len(self.lines) and self.lines[self.pos].lstrip().startswith(">"):
            body.append(re.sub(r"^\s*>\s?", "", self.lines[self.pos]))
            self.pos += 1
        inner, inner_stats = _Renderer(body).run()
        self.stats.words += inner_stats.words
        self.out.append(f"<blockquote>\n{inner}\n</blockquote>")

    def is_table_start(self) -> bool:
        line = self.lines[self.pos]
        nxt = self.lines[self.pos + 1] if self.pos + 1 < len(self.lines) else ""
        return "|" in line and bool(_TABLE_DIVIDER.match(nxt))

    @staticmethod
    def split_row(line: str) -> list[str]:
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        return [cell.strip() for cell in stripped.split("|")]

    def table(self) -> None:
        header = self.split_row(self.lines[self.pos])
        aligns = []
        for spec in self.split_row(self.lines[self.pos + 1]):
            if spec.startswith(":") and spec.endswith(":"):
                aligns.append("center")
            elif spec.endswith(":"):
                aligns.append("right")
            else:
                aligns.append("")
        self.pos += 2

        rows: list[list[str]] = []
        while self.pos < len(self.lines) and "|" in self.lines[self.pos] and self.lines[self.pos].strip():
            rows.append(self.split_row(self.lines[self.pos]))
            self.pos += 1

        self.stats.tables += 1
        for cell in header:
            self.count_words(cell)
        for row in rows:
            for cell in row:
                self.count_words(cell)

        def style(index: int) -> str:
            align = aligns[index] if index < len(aligns) else ""
            return f' style="text-align:{align}"' if align else ""

        self.out.append("<table>")
        self.out.append("  <thead><tr>")
        for index, cell in enumerate(header):
            self.out.append(f"    <th{style(index)}>{render_inline(cell)}</th>")
        self.out.append("  </tr></thead>")
        self.out.append("  <tbody>")
        for row in rows:
            self.out.append("    <tr>")
            for index, cell in enumerate(row):
                self.out.append(f"      <td{style(index)}>{render_inline(cell)}</td>")
            self.out.append("    </tr>")
        self.out.append("  </tbody>")
        self.out.append("</table>")

    def list_block(self, indent: int) -> None:
        first = _LIST_ITEM.match(self.lines[self.pos])
        assert first
        ordered = first.group(2)[0].isdigit()
        tag = "ol" if ordered else "ul"
        self.out.append(f"<{tag}>")

        def same_kind(candidate: re.Match[str] | None) -> bool:
            return bool(candidate) and candidate.group(2)[0].isdigit() == ordered

        while self.pos < len(self.lines):
            line = self.lines[self.pos]
            if not line.strip():
                # A blank line ends the list unless the same kind of list resumes.
                nxt = self.lines[self.pos + 1] if self.pos + 1 < len(self.lines) else ""
                if not same_kind(_LIST_ITEM.match(nxt)):
                    break
                self.pos += 1
                continue
            match = _LIST_ITEM.match(line)
            if not match or not same_kind(match):
                break
            if len(match.group(1)) < indent:
                break

            text = match.group(3).strip()
            self.count_words(text)
            self.pos += 1

            # Collect a nested list, if the following lines are indented items.
            nested_lines: list[str] = []
            while self.pos < len(self.lines):
                nxt = self.lines[self.pos]
                nxt_match = _LIST_ITEM.match(nxt)
                if nxt_match and len(nxt_match.group(1)) > indent:
                    nested_lines.append(nxt)
                    self.pos += 1
                else:
                    break

            if nested_lines:
                dedented = [ln[indent + 2 :] if len(ln) > indent + 2 else ln.strip() for ln in nested_lines]
                inner, inner_stats = _Renderer(dedented).run()
                self.stats.words += inner_stats.words
                self.out.append(f"<li>{render_inline(text)}\n{inner}\n</li>")
            else:
                self.out.append(f"<li>{render_inline(text)}</li>")

        self.out.append(f"</{tag}>")

    def paragraph(self) -> None:
        body: list[str] = []
        while self.pos < len(self.lines):
            line = self.lines[self.pos]
            if not line.strip():
                break
            if (
                _HEADING.match(line)
                or _FENCE.match(line)
                or _LIST_ITEM.match(line)
                or line.lstrip().startswith(">")
                or line.strip().startswith(":::")
                or line.strip() in {"---", "***", "___"}
                or self.is_table_start()
            ):
                break
            body.append(line.strip())
            self.pos += 1
        text = " ".join(body)
        if text:
            self.count_words(text)
            self.out.append(f"<p>{render_inline(text)}</p>")


def finalize(rendered: str, indent: int = 0) -> str:
    """Apply block indentation, leaving preformatted lines untouched."""
    pad = " " * indent
    out = []
    for line in rendered.split("\n"):
        if line.startswith(RAW):
            out.append(line[len(RAW) :])
        elif line.strip():
            out.append(pad + line)
        else:
            out.append(line)
    return "\n".join(out)


def render(markdown_text: str, indent: int = 0) -> tuple[str, RenderStats]:
    """Render Markdown to HTML, returning the HTML and reading-time stats."""
    lines = markdown_text.replace("\r\n", "\n").replace("\t", "    ").split("\n")
    rendered, stats = _Renderer(lines).run()
    return finalize(rendered, indent), stats
