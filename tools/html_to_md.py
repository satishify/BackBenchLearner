"""One-off converter: existing hand-written lesson HTML into lesson Markdown.

Only needs to handle the shape the current pages actually use — a ``.wrapper``
containing a heading, paragraphs, lists, tables, and ``.diagram-wrap`` Mermaid
blocks. Anything unexpected is reported rather than silently dropped.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "source"}
BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "ul", "ol", "table", "div", "blockquote", "pre", "aside"}
SKIP_TAGS = {"nav", "footer", "script", "style"}


@dataclass
class Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list = field(default_factory=list)

    @property
    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())

    def find(self, tag: str, cls: str | None = None):
        for child in self.children:
            if isinstance(child, Node):
                if child.tag == tag and (cls is None or cls in child.classes):
                    return child
                found = child.find(tag, cls)
                if found is not None:
                    return found
        return None


class TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("#root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, {k: (v or "") for k, v in attrs})
        self.stack[-1].children.append(node)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(Node(tag, {k: (v or "") for k, v in attrs}))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------

_ESCAPE_MD = re.compile(r"([*_`\[\]])")


def escape_md(text: str) -> str:
    return _ESCAPE_MD.sub(r"\\\1", text)


def inline(node: Node | str, escape: bool = True) -> str:
    if isinstance(node, str):
        text = node.replace("\n", " ")
        return escape_md(text) if escape else text

    inner = "".join(inline(child, escape) for child in node.children)
    if node.tag in {"strong", "b"}:
        return f"**{inner.strip()}**" if inner.strip() else ""
    if node.tag in {"em", "i"}:
        return f"*{inner.strip()}*" if inner.strip() else ""
    if node.tag == "code":
        return f"`{raw_text(node)}`"
    if node.tag == "a":
        href = node.attrs.get("href", "")
        return f"[{inner.strip()}]({href})"
    if node.tag == "br":
        return " "
    return inner


def raw_text(node: Node | str) -> str:
    if isinstance(node, str):
        return node
    return "".join(raw_text(child) for child in node.children)


def collapse(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


class Converter:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def blocks(self, node: Node, depth: int = 0) -> list[str]:
        out: list[str] = []
        for child in node.children:
            if isinstance(child, str):
                if child.strip():
                    self.warnings.append(f"loose text dropped: {collapse(child)[:60]!r}")
                continue
            out.extend(self.block(child, depth))
        return out

    def block(self, node: Node, depth: int) -> list[str]:
        tag = node.tag
        if tag in SKIP_TAGS:
            return []

        if tag == "div":
            if "diagram-wrap" in node.classes:
                mermaid = node.find("div", "mermaid")
                if mermaid is None:
                    self.warnings.append("diagram-wrap without .mermaid child")
                    return []
                lines = [line.rstrip() for line in html.unescape(raw_text(mermaid)).split("\n")]
                while lines and not lines[0]:
                    lines.pop(0)
                while lines and not lines[-1]:
                    lines.pop()
                return ["```mermaid\n" + "\n".join(lines) + "\n```"]
            return self.blocks(node, depth)

        if tag in {"h2", "h3", "h4"}:
            level = "#" * int(tag[1])
            return [f"{level} {collapse(inline(node))}"]

        if tag == "p":
            text = collapse(inline(node))
            return [text] if text else []

        if tag in {"ul", "ol"}:
            return ["\n".join(self.list_items(node, ordered=(tag == "ol"), depth=depth))]

        if tag == "table":
            return [self.table(node)]

        if tag == "blockquote":
            inner = self.blocks(node, depth)
            return ["\n".join("> " + line for chunk in inner for line in chunk.split("\n"))]

        if tag == "pre":
            code = node.find("code")
            language = ""
            if code is not None:
                for cls in code.classes:
                    if cls.startswith("language-"):
                        language = cls[len("language-") :]
            body = html.unescape(raw_text(node)).strip("\n")
            return [f"```{language}\n{body}\n```"]

        if tag == "hr":
            return ["---"]

        if tag in {"h1", "aside"}:
            return []

        self.warnings.append(f"unhandled <{tag}>")
        return self.blocks(node, depth)

    def list_items(self, node: Node, ordered: bool, depth: int) -> list[str]:
        lines: list[str] = []
        counter = 0
        for child in node.children:
            if not isinstance(child, Node) or child.tag != "li":
                continue
            counter += 1
            marker = f"{counter}." if ordered else "-"
            pad = "  " * depth

            own_inline: list[str] = []
            nested: list[str] = []
            for part in child.children:
                if isinstance(part, Node) and part.tag in {"ul", "ol"}:
                    nested.extend(self.list_items(part, ordered=(part.tag == "ol"), depth=depth + 1))
                elif isinstance(part, Node) and part.tag == "p":
                    own_inline.append(inline(part))
                else:
                    own_inline.append(inline(part))
            text = collapse("".join(own_inline))
            lines.append(f"{pad}{marker} {text}")
            lines.extend(nested)
        return lines

    def table(self, node: Node) -> str:
        rows: list[list[str]] = []
        header: list[str] = []
        for row in self.iter_rows(node):
            cells = [collapse(inline(cell)).replace("|", "\\|") for cell in row]
            is_header = any(
                isinstance(cell, Node) and cell.tag == "th" for cell in row
            )
            if is_header and not header:
                header = cells
            else:
                rows.append(cells)

        width = len(header) if header else (len(rows[0]) if rows else 0)
        if not header:
            header = [""] * width
            self.warnings.append("table without a header row")

        lines = ["| " + " | ".join(header) + " |", "|" + "|".join([" --- "] * width) + "|"]
        for row in rows:
            padded = row + [""] * (width - len(row))
            lines.append("| " + " | ".join(padded[:width]) + " |")
        return "\n".join(lines)

    @staticmethod
    def iter_rows(node: Node):
        def walk(current: Node):
            for child in current.children:
                if isinstance(child, Node):
                    if child.tag == "tr":
                        yield [c for c in child.children if isinstance(c, Node) and c.tag in {"td", "th"}]
                    else:
                        yield from walk(child)

        return walk(node)


@dataclass
class Converted:
    title: str
    description: str
    markdown: str
    warnings: list[str]


def convert(html_text: str) -> Converted:
    builder = TreeBuilder()
    builder.feed(html_text)
    root = builder.root

    description = ""
    for node in iter_nodes(root):
        if node.tag == "meta" and node.attrs.get("name") == "description":
            description = node.attrs.get("content", "").strip()
            break

    wrapper = root.find("div", "wrapper")
    if wrapper is None:
        raise ValueError("no .wrapper element found")

    h1 = wrapper.find("h1")
    title = collapse(inline(h1, escape=False)) if h1 is not None else ""

    converter = Converter()
    blocks = converter.blocks(wrapper)
    markdown = "\n\n".join(block.strip("\n") for block in blocks if block.strip())
    return Converted(title=title, description=description, markdown=markdown + "\n", warnings=converter.warnings)


def iter_nodes(node: Node):
    for child in node.children:
        if isinstance(child, Node):
            yield child
            yield from iter_nodes(child)
