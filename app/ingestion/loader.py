"""Markdown loader that preserves title, section order, and character ranges."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Section:
    title: str | None
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class LoadedDocument:
    source_path: str
    title: str
    content: str
    sections: list[Section]


def load_markdown(source_path: str, content: str | None = None) -> LoadedDocument:
    text = content if content is not None else Path(source_path).read_text(encoding="utf-8")
    title_match = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
    title = title_match.group(1) if title_match else Path(source_path).stem.replace("_", " ")
    headings = list(re.finditer(r"^##\s+(.+?)\s*$", text, re.MULTILINE))
    sections: list[Section] = []
    if not headings:
        return LoadedDocument(source_path, title, text, [Section(None, text.strip(), 0, len(text))])
    preamble_end = headings[0].start()
    preamble = text[:preamble_end].strip()
    if preamble:
        start = text.index(preamble)
        sections.append(Section("Overview", preamble, start, start + len(preamble)))
    for index, heading in enumerate(headings):
        start = heading.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body = text[start:end].strip()
        if body:
            body_start = text.find(body, start, end)
            sections.append(Section(heading.group(1), body, body_start, body_start + len(body)))
    return LoadedDocument(source_path, title, text, sections)
