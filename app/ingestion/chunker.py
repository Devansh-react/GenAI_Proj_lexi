"""Clause-aware chunking: headings first, sentences only for oversized clauses."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import re
from app.ingestion.loader import LoadedDocument, Section


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    source_path: str
    chunk_index: int
    text: str
    section_title: str | None
    char_start: int
    char_end: int
    content_hash: str


def document_id(source_path: str) -> str:
    stem = source_path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")


def _token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _split_section(section: Section, max_tokens: int) -> list[tuple[str, int, int]]:
    if _token_count(section.text) <= max_tokens:
        return [(section.text, section.char_start, section.char_end)]
    sentences = re.split(r"(?<=[.!?])\s+", section.text)
    pieces: list[tuple[str, int, int]] = []
    current: list[str] = []
    for sentence in sentences:
        if current and _token_count(" ".join(current + [sentence])) > max_tokens:
            value = " ".join(current)
            offset = section.text.find(value)
            pieces.append((value, section.char_start + offset, section.char_start + offset + len(value)))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        value = " ".join(current)
        offset = section.text.rfind(value)
        pieces.append((value, section.char_start + offset, section.char_start + offset + len(value)))
    return pieces


def chunk_document(document: LoadedDocument, max_tokens: int = 600) -> list[Chunk]:
    doc_id = document_id(document.source_path)
    output: list[Chunk] = []
    for section in document.sections:
        for text, start, end in _split_section(section, max_tokens):
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            index = len(output)
            output.append(Chunk(f"{doc_id}__{index:04d}__{content_hash[:8]}", doc_id, document.title,
                document.source_path, index, text, section.title, start, end, content_hash))
    return output
