"""Fail-closed marker mapping: only retrieved chunks can become citations."""
from __future__ import annotations
import re
from app.schemas import Citation, RetrievedChunk

def format_citations(raw_answer: str, chunks: list[RetrievedChunk]) -> tuple[str, list[Citation], int, list[int]]:
    invalid: list[int] = []
    def marker(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if not 1 <= index <= len(chunks):
            invalid.append(index)
            return ""
        return f"[{index}]"
    normalized = re.sub(r"\[(\d+)\]", marker, raw_answer)
    # Fail closed per sentence: retain only sentences that have a valid marker.
    # This preserves grounded portions of an otherwise overlong LLM response.
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", normalized) if sentence.strip()]
    supported_sentences = [sentence for sentence in sentences if re.search(r"\[\d+\]", sentence)]
    uncited = len(sentences) - len(supported_sentences)
    answer = " ".join(supported_sentences)
    indexes = []
    for value in re.findall(r"\[(\d+)\]", answer):
        index = int(value)
        if index not in indexes: indexes.append(index)
    citations = [Citation(index=index, chunk_id=chunks[index-1].chunk_id, doc_id=chunks[index-1].doc_id, doc_title=chunks[index-1].doc_title, section_title=chunks[index-1].section_title, char_range=(chunks[index-1].char_start, chunks[index-1].char_end), source_path=chunks[index-1].source_path) for index in indexes]
    return answer.strip(), citations, uncited, invalid
