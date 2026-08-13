"""Stable fusion of Pinecone semantic and local lexical results."""
from __future__ import annotations
from datetime import datetime, timezone
from app.config import Settings
from app.schemas import RetrievedChunk
from app.retrieval.lexical_search import LexicalSearch


def hybrid_search(question: str, semantic_matches: list[dict], local_chunks: list[dict], settings: Settings, semantic_top_k: int, lexical_top_k: int, threshold: float, attempt: int) -> list[RetrievedChunk]:
    merged: dict[str, dict] = {}
    for result in semantic_matches:
        metadata = result["metadata"]
        merged[result["chunk_id"]] = {**metadata, "chunk_id": result["chunk_id"], "text": metadata["text"], "semantic_score": result["score"], "lexical_score": 0.0}
    for chunk, score in LexicalSearch().search(question, local_chunks, lexical_top_k):
        entry = merged.setdefault(chunk["chunk_id"], {**chunk, "semantic_score": 0.0, "lexical_score": 0.0})
        entry["lexical_score"] = score
    output: list[RetrievedChunk] = []
    for item in merged.values():
        semantic, lexical = float(item["semantic_score"]), float(item["lexical_score"])
        strategy = "hybrid" if semantic and lexical else ("semantic" if semantic else "lexical")
        # Pinecone similarity is the primary relevance measure. Lexical overlap
        # resolves close rankings, but must never dilute a strong semantic hit.
        final = semantic + (0.05 * lexical)
        output.append(RetrievedChunk(chunk_id=item["chunk_id"], doc_id=item["doc_id"], doc_title=item["doc_title"], source_path=item["source_path"], chunk_index=int(item["chunk_index"]), char_start=int(item["char_start"]), char_end=int(item["char_end"]), section_title=item.get("section_title"), content_hash=item["content_hash"], embedding_model=item.get("embedding_model", settings.embedding_model), embedding_revision=item.get("embedding_revision", settings.embedding_revision), ingested_at=item.get("ingested_at", datetime.now(timezone.utc).isoformat()), text=item["text"], semantic_score=semantic, lexical_score=lexical, final_score=final, retrieval_strategy=strategy, top_k=semantic_top_k, threshold=threshold, attempt=attempt))
    return sorted(output, key=lambda c: (-c.final_score, c.chunk_id))
