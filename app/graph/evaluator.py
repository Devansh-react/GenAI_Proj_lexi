"""Simple, deterministic evidence gate for a small clause-based corpus."""
from app.schemas import RetrievedChunk

def evaluate(chunks: list[RetrievedChunk], threshold: float) -> tuple[float, str, int]:
    """Accept when at least one retrieved clause is a strong semantic match.

    Legal questions in this assignment are answered by one short clause. An
    average across the remaining (necessarily weaker) top-k results created
    false refusals, so it is intentionally not part of the gate.
    """
    if not chunks:
        return 0.0, "insufficient", 0
    best = max(chunks, key=lambda chunk: (chunk.semantic_score, chunk.lexical_score, chunk.chunk_id))
    score = best.semantic_score
    semantic_hit = any(chunk.semantic_score >= threshold for chunk in chunks)
    # Numbers and short legal clauses can have modest embedding similarity but
    # strong exact-term support. Lexical overlap may corroborate such a match;
    # it never stands alone, which keeps unrelated questions from passing.
    corroborated_hit = any(
        chunk.semantic_score >= 0.50 and chunk.lexical_score > 0.0
        for chunk in chunks
    )
    coverage = sum(
        chunk.semantic_score >= threshold
        or (chunk.semantic_score >= 0.50 and chunk.lexical_score > 0.0)
        for chunk in chunks
    )
    return score, ("sufficient" if semantic_hit or corroborated_hit else "insufficient"), coverage
