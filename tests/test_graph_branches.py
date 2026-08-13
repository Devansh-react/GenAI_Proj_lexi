from app.graph.evaluator import evaluate
from app.graph.build_graph import recursion_limit_for
from app.schemas import RetrievedChunk
def test_low_score_is_insufficient():
    c = RetrievedChunk(chunk_id="c", doc_id="d", doc_title="d", source_path="d", chunk_index=0, char_start=0, char_end=1, content_hash="h", embedding_model="m", embedding_revision="r", ingested_at="n", text="x", final_score=.2)
    assert evaluate([c], .62)[1] == "insufficient"


def test_strong_direct_match_is_sufficient_even_with_noisy_tail():
    direct = RetrievedChunk(chunk_id="direct", doc_id="d", doc_title="d", source_path="d", chunk_index=0, char_start=0, char_end=1, content_hash="h", embedding_model="m", embedding_revision="r", ingested_at="n", text="notice period is 60 days", semantic_score=.68, lexical_score=.375, final_score=.70)
    distractor = RetrievedChunk(chunk_id="noise", doc_id="d", doc_title="d", source_path="d", chunk_index=1, char_start=2, char_end=3, content_hash="i", embedding_model="m", embedding_revision="r", ingested_at="n", text="unrelated", semantic_score=.20, final_score=.20)
    score, verdict, _ = evaluate([direct, distractor], .58)
    assert score == .68
    assert verdict == "sufficient"


def test_lexical_overlap_can_corroborate_a_short_numeric_clause():
    c = RetrievedChunk(chunk_id="c", doc_id="d", doc_title="d", source_path="d", chunk_index=0, char_start=0, char_end=1, content_hash="h", embedding_model="m", embedding_revision="r", ingested_at="n", text="70% of invoices", semantic_score=.55, lexical_score=.60, final_score=.58)
    assert evaluate([c], .58)[1] == "sufficient"


def test_recursion_limit_covers_two_retrieval_attempts_and_refusal():
    # 2 × (retrieve + assemble_context + evaluate) + insufficient_evidence.
    assert recursion_limit_for(2) >= 7
