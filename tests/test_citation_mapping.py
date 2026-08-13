from app.citations.formatter import format_citations
from app.schemas import RetrievedChunk
def chunk():
    return RetrievedChunk(chunk_id="c1", doc_id="d", doc_title="D", source_path="d.md", chunk_index=0, char_start=0, char_end=3, content_hash="x", embedding_model="m", embedding_revision="r", ingested_at="now", text="fact")
def test_invalid_marker_is_dropped_not_invented():
    answer, citations, _, invalid = format_citations("Fact [1]. Bad [9].", [chunk()])
    assert answer == "Fact [1]." and citations[0].chunk_id == "c1" and invalid == [9]
