from app.ingestion.loader import load_markdown
from app.ingestion.chunker import chunk_document
def test_sections_become_clause_chunks():
    doc = load_markdown("agreement.md", "# Agreement\n\n## Notice period\n\n60 days.\n\n## Non-compete\n\n12 months.")
    chunks = chunk_document(doc)
    assert [c.section_title for c in chunks] == ["Overview", "Notice period", "Non-compete"]
    assert chunks[1].text == "60 days."
