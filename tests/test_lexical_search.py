from app.retrieval.lexical_search import LexicalSearch


def test_lexical_search_uses_document_and_section_metadata():
    chunks = [
        {"chunk_id": "lease", "doc_title": "Lease — Unit 4B", "section_title": "Rent and deposit", "text": "Monthly rent: ₹45,000."},
        {"chunk_id": "other", "doc_title": "Agreement", "section_title": "Notice", "text": "60 days."},
    ]
    ranked = LexicalSearch().search("What is the monthly rent for Unit 4B?", chunks, 2)
    assert ranked[0][0]["chunk_id"] == "lease"
    assert ranked[0][1] > 0
