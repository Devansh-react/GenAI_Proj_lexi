"""Regression: a fresh Pinecone index has no namespace until its first upsert."""
from pathlib import Path
from app.config import Settings
from app.ingestion.ledger import IngestionLedger
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.pinecone_client import PineconeStore


class EmptyNamespaceIndex:
    def __init__(self):
        self.delete_called = False

    def describe_index_stats(self):
        return {"namespaces": {}}

    def delete(self, **_kwargs):
        self.delete_called = True
        raise AssertionError("delete must not run for a missing namespace")


class FakeEmbeddings:
    def encode_document(self, _text):
        return [0.0] * 1024


class FirstIngestStore:
    def __init__(self):
        self.index = EmptyNamespaceIndex()
        self.upserted = []

    def delete_by_doc_id(self, doc_id):
        # Exercise PineconeStore's exact namespace guard with a fake index.
        store = PineconeStore(Settings(pinecone_api_key="test"))
        store._index = self.index
        store.delete_by_doc_id(doc_id)

    def upsert(self, vectors):
        self.upserted.extend(vectors)


def test_first_ingest_skips_missing_namespace_delete_and_upserts(tmp_path: Path):
    settings = Settings(ledger_path=str(tmp_path / "ledger.sqlite3"))
    store = FirstIngestStore()
    pipeline = IngestionPipeline(settings, FakeEmbeddings(), store, IngestionLedger(settings.ledger_path))

    result = pipeline.ingest("first.md", "# First\n\n## Clause\n\nThe notice period is 60 days.")

    assert result.status == "new"
    assert not store.index.delete_called
    assert store.upserted
