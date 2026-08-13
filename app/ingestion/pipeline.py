"""Idempotent loader → chunker → embed → Pinecone pipeline."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
from app.config import Settings
from app.ingestion.loader import load_markdown
from app.ingestion.chunker import chunk_document
from app.ingestion.ledger import IngestionLedger
from app.retrieval.embeddings import EmbeddingClient
from app.retrieval.pinecone_client import PineconeStore
from app.schemas import IngestResult


class IngestionPipeline:
    def __init__(self, settings: Settings, embeddings: EmbeddingClient, store: PineconeStore, ledger: IngestionLedger) -> None:
        self.settings, self.embeddings, self.store, self.ledger = settings, embeddings, store, ledger

    def ingest(self, source_path: str, content: str | None = None) -> IngestResult:
        document = load_markdown(source_path, content)
        chunks = chunk_document(document)
        file_hash = hashlib.sha256(document.content.encode()).hexdigest()
        doc_id = chunks[0].doc_id if chunks else "empty-document"
        if self.ledger.unchanged(doc_id, file_hash):
            return IngestResult(doc_id=doc_id, chunks_created=0, status="unchanged")
        existed = self.ledger.has_document(doc_id)
        self.store.delete_by_doc_id(doc_id)
        now = datetime.now(timezone.utc).isoformat()
        vectors = []
        for chunk in chunks:
            metadata = {"doc_id": chunk.doc_id, "doc_title": chunk.doc_title, "source_path": chunk.source_path, "chunk_index": chunk.chunk_index, "char_start": chunk.char_start, "char_end": chunk.char_end, "section_title": chunk.section_title or "", "content_hash": chunk.content_hash, "embedding_model": self.settings.embedding_model, "embedding_revision": self.settings.embedding_revision, "ingested_at": now, "text": chunk.text}
            vectors.append({"id": chunk.chunk_id, "values": self.embeddings.encode_document(chunk.text), "metadata": metadata})
        if vectors:
            self.store.upsert(vectors)
        self.ledger.save(doc_id, file_hash, chunks)
        return IngestResult(doc_id=doc_id, chunks_created=len(chunks), status="updated" if existed else "new")
