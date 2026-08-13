"""Small SQLite ledger for idempotent ingestion and local lexical retrieval."""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from app.ingestion.chunker import Chunk


class IngestionLedger:
    def __init__(self, path: str) -> None:
        self.path = path
        with self._connect() as con:
            con.execute("CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL, chunk_ids TEXT NOT NULL, ingested_at TEXT NOT NULL)")
            con.execute("CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, payload TEXT NOT NULL)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def unchanged(self, doc_id: str, content_hash: str) -> bool:
        with self._connect() as con:
            row = con.execute("SELECT content_hash FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return bool(row and row[0] == content_hash)

    def has_document(self, doc_id: str) -> bool:
        with self._connect() as con:
            return con.execute("SELECT 1 FROM documents WHERE doc_id=?", (doc_id,)).fetchone() is not None

    def save(self, doc_id: str, content_hash: str, chunks: list[Chunk]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as con:
            con.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            con.executemany("INSERT INTO chunks(chunk_id, doc_id, payload) VALUES (?, ?, ?)", [(c.chunk_id, doc_id, json.dumps(c.__dict__)) for c in chunks])
            con.execute("INSERT INTO documents VALUES (?, ?, ?, ?) ON CONFLICT(doc_id) DO UPDATE SET content_hash=excluded.content_hash, chunk_ids=excluded.chunk_ids, ingested_at=excluded.ingested_at", (doc_id, content_hash, json.dumps([c.chunk_id for c in chunks]), now))

    def all_chunks(self) -> list[dict]:
        with self._connect() as con:
            return [json.loads(row[0]) for row in con.execute("SELECT payload FROM chunks ORDER BY doc_id, chunk_id")]
