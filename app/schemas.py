"""Shared, typed API and persistence data shapes."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    doc_id: str
    doc_title: str
    source_path: str
    chunk_index: int
    char_start: int
    char_end: int
    section_title: str | None = None
    content_hash: str
    embedding_model: str
    embedding_revision: str
    ingested_at: str


class RetrievedChunk(ChunkMetadata):
    chunk_id: str
    text: str
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    final_score: float = 0.0
    retrieval_strategy: str = "semantic"
    top_k: int = 0
    threshold: float = 0.0
    attempt: int = 0


class Citation(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    doc_title: str
    section_title: str | None = None
    char_range: tuple[int, int]
    source_path: str


class TraceEvent(BaseModel):
    node: str
    duration_ms: float
    notes: str


class RetrievalDiagnostics(BaseModel):
    verdict: Literal["sufficient", "insufficient"]
    score: float
    attempts: int
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    status: Literal["ok", "insufficient_evidence"]
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieval: RetrievalDiagnostics
    trace: list[TraceEvent] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error_message: str
    trace: list[TraceEvent] = Field(default_factory=list)


class IngestDocument(BaseModel):
    source_path: str = Field(min_length=1)
    content: str | None = None


class IngestRequest(BaseModel):
    documents: list[IngestDocument] = Field(min_length=1)


class IngestResult(BaseModel):
    doc_id: str
    chunks_created: int
    status: Literal["new", "updated", "unchanged"]


class IngestResponse(BaseModel):
    ingested: list[IngestResult]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    pinecone_configured: bool
    generation_configured: bool
