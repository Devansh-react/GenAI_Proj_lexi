"""The single graph state shape, mutated only by graph nodes."""
from typing import Literal, TypedDict
from app.schemas import Citation, RetrievedChunk, TraceEvent

class QAState(TypedDict, total=False):
    question: str
    retrieved_chunks: list[RetrievedChunk]
    retrieval_score: float
    retrieval_verdict: Literal["sufficient", "insufficient"]
    attempt_count: int
    max_attempts: int
    raw_answer: str
    answer: str
    citations: list[Citation]
    uncited_sentence_count: int
    status: Literal["ok", "insufficient_evidence", "error"]
    error_message: str
    trace: list[TraceEvent]
