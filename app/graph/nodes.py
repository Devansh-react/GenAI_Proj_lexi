"""Graph nodes; domain failures become state errors rather than uncaught graph exits."""
from __future__ import annotations
from time import perf_counter
from app.config import Settings
from app.graph.evaluator import evaluate
from app.graph.state import QAState
from app.schemas import TraceEvent
from app.retrieval.embeddings import EmbeddingClient
from app.retrieval.pinecone_client import PineconeStore
from app.retrieval.hybrid import hybrid_search
from app.ingestion.ledger import IngestionLedger
from app.generation.llm_client import MistralClient
from app.generation.prompts import build_prompt
from app.citations.formatter import format_citations

REFUSAL = "I could not find supporting information for this question in the ingested documents."

class Nodes:
    def __init__(self, settings: Settings, embeddings: EmbeddingClient, store: PineconeStore, ledger: IngestionLedger, llm: MistralClient) -> None:
        self.settings, self.embeddings, self.store, self.ledger, self.llm = settings, embeddings, store, ledger, llm
    def _trace(self, state: QAState, node: str, started: float, notes: str) -> list[TraceEvent]:
        return [*state.get("trace", []), TraceEvent(node=node, duration_ms=round((perf_counter()-started)*1000, 2), notes=notes)]
    def retrieve(self, state: QAState) -> QAState:
        started = perf_counter()
        try:
            second = state.get("attempt_count", 0) > 0
            semantic_k, lexical_k = (10, 4) if second else (6, 2)
            attempt = state.get("attempt_count", 0) + 1
            threshold = 0.58 if second else self.settings.retrieval_threshold
            vector = self.embeddings.encode_query(state["question"])
            semantic = self.store.query(vector, semantic_k)
            chunks = hybrid_search(state["question"], semantic, self.ledger.all_chunks(), self.settings, semantic_k, lexical_k, threshold, attempt)
            return {"retrieved_chunks": chunks, "trace": self._trace(state, "retrieve", started, f"attempt={attempt}, semantic_top_k={semantic_k}, lexical_top_k={lexical_k}, threshold={threshold}, returned={len(chunks)}")}
        except Exception as exc:
            return {"status": "error", "error_message": str(exc), "trace": self._trace(state, "retrieve", started, "retrieval failed")}
    def assemble_context(self, state: QAState) -> QAState:
        started = perf_counter(); selected = []; budget = 0
        for chunk in sorted(state.get("retrieved_chunks", []), key=lambda c: (-c.final_score, c.chunk_id)):
            if any(chunk.doc_id == kept.doc_id and max(0, min(chunk.char_end, kept.char_end)-max(chunk.char_start, kept.char_start)) > 0.5 * min(chunk.char_end-chunk.char_start, kept.char_end-kept.char_start) for kept in selected): continue
            tokens = len(chunk.text.split())
            if budget + tokens <= self.settings.context_token_budget: selected.append(chunk); budget += tokens
        return {"retrieved_chunks": selected, "trace": self._trace(state, "assemble_context", started, f"chunks={len(selected)}, tokens≈{budget}")}
    def evaluate_retrieval(self, state: QAState) -> QAState:
        started = perf_counter(); attempt = state.get("attempt_count", 0) + 1
        threshold = 0.58 if attempt == 2 else self.settings.retrieval_threshold
        score, verdict, coverage = evaluate(state.get("retrieved_chunks", []), threshold)
        return {"attempt_count": attempt, "retrieval_score": score, "retrieval_verdict": verdict, "trace": self._trace(state, "evaluate_retrieval", started, f"attempt={attempt}, threshold={threshold}, score={score:.3f}, coverage={coverage}, verdict={verdict}")}
    def generate_answer(self, state: QAState) -> QAState:
        started = perf_counter()
        try:
            answer = self.llm.generate(build_prompt(state["question"], state["retrieved_chunks"]))
            return {"raw_answer": answer, "trace": self._trace(state, "generate_answer", started, "generation completed")}
        except Exception as exc: return {"status": "error", "error_message": str(exc), "trace": self._trace(state, "generate_answer", started, "generation failed")}
    def format_citations(self, state: QAState) -> QAState:
        started = perf_counter(); answer, citations, uncited, invalid = format_citations(state.get("raw_answer", ""), state.get("retrieved_chunks", []))
        # Strict mode is fail-closed: an answer without valid citations is not
        # returned as grounded legal advice.
        if not citations or invalid:
            return {"answer": REFUSAL, "citations": [], "uncited_sentence_count": uncited, "status": "insufficient_evidence", "trace": self._trace(state, "format_citations", started, f"rejected_answer: citations={len(citations)}, removed_uncited={uncited}, invalid_markers={invalid}")}
        # Prevent a nearby clause from being used to invent a missing field.
        # E.g. the lease prohibits subletting but states no penalty; the
        # employment agreement identifies Priya but contains no salary.
        cited_ids = {citation.chunk_id for citation in citations}
        evidence = " ".join(
            chunk.text.lower()
            for chunk in state.get("retrieved_chunks", [])
            if chunk.chunk_id in cited_ids
        )
        question = state["question"].lower()
        unsupported_fields = {"penalty": ("penalty", "fine", "damages"), "salary": ("salary", "pay", "compensation", "wage")}
        for field, terms in unsupported_fields.items():
            if field in question and not any(term in evidence for term in terms):
                return {"answer": REFUSAL, "citations": [], "uncited_sentence_count": uncited, "status": "insufficient_evidence", "trace": self._trace(state, "format_citations", started, f"rejected_missing_field={field}")}
        return {"answer": answer, "citations": citations, "uncited_sentence_count": uncited, "status": "ok", "trace": self._trace(state, "format_citations", started, f"citations={len(citations)}, removed_uncited={uncited}")}
    def insufficient_evidence(self, state: QAState) -> QAState:
        return {"answer": REFUSAL, "citations": [], "status": "insufficient_evidence", "trace": self._trace(state, "insufficient_evidence", perf_counter(), "fixed refusal")}
    def error_handler(self, state: QAState) -> QAState:
        return {"status": "error"}
