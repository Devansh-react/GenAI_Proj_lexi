"""FastAPI HTTP surface for the legal document Q&A service."""
from __future__ import annotations
import logging
from fastapi import FastAPI, HTTPException
from app.config import get_settings
from app.schemas import ErrorResponse, HealthResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse, RetrievalDiagnostics
from app.retrieval.embeddings import EmbeddingClient
from app.retrieval.pinecone_client import PineconeStore, EmbeddingModelMismatch
from app.ingestion.ledger import IngestionLedger
from app.ingestion.pipeline import IngestionPipeline
from app.generation.llm_client import MistralClient
from app.graph.nodes import Nodes
from app.graph.build_graph import build_graph, recursion_limit_for

logger = logging.getLogger(__name__)

settings = get_settings(); embeddings = EmbeddingClient(settings); store = PineconeStore(settings); ledger = IngestionLedger(settings.ledger_path)
pipeline = IngestionPipeline(settings, embeddings, store, ledger); graph = build_graph(Nodes(settings, embeddings, store, ledger, MistralClient(settings)))
app = FastAPI(title="Document-Grounded Legal Q&A API", version="1.0.0")

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok" if settings.pinecone_api_key and settings.mistral_api_key else "degraded", pinecone_configured=bool(settings.pinecone_api_key), generation_configured=bool(settings.mistral_api_key))

@app.post("/ingest", response_model=IngestResponse, responses={502: {"model": ErrorResponse}})
def ingest(request: IngestRequest) -> IngestResponse:
    try: return IngestResponse(ingested=[pipeline.ingest(item.source_path, item.content) for item in request.documents])
    except Exception as exc:
        # Keep the client response stable while retaining the complete upstream
        # traceback in Uvicorn's terminal for operational diagnosis.
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=502, detail=ErrorResponse(error_message=str(exc)).model_dump())

@app.post("/query", response_model=QueryResponse, responses={409: {"model": ErrorResponse}, 502: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def query(request: QueryRequest) -> QueryResponse:
    try:
        state = graph.invoke(
            {"question": request.question, "attempt_count": 0, "max_attempts": settings.max_attempts, "trace": []},
            config={"recursion_limit": recursion_limit_for(settings.max_attempts)},
        )
    except Exception as exc:
        logger.exception("Query graph failed")
        raise HTTPException(status_code=500, detail=ErrorResponse(error_message="query graph execution failed").model_dump())
    if state.get("status") == "error":
        message = state.get("error_message", "internal service error")
        code = 409 if "different embedding model" in message else 502
        raise HTTPException(status_code=code, detail=ErrorResponse(error_message=message, trace=state.get("trace", [])).model_dump())
    return QueryResponse(status=state["status"], answer=state["answer"], citations=state.get("citations", []), retrieval=RetrievalDiagnostics(verdict=state.get("retrieval_verdict", "insufficient"), score=state.get("retrieval_score", 0.0), attempts=state.get("attempt_count", 0), chunks=state.get("retrieved_chunks", [])), trace=state.get("trace", []))
