# Submission checklist

- [x] Python FastAPI API: `POST /ingest`, `POST /query`, `GET /health`
- [x] Real Pinecone index, fixed `legal-docs` namespace, 1024-dimension Qwen embeddings
- [x] Idempotent ingestion ledger and scoped document deletion
- [x] Namespace-safe first ingestion regression test
- [x] Clause-aware Markdown chunking with source/section/character metadata
- [x] LangGraph conditional branch, retry path, and hard termination limit
- [x] Structured citations derived only from query-time retrieved chunks
- [x] Unsupported-question refusal path
- [x] Unit tests in `tests/`
- [x] Live API evaluation suite in `eval/`
- [x] Setup, ingestion, query, and validation instructions in the README

Before submitting, run the commands in the README and include the terminal output from `python eval/run_eval.py` as evidence of the current evaluation score.
