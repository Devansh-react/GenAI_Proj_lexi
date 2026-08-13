# Document-Grounded Legal Q&A API

A FastAPI + LangGraph RAG API for the supplied fictional legal corpus. It retrieves evidence from Pinecone, produces only cited answers, and returns a fixed refusal whenever the documents do not support an answer.

## What is included

- Qwen `Qwen/Qwen3-Embedding-0.6B` embeddings pinned by model id and revision.
- Pinecone cosine index (`docqa-index`, 1024 dimensions) and fixed `legal-docs` namespace.
- Structure-aware Markdown ingestion with a SQLite idempotency ledger.
- Hybrid retrieval: Pinecone semantic search plus deterministic local lexical support.
- Bounded LangGraph retrieval/retry/answer workflow. See [docs/LANGGRAPH.md](docs/LANGGRAPH.md).
- Citation mapping that can cite only chunks retrieved in the same request.
- Unit tests and a progressive live evaluation suite.

## Prerequisites

- Python 3.10+ (tested with Python 3.12)
- A Pinecone API key
- A Mistral API key with access to `ministral-8b-latest` (or change `MISTRAL_MODEL`)

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set these values in `.env`:

```env
PINECONE_API_KEY=your_pinecone_key
MISTRAL_API_KEY=your_mistral_key
```

The first embedding request downloads the Qwen model (about 1.2 GB); this is expected. Pinecone automatically creates `docqa-index` if needed. If the embedding model or revision changes, use a fresh index or re-ingest the corpus.

## Run the API

```powershell
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for Swagger UI.

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Ingest the corpus

Submit this JSON in Swagger `POST /ingest`, or use it with PowerShell. Only ingest the six legal corpus files; do not ingest architecture documents.

```json
{
  "documents": [
    {"source_path": "sample_docs/01_matter_memo_arvind_v_northfield.md"},
    {"source_path": "sample_docs/02_employment_agreement_excerpt.md"},
    {"source_path": "sample_docs/03_hearing_notice_template.md"},
    {"source_path": "sample_docs/04_statute_style_excerpt_fictional.md"},
    {"source_path": "sample_docs/05_counsel_notes_settlement.md"},
    {"source_path": "sample_docs/06_property_lease_clause.md"}
  ]
}
```

Re-submitting unchanged files returns `status: "unchanged"` and avoids both re-embedding and duplicate vectors.

## Query the API

```powershell
Invoke-RestMethod http://127.0.0.1:8000/query -Method Post -ContentType application/json -Body '{"question":"What is the notice period in the employment agreement?"}' | ConvertTo-Json -Depth 10
```

Expected grounded result: an answer containing `60 days` with inline `[1]` and a structured citation. Test refusal behavior with:

```json
{"question":"What is Arvind Mehta's blood type?"}
```

Expected status: `insufficient_evidence`.

## Quality checks

Run unit tests:

```powershell
python -m pytest -q
```

Run the live evaluation after ingestion and while Uvicorn is running:

```powershell
python eval/run_eval.py
```

The suite includes 18 progressive tests: 12 grounded questions across all six documents and six unsupported/adversarial questions. It reports overall score, retrieval hit rate, citation validity, and refusal accuracy.

See [docs/SUBMISSION_CHECKLIST.md](docs/SUBMISSION_CHECKLIST.md) before handoff.
