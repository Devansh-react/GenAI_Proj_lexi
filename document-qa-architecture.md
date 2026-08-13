# Document-Grounded Q&A API — Architecture Design Document

**Scope:** Architecture only, no implementation code. Three candidate designs (Minimal, Balanced, Strong), evaluated against the assignment's hard requirements: Python 3.10+, LangGraph with real branching, Pinecone, FastAPI, ingestion, citation grounding, retrieval-quality gating, and loop/step protection.

---

## 0. Requirements Recap (all three options must satisfy)

| Requirement | Hard constraint |
|---|---|
| Language/runtime | Python 3.10+ |
| Orchestration | LangGraph, with a **real** conditional branch |
| Vector store | Pinecone (real, not mocked) |
| Interface | HTTP API, FastAPI |
| Ingestion | Documents → chunks → embeddings → Pinecone |
| Grounding | Answers cite source chunks |
| Refusal | If unsupported, explicitly say "insufficient evidence" |
| Retrieval evaluation | Retrieve → Evaluate → branch (generate vs. refuse/retry) |
| Safety | Max step count / loop protection in the graph |

Everything below is designed against this common core; the three options differ in *how much infrastructure, evaluation rigor, and operational polish* surrounds that core.

---

## 1. Common LangGraph Skeleton (all options share this shape)

This is the backbone the assignment explicitly asks for. All three options implement this graph; they differ in node internals, not topology (Option 3 adds two optional nodes, marked below).

### 1.1 State Object

```
QAState:
  query: str                      # original user question
  namespace: str                  # Pinecone namespace to search
  retrieved_chunks: list[Chunk]   # candidate evidence
  retrieval_score: float          # aggregate quality score from evaluator
  retrieval_verdict: "sufficient" | "insufficient"
  attempt_count: int              # retry counter (loop protection)
  max_attempts: int = 2           # hard cap
  rewritten_query: str | None     # only used on retry path
  answer: str | None
  citations: list[Citation]
  trace: list[TraceEvent]         # per-node debug log
  status: "ok" | "insufficient_evidence" | "error"
```

### 1.2 Node List

| Node | Responsibility |
|---|---|
| `retrieve` | Query Pinecone (top-k) into `namespace`, populate `retrieved_chunks` |
| `evaluate_retrieval` | Score relevance/coverage of `retrieved_chunks` vs `query`; set `retrieval_score`, `retrieval_verdict` |
| `rewrite_query` *(retry path only)* | Reformulate `query` → `rewritten_query`, increment `attempt_count` |
| `generate_answer` | LLM generates answer strictly from `retrieved_chunks`, attaches citations |
| `verify_citations` *(Option 3 only)* | Post-hoc check that every claim maps to a cited chunk id |
| `insufficient_evidence` | Terminal node producing the explicit refusal response |
| `error_handler` | Catches upstream exceptions, returns structured error |

### 1.3 Graph Topology (ASCII)

```
                    ┌─────────────┐
                    │   retrieve   │
                    └──────┬───────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ evaluate_retrieval  │
                 └──────────┬──────────┘
                             │
              ┌──────────────┴───────────────┐
              │ conditional_edge:              │
              │  retrieval_verdict?            │
              └──────────────┬───────────────┘
        sufficient            │            insufficient
              │                │                 │
              ▼                │                 ▼
     ┌─────────────────┐       │      attempt_count < max_attempts?
     │ generate_answer  │       │        │                  │
     └────────┬─────────┘       │       yes                 no
              │                 │        │                  │
              ▼                 │        ▼                  ▼
   (Opt.3) verify_citations     │  ┌───────────────┐  ┌─────────────────────┐
              │                 │  │ rewrite_query  │  │ insufficient_evidence│
              ▼                 │  └───────┬────────┘  └──────────┬───────────┘
            [END]               │          │                       │
                                 └──────────┘ (loop back to retrieve) [END]
```

### 1.4 Branching Conditions

1. **After `evaluate_retrieval`** — conditional edge on `retrieval_verdict`:
   - `sufficient` → `generate_answer`
   - `insufficient` → check `attempt_count`:
     - `< max_attempts` → `rewrite_query` → back to `retrieve` (bounded retry loop)
     - `>= max_attempts` → `insufficient_evidence` → `END`

2. **Termination conditions** (any one ends the graph):
   - `generate_answer` (and, in Option 3, `verify_citations`) completes → `END`
   - `insufficient_evidence` reached → `END`
   - `attempt_count >= max_attempts` → forced exit to `insufficient_evidence`
   - Unhandled exception in any node → `error_handler` → `END`

3. **Maximum step protection** — two independent layers, not just one:
   - **Semantic cap:** `max_attempts` (default 2, i.e. at most one retry) enforced explicitly inside the conditional edge logic.
   - **Structural cap:** LangGraph's own `recursion_limit` (e.g. set to 8) as a hard backstop, so a bug in the conditional-edge logic can never produce an infinite loop even if the semantic counter is mishandled. This "belt and suspenders" pairing is what the assignment's "maximum step / loop protection" line item is really asking for — a single counter is fragile if a node mutates state incorrectly; the graph-level recursion limit is the guaranteed backstop.

This skeleton is identical across all three options because it *is* the assignment's core deliverable. What changes below is retrieval evaluation sophistication, ingestion robustness, citation verification depth, and operational concerns.

---

## 2. Option 1 — Minimal (satisfies every requirement, nothing more)

**Target:** Fastest path to a fully compliant, honestly-working submission. Good if time is short or as a fallback if Option 2 slips.

### 2.1 Component Diagram

```
┌────────────┐     ┌───────────────┐     ┌─────────────┐
│  FastAPI    │────▶│  LangGraph     │────▶│  Pinecone    │
│  (2 routes) │     │  (7 nodes)     │     │  (1 index)   │
└────────────┘     └───────────────┘     └─────────────┘
       │                    │
       │                    ▼
       │            ┌───────────────┐
       │            │  LLM (1 model) │
       │            │  embed + gen   │
       │            └───────────────┘
       ▼
  ┌───────────┐
  │ local file │ (ingestion input: a folder of .txt/.md docs)
  └───────────┘
```

### 2.2 Data Flow
`docs/*.txt` → chunk (fixed-size) → embed → upsert to Pinecone (single namespace) → API receives question → graph runs → answer + citations returned.

### 2.3 LangGraph Flow
Exactly the common skeleton in §1, with `verify_citations` omitted. `evaluate_retrieval` uses a single, cheap signal: **top-1 cosine similarity score against a fixed threshold** (e.g. 0.75). No LLM-based judgment — keeps cost and latency low and keeps the branch logic trivially explainable in a design review.

### 2.4 Pinecone Interaction
- One index, one namespace (`default`).
- Ingestion: upsert with deterministic ids (see §5).
- Query: `top_k=4`, cosine similarity, no metadata filtering.

### 2.5 Ingestion Pipeline
Synchronous script/endpoint: read files → fixed 500-token chunks, 50-token overlap → embed (batch) → upsert. No dedup beyond deterministic ids (re-running ingestion on the same file is a no-op because ids are content-derived — see §5.4).

### 2.6 Query Pipeline
Single request/response cycle through the graph in §1, synchronous, no streaming.

### 2.7 Citation Strategy
Each retrieved chunk carries `{doc_id, chunk_index, source_path}`. The generation prompt instructs the LLM to reference chunks by id inline (e.g. `[doc1#3]`), and the API layer maps those ids back to metadata for the response's `citations` array. No independent verification that the citation is actually supported — trust the LLM's tagging.

### 2.8 Failure Handling
- Retrieval evaluator threshold miss → one retry, then explicit refusal (per skeleton).
- Pinecone/LLM call errors → caught at node boundary, mapped to HTTP 502 with a structured error body.
- No retries on transient network errors (acceptable for a take-home scope).

### 2.9 Scalability Considerations
Not a design goal here — single index, no batching optimization, synchronous ingestion. Explicitly out of scope; mention in the doc that this is a known, intentional limitation.

### 2.10 Implementation Complexity
**Low.** Roughly: 1 ingestion script, 1 graph module (~7 small nodes), 2 FastAPI routes (`/ingest`, `/query`), no auth, no persistence beyond Pinecone. Realistic to build well in under a day, leaving slack for the rest of a multi-day take-home.

**Risk:** a reviewer may read "minimal" as "did the bare minimum" even though it's fully compliant — the retrieval evaluator being a single similarity threshold is the most likely point of critique ("is this really *evaluating* quality, or just re-thresholding retrieval?").

---

## 3. Option 2 — Balanced (production-inspired, realistic for 3–5 days)

**Target:** The recommended default for most candidates — see §7. Demonstrates real engineering judgment without scope creep.

### 3.1 Component Diagram

```
┌─────────────┐      ┌────────────────────┐      ┌──────────────┐
│  FastAPI     │─────▶│  LangGraph Runner   │─────▶│  Pinecone     │
│  /ingest      │      │  (7-node graph,    │      │  index, ns    │
│  /query       │      │   retry-aware)      │      │  per corpus   │
│  /health      │      └────────┬───────────┘      └──────────────┘
└──────┬───────┘               │
       │                        ▼
       │               ┌────────────────────┐
       │               │  LLM Client          │
       │               │  - embeddings         │
       │               │  - generation          │
       │               │  - retrieval judge     │
       │               └────────────────────┘
       ▼
┌───────────────┐
│ Ingestion       │  (PDF/txt/md loaders, structure-aware chunker)
│ Pipeline        │
└───────────────┘
       │
       ▼
┌───────────────┐
│ Metadata store  │ (lightweight — could be Pinecone metadata itself,
│ (ingestion log) │  or a local SQLite table of doc→chunk→hash mappings)
└───────────────┘
```

Key addition over Option 1: a small **ingestion ledger** (SQLite is enough) that tracks `doc_id → content_hash → chunk_ids`, enabling real idempotency and re-ingestion diffing instead of relying purely on deterministic Pinecone ids.

### 3.2 Data Flow
Multiple file types (PDF, txt, md) → loader normalizes to text with page/section metadata → structure-aware chunker (see §6) → embed in batches → upsert to a **per-corpus namespace** → ledger updated → query time: question → namespace resolution → graph run → response with citations + trace.

### 3.3 LangGraph Flow
Full skeleton from §1 (still without `verify_citations`), but `evaluate_retrieval` upgrades from a raw similarity threshold to a **composite score**:
- similarity score (normalized)
- coverage heuristic: do retrieved chunks span more than one section/doc, or is everything from one weak match?
- an LLM-as-judge call ("given this question and these chunk excerpts, can this be answered? yes/no + reason") — cheap, single small-model call, *not* the same model used for final generation, so the evaluation is a genuinely independent signal rather than the generator grading its own retrieval.

`rewrite_query` node is a real reformulation step (LLM rewrites the query using the reason the judge gave for insufficiency), not just a blind retry with the same query — this is what makes the retry loop meaningfully different from attempt 1, and is a common gap reviewers flag in weaker submissions.

### 3.4 Pinecone Interaction
- One index, **namespace per document corpus/collection** (e.g. `namespace = corpus_id`), so multiple document sets can be ingested and queried independently without cross-contamination.
- Metadata filtering supported (e.g. filter by `doc_type`, `date_ingested`) — plumbed through the API but optional at query time.
- `top_k=6`, similarity + light MMR-style diversity re-ranking done client-side (cheap, no extra service) to avoid returning 6 near-duplicate chunks from the same paragraph.

### 3.5 Ingestion Pipeline
- Multi-format loaders (txt/md native; PDF via a text-extraction library).
- Structure-aware chunking: prefer paragraph/section boundaries over blind fixed windows (see §6).
- Content-hash-based idempotency: re-ingesting an unchanged file is a no-op; a changed file triggers delete-old-chunks + upsert-new-chunks for that `doc_id` (via the ledger).
- Batch embedding calls, basic backoff/retry on rate limits.

### 3.6 Query Pipeline
Same synchronous request/response as Option 1, but the response includes a `trace` array (one entry per node: name, duration, key state deltas) — directly useful for the "reviewer friendliness" and "clarity" goals since it makes the graph's decisions inspectable without reading logs.

### 3.7 Citation Strategy
- Chunk-level citations `{doc_id, chunk_id, section_title?, char_range, source_path}`.
- Generation prompt requires inline citation markers per sentence/claim, not just per answer.
- Response separates `answer_text` (clean prose) from `citations` (structured list), plus an `answer_with_markers` variant showing exactly where each citation applies — good for debugging without cluttering the primary answer.

### 3.8 Failure Handling
- Same two-layer step protection as §1.4.
- Bounded retry only on the *retrieval* side (rewrite+retry), never on the *generation* side (no silent re-generation loops — keeps the graph's control flow easy to reason about).
- Pinecone/embedding transient errors: retry with exponential backoff (2–3 attempts) before surfacing as a 502.
- Malformed/empty documents at ingestion time are logged and skipped, not fatal to the whole batch.

### 3.9 Scalability Considerations
Discussed but not over-built:
- Namespace-per-corpus is itself a scaling lever (query smaller, relevant slices instead of one giant index).
- Ingestion is batchable and could be moved to a background task/queue later; for the take-home it stays synchronous but is written so that swap is a small change (ingestion logic isolated from the FastAPI route).
- Note in the doc: horizontal scaling of the API is standard FastAPI/uvicorn worker scaling and isn't special-cased here.

### 3.10 Implementation Complexity
**Medium.** Extra pieces vs. Option 1: multi-format loaders, ledger table, composite retrieval evaluator, query rewriting, trace field. Realistic for 3–5 days by a strong intern, with margin for testing and the eval framework (§9). This is the sweet spot: every added component maps directly to an assignment requirement or an obvious grading criterion, with no infrastructure added purely for its own sake.

---

## 4. Option 3 — Strong (would score highly in a rigorous technical review)

**Target:** For a candidate confident in their time budget and wanting to demonstrate senior-level judgment: independent verification, evaluation harness, and operational maturity — without turning this into a distributed system.

### 4.1 Component Diagram

```
┌───────────────────────────────────────────────────────────┐
│                        FastAPI Layer                        │
│  /ingest   /query   /health   /corpora   /eval/run           │
└───────────┬───────────────────────────────┬─────────────────┘
            │                                 │
            ▼                                 ▼
 ┌─────────────────────┐          ┌────────────────────────┐
 │  Ingestion Pipeline    │          │  LangGraph Runner        │
 │  - loaders (pdf/md/txt)│          │  - retrieve                │
 │  - structure chunker    │          │  - evaluate_retrieval      │
 │  - dedup + hashing        │          │  - rewrite_query            │
 │  - batch embed + upsert    │          │  - generate_answer            │
 └──────────┬──────────────┘          │  - verify_citations             │
             │                          │  - insufficient_evidence          │
             ▼                          │  - error_handler                    │
 ┌─────────────────────┐          └───────────┬────────────────────┘
 │  Ingestion Ledger       │                       │
 │  (doc/chunk/hash/ns)     │                       ▼
 └──────────┬──────────────┘          ┌────────────────────────┐
             │                          │  LLM Client Layer         │
             ▼                          │  - embed                    │
 ┌─────────────────────┐          │  - generate                  │
 │      Pinecone           │◀─────────│  - judge (retrieval eval)     │
 │  index, ns per corpus     │          │  - judge (citation verify)      │
 └─────────────────────┘          └────────────────────────┘

 ┌─────────────────────────────────────────────────────────┐
 │  Evaluation Harness (offline + /eval/run)                    │
 │  JSON test cases → run graph → score retrieval/answer/citation│
 └─────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow
Same as Option 2, plus: every ingested chunk's embedding and metadata are versioned in the ledger with a `corpus_id` and `ingest_run_id`, enabling reproducible re-ingestion and rollback (delete by `ingest_run_id`) — useful if a bad parse pollutes a namespace.

### 4.3 LangGraph Flow
Full skeleton **including** `verify_citations`:

```
generate_answer → verify_citations → conditional edge:
    all claims supported?  → END (status="ok")
    unsupported claim(s)   → downgrade: strip unsupported
                              sentence(s) OR fall back to
                              insufficient_evidence if the
                              *entire* answer is unsupported
                              → END
```

This closes the loop the assignment cares about most: it's not enough to *retrieve* well, the final answer must be checked against what was actually retrieved before it's returned. `verify_citations` is a second, independent LLM judge call (different prompt, ideally different/cheaper model) that takes `(answer, citations, retrieved_chunks)` and flags any sentence whose claim isn't backed by its cited chunk. This node still respects the same step-protection contract: it runs at most once per generation (no loop back into itself), so it adds a bounded, constant amount of work regardless of `attempt_count`.

`evaluate_retrieval` in this option is the same composite scorer as Option 2, but the score and reasoning are persisted in `trace` for later offline analysis via the evaluation harness — the design explicitly treats the graph's intermediate judgments as data, not just control flow.

### 4.4 Pinecone Interaction
Same as Option 2 (namespace-per-corpus, metadata filtering, MMR-style diversity), plus:
- Metadata includes `ingest_run_id` for rollback.
- Query-time metadata filters exposed in the API (`doc_type`, `date_range`, `source`) so the evaluation harness can run targeted regression tests against specific document subsets.

### 4.5 Ingestion Pipeline
Same as Option 2, plus:
- Explicit duplicate handling policy across *different* source files with overlapping content (near-duplicate detection via a cheap similarity check on chunk embeddings at ingest time, flagged in metadata rather than silently duplicated in results).
- Idempotent re-ingestion is transactional at the ledger level: if the upsert step fails partway, the ledger entry stays in a `pending` state and a re-run resumes cleanly rather than double-upserting.

### 4.6 Query Pipeline
Same as Option 2's traced response, plus:
- `/eval/run` endpoint accepts a JSON test-case file and returns aggregate metrics (see §9) — makes evaluation a first-class API capability, not just an offline script, which is a strong signal in a technical review because it shows the candidate thought about *how this system proves itself*, not just how it answers one question.

### 4.7 Citation Strategy
Same structured citations as Option 2, plus:
- `verify_citations` output is surfaced in the response as `citation_confidence` per claim, and any stripped/unsupported claims are noted in `trace` — full transparency instead of silently editing the answer.

### 4.8 Failure Handling
Same as Option 2, plus:
- Partial-answer degradation path: if some claims are supported and others aren't, the system can return the supported subset with a note, rather than a binary all-or-nothing outcome — configurable behavior (`strict` vs `partial`) exposed as a request parameter, defaulting to `strict` for assignment compliance ("must explicitly say insufficient" implies strict-by-default is the safe reading of the spec).

### 4.9 Scalability Considerations
Actually addressed with concrete, still-realistic levers:
- Namespace-per-corpus + metadata filtering as the primary scale-out mechanism for multi-tenant document sets.
- Ingestion ledger enables incremental re-ingestion (only changed docs are re-embedded), which is the dominant cost driver at scale.
- Graph nodes are stateless and side-effect-isolated (all Pinecone/LLM calls go through the client layer), so moving from synchronous FastAPI to an async task queue for ingestion is a contained change, explicitly called out as the first thing to do post-take-home if this went to production.

### 4.10 Implementation Complexity
**High relative to a take-home**, but each addition is justified: `verify_citations` directly answers the assignment's grounding requirement at the highest rigor level; the evaluation harness directly answers the "how do you know it works" question every reviewer will ask; the ledger directly answers "idempotent ingestion" and "duplicate handling" line items instead of hand-waving them. Real risk: this is the option most likely to run over a 3–5 day budget if the candidate isn't disciplined, especially the near-duplicate detection and partial-answer degradation features, which are the two easiest pieces to cut if time gets tight without weakening the core story.

---

## 5. Pinecone Design (applies across options, differences noted)

### 5.1 Index Schema
- One index for the assignment (`docqa-index`), dimensionality matched to the embedding model in use.
- Metric: cosine.
- Serverless/pod spec is an infra choice, not an architecture one — note it, don't over-specify.

### 5.2 Namespace Strategy
- Option 1: single `default` namespace.
- Option 2/3: `namespace = corpus_id` (one namespace per logical document collection). This is the cleanest way to support "multiple document sets" without needing multiple indexes, and it directly limits blast radius if one corpus needs to be wiped and re-ingested.

### 5.3 Metadata Fields (per vector)
```
{
  doc_id: str,
  doc_title: str,
  source_path: str,
  chunk_index: int,
  char_start: int,
  char_end: int,
  section_title: str | None,
  ingest_run_id: str,          # Option 2/3
  content_hash: str,           # Option 2/3
  ingested_at: iso8601 str
}
```

### 5.4 Chunk ID Strategy
Deterministic, content-derived id: `chunk_id = sha256(f"{doc_id}:{chunk_index}:{content_hash}")[:16]`. Deterministic ids are what make upserts naturally idempotent — re-running ingestion on an unchanged file produces the same ids, so Pinecone's upsert simply overwrites identical vectors rather than duplicating them.

### 5.5 Idempotent Ingestion Strategy
- Option 1: rely purely on deterministic ids (§5.4) — good enough, minimal.
- Option 2/3: deterministic ids **plus** the ledger's content-hash check, so unchanged files are skipped entirely (no re-embedding cost), and changed files trigger a scoped delete-then-upsert for just that `doc_id`'s old chunk ids.

### 5.6 Duplicate Handling
- Same file re-ingested → no-op (idempotent id match).
- Same content under a different `doc_id`/filename → Option 1/2 treat as legitimately distinct documents (simplest, defensible default). Option 3 adds near-duplicate flagging at the embedding-similarity level so query-time results don't return 3 versions of the same paragraph without at least surfacing that they're duplicates.

---

## 6. Chunking Strategy

| Aspect | Recommendation |
|---|---|
| Chunk size | 400–600 tokens (target ~500). Large enough to preserve local context for grounding, small enough to keep citations precise and retrieval focused. |
| Overlap | 10–15% (≈50–75 tokens) — enough to avoid severing a claim across a chunk boundary, not so much that it inflates index size and near-duplicate noise. |
| Boundary preference | Prefer semantic/structural boundaries (paragraph, heading, list item) over blind fixed-width slicing wherever the loader can detect them (Option 2/3); Option 1 can use fixed-width with overlap as a defensible simplification. |
| Metadata attached | doc_id, chunk_index, section_title (if available), char_start/char_end, source_path — this is what makes citations pin-pointable rather than just "somewhere in this document." |
| Citation formatting | `[<doc_title>, chunk #<chunk_index>]` inline in generation, expanded server-side to `{doc_id, doc_title, chunk_id, section_title, source_path, char_range}` in the structured `citations` response field. Keep the inline marker short (cheap for the LLM to emit correctly and consistently); keep the structured object rich (useful for the caller). |

---

## 7. API Design

### 7.1 Endpoints

| Method | Path | Purpose | Option |
|---|---|---|---|
| POST | `/ingest` | Ingest a document (or batch) into a corpus/namespace | 1/2/3 |
| POST | `/query` | Ask a question, get grounded answer + citations | 1/2/3 |
| GET | `/health` | Liveness/readiness | 1/2/3 |
| GET | `/corpora` | List known corpora/namespaces + doc counts | 2/3 |
| POST | `/eval/run` | Run a JSON test-case suite against the live graph | 3 |

### 7.2 Request Schema — `POST /query`
```json
{
  "question": "string, required",
  "corpus_id": "string, optional (defaults to 'default')",
  "top_k": "int, optional, default 6",
  "mode": "strict | partial, optional, default strict"   // Option 3 only
}
```

### 7.3 Response Schema — `POST /query`
```json
{
  "status": "ok | insufficient_evidence | error",
  "answer": "string | null",
  "citations": [
    {
      "chunk_id": "string",
      "doc_id": "string",
      "doc_title": "string",
      "section_title": "string | null",
      "char_range": [0, 0],
      "source_path": "string"
    }
  ],
  "retrieval": {
    "verdict": "sufficient | insufficient",
    "score": 0.0,
    "attempts": 1
  },
  "trace": [
    {"node": "retrieve", "duration_ms": 0, "notes": "string"}
  ]
}
```
`trace` is present in Options 2/3 by default; optional/omittable via a query param in Option 1 if included at all.

### 7.4 Request/Response Schema — `POST /ingest`
```json
// request
{ "corpus_id": "string", "documents": [{"source_path": "string", "content": "string | base64"}] }

// response
{
  "corpus_id": "string",
  "ingested": [{"doc_id": "string", "chunks_created": 0, "status": "new | updated | unchanged"}],
  "ingest_run_id": "string"   // Option 2/3
}
```

### 7.5 Trace/Debug Fields
- Per-node timing and key state deltas (`trace[]` above) — this single field is the highest-leverage addition for "reviewer friendliness": it lets a grader see *why* the graph branched the way it did without reading source code.
- Option 3 additionally exposes `citation_confidence` per claim and `unsupported_claims` (if any were stripped under `partial` mode).

---

## 8. Failure Handling Summary (cross-cutting)

| Failure | Handling |
|---|---|
| No relevant chunks found | `evaluate_retrieval` → insufficient → (bounded retry) → explicit refusal, never a hallucinated answer |
| Pinecone unavailable | Node-level exception → `error_handler` → HTTP 502, structured error body |
| LLM call fails/times out | Retry w/ backoff (Option 2/3), then `error_handler` |
| Malformed document at ingestion | Logged, skipped, batch continues (Option 2/3); Option 1 can fail the whole batch — acceptable but worth calling out as a known simplification |
| Answer contains unsupported claim | Option 3 only: `verify_citations` strips or downgrades to insufficient; Options 1/2 accept generation-time grounding as sufficient (a real, disclosed limitation) |
| Runaway loop | Two-layer step protection (semantic `max_attempts` + LangGraph `recursion_limit`) — identical across all three options |

---

## 9. Evaluation Framework

### 9.1 Retrieval Metrics
- **Hit rate@k** — did the correct source chunk appear in the top-k for a labeled test question?
- **MRR (mean reciprocal rank)** — how high did the correct chunk rank when it appeared?
- **Evaluator agreement** — how often does `evaluate_retrieval`'s verdict match a human label of "answerable from corpus: yes/no" (this directly measures whether the graph's branch is actually doing its job).

### 9.2 Answer Quality Checks
- **Groundedness** — every sentence in the answer traceable to a cited chunk (automatable via the same judge prompt used in `verify_citations`, or manually scored for Options 1/2).
- **Refusal correctness** — for questions deliberately outside the corpus, does the system correctly return `insufficient_evidence` instead of guessing? This is arguably the single most important metric for this assignment, since it's the one explicitly called out in the spec.
- **Answer relevance** — does the answer actually address the question asked (separate from whether it's grounded — an answer can be well-cited and still non-responsive).

### 9.3 Citation Verification
- **Precision** — of the citations returned, what fraction genuinely support the sentence they're attached to?
- **Recall** — of the claims in the answer, what fraction have *any* citation at all (uncited claims are a grounding failure even if the answer happens to be correct).

### 9.4 JSON Test-Case Format
```json
{
  "test_cases": [
    {
      "id": "tc-001",
      "corpus_id": "default",
      "question": "What is the refund window described in the policy?",
      "expected_answerable": true,
      "expected_chunk_ids": ["a1b2c3d4"],
      "expected_answer_contains": ["30 days"]
    },
    {
      "id": "tc-002",
      "corpus_id": "default",
      "question": "What is the CEO's favorite color?",
      "expected_answerable": false
    }
  ]
}
```
Harness runs each case through `/query`, compares `status`/`citations`/`answer` against expectations, and aggregates the §9.1–9.3 metrics. Option 3 exposes this as `/eval/run`; Options 1/2 can run it as an offline script over the same `/query` endpoint — the value of the test format doesn't depend on which option is chosen.

---

## 10. Tradeoff Summary

| Dimension | Option 1 (Minimal) | Option 2 (Balanced) | Option 3 (Strong) |
|---|---|---|---|
| Meets hard requirements | Yes | Yes | Yes |
| Retrieval evaluation rigor | Threshold only | Composite + independent judge | Composite + judge, persisted for offline eval |
| Citation trust model | Trust generation | Trust generation | Independently verified |
| Ingestion idempotency | Deterministic ids only | Ids + content-hash ledger | Ids + ledger + rollback + near-dup detection |
| Build time (realistic) | ~1 day | 3–5 days | 5+ days, real overrun risk |
| Reviewer "wow" factor | Low–medium | Medium–high | High, if finished; low if incomplete |
| Biggest risk | Reads as bare-minimum | None major — best risk/reward | Running out of time before polish |

---

## 11. Recommended Architecture for This Specific Take-Home

**Recommendation: Option 2 (Balanced), with two cheap upgrades borrowed from Option 3 if time allows — the `verify_citations` node and the `trace` field — but without the ledger's near-duplicate detection or the `/eval/run` endpoint.**

This maximizes the five stated goals as follows:

- **Correctness.** Option 2's composite retrieval evaluator (similarity + coverage + an independent LLM judge) is a meaningfully real implementation of "evaluate retrieval quality" — not just re-applying a similarity threshold (Option 1's main weakness) — while staying well short of Option 3's scope, where extra moving parts (ledger transactions, near-dup detection, partial-mode degradation) create more surface area for bugs relative to the time available. Adding `verify_citations` on top is a small, high-value addition: it directly closes the loop between "we retrieved good evidence" and "the answer actually used it," which is the crux of the assignment's grounding requirement, and it's a single extra node — cheap to add without pulling in the rest of Option 3's machinery.

- **Clarity.** The graph topology stays exactly the skeleton in §1 — retrieve → evaluate → branch → (retry or refuse) → generate → (verify) → end — which is easy to draw, easy to explain in five minutes, and easy for a reviewer to trace end-to-end. Option 3's additional endpoints and ledger states are harder to hold in your head all at once during a live walkthrough; Option 1's single-signal evaluator is *too* simple to demonstrate real design thinking about what "retrieval quality" means.

- **Reviewer friendliness.** The `trace` field is the single highest-leverage addition in this whole document: it turns every `/query` response into a self-documenting explanation of what the graph did and why, which is exactly what an AI-engineer reviewer wants to inspect without reading your source. Structured citations with real char ranges, plus a `retrieval.verdict`/`score`/`attempts` block, let a reviewer verify the branch actually fired correctly on a handful of manual test questions in minutes.

- **Implementation speed.** Every component in Option 2 maps to exactly one assignment requirement — multi-format loaders (ingestion), content-hash ledger (idempotent ingestion / duplicate handling), composite evaluator (retrieval quality checking), rewrite-and-retry (loop protection with real behavior, not just a counter), namespace-per-corpus (a natural, low-effort Pinecone design choice). Nothing is included "for polish's sake," which is what keeps it achievable in 3–5 days. Option 3's extra pieces (rollback, near-dup detection, `/eval/run` as a live endpoint) are the parts most likely to be rushed or cut under time pressure, and a rushed strong architecture reads worse in review than a complete balanced one.

- **Extensibility.** Because Pinecone calls, LLM calls, and graph nodes are cleanly separated in Option 2 (as noted in §4.9 for Option 3, but true of Option 2's design already), moving to async ingestion, adding `verify_citations` later, or standing up the full `/eval/run` harness are all incremental additions to the same skeleton rather than a rearchitecture. In other words, Option 2 is a proper strict subset of Option 3 — you can *start* at Option 2 and grow into Option 3 if time permits, but you can't safely shrink from an unfinished Option 3 back down to something submittable.

**Practical build order if you take this recommendation:**
1. Common LangGraph skeleton (§1) with the threshold-only evaluator first — get the branch, retry, and refusal path working end-to-end on a toy corpus. This alone satisfies every hard requirement.
2. Multi-format ingestion + ledger + namespace-per-corpus (§3.5, §5.5).
3. Upgrade `evaluate_retrieval` to the composite/judge version (§3.3).
4. Add `trace` to the response (§3.6) — cheap, high reviewer impact.
5. If time remains: add `verify_citations` (§4.3) and the JSON test-case harness (§9), even as an offline script rather than a live `/eval/run` endpoint — the evaluation *format* matters more to a reviewer than whether it's exposed over HTTP.
6. Only attempt near-duplicate detection, rollback, or `partial` mode if 1–5 are done with a full day to spare.
