# Primmo — Notarial Agent

A conversational API that lets notaries ask natural-language questions over real estate sale dossiers. The corpus is 21 OCR-scanned documents (Google Vision API format) across 3 dossiers. The system answers in French, cites its sources at chunk level, and exposes cost, latency, and retrieval-relevance metrics on every response.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your OPENROUTER_API_KEY (or any OpenAI-compatible key)
make dev                      # uvicorn with hot reload on :8000
```

Chat UI: `http://localhost:8000/ui/`  
Swagger: `http://localhost:8000/docs`  
Docker: `make run` (docker compose up --build)

---

## Corpus

| Dossier | Documents | Location |
|---|---|---|
| Dossier 1 — Paris 75011 | 9 | `documents/dossier_1/` |
| Dossier 2 — Bordeaux | 6 | `documents/dossier_2/` |
| Dossier 3 — Lyon | 6 | `documents/dossier_3/` |

Each dossier contains: compromis de vente, pièces d'identité (CNI), justificatifs de domicile, and a DPE. All files are Google Vision API `fullTextAnnotation` JSON (pages → blocks → paragraphs → words → symbols with per-symbol confidence scores).
---

## Ingestion & preprocessing pipeline

On first boot, the app runs a two-stage pipeline and caches results to `data/` so subsequent restarts are instant.

### Stage 1 — Parsing (`ingestion/parser.py`)

Walks the Vision API structure to reconstruct per-block text and compute mean OCR confidence per block. Classifies each file into a doc type (`compromis`, `identite`, `domicile`, `dpe`) using filename keywords first, then content keywords — with explicit handling for OCR corruption (e.g. `"CARTE MATIONALE"` → `identite`).

Chunking is doc-type-aware:

| Doc type | Strategy | Chunks |
|---|---|---|
| `compromis` | Regex split on section headers (`VENDEUR`, `ACQUEREUR`, `ARTICLE N`) | ~15 per doc |
| `dpe` | Regex split on all-caps lines ≥ 5 chars | ~5 per doc |
| `identite` / `domicile` | Single chunk (short, single-topic) | 1 per doc |

Falls back to Vision API block boundaries, then whole-document, if header regexes match nothing. Produces **76 `Chunk` objects** total, each with a stable human-readable ID (`dossier_1/compromis#VENDEUR`) used as the citation unit throughout.

### Stage 2 — Structured extraction (`ingestion/extractor.py`)

One LLM call per document produces a typed JSON profile (vendeurs, acquéreurs, dates, addresses, expiry flags, DPE class…). Cached to `data/profiles/` — 21 profiles, zero LLM calls on subsequent boots. Used by `get_document_inventory` for lightweight structural queries without full-text retrieval.

---

## Retrieval pipeline (`rag/engine.py`)

Three-stage hybrid search on every `search_documents` call:

**1. Dual retrieval**
- Cosine similarity over 384-dim embeddings (`paraphrase-multilingual-MiniLM-L12-v2`, local via sentence-transformers)
- BM25 (`BM25Okapi`) over `\w+`-tokenized chunk texts — catches exact lexical matches (article numbers, names, legal codes) that cosine alone misses

**2. RRF fusion**
Top-15 candidates from each retriever are merged and scored with Reciprocal Rank Fusion (`1/(60+rank_cosine) + 1/(60+rank_bm25)`). The constant 60 is from the original RRF paper.

**3. Cross-encoder reranking**
All RRF candidates are re-scored as `(query, chunk_text)` pairs by `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual, ~120 MB). The raw logit is sigmoid-normalised to produce the `relevance_score` (0–1) in the response.

Embeddings are cached to `data/embeddings.npy`; the BM25 index and cross-encoder are rebuilt/loaded in-memory on every boot.

---

## Agent design — ReWoo

See [AGENTIC_DESIGN.md](AGENTIC_DESIGN.md) for the full rationale. The short version:

```
Planner LLM  →  tool executor (pure Python)  →  Solver LLM
```

Always exactly **3 LLM calls** per request. No loop, no runaway depth. The planner sees the query and tool schemas; it never sees tool results. The solver sees the query and all tool results; it never calls tools.

**Tools:**

| Tool | When used |
|---|---|
| `search_documents(query, dossier?, doc_type?)` | Targeted lookups — "quel est le prix du bien ?" |
| `get_dossier_documents(dossier)` | Full-dossier retrieval — coherence checks, cross-document inconsistency |
| `get_document_inventory()` | Structural overview — missing pieces, completeness per dossier |

The planner is capped at `MAX_TOOLS_PER_PLAN=5` calls. Multi-turn context is maintained client-side and sent as a `history` array with each request — both planner and solver see it between the system prompt and the current query.

---

## API

### `POST /chat`
```json
{ "query": "Les pièces d'identité sont-elles en ordre dans le dossier 1 ?", "history": [] }
```
Returns JSON with `answer`, `sources` (chunk IDs + relevance scores + OCR confidence), and `metrics` (latency, cost, token counts, per-stage breakdown).

### `POST /chat/stream`
Same input. Returns SSE:
```
event: status   {"stage": "planning"}
event: status   {"stage": "tools"}
event: status   {"stage": "solving"}
event: delta    {"text": "Les pièces..."}   (one per token)
event: done     {"answer": "...", "sources": [...], "metrics": {...}}
```

### `GET /metrics`
Aggregate stats over all interactions (count, mean/p95 latency, total cost, mean relevance).

### `GET /metrics/history`
Full interaction log with per-query breakdown.

### `POST /evaluate`
Manually score an interaction for retrieval relevance.

---

## Observability (optional)

Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env` to enable Langfuse tracing. Each request produces a trace with child spans for planner, tools, and solver — including token usage, cost, and a `retrieval_relevance` score. If keys are absent the app runs identically with no tracing.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | required | OpenRouter key (or any OpenAI-compatible provider) |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | LLM provider base URL |
| `LLM_MODEL` | `anthropic/claude-sonnet-4.5` | Model for planner, solver, and extraction |
| `DOCUMENTS_PATH` | `documents` | Path to OCR JSON corpus |
| `DATA_PATH` | `data` | Cache directory for profiles and embeddings |
| `MAX_TOOLS_PER_PLAN` | `5` | Planner tool call cap |
| `OCR_CONFIDENCE_THRESHOLD` | `0.7` | Below this, agent flags the source as unreliable |
| `LANGFUSE_PUBLIC_KEY` | — | Optional — Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | — | Optional — Langfuse tracing |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | Optional — self-hosted Langfuse |

---

## Tests

```bash
make test    # pytest tests/ (37 tests, no LLM calls — planner/solver are mocked)
make lint    # ruff check app/
make format  # ruff format app/
```

---

## Retrieval relevance vs faithfulness vs answer relevance

These three dimensions measure different things and belong in different places:

| Dimension | What it measures | Where |
|---|---|---|
| **Retrieval relevance** | Are the retrieved chunks semantically close to the question? | Always-on — free byproduct of the cross-encoder reranking step, returned with every response |
| **Faithfulness** | Is the answer grounded in the retrieved documents, or is the LLM adding information that isn't there? | On-demand via `POST /evaluate` — running it inline on every request would add latency and cost with no user benefit |
| **Answer relevance** | Does the answer actually address what was asked? | Offline evaluation — requires a ground-truth Q&A set; not measurable at runtime |

Faithfulness matters most in a notarial context: a hallucinated name or wrong address in a legal document has real consequences. The `POST /evaluate` endpoint exposes LLM-as-judge scoring over stored interactions so the evaluation layer can evolve independently of the pipeline.

---

## Going further

Items not in scope for this implementation:

- **Chunking overlap** — add a sliding window between adjacent sections to avoid losing context at boundaries (relevant for long `compromis` articles that span sections)
- **Automated evaluation** — build a ground-truth Q&A set from the example questions and run RAGAS to track faithfulness and answer relevance over time
- **Metrics persistence** — swap the in-memory store for PostgreSQL + TimescaleDB for time-series analysis of cost and latency across deployments
- **Semantic cache** — cache responses for semantically similar queries (e.g. GPTCache) to reduce cost on repeated or near-duplicate questions

---

## Further reading

- [ARCHITECTURE.md](ARCHITECTURE.md) — module map, startup sequence, full request flow, RAG pipeline detail, tradeoffs table
- [AGENTIC_DESIGN.md](AGENTIC_DESIGN.md) — ReWoo vs ReAct rationale, tool design decisions, source citation strategy
