# Architecture

## Module map

```
app/
├── main.py               Entry point. FastAPI app + lifespan startup hook.
│                         Runs the full ingestion pipeline on boot, stores
│                         results in app.state for the lifetime of the process.
│
├── core/config.py        Pydantic BaseSettings. Single source of truth for
│                         all configuration. Read from .env at import time.
│
├── ingestion/
│   ├── parser.py         Reads OCR JSON files, produces Chunk list.
│   │                     Pure Python, no I/O beyond file reads.
│   └── extractor.py      One LLM call per document → structured JSON profile.
│                         Writes profiles to data/profiles/. Cached: skips
│                         documents that already have a profile on disk.
│
│
├── rag/
│   └── engine.py         Loads sentence-transformers model, encodes all chunks,
│                         saves embeddings to data/embeddings.npy. Also exposes
│                         cosine_search() used by the tools.
│                         Cached: recomputes only if chunk IDs change.
│
├── agent/
│   ├── tools/
│   │   ├── __init__.py               execute_tool() dispatcher — routes by name to the
│   │   │                             correct tool function. Single import surface for
│   │   │                             callers (chat.py only needs execute_tool).
│   │   ├── search_documents.py       Hybrid search: encodes query, calls hybrid_search()
│   │   │                             (cosine + BM25 → RRF → cross-encoder reranking).
│   │   ├── get_dossier_documents.py  Filters app.state.chunks by dossier, returns all
│   │   │                             chunks as dicts with OCR confidence.
│   │   └── get_document_inventory.py Derives structural map from profiles — doc types
│   │                                 present per dossier + per-dossier completeness flags.
│   │                             All three are deterministic Python, no LLM calls.
│   ├── planner.py        ReWoo planner: takes query + tool schemas,
│   │                     returns a bounded list of tool calls (≤ max_tools_per_plan).
│   └── solver.py         ReWoo solver: takes query + all tool results,
│                         returns final answer with source citations.
│
├── metrics/
│   ├── store.py          Thread-safe in-memory list of interaction records.
│   │                     Each record: latency_ms, input_tokens, output_tokens,
│   │                     cost_usd, retrieval_relevance.
│   └── profiler.py       Profiler dataclass — context-manager spans that
│                         accumulate wall-clock ms per label into a breakdown dict.
│
├── api/routes/
│   ├── chat.py           POST /chat — orchestrates planner → tools → solver,
│   │                     records metrics, returns answer + sources + metrics.
│   │                     POST /chat/stream — same pipeline but solver streams
│   │                     via SSE: status / delta / done events.
│   └── metrics.py        GET /metrics, GET /metrics/history, POST /evaluate.
│
└── static/
    └── index.html        Single-page chat UI served at /ui/. Consumes
                          POST /chat/stream via SSE, renders markdown deltas
                          live, displays source citations and cost badges.
                          Maintains client-side history array sent with each request.
```

---

## Central data model: `Chunk`

Defined in `ingestion/parser.py`. Everything downstream operates on lists of `Chunk`.

```python
@dataclass
class Chunk:
    id: str             # "dossier_1/compromis#VENDEUR" — stable, human-readable
    dossier: int        # 1 | 2 | 3
    doc_type: str       # "compromis" | "identite" | "domicile" | "dpe"
    filename: str       # stem of the source file, e.g. "compromis"
    section: str        # section key, e.g. "VENDEUR", "ART_04_PRIX...", "full"
    text: str           # reconstructed text for this section
    ocr_confidence: float  # mean symbol-level confidence from Vision API
```

The chunk `id` is the citation unit: tool results carry it, the solver is instructed to include it in every answer.

---

## Startup sequence (`main.py` lifespan)

```
boot
 │
 ├─ parser.load_chunks(documents_path)
 │    Reads all OCR JSONs → list[Chunk]  (76 chunks, ~instant)
 │
 ├─ extractor.load_or_extract_profiles(chunks, data_path, client)
 │    For each of 21 documents:
 │      if data/profiles/dossier_N_filename.json exists → load from disk
 │      else → LLM call → save to disk
 │    Returns dict["dossier_1/compromis"] = {vendeurs, acquereurs, bien, ...}
 │
 ├─ engine.load_or_compute_embeddings(chunks, data_path)
 │    if data/embeddings.npy exists AND chunk IDs match chunk_index.json → load
 │    else → encode all chunks with sentence-transformers → save
 │    Returns ndarray shape (76, 384), normalized
 │
 ├─ engine.build_bm25_index(chunks)
 │    Tokenizes all chunk texts (regex \w+ lowercase) → BM25Okapi index
 │    In-memory only; rebuilt from chunks on every boot (~instant)
 │
 ├─ engine.load_cross_encoder()
 │    Loads cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (multilingual, ~120 MB)
 │    Weights cached by HuggingFace in ~/.cache/huggingface/ after first boot
 │
 ├─ Langfuse(public_key, secret_key, host)   [optional]
 │    Only initialised if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.
 │    If keys are absent, lf = None and all tracing is skipped silently.
 │    lf.flush() is called on shutdown to drain any buffered events.
 │
 └─ stored in app.state:
      app.state.chunks         list[Chunk]
      app.state.profiles       dict[str, dict]
      app.state.embeddings     ndarray (76, 384)
      app.state.bm25_index     BM25Okapi
      app.state.cross_encoder  CrossEncoder
      app.state.client         OpenAI  (shared LLM client)
      app.state.langfuse       Langfuse | None
```

After boot, all state is in-memory. No database reads at query time.

---

## Request flow (`POST /chat` and `POST /chat/stream`)

Both endpoints share the same three-stage pipeline. `/chat` returns a single JSON
response; `/chat/stream` returns an SSE stream of `status` / `delta` / `done` events.

`ChatRequest`: `{"query": "...", "history": [...]}` — `history` is a client-managed
array of `{role, content}` turns appended after each exchange. Both planner and solver
prepend it between the system prompt and the current user message.

```
POST /chat  (or /chat/stream)
 │
 ├─ Langfuse trace span opened  (skipped if app.state.langfuse is None)
 │
 ├─ Planner LLM (agent/planner.py)
 │    Input:  system prompt + history + query + tool schemas
 │    Output: list of tool calls, e.g.
 │              [get_dossier_documents(1), search_documents("DPE adresse", dossier=1)]
 │    Capped at max_tools_per_plan (default: 5)
 │    Langfuse: generation child span records input messages + token usage
 │
 ├─ Tool executor  (deterministic, in-memory, sub-ms each)
 │    search_documents      → hybrid_search() — three-stage pipeline:
 │                             1. cosine similarity over app.state.embeddings
 │                             2. BM25 over app.state.bm25_index
 │                             3. RRF fusion → top-15 candidates
 │                             4. cross-encoder reranking → final top-5
 │    get_dossier_documents → filter app.state.chunks by dossier
 │    get_document_inventory→ derive from app.state.profiles
 │    Langfuse: retriever child span records plan + result count
 │
 ├─ Solver LLM (agent/solver.py)
 │    Input:  system prompt + history + query + all tool results
 │    Output: answer in French, citing chunk IDs as sources
 │            low-confidence chunks (< 0.70) flagged explicitly
 │    Langfuse: generation child span records input messages + token usage
 │    /chat/stream: solver uses solve_stream() — yields ("delta", str) incrementally,
 │                  then ("done", answer, sources, usage) at the end
 │
 ├─ metrics/store.py
 │    Appends: {latency_ms, tokens, cost_usd, retrieval_relevance, breakdown}
 │
 ├─ Langfuse trace span closed; retrieval_relevance recorded as a Score
 │
 └─ Response
      POST /chat → JSON:
        {"answer": "...", "sources": [...], "metrics": {"latency_ms": 1840, ...}}

      POST /chat/stream → SSE events:
        event: status   {"stage": "planning"|"tools"|"solving"}
        event: delta    {"text": "..."}          (one per streamed token)
        event: done     {"answer": "...", "sources": [...], "metrics": {...}}
```

---

## Ingestion pipeline detail

### `parser.py`

**Input:** `documents/dossier_N/*.json` — Google Vision API `fullTextAnnotation` format.

**Text reconstruction:** walks `responses[0].fullTextAnnotation.pages → blocks → paragraphs → words → symbols`, reconstructing text per block and computing mean symbol confidence per block. The top-level `fullTextAnnotation.text` is used as the full document string.

**Doc type classification:** filename keywords first, then content keywords in the first 500 chars. Handles OCR corruption (`"CARTE MATIONALE"` → `identite`).

**Chunking strategy by doc type:**

| doc_type | Strategy | Result |
|---|---|---|
| `compromis` | Regex split on `LE(S) VENDEUR(S)`, `L(LES) ACQUEREUR(S)`, `ARTICLE N —` | ~15 chunks per doc |
| `dpe` | Regex split on all-caps lines ≥ 5 chars | ~5 chunks per doc |
| `identite` | Always single chunk | 1 chunk per doc |
| `domicile` | Always single chunk | 1 chunk per doc |

**Fallback cascade** (for `compromis`/`dpe` only, if header regex matches nothing):
1. OCR block boundaries from Vision API layout
2. Whole document as one chunk

**Output:** 76 `Chunk` objects for the current corpus.

---

### `extractor.py`

**Input:** `list[Chunk]` + OpenAI client.

**Per document:** reconstructs full document text by joining its chunks, sends to LLM with a doc-type-specific prompt, parses the JSON response. Strips markdown fences if the model wraps its output.

**Cache:** `data/profiles/dossier_N_filename.json`. On subsequent boots, all 21 profiles load from disk with no LLM calls.

**Output schema per doc_type:**

| doc_type | Key fields |
|---|---|
| `compromis` | `vendeurs[]`, `acquereurs[]`, `bien{adresse,type,surface_m2}`, `prix_eur`, `date` |
| `identite` | `nom`, `prenoms`, `dob`, `expire`, `expired` (bool) |
| `domicile` | `titulaire`, `adresse`, `date_document`, `stale` (bool, >3 months) |
| `dpe` | `adresse`, `classe_energie`, `date_etablissement`, `valide_jusqu_au` |

---

### `rag/engine.py`

**Embedding model:** `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers, local). Produces 384-dim normalized vectors. Model weights cached by HuggingFace in `~/.cache/huggingface/`.

**Cache:** `data/embeddings.npy` (float32 array) + `data/chunk_index.json` (ordered chunk ID list). Invalidated if chunk IDs change.

**`cosine_search(query_emb, embeddings, chunks, top_k, dossier?, doc_type?)`**
Computes `embeddings @ query_emb` (dot product of normalized vectors = cosine similarity), sorts descending, applies metadata filters, returns top-k as dicts with `relevance_score`. Still used directly by tests; production calls go through `hybrid_search`.

---

**BM25 index:** `BM25Okapi` (rank-bm25) built over all chunk texts tokenized with `re.findall(r"\w+", text.lower())`. Handles French accented characters. Rebuilt from chunks in-memory on every boot (~instant, not cached).

**Cross-encoder:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (sentence-transformers). Multilingual model trained on mMARCO, scores `(query, document)` pairs jointly — sees both at once, unlike bi-encoders. Weights cached by HuggingFace after first boot.

**`hybrid_search(query, query_emb, embeddings, chunks, bm25_index, cross_encoder, top_k, candidate_k, dossier?, doc_type?)`**

Three-stage pipeline:

1. **Parallel retrieval** — cosine similarity ranks all (filtered) chunks; BM25 independently ranks the same set.
2. **RRF fusion** — takes `candidate_k` (default 15) from each ranked list, unions them, scores every candidate as `1/(60 + cosine_rank) + 1/(60 + bm25_rank)`. The constant 60 is the standard value from the original RRF paper; it dampens the impact of very high ranks without eliminating lower-ranked items.
3. **Cross-encoder reranking** — runs a forward pass over all RRF candidates as `(query, chunk_text)` pairs, re-sorts by the resulting logits, returns top-k. `relevance_score` in the response is the sigmoid-normalized logit (0–1), which is more calibrated than raw cosine similarity.

---

## Agent pattern: ReWoo

Three LLM calls per request, always. No loop, no runaway depth.

```
Planner LLM  →  bounded plan  →  tool executor  →  Solver LLM
```

The Planner sees the query and tool schemas; it never sees tool results. The Solver sees the query and all tool results; it never runs tools. The executor is pure Python.

See [AGENTIC_DESIGN.md](AGENTIC_DESIGN.md) for the rationale vs ReAct.

---

## Tradeoffs

| Decision | Chosen | Alternative | Why | When to revisit |
|---|---|---|---|---|
| Agent pattern | ReWoo (plan → execute → solve, always 3 LLM calls) | ReAct (interleaved reasoning + tool calls, variable) | Query types are predictable enough to plan upfront. Bounded cost, no loop risk. | If queries become open-ended enough that intermediate results need to change the retrieval strategy |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` local via sentence-transformers | Hosted API model (OpenAI, Cohere) | OpenRouter has no embedding endpoint. Local is free, cached, good French support. Heavy Docker image (~1.5GB) is the cost. | If retrieval precision becomes the bottleneck at scale — one-file swap in `rag/engine.py` |
| Vector store | numpy `@` dot product in-memory | ChromaDB / Pinecone / Qdrant | 76 vectors. A vector DB adds a service dependency for zero benefit at this scale. | ~10k+ chunks, or when filtered ANN latency matters |
| Chunk granularity | Section-level for `compromis`/`dpe`, single chunk for `identite`/`domicile` | Full-document chunks or uniform block splitting | Section chunks give precise citation and better retrieval for targeted queries. Short single-topic docs (CNI, EDF bill) have nothing to gain from splitting. | If cross-section context is lost at retrieval time — add overlap between adjacent sections |
| Profile extraction timing | Once at startup, cached to `data/profiles/` | At query time, per request | Profiles are stable — re-extracting on every request would cost 21 LLM calls per boot and redundant calls per query. Cold start on first boot only. | If the corpus becomes dynamic (new documents added at runtime), extraction needs to be incremental rather than startup-triggered |
| Retrieval pipeline | Hybrid: cosine + BM25 fused with RRF, then cross-encoder reranking | Cosine-only (previous) | Cosine alone misses exact lexical matches (article numbers, proper names, legal codes). BM25 catches these. Cross-encoder reranks the fused shortlist by true query-document relevance rather than embedding-space proximity. At 76 chunks the added latency is ~50–150 ms on CPU. | If latency becomes a bottleneck — cross-encoder is the expensive step; a lighter model (e.g. `ms-marco-MiniLM-L-2-v2`) or batched async execution would help |
| Multi-turn history | Client-side `history` array sent with each request | Server-side session store (Redis + session_id) | Zero backend state. Works correctly for the demo; upgrade path is a one-field addition to `ChatRequest` and a dict in `app.state`. | When multi-device session resumption or audit logging of conversation history is needed |

