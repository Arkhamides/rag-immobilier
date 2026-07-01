# Primmo Notarial Agent — Implementation Plan

## Context

Build a conversational API for notarial offices that lets notaries and staff ask natural language questions over real estate sale dossiers. The corpus is 21 Google Vision OCR JSON files across 3 dossiers. The system must also track cost, latency, and response relevance.

---

## Document Corpus

### Dossier 1 — Paris 75011 (9 documents)
| File | Type | Key info |
|------|------|----------|
| `compromis.json` | Compromis de vente | Vendeur: MOREAU Jean-Pierre → Acheteurs: LAURENT Sophie + LAURENT Marc |
| `scan_id_001.json` | CNI | MOREAU, né 15/03/1958 Paris |
| `scan_id_001_v2.json` | CNI | Duplicate of MOREAU's CNI |
| `piece_identite_3.json` | CNI | LAURENT Sophie, née 22/07/1985 Lyon |
| `scan_id_004.json` | CNI | LAURENT Marc, né 03/11/1983 Marseille |
| `diag_dpe.json` | DPE | 8 rue des Acacias, 75011 Paris |
| `justif_001.json` | EDF bill | MOREAU, 12 rue des Lilas |
| `facture_edf_02.json` | EDF bill | Marc LAURENT, 45 rue de la Roquette |
| `avis_imposition.json` | Tax notice | Sophie LAURENT, 45 rue de la Roquette |

### Dossier 2 — Bordeaux (6 documents)
| File | Type | Key info |
|------|------|----------|
| `compromis_vente.json` | Compromis de vente | Vendeuse: DUBOIS Catherine → Acheteur: BENALI Youssef |
| `piece_id_vendeuse.json` | CNI | DUBOIS, née 08/12/1970 Bordeaux |
| `scan_005.json` | CNI | BENALI, né 17/06/1990 Casablanca |
| `dpe_bien.json` | DPE | — |
| `justif_domicile_1.json` | EDF bill | DUBOIS, 24 avenue du Médoc |
| `justif_domicile_2.json` | EDF bill | BENALI, 15 cours de l'Intendance |

### Dossier 3 — Lyon (6 documents)
| File | Type | Key info |
|------|------|----------|
| `scan_012.json` | Compromis de vente | Vendeuse: PETIT Marie-Claire → Acheteur: FONTAINE Alexandre |
| `id_vendeuse.json` | CNI | PETIT, née 30/09/1962 Saint-Étienne |
| `scan_006.json` | CNI | FONTAINE — **severely corrupted OCR** ("FONTAIMF", "FR4MCAISE") |
| `diagnostic_energie.json` | DPE | — |
| `edf_vendeur.json` | EDF bill | PETIT, 7 rue de la République |
| `piece_12.json` | EDF bill | FONTAINE, 22 rue Mercière |

### Notable observations
- `scan_id_001` and `scan_id_001_v2` in dossier 1 are duplicate scans of the same document
- `scan_006` in dossier 3 has severe OCR noise — an intentional test case for coherence/inconsistency questions
- All documents are in French, short (~200–500 words each)

### Intentional inconsistencies in the corpus (test cases)
These are ground truth for the "incohérences" and "pièces en ordre" question types:

| Dossier | Issue | Detail |
|---------|-------|--------|
| 1 | **DPE address mismatch** | DPE says "8 rue des Acacias, 75011 Paris" — compromis property is "12 rue des Lilas, 75011 Paris" |
| 1 | **Expired CNI** | MOREAU's CNI (scan_id_001) expired 10/04/2019 — sale is 15/01/2026, 7 years past expiry |
| 3 | **Stale justificatif** | FONTAINE's EDF bill (piece_12) dated 15/09/2025, compromis is 10/02/2026 — ~5 months, exceeds 3-month rule |
| 3 | **Corrupted OCR** | scan_006 (FONTAINE's CNI) has severe OCR noise making identity verification unreliable |

---

## Why Simple RAG Is Not Enough

A static pipeline (query → embed → top-k → answer) fails for several question types:

| Question type | Why top-k fails |
|---|---|
| "Y a-t-il des incohérences dans le dossier 1 ?" | Needs **all** docs from dossier 1, not just the closest ones |
| "Résumé des pièces manquantes" | Needs to know what is **absent**, not just what matches |
| "Les pièces d'identité sont-elles en ordre ?" | Needs structural inventory, not semantic proximity |

The agent must choose its retrieval strategy based on what the question actually requires.

---

## Agentic Design

### Pattern: ReWoo (Reasoning Without Observation)

See `AGENTIC_DESIGN.md` for the full rationale. Summary: the Planner LLM produces a bounded tool plan in one pass, all tools execute (in parallel where possible), then the Solver LLM synthesizes the final answer. Always exactly 3 LLM calls regardless of query complexity.

```
User query
    │
    ▼
Planner LLM — produces plan: list of tool calls + parameters
    │
    ▼
Execute all tools (in parallel where possible)
    │
    ▼
Solver LLM — synthesizes final answer from all results + source citations
```

This is the right trade-off for this use case: question types are predictable enough that the Planner can determine the correct tools upfront without needing intermediate results. Cost and latency are bounded. The loop depth problem is solved architecturally — there is no loop.

### No regex in the query pipeline

Using regex to detect "dossier 1" in a query is fragile ("le premier dossier", "dossier numéro 1" would silently fail) and redundant — the LLM handles French naturally via tool parameters. The only legitimate use of string parsing is at **ingestion time** to split documents into sections using their consistent capitalized headers.

---

## Tools

### Tool 1 — `search_documents`
```
search_documents(query: str, dossier?: int, doc_type?: "compromis"|"identite"|"domicile"|"dpe")
→ list of matching chunks with IDs and text
```
Semantic search (cosine similarity) over section-level chunks, with optional metadata filters. Used for targeted lookups across or within dossiers.

### Tool 2 — `get_dossier_documents`
```
get_dossier_documents(dossier: int)
→ all chunks from that dossier with IDs and text
```
Returns every section from every document in a dossier. Used for coherence checks, completeness reviews, and cross-document inconsistency detection.

### Tool 3 — `get_document_inventory`
```
get_document_inventory()
→ structured summary: per dossier, which doc types are present
```
Returns a lightweight structural map without full text. Used for "pièces manquantes" questions — the LLM can reason about what is absent without burning tokens on full retrieval.

> **Completeness checklist (baked into system prompt):** A complete dossier de vente is expected to contain the following documents for all parties:
> - 1 compromis de vente (or promesse de vente)
> - 1 pièce d'identité valide per party (vendeur(s) and acquéreur(s))
> - 1 justificatif de domicile de moins de 3 mois per party
> - 1 DPE (Diagnostic de Performance Énergétique) for the property
>
> This checklist is embedded in the system prompt and in the `get_document_inventory` tool response so the agent can reason about what is absent. It reflects the standard dossier requirements under French notarial practice for a vente immobilière (articles L271-1 and following of the Code de la construction). It is stated as an assumption in the README.

---

## Preprocessing Pipeline

### Stage 1 — Section-level chunking (deterministic, at ingestion)

Split each document's OCR text into logical sections using capitalized headers that appear consistently across documents:

- Compromis: `VENDEUR(S)`, `ACQUEREUR(S)`, `BIEN VENDU`, `CONDITIONS FINANCIERES`, etc.
- DPE: `IDENTIFICATION DU BIEN`, `CLASSEMENT ENERGETIQUE`, etc.
- CNI / EDF / tax notice: treated as a single section (already short)

Each section becomes an independently retrievable chunk with a stable, human-readable ID:

```
dossier_1/compromis#VENDEUR
dossier_1/compromis#ACQUEREUR
dossier_1/diag_dpe#IDENTIFICATION_DU_BIEN
dossier_3/scan_006#block_1
```

#### Text reconstruction from Vision API JSON

The actual JSON structure is `responses[0].fullTextAnnotation`, with the tree: `pages → blocks → paragraphs → words → symbols`. Verified against the corpus. Key observations from inspection:

- `fullTextAnnotation.text` — pre-concatenated full-page string, already present
- `block.confidence`, `paragraph.confidence`, `word.confidence`, `symbol.confidence` — confidence at every level
- `word.property` holds `detectedBreak` (not at symbol level — symbols only have `text` and `confidence`)
- `block.blockType` distinguishes text blocks from other elements

Two options for text reconstruction:

1. **Use `fullTextAnnotation.text` directly** — zero cost, already done. Loses block boundary information.
2. **Walk the block tree** — reconstruct text per block by joining `symbol.text` within each `word`, joining words within each `paragraph`, joining paragraphs within each `block`. Use `block` boundaries for Level 2 chunking fallback. Average `symbol.confidence` per block for OCR quality metadata.

**Use option 2.** Block boundaries are needed for the structural fallback and per-chunk confidence scores require walking to the symbol level anyway. A utility `extract_text_and_blocks(ocr_json)` should return: the full text string (from `fta.text`) and a list of blocks each with reconstructed text + mean symbol confidence.

#### Fallback strategy for corrupted OCR

`scan_006` in dossier 3 has severely garbled text ("REPUBLIQUF FR4MCAISE", "FONTAIMF") — capitalized headers may not be recoverable. Section splitting uses a cascade:

1. **Level 1 — Capitalized header splitting (primary)**: detect headers like `VENDEUR(S)`, `BIEN VENDU` in the text. Works for clean documents.
2. **Level 2 — OCR block boundaries (structural fallback)**: the Vision API JSON already provides block-level structure with bounding boxes, derived from page layout rather than text content. If no headers are found, split on block boundaries instead. Already in the data, costs nothing extra.
3. **Level 3 — One chunk per document (last resort)**: if block splitting produces nothing meaningful, treat the whole document as a single chunk. Always works; loses granularity but never fails.

#### OCR confidence metadata

Regardless of which level triggers, compute the average OCR confidence per chunk from the Vision API's symbol-level scores and store it as metadata:

```json
{"id": "dossier_3/scan_006#block_1", "ocr_confidence": 0.41, "dossier": 3, ...}
```

The system prompt instructs the agent to flag low-confidence sources explicitly in its answer:

> "La CNI de M. FONTAINE (scan_006) présente une qualité OCR faible (confiance: 0.41) — les informations peuvent être inexactes."

This turns OCR corruption from a silent failure into a useful signal, directly relevant to the "incohérences" question type.

### Stage 2 — Structured extraction (one-time LLM pass, at ingestion)

Run each document through the LLM once at startup to produce a typed JSON profile:

```json
{
  "dossier": 1,
  "doc_type": "compromis",
  "fields": {
    "vendeurs": [{"nom": "MOREAU Jean-Pierre", "naissance": "15/03/1958", "adresse": "12 rue des Lilas, 75011 Paris"}],
    "acquereurs": [{"nom": "LAURENT Sophie", ...}, {"nom": "LAURENT Marc", ...}],
    "bien": {"adresse": "8 rue des Acacias, 75011 Paris"},
    "prix": 485000,
    "date": "15/01/2026"
  }
}
```

Profiles enable fast structured lookups (names, dates, addresses) without retrieval. They also power `get_document_inventory`. Cost: ~21 LLM calls at startup, run once, persisted to disk.

### Stage 3 — Embeddings

Compute embeddings for each section-level chunk (not full documents). Stored as a numpy array in memory — 21 documents × ~3 sections average = ~60 vectors. No vector database needed at this scale.

---

## Source Citation

The system prompt instructs the LLM to always cite the chunk IDs it used in its answer. Tool results include IDs alongside text, making citation trivial:

> "Le vendeur du dossier 1 est Jean-Pierre MOREAU, né le 15/03/1958 à Paris.
> **Sources:** `dossier_1/compromis#VENDEUR`, `dossier_1/scan_id_001#IDENTITE`"

This is auditable and directly traceable to the original OCR JSON.

---

## Relevance in a RAG System

Relevance has three distinct dimensions — each measures something different and none fully substitutes for the others:

**1. Retrieval relevance** — are the chunks that were retrieved actually related to the question?
Measured by cosine similarity between the query embedding and the retrieved chunk embeddings. Free and always available as a byproduct of retrieval. Weak proxy for answer quality — a high score means semantic closeness, not correctness.

**2. Faithfulness** — is the answer grounded in the retrieved documents, or is the LLM adding information that isn't there?
The most critical dimension for a notarial tool. A hallucinated name or wrong address in a legal context has real consequences. Measured via LLM-as-judge: given the source documents, is this answer fully supported?

**3. Answer relevance** — does the answer actually address what was asked?
A faithful answer can still be off-topic. Hardest to measure without ground truth. Requires a reference set of expected answers.

### What belongs where

| Dimension | Where | Rationale |
|---|---|---|
| Cosine similarity | Pipeline (always-on) | Free byproduct of retrieval, zero extra cost or latency |
| Faithfulness (LLM-as-judge) | Evaluation sweep (on-demand) | Developer/monitoring concern — user doesn't need it; running it inline adds latency and cost to every request for no user benefit |
| Answer relevance | Evaluation sweep (offline) | Requires ground truth; not measurable at runtime |

The evaluation sweep is exposed as a dedicated endpoint (`POST /evaluate`) that takes a list of stored interaction IDs and returns faithfulness scores. This keeps the agentic pipeline clean and lets the evaluation layer evolve independently.

---

## Metrics Tracking

Every interaction records:

| Metric | How |
|---|---|
| **Latency** | End-to-end wall time (ms), measured per LLM call |
| **Cost** | Token counts × per-model price table (USD estimate) |
| **Retrieval relevance** | Cosine similarity score of top retrieved chunk — always-on |

Stored in-memory (thread-safe). Exposed via:
- `GET /metrics` — aggregated stats (mean/median/p95 latency, total cost, mean retrieval relevance)
- `GET /metrics/history` — full interaction log
- `POST /evaluate` — on-demand LLM-as-judge faithfulness scoring over stored interactions

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Main query endpoint |
| `GET` | `/metrics` | Aggregated stats |
| `GET` | `/metrics/history` | Full interaction history |
| `GET` | `/health` | Health + document count |
| `GET` | `/docs` | Auto-generated Swagger UI |

### Request / Response shape

```json
// POST /chat
{
  "query": "Y a-t-il des incohérences dans le dossier 3 ?"
}

// Response
{
  "answer": "Oui, la CNI de M. FONTAINE (scan_006) présente des erreurs OCR significatives...",
  "sources": [
    {"id": "dossier_3/scan_006#IDENTITE", "dossier": 3, "doc_type": "identite", "relevance_score": 0.91},
    {"id": "dossier_3/scan_012#ACQUEREUR", "dossier": 3, "doc_type": "compromis", "relevance_score": 0.87}
  ],
  "metrics": {
    "latency_ms": 1840,
    "input_tokens": 2100,
    "output_tokens": 310,
    "cost_usd": 0.0084,
    "retrieval_relevance": 0.91
  }
}
```

---

## Technical Stack

| Component | Choice | Rationale |
|---|---|---|
| Framework | FastAPI | Required by spec; async-native |
| LLM | OpenAI GPT-4o | Strong French support, mature function-calling API, clear token pricing |
| Embeddings | text-embedding-3-small | Best cost/performance ratio for this size |
| Vector store | numpy in-memory | 60 vectors — a dedicated vector DB would be over-engineered |
| Metrics store | In-memory (Python list + lock) | Sufficient for demo; swap for SQLite/Postgres in production |
| Containerization | Docker Compose | Required by spec |

---

## Tradeoffs

| Decision | Alternative | Why this choice |
|---|---|---|
| Agent with tools | Static RAG | Handles all question types correctly; coherence/inventory queries require it |
| numpy cosine sim | ChromaDB / Pinecone | ~60 vectors; a vector DB adds a dependency and zero benefit at this scale |
| Section-level chunks | Full-document chunks | Finer-grained source citation; more precise retrieval |
| One-time LLM extraction at startup | Re-extract at query time | Avoid redundant LLM calls on every request; profiles are stable |
| In-memory metrics | SQLite / PostgreSQL | Sufficient for the demo scope; noted as upgrade path |
| No regex in query path | Regex dossier detection | LLM handles "le premier dossier", "dossier numéro 1" naturally; regex would be fragile |

---

## Implementation Roadmap

### Step 1 — Project scaffold
- `app/` package structure, `config.py` (pydantic-settings), `requirements.txt`
- `Dockerfile`, `docker-compose.yml`, `Makefile`, `.env.example`

### Step 2 — Ingestion pipeline
- `parser.py`: OCR JSON → section-level chunks with stable IDs
- `extractor.py`: one-time LLM pass → structured document profiles
- Persist profiles and embeddings to disk (avoid re-running on every restart)

### Step 3 — RAG engine
- `rag.py`: load chunks + embeddings into memory, cosine similarity retrieval
- Support metadata filtering by dossier and doc_type

### Step 4 — Tools + agent
- `tools.py`: implement `search_documents`, `get_dossier_documents`, `get_document_inventory`
- `planner.py`: Planner LLM — takes query + tool schemas, returns tool call plan (max 5 calls)
- `solver.py`: Solver LLM — takes query + all tool results, returns final answer with source citations

### Step 5 — API
- `main.py`: FastAPI app, lifespan startup (ingestion + indexing)
- `routes/chat.py`: POST /chat
- `routes/metrics.py`: GET /metrics, GET /metrics/history

### Step 6 — Metrics
- `metrics.py`: thread-safe in-memory store, aggregation logic
- Optional LLM-as-judge relevance scoring

### Step 7 — Documentation
- Update `README.md` with architecture overview, quickstart, API reference, technical choices

---

## Suggestions for Going Further

- **Chunking refinement**: overlap between sections to avoid missing context at boundaries
- **Reranking**: add a cross-encoder after retrieval to improve precision on ambiguous queries
- **Multi-turn conversation**: session-based history (Redis + session_id) for follow-up questions
- **Automated evaluation**: RAGAS framework with ground truth Q&A pairs built from the example questions
- **Streaming**: SSE streaming for long LLM responses
- **Metrics persistence**: PostgreSQL + TimescaleDB for time-series analysis of cost and latency
- **Semantic cache**: cache responses for semantically similar queries (e.g. GPTCache) to reduce cost

