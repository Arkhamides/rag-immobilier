# Primmo Notarial Agent

Primmo is a conversational API that lets notaries ask natural language questions over real estate sale dossiers. The corpus is OCR-scanned documents (Google Vision API format) across multiple dossiers. The system must answer correctly, cite its sources, and expose cost/latency/relevance metrics.

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY (OpenRouter key)
```

## Running

```bash
make dev    # local with hot reload — uvicorn app.main:app --reload
make run    # Docker — docker compose up --build
```

The app runs on `http://localhost:8000`. Swagger UI at `/docs`.

On startup the app runs the ingestion pipeline (parser → extractor → embeddings). Results are cached to `data/` so subsequent restarts are instant.

## Key commands

```bash
make dev      # run locally with hot reload
make run      # run with Docker
make test     # pytest tests/
make lint     # ruff check app/
make format   # ruff format app/
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | required | OpenRouter API key |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | LLM provider base URL |
| `LLM_MODEL` | `anthropic/claude-sonnet-4.5` | Model for planner, solver, and extraction |
| `DOCUMENTS_PATH` | `documents` | Path to OCR JSON corpus |
| `DATA_PATH` | `data` | Cache directory for profiles and embeddings |
| `MAX_TOOLS_PER_PLAN` | `5` | ReWoo planner tool call cap |
| `OCR_CONFIDENCE_THRESHOLD` | `0.7` | Below this, agent flags the source as unreliable |

## Project layout

```
app/
├── main.py                # FastAPI app + lifespan startup
├── core/
│   └── config.py          # Pydantic settings (reads from .env)
├── api/routes/
│   ├── chat.py            # POST /chat
│   └── metrics.py         # GET /metrics, /metrics/history, POST /evaluate
├── ingestion/
│   ├── parser.py          # OCR JSON → Chunk list
│   └── extractor.py       # LLM structured extraction → profiles (cached)
├── rag/
│   └── engine.py          # Embeddings (sentence-transformers) + cosine search
├── agent/
│   ├── tools.py           # search_documents, get_dossier_documents, get_document_inventory
│   ├── planner.py         # ReWoo planner LLM call
│   └── solver.py          # ReWoo solver LLM call
└── metrics/
    └── store.py           # Thread-safe in-memory interaction log

data/                      # Generated at runtime — gitignored
├── profiles/              # One JSON file per document (21 total)
├── embeddings.npy         # 76 × 384 float32 array
└── chunk_index.json       # Ordered chunk ID list (cache invalidation key)

documents/                 # OCR corpus — not modified at runtime
├── dossier_1/             # 9 files
├── dossier_2/             # 6 files
└── dossier_3/             # 6 files
```

## Implementation status

| Step | Module | Status |
|---|---|---|
| Scaffold | `main.py`, `core/`, `api/` | Done |
| Ingestion | `ingestion/parser.py`, `ingestion/extractor.py` | Done |
| RAG engine | `rag/engine.py` | Done (embeddings + search) |
| Agent | `agent/tools.py`, `agent/planner.py`, `agent/solver.py` | Stub |
| Metrics | `metrics/store.py` | Stub |
| API routes | `api/routes/chat.py`, `api/routes/metrics.py` | Stub (501) |
