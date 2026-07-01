from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import anthropic

from app.api.routes import chat, metrics
from app.core.config import settings
from app.ingestion.extractor import load_or_extract_profiles
from app.ingestion.parser import load_chunks
from app.rag.engine import build_bm25_index, load_cross_encoder, load_or_compute_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    chunks = load_chunks(settings.documents_path)
    logger.info("Parser: %d chunks loaded", len(chunks))

    profiles = load_or_extract_profiles(chunks, settings.data_path, client)
    logger.info("Extractor: %d document profiles ready", len(profiles))

    embeddings = load_or_compute_embeddings(chunks, settings.data_path)
    logger.info("Embeddings: %s", embeddings.shape)

    bm25_index = build_bm25_index(chunks)
    cross_encoder = load_cross_encoder()

    lf = None
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        from langfuse import Langfuse
        lf = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
        )
        logger.info("Langfuse: tracing enabled (host=%s)", settings.langfuse_base_url)

    app.state.chunks = chunks
    app.state.profiles = profiles
    app.state.embeddings = embeddings
    app.state.bm25_index = bm25_index
    app.state.cross_encoder = cross_encoder
    app.state.client = client
    app.state.langfuse = lf

    yield

    if lf is not None:
        lf.flush()


app = FastAPI(
    title="Primmo Notarial Agent",
    description="Conversational API for querying real estate sale dossiers.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat.router)
app.include_router(metrics.router)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/ui/")


@app.get("/health")
async def health():
    chunks = getattr(app.state, "chunks", None)
    profiles = getattr(app.state, "profiles", None)
    return {
        "status": "ok",
        "chunks": len(chunks) if chunks else 0,
        "profiles": len(profiles) if profiles else 0,
    }


app.mount("/ui", StaticFiles(directory="app/static", html=True), name="ui")
