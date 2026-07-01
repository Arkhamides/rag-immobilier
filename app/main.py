from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from openai import OpenAI

from app.api.routes import chat, metrics
from app.core.config import settings
from app.ingestion.extractor import load_or_extract_profiles
from app.ingestion.parser import load_chunks
from app.rag.engine import load_or_compute_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    chunks = load_chunks(settings.documents_path)
    logger.info("Parser: %d chunks loaded", len(chunks))

    profiles = load_or_extract_profiles(chunks, settings.data_path, client)
    logger.info("Extractor: %d document profiles ready", len(profiles))

    embeddings = load_or_compute_embeddings(chunks, settings.data_path)
    logger.info("Embeddings: %s", embeddings.shape)

    app.state.chunks = chunks
    app.state.profiles = profiles
    app.state.embeddings = embeddings
    app.state.client = client

    yield


app = FastAPI(
    title="Primmo Notarial Agent",
    description="Conversational API for querying real estate sale dossiers.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat.router)
app.include_router(metrics.router)


@app.get("/health")
async def health():
    chunks = getattr(app.state, "chunks", None)
    profiles = getattr(app.state, "profiles", None)
    return {
        "status": "ok",
        "chunks": len(chunks) if chunks else 0,
        "profiles": len(profiles) if profiles else 0,
    }
