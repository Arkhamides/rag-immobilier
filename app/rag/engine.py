from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.ingestion.parser import Chunk

logger = logging.getLogger(__name__)

_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", _EMBED_MODEL)
        _model = SentenceTransformer(_EMBED_MODEL)
    return _model


def load_or_compute_embeddings(chunks: list[Chunk], data_path: str) -> np.ndarray:
    base = Path(data_path)
    embeddings_path = base / "embeddings.npy"
    index_path = base / "chunk_index.json"

    current_ids = [c.id for c in chunks]

    if embeddings_path.exists() and index_path.exists():
        with open(index_path) as f:
            saved_ids = json.load(f)
        if saved_ids == current_ids:
            logger.info("Embeddings loaded from cache (%d vectors)", len(chunks))
            return np.load(embeddings_path)
        logger.info("Chunk IDs changed — recomputing embeddings")

    logger.info("Computing embeddings for %d chunks...", len(chunks))
    model = _get_model()
    embeddings: np.ndarray = model.encode(
        [c.text for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    base.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    with open(index_path, "w") as f:
        json.dump(current_ids, f)

    logger.info("Embeddings saved: %d × %d", *embeddings.shape)
    return embeddings


def cosine_search(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    chunks: list[Chunk],
    top_k: int = 5,
    dossier: int | None = None,
    doc_type: str | None = None,
) -> list[dict]:
    """Returns top-k chunks by cosine similarity with optional metadata filters."""
    scores: np.ndarray = embeddings @ query_embedding

    results = []
    for idx in np.argsort(scores)[::-1]:
        chunk = chunks[idx]
        if dossier is not None and chunk.dossier != dossier:
            continue
        if doc_type is not None and chunk.doc_type != doc_type:
            continue
        results.append({
            "id": chunk.id,
            "dossier": chunk.dossier,
            "doc_type": chunk.doc_type,
            "filename": chunk.filename,
            "section": chunk.section,
            "text": chunk.text,
            "ocr_confidence": chunk.ocr_confidence,
            "relevance_score": float(scores[idx]),
        })
        if len(results) == top_k:
            break

    return results
