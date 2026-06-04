from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.ingestion.parser import Chunk

logger = logging.getLogger(__name__)

_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

_model: SentenceTransformer | None = None
_cross_encoder: CrossEncoder | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", _EMBED_MODEL)
        _model = SentenceTransformer(_EMBED_MODEL)
    return _model


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_bm25_index(chunks: list[Chunk]) -> BM25Okapi:
    tokenized = [_tokenize(c.text) for c in chunks]
    logger.info("BM25 index built over %d chunks", len(chunks))
    return BM25Okapi(tokenized)


def load_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder: %s", _CROSS_ENCODER_MODEL)
        _cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL)
    return _cross_encoder


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


def hybrid_search(
    query: str,
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    chunks: list[Chunk],
    bm25_index: BM25Okapi,
    cross_encoder: CrossEncoder,
    top_k: int = 5,
    candidate_k: int = 15,
    dossier: int | None = None,
    doc_type: str | None = None,
) -> list[dict]:
    """
    Three-stage retrieval: RRF fusion of cosine + BM25, then cross-encoder reranking.
    candidate_k controls how many top results each retriever contributes before fusion.
    """
    n = len(chunks)
    cosine_scores: np.ndarray = embeddings @ query_embedding
    bm25_scores = np.array(bm25_index.get_scores(_tokenize(query)), dtype=float)

    mask = np.ones(n, dtype=bool)
    if dossier is not None:
        mask &= np.array([c.dossier == dossier for c in chunks])
    if doc_type is not None:
        mask &= np.array([c.doc_type == doc_type for c in chunks])

    filtered = np.where(mask)[0]
    if len(filtered) == 0:
        return []

    cosine_ranked = filtered[np.argsort(cosine_scores[filtered])[::-1]]
    bm25_ranked = filtered[np.argsort(bm25_scores[filtered])[::-1]]

    cosine_rank = {int(idx): rank for rank, idx in enumerate(cosine_ranked)}
    bm25_rank = {int(idx): rank for rank, idx in enumerate(bm25_ranked)}

    k = min(candidate_k, len(filtered))
    candidates = {int(i) for i in cosine_ranked[:k]} | {int(i) for i in bm25_ranked[:k]}

    # RRF constant k=60 is the standard value from the original paper
    K = 60
    rrf_scores = {
        idx: 1.0 / (K + cosine_rank.get(idx, len(filtered)))
             + 1.0 / (K + bm25_rank.get(idx, len(filtered)))
        for idx in candidates
    }
    rrf_ranked = sorted(candidates, key=lambda i: rrf_scores[i], reverse=True)

    pairs = [(query, chunks[i].text) for i in rrf_ranked]
    ce_scores = cross_encoder.predict(pairs)

    reranked = sorted(zip(rrf_ranked, ce_scores), key=lambda x: float(x[1]), reverse=True)

    results = []
    for idx, ce_score in reranked[:top_k]:
        chunk = chunks[idx]
        # Sigmoid normalizes the raw cross-encoder logit to a 0–1 relevance score
        norm_score = round(float(1.0 / (1.0 + np.exp(-float(ce_score)))), 4)
        results.append({
            "id": chunk.id,
            "dossier": chunk.dossier,
            "doc_type": chunk.doc_type,
            "filename": chunk.filename,
            "section": chunk.section,
            "text": chunk.text,
            "ocr_confidence": chunk.ocr_confidence,
            "relevance_score": norm_score,
        })

    return results


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
