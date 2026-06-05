from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.ingestion.parser import Chunk
from app.rag.engine import _get_model, hybrid_search


def search_documents(
    query: str,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    bm25_index: BM25Okapi,
    cross_encoder: CrossEncoder,
    top_k: int = 5,
    dossier: int | None = None,
    doc_type: str | None = None,
) -> list[dict]:
    """Hybrid search: RRF fusion of cosine + BM25, reranked by cross-encoder."""
    model = _get_model()
    query_emb: np.ndarray = model.encode(query, normalize_embeddings=True)
    return hybrid_search(
        query=query,
        query_embedding=query_emb,
        embeddings=embeddings,
        chunks=chunks,
        bm25_index=bm25_index,
        cross_encoder=cross_encoder,
        top_k=top_k,
        dossier=dossier,
        doc_type=doc_type,
    )
