from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.ingestion.parser import Chunk

from .get_document_inventory import get_document_inventory
from .get_dossier_documents import get_dossier_documents
from .search_documents import search_documents

__all__ = ["search_documents", "get_dossier_documents", "get_document_inventory", "execute_tool"]


def execute_tool(
    name: str,
    arguments: dict,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    profiles: dict[str, dict],
    bm25_index: BM25Okapi,
    cross_encoder: CrossEncoder,
) -> object:
    """Dispatch a named tool call. Returns the tool result."""
    if name == "search_documents":
        return search_documents(
            query=arguments["query"],
            chunks=chunks,
            embeddings=embeddings,
            bm25_index=bm25_index,
            cross_encoder=cross_encoder,
            dossier=arguments.get("dossier"),
            doc_type=arguments.get("doc_type"),
        )
    if name == "get_dossier_documents":
        return get_dossier_documents(dossier=arguments["dossier"], chunks=chunks)
    if name == "get_document_inventory":
        return get_document_inventory(profiles=profiles)
    return {"error": f"Unknown tool: {name}"}
