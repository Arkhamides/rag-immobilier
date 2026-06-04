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


def get_dossier_documents(dossier: int, chunks: list[Chunk]) -> list[dict]:
    """Returns all chunks from the given dossier."""
    return [
        {
            "id": c.id,
            "dossier": c.dossier,
            "doc_type": c.doc_type,
            "filename": c.filename,
            "section": c.section,
            "text": c.text,
            "ocr_confidence": c.ocr_confidence,
        }
        for c in chunks
        if c.dossier == dossier
    ]


def get_document_inventory(profiles: dict[str, dict]) -> dict:
    """Returns a structural map of doc types present per dossier, plus completeness flags."""
    REQUIRED_TYPES = {"compromis", "identite", "domicile", "dpe"}

    inventory: dict[int, dict] = {}
    for doc_key, profile in profiles.items():
        d = profile.get("dossier")
        if d not in inventory:
            inventory[d] = {"dossier": d, "documents": []}

        entry: dict = {
            "doc_key": doc_key,
            "doc_type": profile.get("doc_type"),
            "filename": profile.get("filename"),
        }
        doc_type = profile.get("doc_type")
        if doc_type == "identite":
            entry["nom"] = profile.get("nom")
            entry["expired"] = profile.get("expired")
            entry["expire"] = profile.get("expire")
        elif doc_type == "domicile":
            entry["titulaire"] = profile.get("titulaire")
            entry["stale"] = profile.get("stale")
            entry["date_document"] = profile.get("date_document")
        elif doc_type == "compromis":
            entry["vendeurs"] = [v.get("nom") for v in profile.get("vendeurs", [])]
            entry["acquereurs"] = [a.get("nom") for a in profile.get("acquereurs", [])]
            entry["date"] = profile.get("date")
        elif doc_type == "dpe":
            entry["adresse"] = profile.get("adresse")
            entry["classe_energie"] = profile.get("classe_energie")
            entry["valide_jusqu_au"] = profile.get("valide_jusqu_au")

        inventory[d]["documents"].append(entry)

    for data in inventory.values():
        present = {doc["doc_type"] for doc in data["documents"]}
        data["missing_types"] = sorted(REQUIRED_TYPES - present)
        data["complete"] = not data["missing_types"]

    return {
        "dossiers": sorted(inventory.values(), key=lambda x: x["dossier"]),
        "completeness_checklist": [
            "1 compromis de vente",
            "1 pièce d'identité valide par partie",
            "1 justificatif de domicile de moins de 3 mois par partie",
            "1 DPE pour le bien immobilier",
        ],
    }


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
