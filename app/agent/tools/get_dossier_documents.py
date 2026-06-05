from __future__ import annotations

from app.ingestion.parser import Chunk


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
