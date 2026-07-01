from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: str             # e.g. "dossier_1/compromis#VENDEUR"
    dossier: int
    doc_type: str       # compromis | identite | domicile | dpe
    filename: str       # stem, e.g. "compromis"
    section: str        # e.g. "VENDEUR", "block_1", "full"
    text: str
    ocr_confidence: float


def _extract_blocks(ocr_json: dict) -> tuple[str, list[dict]]:
    """Returns (full_text, [{text, confidence}]) from Vision API JSON."""
    fta = ocr_json["responses"][0]["fullTextAnnotation"]
    full_text = fta.get("text", "")

    blocks = []
    for page in fta.get("pages", []):
        for block in page.get("blocks", []):
            symbols: list[dict] = []
            word_texts: list[str] = []
            for para in block.get("paragraphs", []):
                for word in para.get("words", []):
                    syms = word.get("symbols", [])
                    word_texts.append("".join(s["text"] for s in syms))
                    symbols.extend(syms)

            block_text = " ".join(word_texts).strip()
            confidence = (
                sum(s.get("confidence", 1.0) for s in symbols) / len(symbols)
                if symbols else 0.0
            )
            if block_text:
                blocks.append({"text": block_text, "confidence": confidence})

    return full_text, blocks


def _classify_doc_type(filename: str, text: str) -> str:
    fname = filename.lower()
    preview = text[:500].upper()

    if "compromis" in fname or "COMPROMIS DE VENTE" in preview:
        return "compromis"
    if "dpe" in fname or "diag" in fname or "diagnostic" in fname or "DIAGNOSTIC DE PERFORMANCE" in preview:
        return "dpe"
    # Identity cards — handle OCR corruption ("MATIONALE", "IDTITE" in scan_006)
    if (
        any(k in fname for k in ["id_", "_id", "identite", "piece_id", "scan_id"])
        or "CARTE NATIONALE" in preview
        or "CARTE MATIONALE" in preview
        or ("CARTE" in preview and "IDENTIT" in preview)
    ):
        return "identite"
    return "domicile"


# Compromis: split on VENDEUR, ACQUEREUR, and ARTICLE N headers
_COMPROMIS_HEADER = re.compile(
    r"(LE\(S\)\s+VENDEUR\(S\)\s*:"
    r"|L\(?LES?\)?\s+ACQUEREUR\(S\)\s*:"
    r"|ARTICLE\s+\d+\s*[-–]\s*[^\n]+)",
    re.IGNORECASE,
)

# DPE: all-caps lines of 5+ characters
_DPE_HEADER = re.compile(r"^([A-ZÀ-Ü][A-ZÀ-Ü\s\(\)\-]{4,})$", re.MULTILINE)


def _header_to_key(header: str) -> str:
    h = header.strip()
    if re.match(r"LE\(S\)\s+VENDEUR", h, re.IGNORECASE):
        return "VENDEUR"
    if re.match(r"L\(?LES?\)?\s+ACQUEREUR", h, re.IGNORECASE):
        return "ACQUEREUR"
    if m := re.match(r"ARTICLE\s+(\d+)\s*[-–]\s*(.+)", h, re.IGNORECASE):
        num = m.group(1).zfill(2)
        title = re.sub(r"\s+", "_", m.group(2).strip().upper())[:25]
        return f"ART_{num}_{title}"
    return re.sub(r"\W+", "_", h.upper())[:40]


def _split_on_headers(text: str, header_re: re.Pattern, key_fn=None) -> list[tuple[str, str]]:
    parts = header_re.split(text)
    if len(parts) <= 1:
        return []

    sections: list[tuple[str, str]] = []
    i = 1
    while i < len(parts) - 1:
        header = parts[i].strip()
        content = (parts[i + 1].strip() if i + 1 < len(parts) else "")
        key = key_fn(header) if key_fn else re.sub(r"\s+", "_", header.upper())[:40]
        if content:
            sections.append((key, f"{header}\n{content}"))
        i += 2

    return sections


def _make_chunks(
    dossier: int, stem: str, doc_type: str, text: str, blocks: list[dict]
) -> list[Chunk]:
    mean_conf = (
        sum(b["confidence"] for b in blocks) / len(blocks) if blocks else 1.0
    )

    def _c(section: str, content: str, conf: float = mean_conf) -> Chunk:
        return Chunk(
            id=f"dossier_{dossier}/{stem}#{section}",
            dossier=dossier,
            doc_type=doc_type,
            filename=stem,
            section=section,
            text=content,
            ocr_confidence=conf,
        )

    # identite and domicile are short, single-topic documents — always one chunk
    if doc_type in ("identite", "domicile"):
        return [_c("full", text)]

    # Level 1 — header splitting for structured multi-section documents
    if doc_type == "compromis":
        sections = _split_on_headers(text, _COMPROMIS_HEADER, key_fn=_header_to_key)
    else:  # dpe
        sections = _split_on_headers(text, _DPE_HEADER)

    if sections:
        return [_c(key, content) for key, content in sections]

    # Level 2 — OCR block boundaries (fallback: headers unreadable due to OCR corruption)
    if len(blocks) > 1:
        return [
            _c(f"block_{i + 1}", b["text"], b["confidence"])
            for i, b in enumerate(blocks)
            if b["text"].strip()
        ]

    # Level 3 — whole document as one chunk
    return [_c("full", text)]


def load_chunks(documents_path: str) -> list[Chunk]:
    base = Path(documents_path)
    chunks: list[Chunk] = []

    for dossier_dir in sorted(base.iterdir()):
        if not dossier_dir.is_dir() or not dossier_dir.name.startswith("dossier_"):
            continue
        dossier_num = int(dossier_dir.name.split("_")[1])

        for json_file in sorted(dossier_dir.glob("*.json")):
            with open(json_file) as f:
                ocr_json = json.load(f)

            text, blocks = _extract_blocks(ocr_json)
            doc_type = _classify_doc_type(json_file.stem, text)
            chunks.extend(_make_chunks(dossier_num, json_file.stem, doc_type, text, blocks))

    return chunks
