from __future__ import annotations

import json
import logging
import re

from openai import OpenAI
from openai.types.completion_usage import CompletionUsage

from app.core.config import settings
from app.ingestion.parser import Chunk

logger = logging.getLogger(__name__)

# Matches full IDs (dossier_1/compromis#VENDEUR) and bare filenames (dossier_1/scan_id_001).
# Bare matches are resolved to their canonical chunk ID in _extract_sources.
_CHUNK_ID_RE = re.compile(r"dossier_\d+/[a-z0-9_]+(?:#[A-Za-z0-9_]+)?")

_SYSTEM = f"""Tu es un assistant notarial expert. Tu réponds en français aux questions sur des dossiers de vente immobilière.

Règles strictes :
1. Tes réponses doivent être basées UNIQUEMENT sur les documents fournis dans le contexte.
2. Cite les identifiants de chunk EXACTS (tels qu'ils apparaissent entre crochets dans le contexte) entre backticks.
   - Compromis/DPE : `dossier_1/compromis#VENDEUR`, `dossier_1/diag_dpe#IDENTIFICATION_DU_BIEN`
   - Identité/domicile (section toujours `#full`) : `dossier_1/scan_id_001#full`, `dossier_3/piece_12#full`
   N'omet JAMAIS la partie `#section` de l'identifiant.
3. Si un chunk a une confiance OCR inférieure à {settings.ocr_confidence_threshold}, mentionne-le explicitement.
   Ex : "La CNI de M. FONTAINE (scan_006) présente une qualité OCR faible (confiance : 0.38) — les informations peuvent être inexactes."
4. Si l'information demandée n'est pas dans les documents fournis, dis-le clairement.
5. Sois précis, factuel et professionnel."""


def _format_tool_results(tool_results: list[dict]) -> str:
    parts: list[str] = []
    for tr in tool_results:
        name = tr["name"]
        args = tr["arguments"]
        result = tr["result"]

        header = f"Outil : {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"
        parts.append(header)
        parts.append("-" * len(header))

        if isinstance(result, list):
            for item in result:
                conf = item.get("ocr_confidence", 1.0)
                conf_warning = f" [OCR faible : {conf:.2f}]" if conf < settings.ocr_confidence_threshold else ""
                score = item.get("relevance_score")
                score_str = f" [score={score:.3f}]" if score is not None else ""
                parts.append(f"[{item['id']}]{conf_warning}{score_str}")
                parts.append(item.get("text", "").strip())
                parts.append("")
        elif isinstance(result, dict):
            parts.append(json.dumps(result, ensure_ascii=False, indent=2))

        parts.append("")

    return "\n".join(parts)


def _extract_sources(
    answer: str,
    tool_results: list[dict],
    chunks: list[Chunk],
) -> list[dict]:
    """Extract cited chunk IDs from the answer and enrich with metadata."""
    raw_matches = set(_CHUNK_ID_RE.findall(answer))

    # Build lookup: chunk_id -> relevance_score (from search_documents results)
    relevance_map: dict[str, float] = {}
    for tr in tool_results:
        if isinstance(tr["result"], list):
            for item in tr["result"]:
                if "relevance_score" in item:
                    relevance_map[item["id"]] = item["relevance_score"]

    # Build lookup maps
    chunk_map: dict[str, Chunk] = {c.id: c for c in chunks}
    # Bare filename → first matching chunk (identite/domicile docs are always #full)
    prefix_map: dict[str, Chunk] = {}
    for c in chunks:
        key = f"dossier_{c.dossier}/{c.filename}"
        if key not in prefix_map:
            prefix_map[key] = c

    # Resolve bare matches (no #section) to their canonical chunk ID
    cited_ids: set[str] = set()
    for match in raw_matches:
        if "#" in match:
            cited_ids.add(match)
        else:
            resolved = prefix_map.get(match)
            if resolved:
                cited_ids.add(resolved.id)

    sources: list[dict] = []
    for cid in cited_ids:
        chunk = chunk_map.get(cid)
        if chunk is None:
            continue
        entry: dict = {
            "id": cid,
            "dossier": chunk.dossier,
            "doc_type": chunk.doc_type,
            "filename": chunk.filename,
            "section": chunk.section,
            "ocr_confidence": chunk.ocr_confidence,
        }
        if cid in relevance_map:
            entry["relevance_score"] = round(relevance_map[cid], 4)
        sources.append(entry)

    return sorted(sources, key=lambda s: s.get("relevance_score", 0), reverse=True)


def solve(
    query: str,
    tool_results: list[dict],
    chunks: list[Chunk],
    client: OpenAI,
) -> tuple[str, list[dict], CompletionUsage | None]:
    """
    Synthesizes a final answer from tool results.
    Returns (answer, sources, usage).
    """
    context = _format_tool_results(tool_results)
    user_content = f"Question : {query}\n\n=== Résultats des outils ===\n\n{context}\nRéponds à la question en citant les sources pertinentes."

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content or ""
    sources = _extract_sources(answer, tool_results, chunks)

    logger.info("Solver: answer length=%d, sources=%d", len(answer), len(sources))
    return answer, sources, response.usage
