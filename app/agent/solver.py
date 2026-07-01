from __future__ import annotations

import json
import logging
import re
from collections.abc import Generator
from typing import Any

import anthropic

from app.core.config import settings
from app.ingestion.parser import Chunk

logger = logging.getLogger(__name__)

# Matches full IDs (dossier_1/compromis#VENDEUR, dossier_1/diag_dpe#IDENTIFICATION_(DPE))
# and bare filenames (dossier_1/scan_id_001).
# Section names from DPE headers may contain (, ), É etc. — stop at whitespace or backtick.
# Bare matches (no #section) are resolved in _extract_sources via prefix lookup.
_CHUNK_ID_RE = re.compile(r"dossier_\d+/[a-z0-9_]+(?:#[^\s`'\"]+)?")

_SYSTEM = f"""Tu es un assistant notarial expert. Tu réponds en français aux questions sur des dossiers de vente immobilière.

Règles strictes :
1. Tes réponses doivent être basées UNIQUEMENT sur les documents fournis dans le contexte.
2. Cite les identifiants de chunk EXACTS (tels qu'ils apparaissent entre crochets dans le contexte) entre backticks.
   - Compromis/DPE : `dossier_1/compromis#VENDEUR`, `dossier_1/diag_dpe#IDENTIFICATION_DU_BIEN`
   - Identité/domicile (section toujours `#full`) : `dossier_1/scan_id_001#full`, `dossier_3/piece_12#full`
   N'omet JAMAIS la partie `#section` de l'identifiant.
3. Si un chunk a une confiance OCR inférieure à {settings.ocr_confidence_threshold}, mentionne-le explicitement.
   Ex : "La CNI de M. FONTAINE (scan_006) présente une qualité OCR faible (confiance : 0.38) — les informations peuvent être inexactes."
4. Pour tout justificatif de domicile, vérifie que sa date est ≤ 3 mois avant la date du compromis.
   Si la date dépasse ce délai ou est absente, signale-le explicitement comme non conforme.
5. Si l'information demandée n'est pas dans les documents fournis, dis-le clairement.
6. Sois précis, factuel et professionnel."""


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
    client: anthropic.Anthropic,
    history: list[dict] | None = None,
    parent_span: Any = None,
) -> tuple[str, list[dict], Any]:
    """
    Synthesizes a final answer from tool results.
    Returns (answer, sources, usage).
    parent_span: optional Langfuse span; generation is created as an explicit child via
    parent_span.start_observation(), avoiding OTel context-var management entirely.
    """
    context = _format_tool_results(tool_results)
    user_content = f"Question : {query}\n\n=== Résultats des outils ===\n\n{context}\nRéponds à la question en citant les sources pertinentes."

    messages: list[dict] = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    gen = parent_span.start_observation(
        name="solver",
        as_type="generation",
        model=settings.llm_model,
        model_parameters={"temperature": 0},
        input=messages,
    ) if parent_span is not None else None

    response = client.messages.create(
        model=settings.llm_model,
        system=_SYSTEM,
        messages=messages,
        max_tokens=4096,
        temperature=0,
    )

    answer = response.content[0].text if response.content else ""

    if gen is not None:
        usage = response.usage
        gen.update(
            output=answer,
            usage_details={
                "input": getattr(usage, "input_tokens", 0) or 0,
                "output": getattr(usage, "output_tokens", 0) or 0,
            },
        )
        gen.end()

    sources = _extract_sources(answer, tool_results, chunks)

    logger.info("Solver: answer length=%d, sources=%d", len(answer), len(sources))
    return answer, sources, response.usage


def solve_stream(
    query: str,
    tool_results: list[dict],
    chunks: list[Chunk],
    client: anthropic.Anthropic,
    history: list[dict] | None = None,
    parent_span: Any = None,
) -> Generator[tuple, None, None]:
    """
    Streaming solver. Yields:
      ("delta", str)                      — incremental text chunks
      ("done", str, list, usage | None)   — full answer, sources, usage object
    parent_span: optional Langfuse span; generation is created as an explicit child via
    parent_span.start_observation(), safe to use inside generators (no context-var tokens).
    """
    context = _format_tool_results(tool_results)
    user_content = f"Question : {query}\n\n=== Résultats des outils ===\n\n{context}\nRéponds à la question en citant les sources pertinentes."

    messages: list[dict] = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    gen = parent_span.start_observation(
        name="solver",
        as_type="generation",
        model=settings.llm_model,
        model_parameters={"temperature": 0},
        input=messages,
    ) if parent_span is not None else None

    full_answer = ""
    usage = None

    with client.messages.stream(
        model=settings.llm_model,
        system=_SYSTEM,
        messages=messages,
        max_tokens=4096,
        temperature=0,
    ) as stream:
        for text in stream.text_stream:
            if text:
                full_answer += text
                yield ("delta", text)
        message = stream.get_final_message()
        usage = message.usage

    if gen is not None:
        gen.update(
            output=full_answer,
            usage_details={
                "input": getattr(usage, "input_tokens", 0) or 0,
                "output": getattr(usage, "output_tokens", 0) or 0,
            },
        )
        gen.end()

    sources = _extract_sources(full_answer, tool_results, chunks)
    logger.info("Solver (stream): answer length=%d, sources=%d", len(full_answer), len(sources))
    yield ("done", full_answer, sources, usage)
