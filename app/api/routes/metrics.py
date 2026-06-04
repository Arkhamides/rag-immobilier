from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.metrics.store import metrics_store

router = APIRouter(prefix="/metrics", tags=["metrics"])
logger = logging.getLogger(__name__)

_FAITHFULNESS_SYSTEM = """Tu es un juge d'évaluation pour un système RAG notarial.
Tu évalues si une réponse est fidèle aux documents sources.
Réponds uniquement avec du JSON valide, sans texte ni balise supplémentaire."""

_FAITHFULNESS_PROMPT = """Question : {query}

Réponse du système :
{answer}

Documents sources utilisés :
{sources_text}

Évalue la fidélité de la réponse aux sources sur une échelle de 0 à 1 :
- 1.0 = entièrement fondée sur les sources, aucune hallucination
- 0.0 = complètement inventée, non fondée sur les sources

Réponds avec ce JSON :
{{"score": 0.0, "explanation": "..."}}"""


@router.get("")
def get_metrics() -> dict:
    return metrics_store.get_aggregated()


@router.get("/history")
def get_metrics_history() -> list[dict]:
    return metrics_store.get_history()


class EvaluateRequest(BaseModel):
    interaction_ids: list[str] = []


@router.post("/evaluate")
def evaluate(body: EvaluateRequest, request: Request) -> dict:
    """LLM-as-judge faithfulness scoring over stored interactions."""
    client = request.app.state.client
    from app.core.config import settings

    if body.interaction_ids:
        records = metrics_store.get_by_ids(body.interaction_ids)
    else:
        records = metrics_store.get_all()

    if not records:
        return {"results": [], "evaluated": 0}

    results = []
    for rec in records:
        sources_text = "\n".join(
            f"[{s['id']}] (dossier={s['dossier']}, type={s['doc_type']})"
            for s in rec.sources
        ) or "Aucune source récupérée."

        prompt = _FAITHFULNESS_PROMPT.format(
            query=rec.query,
            answer=rec.answer,
            sources_text=sources_text,
        )

        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": _FAITHFULNESS_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            content = content.strip()
            if content.startswith("```"):
                lines = content.splitlines()
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            judge = json.loads(content)
            score = float(judge.get("score", 0.0))
            explanation = judge.get("explanation", "")
        except Exception as exc:
            logger.warning("Faithfulness evaluation failed for %s: %s", rec.id, exc)
            score = -1.0
            explanation = f"Evaluation error: {exc}"

        results.append({
            "interaction_id": rec.id,
            "query": rec.query,
            "faithfulness_score": score,
            "explanation": explanation,
        })

    return {"results": results, "evaluated": len(results)}
