from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from app.core.config import settings

logger = logging.getLogger(__name__)

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "search_documents",
        "description": (
            "Recherche sémantique sur les sections de documents par similarité cosinus. "
            "À utiliser pour des questions ciblées sur un sujet, un nom ou une valeur spécifique."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La requête de recherche en français.",
                },
                "dossier": {
                    "type": "integer",
                    "description": "Filtrer sur un dossier spécifique (1, 2 ou 3). Optionnel.",
                },
                "doc_type": {
                    "type": "string",
                    "enum": ["compromis", "identite", "domicile", "dpe"],
                    "description": "Filtrer par type de document. Optionnel.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_dossier_documents",
        "description": (
            "Retourne toutes les sections de tous les documents d'un dossier. "
            "À utiliser pour les vérifications de cohérence, les incohérences entre documents, "
            "ou les questions nécessitant une vision complète du dossier."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dossier": {
                    "type": "integer",
                    "description": "Numéro du dossier (1, 2 ou 3).",
                },
            },
            "required": ["dossier"],
        },
    },
    {
        "name": "get_document_inventory",
        "description": (
            "Retourne la liste structurée des types de documents présents dans chaque dossier "
            "sans le texte intégral. À utiliser pour les questions sur les pièces manquantes "
            "ou la complétude des dossiers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

_SYSTEM = (
    "Tu es le planificateur d'un agent notarial. "
    "Ta seule tâche est de sélectionner les outils nécessaires pour répondre à la question. "
    f"Tu peux appeler au maximum {settings.max_tools_per_plan} outils en parallèle. "
    "Ne réponds pas à la question — choisis uniquement les outils appropriés.\n\n"
    "Corpus de dossiers :\n"
    "- Dossier 1 (Paris 75011) : vendeur MOREAU Jean-Pierre, acquéreurs LAURENT Sophie et LAURENT Marc\n"
    "- Dossier 2 (Bordeaux) : vendeuse DUBOIS Catherine, acquéreur BENALI Youssef\n"
    "- Dossier 3 (Lyon) : vendeuse PETIT Marie-Claire, acquéreur FONTAINE Alexandre\n\n"
    "Règles de sélection (OBLIGATOIRES) :\n"
    "1. `get_dossier_documents(N)` : utilise TOUJOURS cet outil quand :\n"
    "   - Une personne est nommée (MOREAU→1, LAURENT→1, DUBOIS→2, BENALI→2, PETIT→3, FONTAINE→3)\n"
    "   - La question porte sur les incohérences, la cohérence, ou une relecture complète d'un dossier\n"
    "   - La recherche sémantique ne retrouve PAS fiablement les sections VENDEUR/ACQUEREUR\n"
    "     ni les justificatifs d'une personne spécifique.\n"
    "2. `search_documents` : uniquement pour des recherches thématiques précises "
    "(prix, adresse d'un bien, clause légale) quand le dossier cible est inconnu.\n"
    "3. `get_document_inventory` : pour les questions sur les pièces manquantes "
    "ou la complétude des dossiers."
)


def plan(
    query: str,
    client: anthropic.Anthropic,
    history: list[dict] | None = None,
    parent_span: Any = None,
) -> tuple[list[dict], Any]:
    """
    Calls the planner LLM and returns (tool_call_list, usage).
    parent_span: optional Langfuse span; generation is created as an explicit child via
    parent_span.start_observation(), avoiding OTel context-var management entirely.
    """
    messages: list[dict] = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})

    gen = parent_span.start_observation(
        name="planner",
        as_type="generation",
        model=settings.llm_model,
        model_parameters={"temperature": 0},
        input=messages,
    ) if parent_span is not None else None

    response = client.messages.create(
        model=settings.llm_model,
        system=_SYSTEM,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice={"type": "any"},
        max_tokens=1024,
        temperature=0,
    )

    raw_calls = [block for block in response.content if block.type == "tool_use"]
    plan_list: list[dict] = []
    for tc in raw_calls[: settings.max_tools_per_plan]:
        plan_list.append({"name": tc.name, "arguments": tc.input})

    if gen is not None:
        usage = response.usage
        gen.update(
            output=plan_list,
            usage_details={
                "input": getattr(usage, "input_tokens", 0) or 0,
                "output": getattr(usage, "output_tokens", 0) or 0,
            },
        )
        gen.end()

    logger.info(
        "Planner: %d tool call(s) — %s",
        len(plan_list),
        [p["name"] for p in plan_list],
    )
    return plan_list, response.usage
