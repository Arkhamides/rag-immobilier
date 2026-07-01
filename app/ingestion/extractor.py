from __future__ import annotations

import json
import logging
from pathlib import Path

import anthropic

from app.core.config import settings
from app.ingestion.parser import Chunk

logger = logging.getLogger(__name__)

_SYSTEM = (
    "Tu es un assistant notarial expert. "
    "Extrais les informations structurées du document fourni. "
    "Réponds uniquement avec du JSON valide, sans texte ni balise supplémentaire."
)

_PROMPTS: dict[str, str] = {
    "compromis": """Extrais les informations du compromis de vente et retourne ce JSON :
{{
  "doc_type": "compromis",
  "date": "JJ/MM/AAAA",
  "notaire": "...",
  "vendeurs": [{{"nom": "...", "dob": "JJ/MM/AAAA", "lieu_naissance": "...", "adresse": "..."}}],
  "acquereurs": [{{"nom": "...", "dob": "JJ/MM/AAAA", "lieu_naissance": "...", "adresse": "..."}}],
  "bien": {{"adresse": "...", "type": "...", "surface_m2": 0}},
  "prix_eur": 0
}}

Document :
{text}""",

    "identite": """Extrais les informations de la pièce d'identité et retourne ce JSON :
{{
  "doc_type": "identite",
  "nom": "...",
  "prenoms": "...",
  "dob": "JJ/MM/AAAA",
  "lieu_naissance": "...",
  "numero": "...",
  "delivree": "JJ/MM/AAAA",
  "expire": "JJ/MM/AAAA",
  "expired": false
}}
Si la date d'expiration est antérieure au 01/01/2026, mets expired=true.
Si le document est illisible (OCR corrompu), retourne les champs lisibles et null pour les autres.

Document :
{text}""",

    "domicile": """Extrais les informations du justificatif de domicile et retourne ce JSON :
{{
  "doc_type": "domicile",
  "type_document": "facture_edf | avis_imposition | autre",
  "titulaire": "...",
  "adresse": "...",
  "date_document": "JJ/MM/AAAA",
  "stale": false
}}
Mets stale=true si la date est antérieure de plus de 3 mois au 10/02/2026.

Document :
{text}""",

    "dpe": """Extrais les informations du DPE et retourne ce JSON :
{{
  "doc_type": "dpe",
  "adresse": "...",
  "type_bien": "...",
  "surface_m2": 0,
  "classe_energie": "A|B|C|D|E|F|G",
  "date_etablissement": "JJ/MM/AAAA",
  "valide_jusqu_au": "JJ/MM/AAAA"
}}

Document :
{text}""",
}


def _parse_llm_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(content.strip())


def _extract_one(doc_type: str, text: str, client: anthropic.Anthropic) -> dict:
    response = client.messages.create(
        model=settings.llm_model,
        system=_SYSTEM,
        messages=[
            {"role": "user", "content": _PROMPTS[doc_type].format(text=text)},
        ],
        max_tokens=2048,
        temperature=0,
    )
    return _parse_llm_json(response.content[0].text)


def load_or_extract_profiles(
    chunks: list[Chunk], data_path: str, client: anthropic.Anthropic
) -> dict[str, dict]:
    """
    Returns profiles keyed by "dossier_{n}/{stem}".
    Loads from disk if cached; runs LLM extraction otherwise.
    """
    profiles_dir = Path(data_path) / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # One representative chunk per document (dossier + filename)
    docs: dict[tuple[int, str], Chunk] = {}
    for chunk in chunks:
        key = (chunk.dossier, chunk.filename)
        if key not in docs:
            docs[key] = chunk

    profiles: dict[str, dict] = {}

    for (dossier, filename), rep in docs.items():
        cache_file = profiles_dir / f"dossier_{dossier}_{filename}.json"
        doc_key = f"dossier_{dossier}/{filename}"

        if cache_file.exists():
            with open(cache_file) as f:
                profiles[doc_key] = json.load(f)
            logger.info("Profile loaded from cache: %s", cache_file.name)
            continue

        doc_chunks = [c for c in chunks if c.dossier == dossier and c.filename == filename]
        full_text = "\n\n".join(c.text for c in doc_chunks)

        logger.info("Extracting profile: dossier_%d/%s (%s)", dossier, filename, rep.doc_type)
        try:
            profile = _extract_one(rep.doc_type, full_text, client)
        except Exception as exc:
            logger.warning("Extraction failed for %s: %s", filename, exc)
            profile = {}

        profile["dossier"] = dossier
        profile["filename"] = filename
        profile["doc_type"] = rep.doc_type

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

        profiles[doc_key] = profile

    return profiles
