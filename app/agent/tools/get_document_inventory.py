from __future__ import annotations


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
