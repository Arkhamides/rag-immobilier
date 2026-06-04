"""
Ground-truth evaluation tests.

These tests call the real LLM and assert that the agent surfaces the known
inconsistencies documented in plan.md § Intentional inconsistencies.

Run with:   pytest -m slow
Skip in CI: pytest -m "not slow"   (default via `make test`)
"""
from __future__ import annotations

import pytest


def _answer(http_client, query: str) -> str:
    r = http_client.post("/chat", json={"query": query})
    assert r.status_code == 200
    return r.json()["answer"].lower()


def _source_ids(http_client, query: str) -> set[str]:
    r = http_client.post("/chat", json={"query": query})
    return {s["id"] for s in r.json()["sources"]}


# ---------------------------------------------------------------------------
# Dossier 1 — DPE address mismatch
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_dpe_address_mismatch_detected(http_client):
    """Agent must identify that the DPE address differs from the compromis address."""
    answer = _answer(http_client, "L'adresse du DPE correspond-elle à l'adresse du compromis dans le dossier 1 ?")
    assert "acacias" in answer, "Expected DPE address (rue des Acacias) to be mentioned"
    assert "lilas" in answer, "Expected compromis address (rue des Lilas) to be mentioned"


@pytest.mark.slow
def test_dpe_address_mismatch_sources(http_client):
    """Both the DPE and compromis chunks must be cited as sources."""
    ids = _source_ids(http_client, "L'adresse du DPE correspond-elle à l'adresse du compromis dans le dossier 1 ?")
    assert any("diag_dpe" in sid or "dpe" in sid for sid in ids), "DPE chunk not cited"
    assert any("compromis" in sid for sid in ids), "Compromis chunk not cited"


# ---------------------------------------------------------------------------
# Dossier 1 — MOREAU expired CNI
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_moreau_expired_cni_detected(http_client):
    """Agent must flag MOREAU's CNI as expired."""
    answer = _answer(http_client, "Les pièces d'identité sont-elles en ordre dans le dossier 1 ?")
    assert "moreau" in answer, "MOREAU must be named"
    assert any(kw in answer for kw in ("expir", "2019")), (
        "Answer must mention expiry or the expiry year 2019"
    )


@pytest.mark.slow
def test_expired_ids_only_in_dossier_1(http_client):
    """Only dossier 1 has expired identity documents."""
    answer = _answer(http_client, "Quels dossiers ont des pièces d'identité expirées ?")
    assert "dossier 1" in answer or "dossier1" in answer
    assert "moreau" in answer


# ---------------------------------------------------------------------------
# Dossier 3 — FONTAINE stale domicile
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_fontaine_stale_domicile_detected(http_client):
    """Agent must flag FONTAINE's EDF bill (piece_12) as too old."""
    answer = _answer(http_client, "Le justificatif de domicile de FONTAINE est-il en ordre ?")
    assert "fontaine" in answer
    assert any(kw in answer for kw in ("périm", "3 mois", "ancienne", "09/2025", "stale")), (
        "Answer must flag the domicile document as outdated"
    )


@pytest.mark.slow
def test_fontaine_domicile_source_cited(http_client):
    """piece_12 (FONTAINE's EDF) must appear in sources."""
    ids = _source_ids(http_client, "Le justificatif de domicile de FONTAINE est-il en ordre ?")
    assert any("piece_12" in sid for sid in ids), f"piece_12 not in sources: {ids}"


# ---------------------------------------------------------------------------
# Dossier 3 — scan_006 OCR corruption
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_scan_006_ocr_corruption_flagged(http_client):
    """Agent must flag scan_006 as low-quality OCR when reviewing dossier 3."""
    answer = _answer(http_client, "Y a-t-il des incohérences dans le dossier 3 ?")
    assert any(kw in answer for kw in ("scan_006", "ocr", "qualité", "confiance", "corrompu")), (
        "scan_006 OCR corruption must be mentioned"
    )


# ---------------------------------------------------------------------------
# All dossiers — structural completeness
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_all_dossiers_structurally_complete(http_client):
    """No required doc type should be missing from any dossier."""
    answer = _answer(http_client, "Résumé des pièces manquantes dans tous les dossiers")
    assert "complet" in answer, "Answer should confirm all dossiers are structurally complete"


# ---------------------------------------------------------------------------
# Named-party lookup (regression: planner must use get_dossier_documents)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_vendeur_dossier_1_identified(http_client):
    answer = _answer(http_client, "Qui est le vendeur du dossier 1 ?")
    assert "moreau" in answer


@pytest.mark.slow
def test_acquereur_dossier_2_identified(http_client):
    answer = _answer(http_client, "Qui sont les acquéreurs du dossier 2 ?")
    assert "benali" in answer
