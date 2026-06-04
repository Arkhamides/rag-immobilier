from __future__ import annotations

import pytest

from app.agent.tools import get_dossier_documents, get_document_inventory


def test_get_dossier_documents_returns_only_requested_dossier(chunks):
    result = get_dossier_documents(1, chunks)
    assert result
    assert all(r["dossier"] == 1 for r in result)


def test_get_dossier_documents_no_cross_contamination(chunks):
    d1_ids = {r["id"] for r in get_dossier_documents(1, chunks)}
    d2_ids = {r["id"] for r in get_dossier_documents(2, chunks)}
    d3_ids = {r["id"] for r in get_dossier_documents(3, chunks)}
    assert d1_ids.isdisjoint(d2_ids)
    assert d1_ids.isdisjoint(d3_ids)
    assert d2_ids.isdisjoint(d3_ids)


def test_get_dossier_documents_result_shape(chunks):
    result = get_dossier_documents(1, chunks)
    required_keys = {"id", "dossier", "doc_type", "filename", "section", "text", "ocr_confidence"}
    for item in result:
        assert required_keys <= item.keys()


def test_get_document_inventory_covers_all_dossiers(profiles):
    inv = get_document_inventory(profiles)
    dossier_nums = {d["dossier"] for d in inv["dossiers"]}
    assert dossier_nums == {1, 2, 3}


def test_get_document_inventory_all_dossiers_structurally_complete(profiles):
    """All dossiers have at least one of each required doc type."""
    inv = get_document_inventory(profiles)
    for dossier in inv["dossiers"]:
        assert dossier["complete"] is True, (
            f"Dossier {dossier['dossier']} missing: {dossier['missing_types']}"
        )


def test_get_document_inventory_moreau_cni_flagged_expired(profiles):
    """MOREAU's CNI (dossier 1) must be flagged as expired — ground truth from plan.md."""
    inv = get_document_inventory(profiles)
    d1 = next(d for d in inv["dossiers"] if d["dossier"] == 1)
    expired_ids = [
        doc for doc in d1["documents"]
        if doc["doc_type"] == "identite" and doc.get("expired")
    ]
    assert expired_ids, "Expected at least one expired CNI in dossier 1 (MOREAU)"


def test_get_document_inventory_fontaine_domicile_flagged_stale(profiles):
    """FONTAINE's EDF bill (piece_12) must be flagged stale — ground truth from plan.md."""
    inv = get_document_inventory(profiles)
    d3 = next(d for d in inv["dossiers"] if d["dossier"] == 3)
    stale_docs = [
        doc for doc in d3["documents"]
        if doc["doc_type"] == "domicile" and doc.get("stale")
    ]
    assert stale_docs, "Expected piece_12 (FONTAINE's EDF) to be flagged stale in dossier 3"


def test_get_document_inventory_includes_completeness_checklist(profiles):
    inv = get_document_inventory(profiles)
    assert "completeness_checklist" in inv
    assert len(inv["completeness_checklist"]) == 4
