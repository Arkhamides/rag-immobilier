from __future__ import annotations

import re

import pytest

from app.ingestion.parser import load_chunks
from app.core.config import settings


@pytest.fixture(scope="module")
def chunks():
    return load_chunks(settings.documents_path)


def test_total_chunk_count(chunks):
    assert len(chunks) == 76


def test_all_three_dossiers_present(chunks):
    for dossier in (1, 2, 3):
        assert any(c.dossier == dossier for c in chunks)


def test_chunk_ids_are_unique(chunks):
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_id_format(chunks):
    # Section names from DPE headers can contain (  ) É etc. — allow any non-whitespace after #
    pattern = re.compile(r"^dossier_\d+/[a-z0-9_]+#\S+$")
    for chunk in chunks:
        assert pattern.match(chunk.id), f"Bad ID format: {chunk.id}"


def test_compromis_has_vendeur_and_acquereur_sections(chunks):
    ids = {c.id for c in chunks}
    assert "dossier_1/compromis#VENDEUR" in ids
    assert "dossier_1/compromis#ACQUEREUR" in ids


def test_scan_006_has_low_ocr_confidence(chunks):
    """scan_006 is the intentionally corrupted CNI in dossier 3."""
    scan = next((c for c in chunks if c.filename == "scan_006"), None)
    assert scan is not None
    assert scan.ocr_confidence < settings.ocr_confidence_threshold


def test_identite_and_domicile_are_single_chunks(chunks):
    """Short single-topic docs must produce exactly one chunk each."""
    from collections import Counter
    per_file: Counter = Counter()
    for c in chunks:
        if c.doc_type in ("identite", "domicile"):
            per_file[(c.dossier, c.filename)] += 1
    for key, count in per_file.items():
        assert count == 1, f"{key} produced {count} chunks (expected 1)"


def test_dossier_directory_regex_accepts_underscore_form():
    assert re.fullmatch(r"dossier_?(\d+)", "dossier_3") is not None


def test_dossier_directory_regex_accepts_no_underscore_form():
    assert re.fullmatch(r"dossier_?(\d+)", "dossier3") is not None


def test_dossier_directory_regex_rejects_name_only():
    assert re.fullmatch(r"dossier_?(\d+)", "dossier_samanta") is None


def test_dossier_directory_regex_rejects_trailing_name():
    assert re.fullmatch(r"dossier_?(\d+)", "dossier_samanta3") is None
