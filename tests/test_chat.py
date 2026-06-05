"""
Integration tests for the /chat and /metrics routes.
The planner and solver are patched so no LLM calls are made.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.conftest import mock_planner_response, mock_solver_response

_MOCK_ANSWER = "Réponse de test. `dossier_1/compromis#VENDEUR`"
_MOCK_SOURCES = [
    {
        "id": "dossier_1/compromis#VENDEUR",
        "dossier": 1,
        "doc_type": "compromis",
        "filename": "compromis",
        "section": "VENDEUR",
        "ocr_confidence": 0.95,
    }
]


@pytest.fixture()
def mock_llm(monkeypatch):
    """Patch planner.plan and solver.solve to avoid real LLM calls."""
    planner_resp = mock_planner_response("get_document_inventory", "{}")
    solver_resp = mock_solver_response(_MOCK_ANSWER)

    monkeypatch.setattr(
        "app.agent.planner.plan",
        lambda query, client, **kwargs: ([{"name": "get_document_inventory", "arguments": {}}], planner_resp.usage),
    )
    monkeypatch.setattr(
        "app.agent.solver.solve",
        lambda query, tool_results, chunks, client, **kwargs: (_MOCK_ANSWER, _MOCK_SOURCES, solver_resp.usage),
    )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health(http_client):
    r = http_client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["chunks"] == 76
    assert data["profiles"] == 21


# ---------------------------------------------------------------------------
# POST /chat — response shape
# ---------------------------------------------------------------------------

def test_chat_response_has_required_keys(http_client, mock_llm):
    r = http_client.post("/chat", json={"query": "test"})
    assert r.status_code == 200
    data = r.json()
    assert {"answer", "sources", "metrics"} <= data.keys()


def test_chat_metrics_shape(http_client, mock_llm):
    r = http_client.post("/chat", json={"query": "test"})
    m = r.json()["metrics"]
    assert {"latency_ms", "input_tokens", "output_tokens", "cost_usd", "retrieval_relevance"} <= m.keys()
    assert m["latency_ms"] >= 0
    assert m["cost_usd"] >= 0


def test_chat_sources_list(http_client, mock_llm):
    r = http_client.post("/chat", json={"query": "test"})
    sources = r.json()["sources"]
    assert isinstance(sources, list)
    for s in sources:
        assert "id" in s
        assert "dossier" in s


def test_chat_answer_is_string(http_client, mock_llm):
    r = http_client.post("/chat", json={"query": "test"})
    assert isinstance(r.json()["answer"], str)
    assert len(r.json()["answer"]) > 0


def test_chat_missing_query_returns_422(http_client):
    r = http_client.post("/chat", json={})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /metrics — recorded after chat
# ---------------------------------------------------------------------------

def test_metrics_count_increments_after_chat(http_client, mock_llm):
    http_client.post("/chat", json={"query": "q1"})
    http_client.post("/chat", json={"query": "q2"})
    r = http_client.get("/metrics")
    assert r.status_code == 200
    assert r.json()["count"] == 2


def test_metrics_aggregated_shape(http_client, mock_llm):
    http_client.post("/chat", json={"query": "q"})
    data = http_client.get("/metrics").json()
    assert "latency_ms" in data
    assert "cost_usd" in data
    assert "retrieval_relevance" in data
    assert "tokens" in data


# ---------------------------------------------------------------------------
# GET /metrics/history
# ---------------------------------------------------------------------------

def test_metrics_history_contains_query(http_client, mock_llm):
    http_client.post("/chat", json={"query": "specific question"})
    history = http_client.get("/metrics/history").json()
    assert any(r["query"] == "specific question" for r in history)


def test_metrics_history_each_record_has_required_fields(http_client, mock_llm):
    http_client.post("/chat", json={"query": "q"})
    for record in http_client.get("/metrics/history").json():
        assert {"id", "query", "answer", "latency_ms", "cost_usd"} <= record.keys()
