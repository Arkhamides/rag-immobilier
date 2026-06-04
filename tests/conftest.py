from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.ingestion.parser import load_chunks
from app.metrics.store import metrics_store


# ---------------------------------------------------------------------------
# Data fixtures (session-scoped — loaded once per test run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def chunks():
    return load_chunks(settings.documents_path)


@pytest.fixture(scope="session")
def profiles():
    """Load profiles from disk cache — no LLM calls."""
    profiles_dir = Path(settings.data_path) / "profiles"
    result: dict[str, dict] = {}
    for f in sorted(profiles_dir.glob("*.json")):
        with open(f) as fp:
            p = json.load(fp)
        key = f"dossier_{p['dossier']}/{p['filename']}"
        result[key] = p
    return result


# ---------------------------------------------------------------------------
# HTTP client fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def http_client():
    """
    TestClient with full app startup (profiles/embeddings from disk cache).
    Shared across all tests; planner/solver are patched per-test where needed.
    """
    from app.main import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Metrics isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_metrics_store():
    """Clear the in-memory metrics store before every test."""
    metrics_store.reset()
    yield


# ---------------------------------------------------------------------------
# Mock LLM helpers
# ---------------------------------------------------------------------------

def _make_tool_call(name: str, arguments: str) -> MagicMock:
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def mock_planner_response(tool_name: str = "get_document_inventory", arguments: str = "{}") -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 20
    resp = MagicMock()
    resp.choices[0].message.tool_calls = [_make_tool_call(tool_name, arguments)]
    resp.usage = usage
    return resp


def mock_solver_response(answer: str = "Réponse de test. `dossier_1/compromis#VENDEUR`") -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = 200
    usage.completion_tokens = 50
    resp = MagicMock()
    resp.choices[0].message.content = answer
    resp.usage = usage
    return resp
