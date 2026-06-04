from __future__ import annotations

from app.metrics.store import MetricsStore


def _record(store: MetricsStore, latency_ms: float = 500.0, **kwargs):
    defaults = dict(
        query="q", answer="a", input_tokens=10, output_tokens=5,
        cost_usd=0.001, retrieval_relevance=0.8, sources=[]
    )
    defaults.update(kwargs)
    return store.record(latency_ms=latency_ms, **defaults)


def test_empty_store_aggregated():
    store = MetricsStore()
    agg = store.get_aggregated()
    assert agg["count"] == 0
    assert agg["latency_ms"] == {}


def test_record_appears_in_history():
    store = MetricsStore()
    _record(store, query="test query")
    history = store.get_history()
    assert len(history) == 1
    assert history[0]["query"] == "test query"


def test_record_returns_unique_ids():
    store = MetricsStore()
    id1 = _record(store)
    id2 = _record(store)
    assert id1 != id2


def test_get_by_ids_filters_correctly():
    store = MetricsStore()
    id1 = _record(store, query="first")
    _record(store, query="second")
    records = store.get_by_ids([id1])
    assert len(records) == 1
    assert records[0].query == "first"


def test_aggregated_latency_stats():
    store = MetricsStore()
    for ms in [100.0, 200.0, 300.0]:
        _record(store, latency_ms=ms)
    agg = store.get_aggregated()
    assert agg["count"] == 3
    assert agg["latency_ms"]["mean"] == 200.0
    assert agg["latency_ms"]["median"] == 200.0


def test_aggregated_cost_total():
    store = MetricsStore()
    _record(store, cost_usd=0.01)
    _record(store, cost_usd=0.02)
    agg = store.get_aggregated()
    assert abs(agg["cost_usd"]["total"] - 0.03) < 1e-9


def test_aggregated_token_counts():
    store = MetricsStore()
    _record(store, input_tokens=100, output_tokens=50)
    _record(store, input_tokens=200, output_tokens=100)
    agg = store.get_aggregated()
    assert agg["tokens"]["total_input"] == 300
    assert agg["tokens"]["total_output"] == 150


def test_reset_clears_records():
    store = MetricsStore()
    _record(store)
    _record(store)
    store.reset()
    assert store.get_aggregated()["count"] == 0
    assert store.get_history() == []
