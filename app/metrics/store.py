from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class InteractionRecord:
    id: str
    query: str
    answer: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    retrieval_relevance: float
    sources: list[dict]
    breakdown: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[InteractionRecord] = []

    def record(
        self,
        query: str,
        answer: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        retrieval_relevance: float,
        sources: list[dict],
        breakdown: dict[str, float] | None = None,
    ) -> str:
        rec = InteractionRecord(
            id=str(uuid.uuid4()),
            query=query,
            answer=answer,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            retrieval_relevance=retrieval_relevance,
            sources=sources,
            breakdown=breakdown or {},
        )
        with self._lock:
            self._records.append(rec)
        return rec.id

    def get_history(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "id": r.id,
                    "query": r.query,
                    "answer": r.answer,
                    "latency_ms": r.latency_ms,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cost_usd": r.cost_usd,
                    "retrieval_relevance": r.retrieval_relevance,
                    "sources": r.sources,
                    "breakdown": r.breakdown,
                    "timestamp": r.timestamp,
                }
                for r in self._records
            ]

    def get_aggregated(self) -> dict:
        with self._lock:
            records = list(self._records)

        if not records:
            return {"count": 0, "latency_ms": {}, "cost_usd": {}, "retrieval_relevance": {}, "tokens": {}}

        n = len(records)
        latencies = sorted(r.latency_ms for r in records)
        p95_idx = min(int(n * 0.95), n - 1)

        return {
            "count": n,
            "latency_ms": {
                "mean": round(sum(latencies) / n, 1),
                "median": round(latencies[n // 2], 1),
                "p95": round(latencies[p95_idx], 1),
            },
            "cost_usd": {
                "total": round(sum(r.cost_usd for r in records), 6),
                "mean": round(sum(r.cost_usd for r in records) / n, 6),
            },
            "retrieval_relevance": {
                "mean": round(sum(r.retrieval_relevance for r in records) / n, 4),
            },
            "tokens": {
                "total_input": sum(r.input_tokens for r in records),
                "total_output": sum(r.output_tokens for r in records),
            },
        }

    def get_by_ids(self, ids: list[str]) -> list[InteractionRecord]:
        id_set = set(ids)
        with self._lock:
            return [r for r in self._records if r.id in id_set]

    def get_all(self) -> list[InteractionRecord]:
        with self._lock:
            return list(self._records)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()


metrics_store = MetricsStore()
