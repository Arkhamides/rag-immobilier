from __future__ import annotations

import json as _json
from contextlib import contextmanager
from typing import Generator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent import planner, solver
from app.agent.tools import execute_tool
from app.metrics.profiler import Profiler
from app.metrics.store import metrics_store


@contextmanager
def _nullctx() -> Generator[None, None, None]:
    """No-op context manager used when Langfuse is disabled."""
    yield None

router = APIRouter(tags=["chat"])

# Price per 1M tokens (USD). Used for cost estimation.
_PRICE_PER_1M: dict[str, dict[str, float]] = {
    "anthropic/claude-sonnet-4.5": {"input": 3.0, "output": 15.0},
    "anthropic/claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    "openai/gpt-4o": {"input": 2.5, "output": 10.0},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.6},
}
_DEFAULT_PRICE = {"input": 3.0, "output": 15.0}


def _estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    prices = _PRICE_PER_1M.get(model, _DEFAULT_PRICE)
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []


@router.post("/chat")
def chat(body: ChatRequest, request: Request) -> dict:
    profiler = Profiler()

    state = request.app.state
    client = state.client
    chunks = state.chunks
    embeddings = state.embeddings
    profiles = state.profiles
    bm25_index = state.bm25_index
    cross_encoder = state.cross_encoder
    lf = getattr(state, "langfuse", None)

    from app.core.config import settings

    with (lf.start_as_current_observation(name="chat", as_type="span", input={"query": body.query}) if lf is not None else _nullctx()) as trace_span:
        # Step 1 — Planner
        with profiler.span("planner"):
            tool_plan, planner_usage = planner.plan(body.query, client, history=body.history, langfuse=lf)

        # Step 2 — Execute tools (sequentially; all are in-memory, sub-ms each)
        tool_results: list[dict] = []
        with profiler.span("tools"):
            with (lf.start_as_current_observation(name="tools", as_type="span", input={"plan": tool_plan}) if lf is not None else _nullctx()) as tool_span:
                for tc in tool_plan:
                    result = execute_tool(tc["name"], tc["arguments"], chunks, embeddings, profiles, bm25_index, cross_encoder)
                    tool_results.append({"name": tc["name"], "arguments": tc["arguments"], "result": result})
                if tool_span is not None:
                    tool_span.update(output={"count": len(tool_results)})

        # Step 3 — Solver
        with profiler.span("solver"):
            answer, sources, solver_usage = solver.solve(body.query, tool_results, chunks, client, history=body.history, langfuse=lf)

        latency_ms = profiler.total_ms

        # Aggregate token counts from both LLM calls
        planner_input = getattr(planner_usage, "prompt_tokens", 0) or 0
        planner_output = getattr(planner_usage, "completion_tokens", 0) or 0
        solver_input = getattr(solver_usage, "prompt_tokens", 0) or 0
        solver_output = getattr(solver_usage, "completion_tokens", 0) or 0
        total_input = planner_input + solver_input
        total_output = planner_output + solver_output
        cost_usd = _estimate_cost(total_input, total_output, settings.llm_model)

        # Retrieval relevance — best cosine score from search_documents, or 1.0 if
        # only get_dossier_documents was used (all docs explicitly retrieved).
        retrieval_relevance: float | None = None
        used_get_dossier = False
        for tr in tool_results:
            if tr["name"] == "search_documents" and isinstance(tr["result"], list):
                for item in tr["result"]:
                    score = item.get("relevance_score", 0.0)
                    if retrieval_relevance is None or score > retrieval_relevance:
                        retrieval_relevance = score
            elif tr["name"] == "get_dossier_documents":
                used_get_dossier = True
        if retrieval_relevance is None:
            retrieval_relevance = 1.0 if used_get_dossier else 0.0

        if trace_span is not None:
            trace_span.update(
                output={"answer": answer},
                metadata={
                    "cost_usd": round(cost_usd, 6),
                    "latency_ms": latency_ms,
                    "retrieval_relevance": round(retrieval_relevance, 4),
                },
            )

    metrics_store.record(
        query=body.query,
        answer=answer,
        latency_ms=latency_ms,
        input_tokens=total_input,
        output_tokens=total_output,
        cost_usd=round(cost_usd, 6),
        retrieval_relevance=round(retrieval_relevance, 4),
        sources=sources,
        breakdown=profiler.breakdown,
    )

    return {
        "answer": answer,
        "sources": sources,
        "metrics": {
            "latency_ms": latency_ms,
            "breakdown": profiler.breakdown,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cost_usd": round(cost_usd, 6),
            "retrieval_relevance": round(retrieval_relevance, 4),
        },
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {_json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    state = request.app.state
    client = state.client
    chunks = state.chunks
    embeddings = state.embeddings
    profiles = state.profiles
    bm25_index = state.bm25_index
    cross_encoder = state.cross_encoder
    lf = getattr(state, "langfuse", None)

    from app.core.config import settings

    def event_stream():
        profiler = Profiler()

        with (lf.start_as_current_observation(name="chat/stream", as_type="span", input={"query": body.query}) if lf is not None else _nullctx()) as trace_span:
            # Step 1 — Planner
            yield _sse("status", {"stage": "planning"})
            with profiler.span("planner"):
                tool_plan, planner_usage = planner.plan(body.query, client, history=body.history, langfuse=lf)

            # Step 2 — Tools
            yield _sse("status", {"stage": "tools"})
            tool_results: list[dict] = []
            with profiler.span("tools"):
                with (lf.start_as_current_observation(name="tools", as_type="span", input={"plan": tool_plan}) if lf is not None else _nullctx()) as tool_span:
                    for tc in tool_plan:
                        result = execute_tool(tc["name"], tc["arguments"], chunks, embeddings, profiles, bm25_index, cross_encoder)
                        tool_results.append({"name": tc["name"], "arguments": tc["arguments"], "result": result})
                    if tool_span is not None:
                        tool_span.update(output={"count": len(tool_results)})

            # Step 3 — Solver (streaming)
            yield _sse("status", {"stage": "solving"})
            answer = ""
            sources: list[dict] = []
            solver_usage = None
            with profiler.span("solver"):
                for item in solver.solve_stream(body.query, tool_results, chunks, client, history=body.history, langfuse=lf):
                    if item[0] == "delta":
                        yield _sse("delta", {"text": item[1]})
                    elif item[0] == "done":
                        _, answer, sources, solver_usage = item

            latency_ms = profiler.total_ms

            planner_input = getattr(planner_usage, "prompt_tokens", 0) or 0
            planner_output = getattr(planner_usage, "completion_tokens", 0) or 0
            solver_input = getattr(solver_usage, "prompt_tokens", 0) or 0
            solver_output = getattr(solver_usage, "completion_tokens", 0) or 0
            total_input = planner_input + solver_input
            total_output = planner_output + solver_output
            cost_usd = _estimate_cost(total_input, total_output, settings.llm_model)

            retrieval_relevance: float | None = None
            used_get_dossier = False
            for tr in tool_results:
                if tr["name"] == "search_documents" and isinstance(tr["result"], list):
                    for item in tr["result"]:
                        score = item.get("relevance_score", 0.0)
                        if retrieval_relevance is None or score > retrieval_relevance:
                            retrieval_relevance = score
                elif tr["name"] == "get_dossier_documents":
                    used_get_dossier = True
            if retrieval_relevance is None:
                retrieval_relevance = 1.0 if used_get_dossier else 0.0

            if trace_span is not None:
                trace_span.update(
                    output={"answer": answer},
                    metadata={
                        "cost_usd": round(cost_usd, 6),
                        "latency_ms": latency_ms,
                        "retrieval_relevance": round(retrieval_relevance, 4),
                    },
                )

        metrics_store.record(
            query=body.query,
            answer=answer,
            latency_ms=latency_ms,
            input_tokens=total_input,
            output_tokens=total_output,
            cost_usd=round(cost_usd, 6),
            retrieval_relevance=round(retrieval_relevance, 4),
            sources=sources,
            breakdown=profiler.breakdown,
        )

        yield _sse("done", {
            "answer": answer,
            "sources": sources,
            "metrics": {
                "latency_ms": latency_ms,
                "breakdown": profiler.breakdown,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "cost_usd": round(cost_usd, 6),
                "retrieval_relevance": round(retrieval_relevance, 4),
            },
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")
