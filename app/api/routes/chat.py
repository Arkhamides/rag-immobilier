from __future__ import annotations

import json as _json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent import planner, solver
from app.agent.tools import execute_tool
from app.metrics.profiler import Profiler
from app.metrics.store import metrics_store

router = APIRouter(tags=["chat"])

# Price per 1M tokens (USD). Used for cost estimation.
_PRICE_PER_1M: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-opus-4-8": {"input": 5.0, "output": 25.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    "anthropic/claude-sonnet-4.5": {"input": 3.0, "output": 15.0},
    "anthropic/claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
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

    # Root agent span — use start_observation (imperative, no context token) so there
    # are no OTel context-var tokens to detach. Child spans are linked explicitly by
    # calling parent_span.start_observation() in planner/solver/tools.
    trace_span = lf.start_observation(
        name="chat",
        as_type="agent",
        input={"query": body.query},
    ) if lf is not None else None

    try:
        # Step 1 — Planner
        with profiler.span("planner"):
            tool_plan, planner_usage = planner.plan(
                body.query, client, history=body.history, parent_span=trace_span
            )

        # Step 2 — Execute tools
        tool_results: list[dict] = []
        tools_span = trace_span.start_observation(
            name="tools",
            as_type="retriever",
            input={"plan": tool_plan},
        ) if trace_span is not None else None
        try:
            with profiler.span("tools"):
                for tc in tool_plan:
                    result = execute_tool(
                        tc["name"], tc["arguments"], chunks, embeddings, profiles, bm25_index, cross_encoder
                    )
                    tool_results.append({"name": tc["name"], "arguments": tc["arguments"], "result": result})
        finally:
            if tools_span is not None:
                tools_span.update(output={"count": len(tool_results)})
                tools_span.end()

        # Step 3 — Solver
        with profiler.span("solver"):
            answer, sources, solver_usage = solver.solve(
                body.query, tool_results, chunks, client, history=body.history, parent_span=trace_span
            )

        latency_ms = profiler.total_ms

        # Aggregate token counts from both LLM calls
        planner_input = getattr(planner_usage, "input_tokens", 0) or 0
        planner_output = getattr(planner_usage, "output_tokens", 0) or 0
        solver_input = getattr(solver_usage, "input_tokens", 0) or 0
        solver_output = getattr(solver_usage, "output_tokens", 0) or 0
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
                },
            )
            # Record retrieval_relevance as a proper Langfuse score — shows up in the
            # Scores panel and can be filtered/aggregated across traces.
            trace_span.score_trace(name="retrieval_relevance", value=round(retrieval_relevance, 4))

    finally:
        if trace_span is not None:
            trace_span.end()

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

        # Root agent span — imperative form only; no context managers inside this
        # generator to avoid the OTel context-var detach error across yield points.
        trace_span = lf.start_observation(
            name="chat/stream",
            as_type="agent",
            input={"query": body.query},
        ) if lf is not None else None

        answer = ""
        sources: list[dict] = []
        solver_usage = None
        planner_usage = None
        tool_results: list[dict] = []

        try:
            # Step 1 — Planner
            yield _sse("status", {"stage": "planning"})
            with profiler.span("planner"):
                tool_plan, planner_usage = planner.plan(
                    body.query, client, history=body.history, parent_span=trace_span
                )

            # Step 2 — Tools
            yield _sse("status", {"stage": "tools"})
            tools_span = trace_span.start_observation(
                name="tools",
                as_type="retriever",
                input={"plan": tool_plan},
            ) if trace_span is not None else None
            try:
                with profiler.span("tools"):
                    for tc in tool_plan:
                        result = execute_tool(
                            tc["name"], tc["arguments"], chunks, embeddings, profiles, bm25_index, cross_encoder
                        )
                        tool_results.append({"name": tc["name"], "arguments": tc["arguments"], "result": result})
            finally:
                if tools_span is not None:
                    tools_span.update(output={"count": len(tool_results)})
                    tools_span.end()

            # Step 3 — Solver (streaming)
            yield _sse("status", {"stage": "solving"})
            with profiler.span("solver"):
                for item in solver.solve_stream(
                    body.query, tool_results, chunks, client, history=body.history, parent_span=trace_span
                ):
                    if item[0] == "delta":
                        yield _sse("delta", {"text": item[1]})
                    elif item[0] == "done":
                        _, answer, sources, solver_usage = item

            latency_ms = profiler.total_ms

            planner_input = getattr(planner_usage, "input_tokens", 0) or 0
            planner_output = getattr(planner_usage, "output_tokens", 0) or 0
            solver_input = getattr(solver_usage, "input_tokens", 0) or 0
            solver_output = getattr(solver_usage, "output_tokens", 0) or 0
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
                    },
                )
                trace_span.score_trace(name="retrieval_relevance", value=round(retrieval_relevance, 4))

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

        finally:
            if trace_span is not None:
                trace_span.end()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
