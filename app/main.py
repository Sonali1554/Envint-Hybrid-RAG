"""
FastAPI application: the public HTTP surface.

Endpoints:
  GET  /health   -> liveness check (used by docker-compose healthcheck).
  GET  /metrics  -> process-wide request/error counters.
  POST /query    -> the main RAG endpoint: retrieve + answer + cite, traced.

The heavy RetrievalEngine (which builds the indexes) is created ONCE at startup
and reused for every request.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.config import settings
from app.llm import client
from app.obs.middleware import METRICS, Tracer, observability_middleware
from app.retrieval.engine import VALID_STRATEGIES, RetrievalEngine
from app.schema import QueryRequest, QueryResponse

app = FastAPI(title="Envint Hybrid Retrieval & Reranking Benchmark")

# Register the observability middleware (times every request, counts errors).
app.middleware("http")(observability_middleware)

# Built on startup and stored on the app so every request reuses it.
_engine: RetrievalEngine | None = None


@app.on_event("startup")
def _startup() -> None:
    global _engine
    _engine = RetrievalEngine()


def get_engine() -> RetrievalEngine:
    # Lazy build as a safety net (e.g. if startup hook didn't run in a test).
    global _engine
    if _engine is None:
        _engine = RetrievalEngine()
    return _engine


# Path to the single-page web UI (served at "/").
_UI_FILE = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    """Serve the browser UI. Visit http://localhost:8000/ to use the RAG."""
    return _UI_FILE.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "embedding_version": settings.embed_version}


@app.get("/metrics")
def metrics() -> dict:
    return {
        "requests": METRICS.requests,
        "errors": METRICS.errors,
        "error_rate": round(METRICS.error_rate(), 4),
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    # --- Input validation (negative-input guardrail). ---
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    if req.strategy not in VALID_STRATEGIES:
        raise HTTPException(
            status_code=422,
            detail=f"strategy must be one of {sorted(VALID_STRATEGIES)}",
        )

    engine = get_engine()
    tracer = Tracer()

    # --- Retrieve (timed). ---
    with tracer.span(f"retrieve::{req.strategy}"):
        retrieved = engine.retrieve(req.query, strategy=req.strategy, top_k=req.top_k)

    # --- Answer with guardrails (timed). ---
    with tracer.span("answer"):
        result = client.answer(req.query, retrieved)

    tracer.record_tokens(result["token_usage"])

    return QueryResponse(
        query=req.query,
        strategy=req.strategy,
        answer=result["answer"],
        refused=result["refused"],
        citations=result["citations"],
        conflicts=result["conflicts"],
        injected_sources=result.get("injected_sources", []),
        latency_ms=tracer.total_ms(),
        token_usage=result["token_usage"],
    )
