"""
Lightweight, dependency-free observability (the PDF explicitly allows "custom
middleware logs").

Two pieces:
  * Tracer     -> times individual steps of a single request (embed, search,
                  rerank, llm, ...) and totals them. Attach token counts too.
  * Metrics    -> process-wide counters: total requests and errors, so we can
                  report an error rate. A FastAPI middleware updates these and
                  logs one structured line per request.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List

logger = logging.getLogger("envint.obs")
logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class Tracer:
    """Collects per-step timings (in milliseconds) for one request."""

    steps: List[Dict[str, float]] = field(default_factory=list)
    tokens: Dict[str, int] = field(default_factory=dict)

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Time a block of work: `with tracer.span("vector_search"): ...`."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.steps.append({"step": name, "latency_ms": round(elapsed_ms, 2)})

    def record_tokens(self, tokens: Dict[str, int]) -> None:
        self.tokens = tokens

    def total_ms(self) -> float:
        return round(sum(s["latency_ms"] for s in self.steps), 2)

    def summary(self) -> Dict[str, object]:
        """A JSON-friendly view of everything we traced."""
        return {
            "steps": self.steps,
            "total_latency_ms": self.total_ms(),
            "tokens": self.tokens,
        }


class Metrics:
    """Process-wide request/error counters (for an error-rate figure)."""

    def __init__(self) -> None:
        self.requests = 0
        self.errors = 0

    def error_rate(self) -> float:
        return (self.errors / self.requests) if self.requests else 0.0


# One shared metrics object for the whole process.
METRICS = Metrics()


async def observability_middleware(request, call_next):
    """FastAPI middleware: time each request, count errors, log one line."""
    start = time.perf_counter()
    METRICS.requests += 1
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception:
        METRICS.errors += 1
        raise
    finally:
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        if status >= 500:
            METRICS.errors += 1
        logger.info(
            json.dumps(
                {
                    "event": "request",
                    "method": request.method,
                    "path": request.url.path,
                    "status": status,
                    "latency_ms": latency_ms,
                    "error_rate": round(METRICS.error_rate(), 4),
                }
            )
        )
