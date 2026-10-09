"""
Cost-per-query / token model.

We use NO paid API, so our real spend is $0 for API calls — the only cost is the
compute time to run the pipeline. This module turns measured latency into a
dollar figure using an assumed machine price (config.compute_usd_per_hour), and
also shows what the SAME token volume would cost on a typical paid API, so the
trade-off is explicit in the README.
"""

from __future__ import annotations

from app.config import settings

# Illustrative published rates for a small hosted model, USD per 1,000 tokens.
# These are ONLY used to show the paid-equivalent comparison; we never call them.
PAID_INPUT_PER_1K = 0.00015
PAID_OUTPUT_PER_1K = 0.00060


def compute_cost_usd(latency_ms: float) -> float:
    """Dollar cost of the compute time for one query, given its latency."""
    hours = latency_ms / 1000.0 / 3600.0
    return hours * settings.compute_usd_per_hour


def paid_equivalent_usd(input_tokens: int, output_tokens: int) -> float:
    """What this query WOULD cost on a paid per-token API (for comparison only)."""
    return (
        input_tokens / 1000.0 * PAID_INPUT_PER_1K
        + output_tokens / 1000.0 * PAID_OUTPUT_PER_1K
    )


def cost_breakdown(latency_ms: float, input_tokens: int, output_tokens: int) -> dict:
    """A single structured cost record for one query."""
    return {
        "latency_ms": round(latency_ms, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "compute_cost_usd": round(compute_cost_usd(latency_ms), 8),
        "api_cost_usd": 0.0,  # we use a free local model
        "paid_equivalent_usd": round(
            paid_equivalent_usd(input_tokens, output_tokens), 8
        ),
    }
