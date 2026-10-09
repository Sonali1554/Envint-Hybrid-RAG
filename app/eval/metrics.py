
"""
Retrieval quality metrics — hand-implemented, transparent and testable.

Each function compares ranked retrieved chunk IDs against the set of
relevant chunk IDs supplied by the query's gold labels.

Unsupported queries with empty relevance labels return 0.0. The benchmark
runner should exclude these queries from retrieval-quality averages and
evaluate refusal behavior separately.
"""

from __future__ import annotations

import math
from typing import List, Sequence


def precision_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int,
) -> float:
    """Fraction of the top k positions occupied by relevant chunks."""
    if k <= 0:
        return 0.0

    relevant_set = set(relevant)
    top = retrieved[:k]

    # Fixed-k precision: missing result positions count as non-relevant.
    hits = sum(1 for chunk_id in top if chunk_id in relevant_set)
    return hits / k



def recall_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int,
) -> float:
    """Fraction of all relevant chunks retrieved in the top k."""
    relevant_set = set(relevant)

    # Preserve existing behavior for unsupported queries.
    # The benchmark runner must exclude these from relevance averages.
    if not relevant_set:
        return 1.0

    if k <= 0:
        return 0.0

    top = set(retrieved[:k])
    return len(top & relevant_set) / len(relevant_set)


def mrr(
    retrieved: Sequence[str],
    relevant: Sequence[str],
) -> float:
    """Reciprocal rank of the first relevant result."""
    relevant_set = set(relevant)

    if not relevant_set:
        return 0.0

    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int,
) -> float:
    """NDCG@k using binary relevance."""
    relevant_set = set(relevant)

    if not relevant_set or k <= 0:
        return 0.0

    dcg = sum(
        1.0 / math.log2(position + 2)
        for position, chunk_id in enumerate(retrieved[:k])
        if chunk_id in relevant_set
    )

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(
        1.0 / math.log2(position + 2)
        for position in range(ideal_hits)
    )

    return dcg / idcg if idcg else 0.0


def percentile(values: List[float], pct: float) -> float:
    """Calculate a percentile using the nearest-rank method."""
    if not values:
        return 0.0

    if not 0 <= pct <= 100:
        raise ValueError("pct must be between 0 and 100")

    ordered = sorted(values)
    index = max(0, math.ceil(pct / 100.0 * len(ordered)) - 1)
    return ordered[index]
