"""
Custom Reciprocal Rank Fusion (RRF).

PROBLEM: vector search and BM25 each return their own ranked list, with scores
on completely different scales (cosine similarity vs BM25 score). We cannot just
add the scores. RRF solves this by ignoring the raw scores and using only the
RANK (position) of each item in each list.

FORMULA: for a chunk that appears at rank r (1-based) in a list, it earns
    1 / (k + r)
points from that list. We sum those points across all lists. A chunk ranked
highly by BOTH retrievers accumulates the most points and rises to the top.

`k` (default 60, a well-known value from the original RRF paper) softens the
difference between the very top ranks so one list cannot completely dominate.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Fuse several ranked lists into one.

    Args:
        ranked_lists: each inner list is [(chunk_id, score), ...] already sorted
                      best-first. The scores are ignored — only order matters.
        k: the RRF constant.

    Returns:
        A single [(chunk_id, fused_score), ...] list sorted best-first.
    """
    fused: Dict[str, float] = {}

    for ranked in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(ranked, start=1):
            # Add this list's contribution for the chunk.
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)

    # Highest fused score first.
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
