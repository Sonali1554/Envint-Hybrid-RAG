"""
Tier 3 building block: CROSS-ENCODER RERANKER.

The first two tiers score the query and each chunk *separately* (fast, but the
model never sees them together). A cross-encoder instead reads the query and a
chunk AS A PAIR and outputs a single relevance score. This is slower (it runs
the model once per candidate) but much more precise, so we only use it to
re-order the handful of candidates that fusion already surfaced.

Two backends (chosen in config.py), same as embeddings:
  * RealReranker -> open-source BAAI/bge-reranker-* cross-encoder (free).
  * FakeReranker -> a lexical word-overlap score (offline, for tests/CI).
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

from app.config import settings
from app.retrieval.bm25 import tokenize


class FakeReranker:
    """Offline stand-in: score a pair by how many query words the chunk contains.

    Deterministic and dependency-free, so tests can assert on it without a model.
    """

    def score(self, query: str, texts: List[str]) -> List[float]:
        q_words = set(tokenize(query))
        scores = []
        for text in texts:
            t_words = set(tokenize(text))
            overlap = len(q_words & t_words)
            scores.append(overlap / (len(q_words) + 1e-9))
        return scores


class RealReranker:
    """Wraps a real sentence-transformers CrossEncoder (imported lazily)."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: List[str]) -> List[float]:
        pairs = [(query, text) for text in texts]
        return [float(s) for s in self.model.predict(pairs)]


@lru_cache(maxsize=1)
def get_reranker():
    if settings.rerank_backend == "fake":
        return FakeReranker()
    return RealReranker(settings.rerank_model)


def rerank(
    query: str,
    candidates: List[Tuple[str, str]],
) -> List[Tuple[str, float]]:
    """Re-order candidates by cross-encoder relevance.

    Args:
        query: the user question.
        candidates: list of (chunk_id, text) to score.

    Returns:
        [(chunk_id, rerank_score), ...] sorted best-first.
    """
    if not candidates:
        return []

    reranker = get_reranker()
    chunk_ids = [cid for cid, _ in candidates]
    texts = [text for _, text in candidates]
    scores = reranker.score(query, texts)

    ranked = sorted(zip(chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
    return [(cid, float(score)) for cid, score in ranked]
