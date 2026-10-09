"""
Tier 2 building block: BM25 keyword search.

BM25 is a classic "bag of words" ranking function. It rewards documents that
contain the exact query words, weighting rare words more than common ones. It is
the opposite strength of vector search: it is literal. That is why it shines on
things vector search can miss — exact clause IDs ("SEC-1.1"), specific numbers
("90 days"), and precise policy wording.

We combine BM25 with vector search later (that is the "Hybrid" tier).
"""

from __future__ import annotations

import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from app.schema import Chunk


def tokenize(text: str) -> List[str]:
    """Split text into lowercase word/number tokens.

    We keep alphanumerics together so tokens like 'sec' , '1', '90', 'aes256'
    survive. Simple and predictable — good enough for a compliance corpus.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """In-memory BM25 index over the chunk texts."""

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: List[str] = []
        self._indexed = False  # distinguishes "never indexed" from "indexed empty"

    def index(self, chunks: List[Chunk]) -> None:
        self._indexed = True
        self._chunk_ids = [c.chunk_id for c in chunks]
        # BM25Okapi cannot be built from an empty corpus, so we only build it
        # when there is something to index. An empty index yields no results.
        if not chunks:
            self._bm25 = None
            return
        tokenized_corpus = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Return the top-k chunks by BM25 score as (chunk_id, score)."""
        # Searching before any index() call is a programming error.
        if not self._indexed:
            raise RuntimeError("BM25 index is empty — call index() first.")
        # Indexed with an empty corpus -> legitimately empty result set.
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        # Pair each chunk with its score, sort high-to-low, take k.
        ranked = sorted(
            zip(self._chunk_ids, scores), key=lambda pair: pair[1], reverse=True
        )
        return [(cid, float(score)) for cid, score in ranked[:k]]
