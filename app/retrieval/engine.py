"""
The unified 3-tier retrieval engine — the heart of Option B.

It builds all the indexes once, then exposes a single `retrieve(query, strategy)`
method. The `strategy` picks which of the three tiers to run:

  strategy="vector"  -> Tier 1: vector-only semantic search.
  strategy="hybrid"  -> Tier 2: BM25 + vector, fused with custom RRF.
  strategy="rerank"  -> Tier 3: Tier 2 candidates re-ordered by a cross-encoder.

Each tier returns a list of RetrievedChunk objects carrying full provenance
(doc id, clause id, snippet, score, rank) so the answer layer can cite sources.
"""

from __future__ import annotations

from typing import List

from app.config import settings
from app.ingest import load_corpus
from app.retrieval.bm25 import BM25Index
from app.retrieval.rerank import rerank
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.vector import VectorStore
from app.schema import Chunk, RetrievedChunk

VALID_STRATEGIES = {"vector", "hybrid", "rerank"}


class RetrievalEngine:
    def __init__(self, chunks: List[Chunk] | None = None):
        # Load corpus (or accept an injected one — handy for tests).
        self.chunks: List[Chunk] = chunks if chunks is not None else load_corpus()

        # --- Embedding versioning handling ---
        # We are about to (re-)embed every chunk with the CURRENTLY configured
        # model. So we stamp each chunk's metadata with that model's version.
        # This guarantees the stored `embedding_version` always matches the
        # vectors that actually exist — you can never mix vectors produced by
        # two different models. If the configured model changes, the next build
        # re-embeds and re-stamps automatically.
        for c in self.chunks:
            c.metadata.embedding_version = settings.embed_version

        # Fast lookup from chunk_id back to the full Chunk.
        self._by_id = {c.chunk_id: c for c in self.chunks}

        # Build the two base indexes.
        self.vector = VectorStore()
        self.vector.index(self.chunks)
        self.bm25 = BM25Index()
        self.bm25.index(self.chunks)

    # ---------------------------------------------------------------- #
    # Helper: turn (chunk_id, score) pairs into rich RetrievedChunk objects.
    # ---------------------------------------------------------------- #
    def _materialise(self, scored: List[tuple[str, float]], top_k: int) -> List[RetrievedChunk]:
        results: List[RetrievedChunk] = []
        for rank, (chunk_id, score) in enumerate(scored[:top_k], start=1):
            chunk = self._by_id.get(chunk_id)
            if chunk is None:
                continue  # defensive: skip ids we somehow don't recognise
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.metadata.doc_id,
                    clause_id=chunk.metadata.clause_id,
                    snippet=chunk.text,
                    score=score,
                    metadata=chunk.metadata,
                    rank=rank,
                )
            )
        return results

    # ---------------------------------------------------------------- #
    # The public entry point.
    # ---------------------------------------------------------------- #
    def retrieve(
        self,
        query: str,
        strategy: str = "rerank",
        top_k: int | None = None,
    ) -> List[RetrievedChunk]:
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Use one of {sorted(VALID_STRATEGIES)}."
            )
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        top_k = top_k or settings.top_k
        cand_k = settings.candidate_k

        # --- Tier 1: vector-only ---
        if strategy == "vector":
            scored = self.vector.search(query, cand_k)
            return self._materialise(scored, top_k)

        # --- Tier 2 & 3 both start from BM25 + vector fused by RRF ---
        vector_hits = self.vector.search(query, cand_k)
        bm25_hits = self.bm25.search(query, cand_k)
        fused = reciprocal_rank_fusion([vector_hits, bm25_hits], k=settings.rrf_k)

        if strategy == "hybrid":
            return self._materialise(fused, top_k)

        # --- Tier 3: rerank the fused candidates with the cross-encoder ---
        candidate_ids = [cid for cid, _ in fused[:cand_k]]
        candidates = [(cid, self._by_id[cid].text) for cid in candidate_ids if cid in self._by_id]
        reranked = rerank(query, candidates)
        return self._materialise(reranked, top_k)
