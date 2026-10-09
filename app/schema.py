"""
Data shapes used across the app, defined with Pydantic.

WHY: having one explicit schema means every layer (ingestion, retrieval, API,
tests) agrees on what a "chunk" or a "result" looks like. The PDF asks for an
"explicit payload schema" and "chunk metadata tagging" — this file is that.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Everything we know about a chunk *besides* its text.

    This is exactly what gets stored as the Qdrant point "payload", so it is
    our explicit payload schema.
    """

    doc_id: str = Field(..., description="Which source document this came from, e.g. 'SEC'")
    clause_id: str = Field(..., description="Fine-grained clause label, e.g. 'SEC-1.1'")
    title: str = Field(..., description="Human-readable section title")
    source: str = Field(..., description="Original file / manual name")
    effective_date: Optional[str] = Field(
        None, description="ISO date the clause took effect (used for temporal conflicts)"
    )
    embedding_version: str = Field(
        ..., description="Which embedding model produced this vector (for safe re-indexing)"
    )


class Chunk(BaseModel):
    """A single retrievable passage: an id, the text, and its metadata."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    """A chunk returned by a retriever, together with its score and provenance.

    The PDF requires "strict source provenance (Doc ID, snippet, similarity
    score)" — that is `doc_id`, `snippet`, and `score` below.
    """

    chunk_id: str
    doc_id: str
    clause_id: str
    snippet: str                 # the text, possibly shortened for display
    score: float                 # higher = more relevant
    metadata: ChunkMetadata
    rank: Optional[int] = None    # 1-based position in the result list


class QueryRequest(BaseModel):
    """Body of a POST /query request."""

    query: str
    strategy: str = Field(
        "rerank",
        description="Which retrieval tier to use: 'vector', 'hybrid', or 'rerank'",
    )
    top_k: Optional[int] = None


class Citation(BaseModel):
    """A source the answer is based on — shown to the user for trust."""

    doc_id: str
    clause_id: str
    snippet: str
    score: float


class QueryResponse(BaseModel):
    """What POST /query returns."""

    query: str
    strategy: str
    answer: str
    refused: bool                 # True if we declined (no supporting evidence)
    citations: list[Citation]
    conflicts: list[str] = []     # human-readable notes when sources disagree
    injected_sources: list[str] = []  # chunk_ids flagged as prompt-injection (neutralized)
    latency_ms: float
    token_usage: dict = {}        # {"input": N, "output": M}
