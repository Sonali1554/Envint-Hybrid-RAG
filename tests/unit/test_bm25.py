"""Unit tests for BM25 keyword search and its tokenizer."""

import pytest

from app.retrieval.bm25 import BM25Index, tokenize
from app.schema import Chunk, ChunkMetadata


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=text,
        metadata=ChunkMetadata(
            doc_id="D", clause_id=cid, title="t", source="s",
            effective_date="2024-01-01", embedding_version="v1",
        ),
    )


def test_tokenize_lowercases_and_splits():
    assert tokenize("Password-90 Days!") == ["password", "90", "days"]


def test_bm25_finds_exact_keyword():
    idx = BM25Index()
    idx.index([
        _chunk("a", "passwords rotate every 90 days"),
        _chunk("b", "flights booked 14 days in advance"),
    ])
    top = idx.search("password rotation", 1)
    assert top[0][0] == "a"


def test_bm25_raises_before_indexing():
    idx = BM25Index()
    with pytest.raises(RuntimeError):
        idx.search("anything", 3)


def test_bm25_respects_k():
    idx = BM25Index()
    idx.index([_chunk("a", "alpha"), _chunk("b", "beta"), _chunk("c", "gamma")])
    assert len(idx.search("alpha beta gamma", 2)) == 2
