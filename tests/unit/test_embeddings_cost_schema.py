"""Unit tests for the fake embedder, cost model, extractive LLM, and schema."""

import numpy as np
import pytest

from app.embeddings import FakeEmbedder
from app.eval import cost
from app.llm.client import ExtractiveBackend
from app.schema import Chunk, ChunkMetadata


# ----------------------------- embeddings ----------------------------- #
def test_fake_embedder_is_deterministic():
    emb = FakeEmbedder(dim=16)
    a = emb.encode(["hello world"])
    b = emb.encode(["hello world"])
    assert np.allclose(a, b)


def test_fake_embedder_shape_and_unit_norm():
    emb = FakeEmbedder(dim=16)
    vecs = emb.encode(["policy clause", "another one"])
    assert vecs.shape == (2, 16)
    # Each non-empty vector should be unit length.
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


# ------------------------------- cost --------------------------------- #
def test_compute_cost_scales_with_latency():
    assert cost.compute_cost_usd(2000) > cost.compute_cost_usd(1000)


def test_paid_equivalent_nonnegative():
    assert cost.paid_equivalent_usd(100, 50) > 0


def test_cost_breakdown_keys():
    row = cost.cost_breakdown(1000, 100, 50)
    assert row["api_cost_usd"] == 0.0
    assert "compute_cost_usd" in row and "paid_equivalent_usd" in row


# --------------------------- extractive LLM --------------------------- #
def test_extractive_backend_picks_relevant_sentence():
    from app.schema import RetrievedChunk

    rc = RetrievedChunk(
        chunk_id="a", doc_id="D", clause_id="a",
        snippet="Passwords rotate every 90 days. The sky is blue.",
        score=1.0,
        metadata=ChunkMetadata(doc_id="D", clause_id="a", title="t", source="s",
                               effective_date="2024-01-01", embedding_version="v1"),
    )
    text, tokens = ExtractiveBackend().generate("how often rotate passwords", [rc])
    assert "90 days" in text
    assert tokens["output"] > 0


# ------------------------------ schema -------------------------------- #
def test_chunk_schema_rejects_missing_field():
    with pytest.raises(Exception):
        Chunk(chunk_id="a", text="t", metadata=ChunkMetadata(doc_id="D"))  # type: ignore
