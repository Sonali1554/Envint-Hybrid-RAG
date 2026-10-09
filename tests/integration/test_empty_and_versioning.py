"""Integration tests: empty results, and embedding-version handling."""

from app.config import settings
from app.llm import client
from app.retrieval.engine import RetrievalEngine
from app.schema import Chunk, ChunkMetadata


def _chunk(cid, text, version="stale-version-0"):
    return Chunk(
        chunk_id=cid,
        text=text,
        metadata=ChunkMetadata(
            doc_id="D", clause_id=cid, title="t", source="s",
            effective_date="2024-01-01", embedding_version=version,
        ),
    )


# ------------------------------ empty results ------------------------------ #
def test_empty_corpus_returns_no_results_for_all_strategies():
    engine = RetrievalEngine(chunks=[])
    for strategy in ("vector", "hybrid", "rerank"):
        assert engine.retrieve("anything at all", strategy=strategy) == []


def test_empty_results_lead_to_refusal():
    engine = RetrievalEngine(chunks=[])
    retrieved = engine.retrieve("anything at all", strategy="hybrid")
    result = client.answer("anything at all", retrieved)
    assert result["refused"] is True
    assert result["citations"] == []


# --------------------------- embedding versioning -------------------------- #
def test_engine_restamps_embedding_version_to_active_model():
    # The chunk arrives tagged with a stale version. After the engine (re-)embeds
    # it with the configured model, the stored version must match that model.
    chunk = _chunk("x", "passwords rotate every 90 days", version="stale-version-0")
    engine = RetrievalEngine(chunks=[chunk])
    assert engine.chunks[0].metadata.embedding_version == settings.embed_version
