"""
Shared pytest setup.

The three environment variables below are set BEFORE any `app.*` module is
imported, which forces the whole app into its free, offline, deterministic mode:
  * fake embeddings  (no Hugging Face download, no internet)
  * fake reranker    (lexical overlap, no model)
  * extractive LLM   (no Ollama needed)
This is what lets the entire test suite run anywhere for $0.
"""

from __future__ import annotations

import os

os.environ.setdefault("EMBED_BACKEND", "fake")
os.environ.setdefault("RERANK_BACKEND", "fake")
os.environ.setdefault("LLM_BACKEND", "extractive")
os.environ.setdefault("QDRANT_URL", "")  # force in-memory Qdrant

import pytest  # noqa: E402

from app.ingest import load_corpus  # noqa: E402
from app.retrieval.engine import RetrievalEngine  # noqa: E402


@pytest.fixture(scope="session")
def corpus():
    """The loaded sample corpus (shared across tests)."""
    return load_corpus()


@pytest.fixture(scope="session")
def engine(corpus):
    """A fully-built retrieval engine, built once for the whole test session."""
    return RetrievalEngine(chunks=corpus)
