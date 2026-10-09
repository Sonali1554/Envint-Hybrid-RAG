"""
Central configuration for the whole app.

WHY THIS FILE EXISTS
--------------------
Every tunable thing (which model to use, how many results to fetch, where Qdrant
lives) is defined ONCE here. Other files import `settings` instead of hard-coding
values. You can override any field with an environment variable of the same name
(that is what `pydantic-settings` gives us), which is exactly how docker-compose
and CI pass in different values without touching the code.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Read env vars; ignore ones we don't define; names are case-insensitive.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # ------------------------------------------------------------------ #
    # Embeddings (turning text into vectors for semantic search)
    # ------------------------------------------------------------------ #
    # "real"  -> downloads a small open-source model from Hugging Face.
    # "fake"  -> a deterministic, offline hash-based vector. No download,
    #            no internet. We use "fake" in tests/CI so they stay free & fast.
    embed_backend: str = "real"
    embed_model: str = "BAAI/bge-small-en-v1.5"   # free, CPU-friendly, 384-dim
    embed_dim: int = 384                          # vector size (must match the model)
    embed_version: str = "bge-small-en-v1.5-v1"   # stamped onto every chunk (see §embedding versioning)

    # ------------------------------------------------------------------ #
    # Cross-encoder reranker (tier 3 — re-scores the top candidates)
    # ------------------------------------------------------------------ #
    # "real" -> open-source cross-encoder from Hugging Face.
    # "fake" -> lightweight lexical-overlap scorer (offline, for tests/CI).
    rerank_backend: str = "real"
    # bge-reranker-large is the model named in the PDF but it is heavy on CPU.
    # bge-reranker-base is the same family, much faster — good default for a laptop.
    rerank_model: str = "BAAI/bge-reranker-base"

    # ------------------------------------------------------------------ #
    # Vector database (Qdrant)
    # ------------------------------------------------------------------ #
    # If qdrant_url is empty we run Qdrant *in memory* inside this process
    # (perfect for dev/tests — nothing to install). In docker-compose we set
    # QDRANT_URL=http://qdrant:6333 to use the real container service.
    qdrant_url: str = ""
    qdrant_collection: str = "policies"

    # ------------------------------------------------------------------ #
    # Retrieval sizes
    # ------------------------------------------------------------------ #
    top_k: int = 5            # how many results the user ultimately gets
    candidate_k: int = 20     # how many we fetch per retriever before fusing/reranking
    rrf_k: int = 60           # the RRF constant (explained in retrieval/rrf.py)

    # ------------------------------------------------------------------ #
    # Downstream LLM (answer generation)
    # ------------------------------------------------------------------ #
    # "auto"       -> use Ollama if it is reachable, otherwise fall back to extractive.
    # "ollama"     -> force the free local LLM (must be running).
    # "extractive" -> no LLM at all; build the answer from retrieved sentences.
    #                 Deterministic, offline — this is what CI uses.
    llm_backend: str = "auto"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # ------------------------------------------------------------------ #
    # Cost model (we have no paid API, so "cost" = compute time).
    # ------------------------------------------------------------------ #
    # Assumed price of the machine we'd run on, used only to turn latency
    # into a dollar figure in the cost report. Stated as an assumption.
    compute_usd_per_hour: float = 0.10   # ~ a small commodity CPU VM

    # Paths to the sample data (relative to repo root).
    corpus_path: str = "data/corpus/corpus.json"
    dev_queries_path: str = "data/queries/dev_queries.json"
    eval_queries_path: str = "data/queries/eval_queries.json"


# One shared instance imported everywhere: `from app.config import settings`.
settings = Settings()
