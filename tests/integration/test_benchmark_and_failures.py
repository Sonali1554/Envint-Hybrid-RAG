"""Integration tests: benchmark runner, malformed inputs, and dependency failures."""

import json

import pytest

from app.eval.benchmark import run_benchmark
from app.ingest import load_corpus
from app.llm import client
from app.llm.client import ExtractiveBackend, OllamaBackend, _select_backend


# ------------------------------ benchmark ------------------------------ #
def test_benchmark_produces_full_matrix():
    report = run_benchmark("dev")
    matrix = report["matrix"]
    assert set(matrix.keys()) == {"vector", "hybrid", "rerank"}
    for row in matrix.values():
        for key in ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_5",
                    "p50_latency_ms", "p95_latency_ms", "peak_rss_mb"]:
            assert key in row


# --------------------------- malformed inputs --------------------------- #
def test_load_corpus_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_corpus("data/corpus/does_not_exist.json")


def test_load_corpus_malformed_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json ", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_corpus(str(bad))


def test_load_corpus_missing_required_field(tmp_path):
    # A chunk record missing clause_id must be rejected by the schema.
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"chunk_id": "x", "text": "t",
                                "metadata": {"doc_id": "D"}}]), encoding="utf-8")
    with pytest.raises(Exception):
        load_corpus(str(bad))


# ------------------------- dependency failures -------------------------- #
def test_auto_backend_falls_back_when_ollama_down(monkeypatch):
    # Simulate a dependency outage: Ollama is unreachable.
    monkeypatch.setattr(client.settings, "llm_backend", "auto")
    monkeypatch.setattr(OllamaBackend, "available", lambda self: False)
    assert isinstance(_select_backend(), ExtractiveBackend)


def test_api_reports_500_on_engine_failure(monkeypatch):
    # Simulate the retrieval dependency throwing; API must surface an error,
    # not hang or return a wrong 200.
    from fastapi.testclient import TestClient

    import app.main as main

    def boom(*args, **kwargs):
        raise RuntimeError("vector store timeout")

    # Enter the client first so the startup hook builds the engine, THEN patch
    # that live engine's retrieve() to simulate a vector-store outage.
    with TestClient(app=main.app, raise_server_exceptions=False) as c:
        monkeypatch.setattr(main._engine, "retrieve", boom)
        r = c.post("/query", json={"query": "passwords", "strategy": "vector"})
    assert r.status_code == 500
