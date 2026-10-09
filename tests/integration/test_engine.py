"""Integration tests: the full retrieval engine over the real sample corpus."""

import pytest


def test_vector_strategy_finds_password_clause(engine):
    results = engine.retrieve("How often must passwords be rotated?", strategy="vector")
    assert "SEC-1.1" in [r.chunk_id for r in results]


def test_hybrid_strategy_finds_password_clause(engine):
    results = engine.retrieve("How often must passwords be rotated?", strategy="hybrid")
    assert "SEC-1.1" in [r.chunk_id for r in results]


def test_rerank_strategy_ranks_gold_first(engine):
    # The cross-encoder (fake lexical here) should pull the exact clause to #1.
    results = engine.retrieve("meal per diem for business travel", strategy="rerank")
    assert results[0].chunk_id == "TE-1.1"


def test_results_carry_full_provenance(engine):
    results = engine.retrieve("vendor onboarding security review", strategy="hybrid")
    top = results[0]
    assert top.doc_id and top.clause_id and top.snippet and top.rank == 1


def test_empty_query_raises(engine):
    with pytest.raises(ValueError):
        engine.retrieve("   ", strategy="vector")


def test_invalid_strategy_raises(engine):
    with pytest.raises(ValueError):
        engine.retrieve("anything", strategy="magic")


def test_hard_negative_password_retention(engine):
    # D4 lure: "password retention" should still return the password clause,
    # not the (keyword-overlapping) data-retention clauses.
    results = engine.retrieve("What is the retention period for passwords?", strategy="rerank")
    assert "SEC-1.1" in [r.chunk_id for r in results[:3]]
