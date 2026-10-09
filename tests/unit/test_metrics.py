"""Unit tests for the hand-implemented retrieval metrics."""

import math

from app.eval import metrics


def test_precision_at_k_perfect():
    assert metrics.precision_at_k(["a", "b"], ["a", "b"], 2) == 1.0


def test_precision_at_k_half():
    assert metrics.precision_at_k(["a", "x"], ["a"], 2) == 0.5


def test_recall_at_k_partial():
    # 1 of 2 relevant items found in the top 2.
    assert metrics.recall_at_k(["a", "x"], ["a", "b"], 2) == 0.5


def test_recall_at_k_no_relevant_is_one():
    # Unsupported query (no gold) -> nothing missed -> 1.0.
    assert metrics.recall_at_k(["a"], [], 2) == 1.0


def test_mrr_first_position():
    assert metrics.mrr(["a", "b"], ["a"]) == 1.0


def test_mrr_second_position():
    assert metrics.mrr(["x", "a"], ["a"]) == 0.5


def test_mrr_not_found():
    assert metrics.mrr(["x", "y"], ["a"]) == 0.0


def test_ndcg_perfect_is_one():
    assert math.isclose(metrics.ndcg_at_k(["a", "b"], ["a", "b"], 5), 1.0)


def test_ndcg_lower_when_relevant_is_ranked_lower():
    top = metrics.ndcg_at_k(["a", "x"], ["a"], 5)
    low = metrics.ndcg_at_k(["x", "a"], ["a"], 5)
    assert top > low


def test_percentile_basic():
    values = [10, 20, 30, 40, 50]
    assert metrics.percentile(values, 50) == 30
    assert metrics.percentile(values, 95) == 50


def test_percentile_empty():
    assert metrics.percentile([], 95) == 0.0
