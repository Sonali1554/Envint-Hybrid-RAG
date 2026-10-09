"""Unit tests for custom Reciprocal Rank Fusion."""

from app.retrieval.rrf import reciprocal_rank_fusion


def test_rrf_rewards_items_ranked_high_in_both_lists():
    # 'a' is #1 in list1 and #2 in list2 -> should win overall.
    list1 = [("a", 0.9), ("b", 0.8), ("c", 0.1)]
    list2 = [("b", 5.0), ("a", 4.0), ("d", 1.0)]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert fused[0][0] == "a"


def test_rrf_includes_items_from_any_list():
    fused = reciprocal_rank_fusion([[("a", 1.0)], [("b", 1.0)]], k=60)
    ids = {cid for cid, _ in fused}
    assert ids == {"a", "b"}


def test_rrf_uses_rank_not_raw_score():
    # Even with a huge raw score, a #2 rank contributes less than a #1 rank.
    list1 = [("x", 0.01), ("y", 0.0)]        # x is rank 1
    list2 = [("y", 1000.0), ("x", 999.0)]    # y is rank 1
    fused = dict(reciprocal_rank_fusion([list1, list2], k=60))
    # Both are rank-1 once and rank-2 once -> equal fused score.
    assert abs(fused["x"] - fused["y"]) < 1e-9


def test_rrf_empty_input_returns_empty():
    assert reciprocal_rank_fusion([]) == []


def test_rrf_scores_sorted_descending():
    fused = reciprocal_rank_fusion([[("a", 1), ("b", 1), ("c", 1)]], k=60)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)
