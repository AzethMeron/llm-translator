"""Tests for the pure fusion math: Reciprocal Rank Fusion and Maximal Marginal Relevance.

No I/O, no server, no numpy -- these are plain algorithms over hashable candidates, so every
case is pinned exactly with hand-picked inputs.
"""
from __future__ import annotations

import pytest

from translator.retrieval.fusion import RRF_K, best_possible_rrf, mmr_order, rrf

CANDIDATES = ["a", "b", "c"]


class TestRRF:
    def test_a_single_ranking_scores_by_position(self) -> None:
        scores = rrf([["a", "b", "c"]])
        assert scores["a"] > scores["b"] > scores["c"]

    def test_overlapping_candidates_across_rankings_accumulate(self) -> None:
        # 'a' and 'b' each appear once at rank 1 and once at rank 2, in opposite rankings, so
        # their fused scores are equal -- and each beats appearing in only one ranking.
        both = rrf([["a", "b"], ["b", "a"]])
        assert both["a"] == pytest.approx(both["b"])
        only_once = rrf([["a"]])
        assert both["a"] > only_once["a"]

    def test_empty_rankings_produce_an_empty_result(self) -> None:
        assert rrf([]) == {}
        assert rrf([[]]) == {}
        assert rrf([[], []]) == {}

    def test_default_k_is_60(self) -> None:
        scores = rrf([["a"]])
        assert scores["a"] == pytest.approx(1.0 / 61.0)

    def test_a_smaller_k_gives_a_larger_score(self) -> None:
        # Smaller k discounts position less, so the same rank-1 hit scores higher.
        assert rrf([["a"]], k=1)["a"] > rrf([["a"]], k=60)["a"]

    def test_deterministic_for_the_same_input(self) -> None:
        rankings = [["x", "y", "z"], ["y", "z", "x"]]
        assert rrf(rankings) == rrf(rankings)

    def test_k_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="k"):
            rrf([["a"]], k=0)


class TestBestPossibleRRF:
    """The fixed denominator that gives a fused score a query-independent meaning."""

    def test_it_is_the_score_of_a_candidate_every_ranking_puts_first(self) -> None:
        both_first = rrf([["x"], ["x"]])["x"]
        assert best_possible_rrf(2) == pytest.approx(both_first)

    def test_it_actually_bounds_a_real_fusion(self) -> None:
        # No arrangement of two rankings can exceed it -- the property the [0, 1] scale needs.
        ceiling = best_possible_rrf(2)
        for rankings in ([["x"], ["x"]], [["x", "y"], ["x", "y"]], [["x"], []]):
            assert max(rrf(rankings).values()) <= ceiling + 1e-12

    def test_one_of_two_rankings_finding_it_first_scores_half(self) -> None:
        assert rrf([["x"], []])["x"] / best_possible_rrf(2) == pytest.approx(0.5)

    def test_it_scales_with_the_ranking_count_and_honours_a_custom_k(self) -> None:
        assert best_possible_rrf(3) == pytest.approx(3 * best_possible_rrf(1))
        assert best_possible_rrf(1, k=1) == pytest.approx(0.5)
        assert best_possible_rrf(1) == pytest.approx(1.0 / (RRF_K + 1))

    def test_bad_arguments_are_refused(self) -> None:
        with pytest.raises(ValueError, match="ranking_count"):
            best_possible_rrf(0)
        with pytest.raises(ValueError, match="k"):
            best_possible_rrf(2, k=0)


class TestMMROrder:
    def test_lambda_one_reproduces_relevance_order(self) -> None:
        relevance = {"a": 0.5, "b": 0.9, "c": 0.1}
        order = mmr_order(CANDIDATES, relevance, similarity=lambda x, y: 1.0,
                          lambda_=1.0, k=3)
        assert order == ["b", "a", "c"]

    def test_lambda_zero_maximises_diversity_and_ignores_relevance(self) -> None:
        # 'a' and 'b' are near-duplicates (sim ~1); 'c' is unrelated to both (sim 0). At
        # lambda_=0 every candidate scores 0 before anything is selected (redundancy has
        # nothing to measure against yet), so the FIRST pick is decided by input order alone,
        # not by relevance -- 'a' has the highest relevance here but that must not matter.
        relevance = {"a": 0.9, "b": 0.1, "c": 0.5}

        def similarity(x: str, y: str) -> float:
            return 0.95 if {x, y} == {"a", "b"} else 0.0

        order = mmr_order(CANDIDATES, relevance, similarity=similarity, lambda_=0.0, k=3)
        assert order[0] == "a"   # first-seen wins the all-zero tie, not the most "relevant"
        assert order[1] == "c"   # 'c' (sim 0 to 'a') beats 'b' (sim 0.95 to 'a', redundant)
        assert order[2] == "b"

    def test_redundancy_suppresses_a_near_duplicate(self) -> None:
        # A middling-relevance but dissimilar candidate can outrank a highly-relevant one that
        # duplicates something already picked.
        relevance = {"a": 0.9, "b": 0.85, "c": 0.5}

        def similarity(x: str, y: str) -> float:
            return 0.99 if {x, y} == {"a", "b"} else 0.0

        order = mmr_order(CANDIDATES, relevance, similarity=similarity, lambda_=0.5, k=3)
        assert order == ["a", "c", "b"]  # 'b' demoted below 'c' for duplicating 'a'

    def test_k_zero_returns_nothing(self) -> None:
        assert mmr_order(CANDIDATES, {"a": 1.0, "b": 1.0, "c": 1.0},
                         similarity=lambda x, y: 0.0, lambda_=1.0, k=0) == []

    def test_k_greater_than_the_candidate_count_returns_all(self) -> None:
        relevance = {"a": 0.5, "b": 0.9, "c": 0.1}
        order = mmr_order(CANDIDATES, relevance, similarity=lambda x, y: 0.0,
                          lambda_=1.0, k=99)
        assert sorted(order) == sorted(CANDIDATES)
        assert len(order) == 3

    def test_ties_are_broken_by_input_order(self) -> None:
        relevance = {"a": 0.5, "b": 0.5, "c": 0.5}
        order = mmr_order(["c", "a", "b"], relevance, similarity=lambda x, y: 0.0,
                          lambda_=1.0, k=3)
        assert order == ["c", "a", "b"]

    def test_lambda_out_of_range_is_refused(self) -> None:
        with pytest.raises(ValueError, match="lambda_"):
            mmr_order(CANDIDATES, {}, similarity=lambda x, y: 0.0, lambda_=1.5, k=1)
        with pytest.raises(ValueError, match="lambda_"):
            mmr_order(CANDIDATES, {}, similarity=lambda x, y: 0.0, lambda_=-0.1, k=1)


class TestNonFiniteInputsFailLoud:
    """A NaN must never be allowed to *reorder* a result list quietly.

    Every comparison against NaN is false, so a NaN score neither wins nor loses a ``>``
    test: the old ``mmr_order`` left ``best_index`` at its ``-1`` sentinel and then
    ``remaining.pop(-1)`` selected the LAST candidate -- a plausible-looking but wrong order,
    returned with no error at all. These pin that every such input now raises instead.
    """

    def test_a_nan_relevance_raises_instead_of_reordering(self) -> None:
        relevance = {"a": 0.9, "b": float("nan"), "c": 0.1}
        with pytest.raises(ValueError, match=r"relevance must be finite.*'b'"):
            mmr_order(CANDIDATES, relevance, similarity=lambda x, y: 0.0, lambda_=1.0, k=3)

    def test_a_lone_nan_relevance_raises_rather_than_selecting_the_last_candidate(self) -> None:
        # The starkest form of the old bug: with only NaN scores nothing ever beat -inf, so
        # pop(-1) walked the list backwards and returned a full, entirely meaningless ranking.
        relevance = dict.fromkeys(CANDIDATES, float("nan"))
        with pytest.raises(ValueError, match="relevance must be finite"):
            mmr_order(CANDIDATES, relevance, similarity=lambda x, y: 0.0, lambda_=1.0, k=3)

    @pytest.mark.parametrize("value", [float("inf"), float("-inf")])
    def test_an_infinite_relevance_raises(self, value: float) -> None:
        # +inf pins one candidate at the top forever and -inf can never be selected on merit;
        # neither is a relevance, so both are refused rather than silently dominating MMR.
        relevance = {"a": 0.9, "b": value, "c": 0.1}
        with pytest.raises(ValueError, match="relevance must be finite"):
            mmr_order(CANDIDATES, relevance, similarity=lambda x, y: 0.0, lambda_=1.0, k=3)

    def test_a_nan_similarity_raises_and_names_both_candidates(self) -> None:
        # The same hazard through the redundancy term: max() over a NaN is order-dependent
        # garbage, which poisons the score of a candidate whose relevance was perfectly fine.
        relevance = {"a": 0.9, "b": 0.8, "c": 0.1}

        def similarity(x: str, y: str) -> float:
            return float("nan") if {x, y} == {"a", "b"} else 0.0

        with pytest.raises(ValueError, match="similarity must be finite"):
            mmr_order(CANDIDATES, relevance, similarity=similarity, lambda_=0.5, k=3)

    def test_a_candidate_with_no_relevance_is_named(self) -> None:
        # Previously a bare KeyError from deep inside the loop, with no hint of which caller
        # or which candidate; a reranker returning duplicate indices produced exactly this.
        with pytest.raises(ValueError, match="no relevance for candidate 'c'"):
            mmr_order(CANDIDATES, {"a": 0.9, "b": 0.5},
                      similarity=lambda x, y: 0.0, lambda_=1.0, k=3)

    def test_a_finite_negative_similarity_still_lowers_redundancy(self) -> None:
        # Guards the fix itself: validating similarity must not clamp it. Anti-parallel
        # candidates have a NEGATIVE cosine, which is a genuine diversity bonus, and the
        # redundancy term must keep it rather than flooring it at 0.
        # Tuned so the two readings disagree: after 'a', unclamped scores are b = 0.75 (0.25
        # relevance + 0.5 anti-redundancy bonus) vs c = 0.30, but clamping b's -1.0 to 0 gives
        # b = 0.25 vs c = 0.30 and flips the winner.
        relevance = {"a": 0.9, "b": 0.5, "c": 0.6}

        def similarity(x: str, y: str) -> float:
            return -1.0 if {x, y} == {"a", "b"} else 0.0

        order = mmr_order(CANDIDATES, relevance, similarity=similarity, lambda_=0.5, k=3)
        assert order == ["a", "b", "c"]  # 'b' wins on its negative (rewarded) similarity to 'a'

    def test_rrf_refuses_a_nan_k(self) -> None:
        # A NaN k slips past a plain `k < 1` guard and makes every fused score NaN, which then
        # sorts into an arbitrary order without raising anywhere.
        with pytest.raises(ValueError, match="k must be finite"):
            rrf([["a", "b"]], k=float("nan"))  # type: ignore[arg-type]

    def test_rrf_refuses_an_infinite_k(self) -> None:
        with pytest.raises(ValueError, match="k must be finite"):
            rrf([["a", "b"]], k=float("inf"))  # type: ignore[arg-type]

    def test_best_possible_rrf_refuses_non_finite_arguments(self) -> None:
        # It is the denominator of the hybrid's no-reranker relevance scale: a NaN here makes
        # every relevance NaN, and every min_score comparison silently false.
        with pytest.raises(ValueError, match="ranking_count must be finite"):
            best_possible_rrf(float("nan"), k=RRF_K)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="k must be finite"):
            best_possible_rrf(2, k=float("nan"))  # type: ignore[arg-type]
