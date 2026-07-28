"""Property-based tests for the retrieval-fusion math.

Fusion decides *which* reference translations the model is shown, and every failure mode here
is silent: RRF returns a plain dict of floats and MMR a plain list, so a mis-ordering produces
a worse translation rather than an error anyone can see. That is precisely the shape of defect
property testing catches -- an ordering is only ever wrong relative to a rule, and the rules
(order-independence, boundedness, monotonicity, permutation-ness, determinism) are stateable
without restating the implementation.

The module was recently hardened so that a non-finite relevance, similarity, or ``k`` raises
:class:`ValueError` instead of being ranked wherever the comparison chain happened to leave it
(every comparison against NaN is false, so a NaN neither wins nor loses a maximum). Those
guards are load-bearing and are pinned here too.

Each test states the invariant it defends, not the implementation it happens to match.
"""
from __future__ import annotations

import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from translator.retrieval.fusion import RRF_K, best_possible_rrf, mmr_order, rrf

# Candidates are opaque hashables to the module under test, so single letters are enough and
# keep counterexamples readable.
CANDIDATE = st.sampled_from("ABCDEFGH")

# A *ranking* is best-first and, by contract, mentions each candidate at most once -- a
# candidate listed twice would be counted twice (see TestRrfPreconditions).
RANKING = st.lists(CANDIDATE, unique=True, max_size=8)
RANKINGS = st.lists(RANKING, max_size=4)
VALID_K = st.integers(min_value=1, max_value=10_000)

FINITE = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
NON_FINITE = st.sampled_from([math.nan, math.inf, -math.inf])


def _relevance_for(candidates: list[str], values: list[float]) -> dict[str, float]:
    """A total relevance map over ``candidates``, cycling ``values`` when it is short."""
    return {c: values[i % len(values)] for i, c in enumerate(dict.fromkeys(candidates))}


class TestRrf:
    """Reciprocal Rank Fusion: the only thing combining the lexical and dense arms of the
    hybrid retriever. Its output feeds a *fixed* denominator (``best_possible_rrf``) that turns
    a fused score into an absolute [0, 1] relevance, so both the scores and their ceiling have
    to hold together or the ``min_score`` floor stops meaning anything."""

    @given(RANKINGS, VALID_K)
    def test_is_independent_of_the_order_the_rankings_are_passed_in(
            self, rankings: list[list[str]], k: int) -> None:
        """Fusing [A, B] must equal fusing [B, A], to within floating-point summation error.

        RRF's entire justification is that it needs no calibration between the arms because it
        treats them symmetrically. If argument order leaked into the scores, "which retriever
        was passed first" would silently decide retrieval quality -- and the caller
        (hybrid.HybridRetriever) passes lexical first by nothing more than habit.

        The exact-equality form of this invariant is defended (and currently fails) in
        ``TestRrfPreconditions.test_ranking_order_does_not_perturb_scores_in_the_last_bit``.
        """
        fused, reversed_fused = rrf(rankings, k=k), rrf(list(reversed(rankings)), k=k)
        assert fused.keys() == reversed_fused.keys()
        for candidate, score in fused.items():
            assert math.isclose(score, reversed_fused[candidate], rel_tol=1e-12)

    @given(RANKINGS, VALID_K)
    def test_every_score_is_positive_and_within_the_published_ceiling(
            self, rankings: list[list[str]], k: int) -> None:
        """Scores live in (0, len(rankings) / (k + 1)].

        Positivity: a fused score of 0 is indistinguishable from "never retrieved", and the
        caller reads absence from the dict, not a zero in it. The upper bound is what
        ``best_possible_rrf`` promises; if a real score could exceed it, the normalised
        relevance would exceed 1.0 and a caller's floor would admit everything.

        The ceiling holds to within a rounding step, not exactly: ``rrf`` *accumulates* the
        per-ranking terms while the ceiling is one closed-form division, so identical rankings
        can land one ULP above it (``[['A'], ['A'], ['A']]`` at ``k=4`` sums to
        0.6000000000000001 against a ceiling of 0.6). The same last-bit accumulation order
        recorded in ``TestRrfPreconditions``; harmless here because the only consumer,
        ``HybridRetriever``, divides by the ceiling under a ``min(1.0, ...)`` clamp.
        """
        fused = rrf(rankings, k=k)
        assume(fused)
        ceiling = len(rankings) / (k + 1)
        for score in fused.values():
            assert score > 0.0
            assert score <= ceiling or math.isclose(score, ceiling, rel_tol=1e-12)

    @given(RANKINGS, VALID_K, VALID_K)
    def test_no_score_exceeds_best_possible_rrf_for_any_valid_k(
            self, rankings: list[list[str]], k: int, other_k: int) -> None:
        """The ceiling function must agree with the fusion for the ``k`` it is asked about.

        ``other_k`` is drawn independently to pin that the two are only consistent when the
        same ``k`` is used -- a caller normalising with a mismatched ``k`` is a bug, but a
        caller normalising with the *matching* ``k`` must never see a ratio above 1.
        """
        fused = rrf(rankings, k=k)
        assume(fused)
        ceiling = best_possible_rrf(len(rankings), k=k)
        # Within a rounding step, for the accumulation reason documented on the test above.
        assert max(fused.values()) <= ceiling or math.isclose(max(fused.values()), ceiling,
                                                              rel_tol=1e-12)
        # Sanity on the other side: the ceiling is monotone in k the way the scores are.
        assert (best_possible_rrf(1, k=other_k) > best_possible_rrf(1, k=other_k + 1))

    @given(RANKINGS, st.integers(0, 7), st.integers(0, 7), VALID_K)
    def test_promoting_a_candidate_never_lowers_its_fused_score(
            self, rankings: list[list[str]], which: int, position: int, k: int) -> None:
        """Moving a candidate *earlier* in any one ranking can only help it.

        This is the property that makes RRF a fusion of *rankings* at all: if a retriever
        improving its opinion of a candidate could demote it, the fused order would not be
        interpretable as consensus, and tuning either arm would have unpredictable effects.
        """
        assume(rankings)
        which %= len(rankings)
        ranking = rankings[which]
        assume(len(ranking) >= 2)
        position %= len(ranking)
        assume(position > 0)
        candidate = ranking[position]
        promoted = list(rankings)
        moved = list(ranking)
        moved.insert(0, moved.pop(position))
        promoted[which] = moved
        assert rrf(promoted, k=k)[candidate] >= rrf(rankings, k=k)[candidate]

    @given(VALID_K)
    def test_no_rankings_and_only_empty_rankings_fuse_to_nothing(self, k: int) -> None:
        """An empty fusion must be an empty dict, not a dict of zeros: the caller reads
        "retrieved at all" from membership."""
        assert rrf([], k=k) == {}
        assert rrf([[], [], []], k=k) == {}

    @given(RANKING, VALID_K)
    def test_a_single_ranking_reduces_to_its_reciprocal_ranks(
            self, ranking: list[str], k: int) -> None:
        """With one arm, fusion must be a pure order-preserving re-scaling of that arm.

        Fusing one ranking has to leave its order intact, or enabling a second arm and then
        disabling it again would not return to the original behaviour.
        """
        fused = rrf([ranking], k=k)
        assert fused == {c: 1.0 / (k + i) for i, c in enumerate(ranking, start=1)}
        assert sorted(fused, key=lambda c: -fused[c]) == ranking

    @given(st.lists(RANKING, min_size=5, max_size=12), VALID_K)
    def test_agreement_across_many_rankings_beats_a_single_top_placement(
            self, rankings: list[list[str]], k: int) -> None:
        """A candidate every ranking mentions outscores one only the first ranking placed
        first, once enough rankings agree -- the whole point of rank fusion."""
        agreed, loner = "Y", "Z"
        rankings = [[loner, *r, agreed] if i == 0 else [*r, agreed]
                    for i, r in enumerate(rankings)]
        fused = rrf(rankings, k=k)
        assert fused[agreed] > fused[loner]

    @given(RANKINGS, VALID_K)
    def test_is_deterministic(self, rankings: list[list[str]], k: int) -> None:
        """One retriever instance is shared by every worker in a run; a fusion that varied
        between identical calls would make a run irreproducible."""
        assert rrf(rankings, k=k) == rrf(rankings, k=k)

    @given(RANKINGS)
    def test_uses_the_documented_default_k(self, rankings: list[list[str]]) -> None:
        assert rrf(rankings) == rrf(rankings, k=RRF_K)


class TestRrfPreconditions:
    """The unvalidated edge of the contract, pinned so a change to it is deliberate."""

    @given(VALID_K)
    def test_a_candidate_repeated_within_one_ranking_is_counted_twice(self, k: int) -> None:
        """Documented, NOT endorsed: ``rrf`` does not check that a ranking is duplicate-free,
        so a repeat breaks the ``best_possible_rrf`` ceiling.

        Reachable in principle -- a corpus holding two equal ``ReferenceEntry`` values (a
        frozen dataclass, so equal and hash-equal) makes one retriever arm return the same
        key twice. It is defused downstream only because ``HybridRetriever._relevance``
        clamps with ``min(1.0, ...)``; nothing in this module prevents it. This test exists so
        that if the precondition is ever enforced, the change shows up here.
        """
        assert rrf([["A", "A"]], k=k)["A"] > best_possible_rrf(1, k=k)

    @pytest.mark.xfail(strict=True,
                       reason="known defect, found by this property suite: fused scores are "
                              "accumulated in argument order, so with THREE OR MORE rankings "
                              "float addition is not associative and the same fusion passed in "
                              "a different order differs in the last bit. Minimal repro: "
                              "rrf([['A'],['B','A'],['B','A'],['B','A']], k=1) gives "
                              "A=1.4999999999999998 while the reversed argument list gives "
                              "A=1.5 -- so A ties with B in one order and loses to it in the "
                              "other. That is precisely the argument-order dependence the "
                              "module promises it does not have, and it silently defeats "
                              "HybridRetriever's deliberate tie-break on arm confidence, which "
                              "only fires when two fused scores are EXACTLY equal. Latent "
                              "today because hybrid.py fuses exactly two arms (_ARM_COUNT = 2) "
                              "and two-term float addition is commutative; it becomes live the "
                              "moment a third arm is added. Fix: accumulate in a "
                              "candidate-keyed, order-independent way (e.g. math.fsum over "
                              "each candidate's contributions). Fix pending (do not fix in "
                              "src).")
    def test_ranking_order_does_not_perturb_scores_in_the_last_bit(self) -> None:
        rankings = [["A"], ["B", "A"], ["B", "A"], ["B", "A"]]
        assert rrf(rankings, k=1) == rrf(list(reversed(rankings)), k=1)

    @pytest.mark.parametrize("k", [0, -1, -1000])
    def test_a_k_below_one_is_rejected(self, k: int) -> None:
        """k = 0 would make a rank-1 hit score 1.0 and a negative k can divide by zero; both
        are refused rather than producing a plausible-looking number."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            rrf([["A"]], k=k)
        with pytest.raises(ValueError, match="k must be >= 1"):
            best_possible_rrf(1, k=k)

    @given(NON_FINITE)
    def test_a_non_finite_k_raises_rather_than_scoring_everything_nan(self, k: float) -> None:
        """A NaN ``k`` slips past a plain ``k < 1`` test and makes every fused score NaN,
        which then sorts into an arbitrary order without a word. It must raise."""
        with pytest.raises(ValueError, match="k must be finite"):
            rrf([["A", "B"]], k=k)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="k must be finite"):
            best_possible_rrf(1, k=k)  # type: ignore[arg-type]

    @given(NON_FINITE)
    def test_a_non_finite_ranking_count_raises(self, ranking_count: float) -> None:
        with pytest.raises(ValueError, match="ranking_count must be finite"):
            best_possible_rrf(ranking_count)  # type: ignore[arg-type]

    @pytest.mark.parametrize("ranking_count", [0, -1])
    def test_a_ranking_count_below_one_is_rejected(self, ranking_count: int) -> None:
        """A ceiling of 0 would make the normalised relevance a division by zero."""
        with pytest.raises(ValueError, match="ranking_count must be >= 1"):
            best_possible_rrf(ranking_count)

    @pytest.mark.xfail(strict=True,
                       reason="known defect, found by this property suite: _require_finite "
                              "calls math.isfinite(k), which raises OverflowError -- not the "
                              "ValueError the docstring promises -- for an int too large to "
                              "convert to float (k = 10**400). The guard exists to turn every "
                              "bad k into one structured, catchable error; an OverflowError "
                              "escaping from it defeats that, and it leaks from "
                              "best_possible_rrf too. Fix pending (do not fix in src).")
    def test_an_enormous_integer_k_raises_valueerror_not_overflowerror(self) -> None:
        with pytest.raises(ValueError):
            rrf([["A"]], k=10**400)


class TestMmrOrder:
    """Greedy Maximal Marginal Relevance: the last stage before the prompt, deciding which of
    the surviving candidates the model actually sees. It re-orders and truncates, so the two
    things it must never do are invent a candidate and lose one it should have kept."""

    @given(st.lists(CANDIDATE, unique=True, max_size=8),
           st.lists(FINITE, min_size=1, max_size=8),
           st.floats(0.0, 1.0), st.integers(0, 10))
    def test_output_is_a_subset_permutation_of_the_input(
            self, candidates: list[str], values: list[float], lambda_: float, k: int) -> None:
        """Every element out came in, none is duplicated, and nothing is invented.

        MMR's job is selection; a candidate appearing twice would show the model the same
        example twice, and one appearing that was never a candidate would mean the greedy
        loop's bookkeeping had lost track of ``remaining``.
        """
        relevance = _relevance_for(candidates, values)
        out = mmr_order(candidates, relevance, lambda a, b: 0.0, lambda_=lambda_, k=k)
        assert len(set(out)) == len(out)
        assert set(out) <= set(candidates)

    @given(st.lists(CANDIDATE, unique=True, max_size=8),
           st.lists(FINITE, min_size=1, max_size=8),
           st.floats(0.0, 1.0), st.integers(0, 10))
    def test_returns_exactly_min_k_and_len_candidates(
            self, candidates: list[str], values: list[float], lambda_: float, k: int) -> None:
        """Under-filling silently starves the prompt of context the caller asked for; the
        caller has no way to tell "there were only two" from "the loop stopped early"."""
        relevance = _relevance_for(candidates, values)
        out = mmr_order(candidates, relevance, lambda a, b: 0.5, lambda_=lambda_, k=k)
        assert len(out) == min(k, len(candidates))

    @given(st.lists(CANDIDATE, unique=True, max_size=8),
           st.lists(FINITE, min_size=1, max_size=8), st.integers(0, 10))
    def test_lambda_one_is_exactly_relevance_descending_with_input_order_ties(
            self, candidates: list[str], values: list[float], k: int) -> None:
        """``lambda_ = 1`` must degrade to plain relevance ranking, ties broken by input order.

        This is the documented escape hatch -- how a caller turns diversification off -- so it
        has to be *exactly* the un-diversified order, including the tie-break, or "disable
        MMR" would still quietly reshuffle the shortlist.
        """
        relevance = _relevance_for(candidates, values)
        expected = sorted(range(len(candidates)),
                          key=lambda i: (-relevance[candidates[i]], i))[:k]
        out = mmr_order(candidates, relevance, lambda a, b: 0.7, lambda_=1.0, k=k)
        assert out == [candidates[i] for i in expected]

    @given(st.lists(CANDIDATE, unique=True, max_size=8),
           st.lists(FINITE, min_size=1, max_size=8), st.floats(0.0, 1.0))
    def test_k_zero_or_negative_selects_nothing_and_k_beyond_the_input_selects_everything(
            self, candidates: list[str], values: list[float], lambda_: float) -> None:
        relevance = _relevance_for(candidates, values)
        for k in (0, -1, -99):
            assert mmr_order(candidates, relevance, lambda a, b: 0.1,
                             lambda_=lambda_, k=k) == []
        big = mmr_order(candidates, relevance, lambda a, b: 0.1,
                        lambda_=lambda_, k=len(candidates) + 5)
        assert sorted(big) == sorted(candidates)

    @given(st.lists(CANDIDATE, unique=True, max_size=8),
           st.lists(FINITE, min_size=1, max_size=8),
           st.floats(0.0, 1.0), st.integers(0, 10))
    def test_is_deterministic_across_repeated_calls(
            self, candidates: list[str], values: list[float], lambda_: float, k: int) -> None:
        """Deterministic inputs must give byte-identical output: one retriever is shared by
        every worker, and a shortlist that differed per call would make a run irreproducible
        and its failures unbisectable."""
        relevance = _relevance_for(candidates, values)

        def similarity(a: str, b: str) -> float:
            return (hash(a) ^ hash(b)) % 97 / 97.0

        first = mmr_order(candidates, relevance, similarity, lambda_=lambda_, k=k)
        for _ in range(3):
            assert mmr_order(candidates, relevance, similarity, lambda_=lambda_, k=k) == first

    def test_a_near_duplicate_is_demoted_below_a_less_relevant_but_novel_candidate(self) -> None:
        """The behaviour MMR exists for, pinned on a hand-built case: with the diversity term
        dominant, the runner-up must not be the top hit's twin."""
        relevance = {"top": 1.0, "twin": 0.99, "novel": 0.5}

        def similarity(a: str, b: str) -> float:
            return 1.0 if {a, b} == {"top", "twin"} else 0.0

        assert mmr_order(["top", "twin", "novel"], relevance, similarity,
                         lambda_=0.5, k=2) == ["top", "novel"]

    def test_lambda_zero_ignores_relevance_entirely(self) -> None:
        """Documented: at ``lambda_ = 0`` every pre-selection score is 0, so the first pick is
        the first input, and thereafter selection is pure anti-redundancy."""
        relevance = {"a": 0.0, "b": 1.0, "c": 0.5}
        assert mmr_order(["a", "b", "c"], relevance, lambda x, y: 0.0,
                         lambda_=0.0, k=1) == ["a"]


class TestMmrOrderPreconditions:
    """Every guard that stands between a NaN and a silently plausible ordering."""

    @given(NON_FINITE)
    def test_a_non_finite_relevance_raises(self, bad: float) -> None:
        """A NaN relevance loses every ``>`` comparison, so the candidate is ranked wherever
        the loop leaves it -- a wrong-but-plausible order, the one failure mode that must
        never be silent."""
        with pytest.raises(ValueError, match="relevance must be finite"):
            mmr_order(["a", "b"], {"a": bad, "b": 1.0}, lambda x, y: 0.0,
                      lambda_=0.5, k=2)

    @given(NON_FINITE)
    def test_a_non_finite_similarity_raises(self, bad: float) -> None:
        """Same argument on the redundancy side; the similarity callable is supplied by the
        caller (a cosine over embeddings), so a degenerate vector can produce this for real."""
        with pytest.raises(ValueError, match="similarity must be finite"):
            mmr_order(["a", "b"], {"a": 1.0, "b": 2.0}, lambda x, y: bad,
                      lambda_=0.5, k=2)

    @given(NON_FINITE)
    def test_a_non_finite_lambda_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"lambda_ must be in \[0, 1\]"):
            mmr_order(["a"], {"a": 1.0}, lambda x, y: 0.0, lambda_=bad, k=1)

    @given(st.floats(allow_nan=False, allow_infinity=False).filter(
        lambda v: not 0.0 <= v <= 1.0))
    def test_a_lambda_outside_the_unit_interval_raises(self, bad: float) -> None:
        """Outside [0, 1] the two terms stop being a convex combination and the "score" is no
        longer a marginal relevance at all."""
        with pytest.raises(ValueError, match=r"lambda_ must be in \[0, 1\]"):
            mmr_order(["a"], {"a": 1.0}, lambda x, y: 0.0, lambda_=bad, k=1)

    def test_a_candidate_with_no_relevance_raises_and_names_it(self) -> None:
        """A missing key is a caller bug (the relevance map and the candidate list came from
        different stages); defaulting it to 0.0 would silently bury that candidate."""
        with pytest.raises(ValueError, match="no relevance for candidate 'ghost'"):
            mmr_order(["a", "ghost"], {"a": 1.0}, lambda x, y: 0.0, lambda_=0.5, k=2)

    def test_all_candidates_are_validated_before_any_selection_happens(self) -> None:
        """Validation must be a precondition, not something the loop trips over half way
        through -- otherwise a partially-built shortlist could be observed by a caller that
        catches the error."""
        with pytest.raises(ValueError, match="relevance must be finite"):
            mmr_order(["a", "b"], {"a": 1.0, "b": math.nan}, lambda x, y: 0.0,
                      lambda_=0.5, k=1)  # k=1: 'b' would never be scored by the loop

    @pytest.mark.xfail(strict=True,
                       reason="known defect, found by this property suite: mmr_order does not "
                              "check that k is finite, unlike rrf. k = NaN fails both 'k <= 0' "
                              "and 'len(selected) < k' (every NaN comparison is false), so the "
                              "loop body never runs and the caller silently receives an EMPTY "
                              "shortlist -- every reference example dropped from the prompt, no "
                              "error, a plausible-looking 'nothing was relevant'. This is the "
                              "same class of failure the module's other _require_finite guards "
                              "were added to close. Fix pending (do not fix in src).")
    def test_a_non_finite_k_raises_rather_than_silently_selecting_nothing(self) -> None:
        with pytest.raises(ValueError):
            mmr_order(["a", "b"], {"a": 1.0, "b": 2.0}, lambda x, y: 0.0,
                      lambda_=0.5, k=math.nan)  # type: ignore[arg-type]
