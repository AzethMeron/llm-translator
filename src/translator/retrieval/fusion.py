"""Pure retrieval-fusion math: Reciprocal Rank Fusion and Maximal Marginal Relevance.

Stdlib-only, no I/O, no numpy -- these are plain list/dict algorithms over whatever hashable
identity the caller uses for a candidate (an entry, an id, anything ``==``/``hash``-stable).
Kept separate from the retrievers that use them (:mod:`transunit.reference`,
:mod:`translator.retrieval.embedding`, :mod:`translator.retrieval.hybrid`) so each is testable
and provable in isolation, per the project's single-source-of-truth rule.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

H = TypeVar("H")


def _require_finite(value: float, *, what: str, where: str) -> float:
    """Reject NaN/±inf at a boundary, naming what and where it came from.

    Non-finite numbers do not merely give a wrong number here, they give a wrong *order*
    silently: every comparison against NaN is ``False``, so a NaN never wins a maximum and
    never loses one either -- it is simply ranked wherever the loop happens to leave it. A
    ranking that is quietly wrong is worse than one that stops, hence this fails loud.
    """
    if not math.isfinite(value):
        raise ValueError(f"{what} must be finite, got {value!r} ({where})")
    return value

RRF_K = 60
"""Cormack et al.'s rank-fusion constant, and the conventional default. Exposed because a caller
that wants to interpret a fused score on an absolute scale needs the same ``k`` the fusion used --
see :func:`best_possible_rrf`."""


def rrf(rankings: Sequence[Sequence[H]], *, k: int = RRF_K) -> dict[H, float]:
    """Reciprocal Rank Fusion over any number of best-first rankings.

    Each ranking contributes ``1 / (k + rank)`` to every candidate it contains (rank is
    1-based), summed across rankings -- so a candidate near the top of several rankings
    outscores one near the top of only one. ``k`` (Cormack et al.'s constant, conventionally
    60) discounts how much a single ranking's exact position matters, so the fusion rewards
    *agreement* across rankings rather than one ranking's raw rank. Deterministic and
    stateless: the same rankings always fuse to the same scores, and unlike weighted-score
    fusion this needs no calibration between the rankings' otherwise incomparable scales.

    Raises :class:`ValueError` for a non-finite or out-of-range ``k``. ``k`` is typed ``int``,
    but nothing enforces that at runtime, and a NaN ``k`` slips past a plain ``k < 1`` test
    (every comparison with NaN is false) to make *every* fused score NaN -- an all-NaN fusion
    then sorts into an arbitrary order without ever raising.
    """
    _require_finite(k, what="k", where="rrf")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    fused: dict[H, float] = {}
    for ranking in rankings:
        for rank, candidate in enumerate(ranking, start=1):
            fused[candidate] = fused.get(candidate, 0.0) + 1.0 / (k + rank)
    return fused


def best_possible_rrf(ranking_count: int, *, k: int = RRF_K) -> float:
    """The largest score :func:`rrf` can assign: every ranking placing a candidate first.

    A fused score means nothing on its own -- it is a sum of reciprocal ranks, not a similarity.
    Dividing by this gives a **query-independent** ``[0, 1]`` scale on which a floor keeps a
    stable meaning: 1.0 is "every retriever's top hit", 0.5 (with two rankings) is "one
    retriever's top hit, the other never found it", and a candidate deep in one ranking scores
    low. Deliberately not min-max over the query's own candidates, which would hand the best of
    three terrible candidates a perfect 1.0 and the worst of three excellent ones a 0.0.
    """
    _require_finite(ranking_count, what="ranking_count", where="best_possible_rrf")
    _require_finite(k, what="k", where="best_possible_rrf")
    if ranking_count < 1:
        raise ValueError(f"ranking_count must be >= 1, got {ranking_count}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return ranking_count / (k + 1)


def mmr_order(candidates: Sequence[H], relevance: Mapping[H, float],
              similarity: Callable[[H, H], float], *, lambda_: float, k: int) -> list[H]:
    """Greedy Maximal Marginal Relevance selection.

    Repeatedly picks the remaining candidate maximising::

        lambda_ * relevance[c] - (1 - lambda_) * max(similarity(c, s) for s in selected)

    so once a candidate has been picked, a later one closely resembling it is penalised --
    the standard way to keep a top-k list from being dominated by near-duplicates.
    ``lambda_ = 1`` reduces to plain relevance ranking (redundancy never has a first
    selection to be measured against). ``lambda_ = 0`` ignores relevance entirely -- every
    candidate's score is ``0`` before anything is selected, and thereafter it is pure
    anti-redundancy. Ties are broken by input order (the first-seen candidate wins), so the
    result is deterministic for deterministic ``relevance``/``similarity`` callables.

    Every candidate must have a finite relevance, and ``similarity`` must return a finite
    value, or :class:`ValueError` names the offender. A NaN anywhere in the arithmetic would
    otherwise lose every ``>`` comparison and be shuffled to an arbitrary rank without a word
    -- a wrong-but-plausible ordering, which is the one failure this must never produce.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")  # NaN/±inf fail this too
    if k <= 0:
        return []
    remaining = list(candidates)
    for candidate in remaining:
        if candidate not in relevance:
            raise ValueError(f"no relevance for candidate {candidate!r} (mmr_order)")
        _require_finite(relevance[candidate],
                        what="relevance", where=f"mmr_order candidate {candidate!r}")
    selected: list[H] = []
    while remaining and len(selected) < k:
        # max() over indices, rather than a running best with a sentinel index: it takes the
        # FIRST maximal element, giving the documented input-order tie-break, and it cannot
        # leave an "unset" index behind for a degenerate score to smuggle through.
        scores = [_mmr_score(candidate, selected, relevance, similarity, lambda_)
                  for candidate in remaining]
        selected.append(remaining.pop(max(range(len(remaining)), key=scores.__getitem__)))
    return selected


def _mmr_score(candidate: H, selected: Sequence[H], relevance: Mapping[H, float],
               similarity: Callable[[H, H], float], lambda_: float) -> float:
    """One candidate's marginal-relevance score against what is already selected."""
    # default=0.0 (not a 0.0 seed in the max) so a wholly *dissimilar* set, whose cosines are
    # all negative, keeps its negative -- i.e. a genuine anti-redundancy bonus -- while an
    # empty selection still contributes nothing.
    redundancy = max((_require_finite(similarity(candidate, other), what="similarity",
                                      where=f"mmr_order between {candidate!r} and {other!r}")
                      for other in selected), default=0.0)
    return lambda_ * relevance[candidate] - (1.0 - lambda_) * redundancy
