# Audit — hybrid retrieval, reranking, and continuous-blob context packing (2026-07-26)

Covers the feature built from `docs/feature-requests/hybrid-retrieval-reranking-rag.md`. Recorded
here because the interesting output of this work was not the feature — it was four defects the
audit and the live evaluation turned up, three of which were **silent**.

## Defects found and fixed

**1. The relevance scale was per-query, silently redefining a documented knob.**
Without a reranker, the hybrid's final score was min-max normalised over the query's own
candidate pool. So the best of three *terrible* candidates scored exactly `1.0` and the worst of
three *excellent* ones exactly `0.0` — meaning `reference_min_score`, documented and implemented
everywhere else as an absolute relevance floor, silently meant "relative position in this pool".
Any positive floor always dropped the bottom candidate and always kept the top one, regardless of
quality. Fixed by dividing by `best_possible_rrf()` (a fixed denominator), which also turned the
floor into a useful dial: above `0.5` it keeps only candidates *both* arms agree on.

**2. RRF ties were broken by argument order.**
Whenever the two arms disagree outright — each arm's top hit absent from the other's list — both
candidates score exactly `1/(k+1)`. A plain sort then settled a retrieval-quality question by
which ranking happened to be passed first, systematically handing *every* disagreement to the same
arm. Fixed by breaking ties on each arm's headroom above its own calibrated floor. Honest note:
this corrected the bias but did **not** move the aggregate (58% → 57%, noise) — defect 4 below is
the real cause.

**3. The floor warning gave contradictory, actively harmful advice.**
Three modes put three different scales into `reference_min_score` (embedding cosine / fused rank
agreement / reranker sigmoid), but one warning covered all of them. A correctly-calibrated
reranked run at `0.30` was told to raise its floor to `0.55` — which, measured, would discard good
matches (a correct cross-script match reranks to a sigmoid of ~0.44). Fixed by making the warning
mode-aware and mutually exclusive.

**4. The feature's headline claim did not hold, and now says so.**
`tools/eval_retrieval.sh` (built as part of this work) measured all four variants on two
independent real Japanese corpora. Bare RRF hybrid never beat its own best arm; on one corpus it
fell below *both*. See `docs/reference-translations.md` for the numbers, both mechanisms, and the
resulting recommendation (embedding alone, or hybrid **with** a reranker). Nothing was tuned to
make the result look better; the gate was instead narrowed to what the evidence supports, with the
unfavourable comparison reported rather than asserted away.

## Verified live, against real servers

Every failure mode of the new code was exercised against actual llama.cpp servers, not only mocks:
flag misuse (`--hybrid` without `--embedding-url`, `--rerank-url` without `--hybrid`), a dead
embedding endpoint at construction, a dead rerank endpoint, and — the important one — **SIGKILLing
the reranker mid-run**. That last case stopped the run with exit 1, named the endpoint, wrote
"run aborted" to the `--log-file`, and left all 11 already-translated units intact in the journal.
No silent fallback to unreranked order.

## Documentation drift

A dedicated sweep found 14 stale or incomplete statements, mostly a single class: several
docstrings and docs still described the *old* context rendering ("neighbours shown in the target
language rather than the source"), which the continuous-passage format changed — a neighbour's
source now always appears in the passage, with its translation listed separately. All fixed. The
lesson worth keeping: a rendering change invalidates prose in modules that do not render anything
(`memory.py`, `roles.py`, the carrier-adapter guide), and those are the ones a feature-focused
sweep misses.

## Known limitation, deliberately not addressed

Weighted fusion. There is now evidence motivating it (the arms differ sharply in quality on real
corpora), but choosing a default weight needs its own multi-corpus evaluation, and an untuned knob
would be worse than none.
