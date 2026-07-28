"""The hybrid retriever: lexical + dense fusion, optional reranking, MMR diversity.

Composes the two existing arms behind one :class:`~transunit.reference.Retriever` -- the
stdlib-only :class:`~transunit.reference.LexicalRetriever` (exact-term/"sparse" signal: strong
on names, IDs, and recurring terminology) and :class:`~translator.retrieval.embedding.
EmbeddingRetriever` (dense/semantic: strong on paraphrase and cross-script matches) -- rather
than forking either. Per query:

1. Each arm retrieves its own top ``candidate_pool``, gated by its own calibrated recall floor
   (the two floors are on different score scales, so each arm keeps its own).
2. The two rankings are fused with :func:`~translator.retrieval.fusion.rrf` (Reciprocal Rank
   Fusion) -- a candidate both arms agree on outranks one only one arm found, with no need to
   calibrate the arms' otherwise incomparable scores against each other.
3. If a reranker is configured, the fused candidate pool is re-scored by a cross-encoder
   (:class:`~translator.retrieval.rerank.RerankClient`) -- the accurate but expensive stage,
   affordable only because it only ever sees a short list. Its raw score is squashed through a
   sigmoid to ``[0, 1]``. Without a reranker, the fused RRF scores are min-max normalised to
   ``[0, 1]`` instead.
4. Candidates below the caller's ``min_score`` (on that final relevance) are dropped.
5. The survivors are diversified with :func:`~translator.retrieval.fusion.mmr_order` (Maximal
   Marginal Relevance), so a top-k list of near-duplicate lines from the same passage does not
   crowd out a more varied set of examples -- using the dense arm's own embeddings for the
   redundancy term (:meth:`~translator.retrieval.embedding.EmbeddingRetriever.
   pairwise_similarity`), so no extra embedding round-trip is needed to compare two candidates.

**Fails loud.** If the embedding endpoint is down, construction raises
:class:`~translator.retrieval.embedding.EmbeddingError`. If the reranker endpoint dies mid-run,
:class:`~translator.retrieval.rerank.RerankError` propagates straight out of ``by_source``/
``by_target`` -- there is no silent fallback to the fused-but-unreranked order, because a
misbehaving reranker demands attention, not a quietly degraded answer.

**Read-only.** Like :class:`~translator.retrieval.embedding.EmbeddingRetriever`, this does not
implement :class:`~transunit.reference.WritableRetriever`; self-population stays the lexical
retriever's job.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import httpx

from transunit.reference import LexicalRetriever, ReferenceEntry, Retrieved

from .embedding import EmbeddingRetriever
from .fusion import best_possible_rrf, mmr_order, rrf
from .rerank import RerankClient

_ARM_COUNT = 2
"""The rankings fused per query: one lexical, one dense. Named because the relevance scale
without a reranker is defined against it (see :meth:`HybridRetriever._relevance`)."""


def _sigmoid(x: float) -> float:
    """Squash a cross-encoder's unbounded logit into ``[0, 1]``, for any finite ``x``.

    Branching on the sign is not micro-optimisation, it is the only correct form: the naive
    ``1 / (1 + exp(-x))`` calls ``exp`` on a large *positive* argument for very negative ``x``
    and dies with ``OverflowError`` past about -710, so one outlier logit from a reranker would
    kill a whole run with an unstructured error. Each branch below only ever exponentiates a
    non-positive number, which underflows harmlessly to 0.0 instead of overflowing -- so the
    full float range, out to ±1e308, maps into ``[0, 1]``.

    ``NaN`` raises :class:`ValueError`: there is no relevance that a "not a number" score could
    honestly stand for, and passing it on would silently poison the ranking downstream (a NaN
    relevance never compares greater than anything, so it neither wins nor loses a sort).
    """
    if math.isnan(x):
        raise ValueError(f"cannot squash a NaN score to a relevance in [0, 1] (_sigmoid, {x!r})")
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exponential = math.exp(x)
    return exponential / (1.0 + exponential)


class HybridRetriever:
    """Lexical + dense fusion, optional rerank, MMR diversity -- one :class:`Retriever`."""

    def __init__(self, entries: Iterable[ReferenceEntry], *, embedding_base_url: str,
                 embedding_model: str = "local", rerank_base_url: str = "",
                 rerank_model: str = "local", index_source: bool = True,
                 index_target: bool = False, candidate_pool: int = 40, mmr_lambda: float = 0.7,
                 lexical_min_score: float = 0.30, embedding_min_score: float = 0.55,
                 ngram: int = 3, batch_size: int = 64, timeout_seconds: float = 120.0,
                 embedding_client: httpx.Client | None = None,
                 rerank_client: httpx.Client | None = None) -> None:
        if candidate_pool < 1:
            raise ValueError(f"candidate_pool must be >= 1, got {candidate_pool}")
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError(f"mmr_lambda must be in [0, 1], got {mmr_lambda}")
        self._entries = tuple(entries)
        self._candidate_pool = candidate_pool
        self._mmr_lambda = mmr_lambda
        self._lexical_min_score = lexical_min_score
        self._embedding_min_score = embedding_min_score
        # Built in an order chosen for RAII: the lexical arm and the rerank client do no network
        # I/O at construction (a rerank client only talks to its endpoint when rerank() is
        # called), so neither can fail here. Only the dense arm's corpus embed can, so it is the
        # one step wrapped for cleanup -- if it fails, an already-built rerank client would
        # otherwise be leaked (nothing else would ever close it, since this object is discarded).
        # Its close() only releases a client it opened itself, so an injected one is left alone.
        self._lexical = LexicalRetriever(self._entries, index_source=index_source,
                                         index_target=index_target, ngram=ngram)
        self._rerank = (RerankClient(base_url=rerank_base_url, model=rerank_model,
                                     timeout_seconds=timeout_seconds, client=rerank_client)
                        if rerank_base_url else None)
        try:
            self._dense = EmbeddingRetriever(
                self._entries, base_url=embedding_base_url, model=embedding_model,
                index_source=index_source, index_target=index_target, batch_size=batch_size,
                timeout_seconds=timeout_seconds, client=embedding_client)
        except BaseException:
            if self._rerank is not None:
                self._rerank.close()
            raise

    def by_source(self, query: str, *, k: int,
                  min_score: float = 0.0) -> tuple[Retrieved, ...]:
        return self._search(query, k=k, min_score=min_score, target=False)

    def by_target(self, query: str, *, k: int,
                  min_score: float = 0.0) -> tuple[Retrieved, ...]:
        return self._search(query, k=k, min_score=min_score, target=True)

    def _search(self, query: str, *, k: int, min_score: float,
                target: bool) -> tuple[Retrieved, ...]:
        if k <= 0:
            return ()
        lexical_fn = self._lexical.by_target if target else self._lexical.by_source
        dense_fn = self._dense.by_target if target else self._dense.by_source
        lex_hits = lexical_fn(query, k=self._candidate_pool, min_score=self._lexical_min_score)
        dense_hits = dense_fn(query, k=self._candidate_pool, min_score=self._embedding_min_score)
        fused = rrf([[h.entry for h in lex_hits], [h.entry for h in dense_hits]])
        if not fused:
            return ()
        # Rank on the fused score, but break its ties on each arm's own confidence rather than
        # on argument order. Ties are not rare: whenever the arms disagree outright -- each
        # arm's top hit absent from the other's list -- both candidates score exactly
        # 1/(k+1), and a plain sort would then settle a retrieval-quality question by which
        # ranking happened to be passed first, silently handing every disagreement to the same
        # arm regardless of which one is actually better on this corpus.
        confidence = self._confidence(lex_hits, dense_hits)
        candidates = sorted(fused,
                            key=lambda entry: (-fused[entry], -confidence[entry]))
        candidates = candidates[:self._candidate_pool]
        relevance = self._relevance(query, candidates, fused, target=target)
        survivors = [entry for entry in candidates if relevance[entry] >= min_score]
        if not survivors:
            return ()

        def similarity(a: ReferenceEntry, b: ReferenceEntry) -> float:
            return self._dense.pairwise_similarity(a, b, target=target)

        ordered = mmr_order(survivors, relevance, similarity, lambda_=self._mmr_lambda, k=k)
        return tuple(Retrieved(entry=entry, score=relevance[entry]) for entry in ordered)

    def _confidence(self, lex_hits: tuple[Retrieved, ...],
                    dense_hits: tuple[Retrieved, ...]) -> dict[ReferenceEntry, float]:
        """How far above its own floor each arm placed a candidate, best arm winning.

        The arms' raw scores are not comparable (a lexical n-gram cosine and an embedding
        cosine sit on different scales, which is exactly why fusion is rank-based). Their
        *headroom* above their own calibrated floor is: ``0.0`` means "only just cleared the
        bar this arm is trusted at", ``1.0`` means "a perfect match". That makes it a fair
        secondary signal for separating candidates the fused rank cannot -- and it is derived
        entirely from scores already computed, needing no new knob to tune.
        """
        confidence: dict[ReferenceEntry, float] = {}
        for hits, floor in ((lex_hits, self._lexical_min_score),
                            (dense_hits, self._embedding_min_score)):
            headroom = max(1e-9, 1.0 - floor)  # a floor of 1.0 would otherwise divide by zero
            for hit in hits:
                scaled = min(1.0, max(0.0, (hit.score - floor) / headroom))
                if scaled > confidence.get(hit.entry, -1.0):
                    confidence[hit.entry] = scaled
        return confidence

    def _relevance(self, query: str, candidates: list[ReferenceEntry],
                   fused: dict[ReferenceEntry, float], *,
                   target: bool) -> dict[ReferenceEntry, float]:
        """Final ``[0, 1]`` relevance per candidate, on a scale that does not shift per query.

        With a reranker: the sigmoid of the cross-encoder's raw score (it emits an unbounded
        logit, and is the only stage here that judges relevance in absolute terms at all).
        Without one: the fused score over :func:`~translator.retrieval.fusion.best_possible_rrf`
        -- a fixed denominator, so a floor keeps the same meaning for every query. It is
        deliberately *not* min-max over the query's own candidates: that would score the best of
        three terrible candidates 1.0 and the worst of three excellent ones 0.0, turning an
        absolute-looking floor into a relative one that always keeps the top hit and always
        drops the bottom one however good or bad the pool actually is.
        """
        if self._rerank is not None:
            text_of = (lambda entry: entry.target) if target else (lambda entry: entry.source)
            scored = self._rerank.rerank(query, [text_of(entry) for entry in candidates])
            return {candidates[index]: _sigmoid(score) for index, score in scored}
        ceiling = best_possible_rrf(_ARM_COUNT)
        return {entry: min(1.0, fused[entry] / ceiling) for entry in candidates}

    def close(self) -> None:
        """Close the arms this retriever built. Each releases only the client it owns.

        So an ``embedding_client``/``rerank_client`` injected by the caller stays open -- it is
        the injector's to close -- while a client this retriever opened for itself is released
        here. The same rule in :class:`~translator.retrieval.embedding.EmbeddingRetriever` and
        :class:`~translator.retrieval.rerank.RerankClient`.
        """
        self._dense.close()
        if self._rerank is not None:
            self._rerank.close()

    @property
    def source_entries(self) -> int:
        return self._lexical.source_entries

    def __len__(self) -> int:
        return len(self._lexical)
