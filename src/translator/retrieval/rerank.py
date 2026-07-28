"""Optional cross-encoder reranker: a second-stage relevance scorer over a local ``/v1/rerank``
endpoint (llama.cpp built with ``--reranking``, or any Jina/TEI-compatible rerank server).

A retriever's first stage (lexical, dense, or their fusion) has to be cheap enough to run over
the whole candidate pool; a reranker is the opposite trade -- expensive but precise, so it only
ever scores a short list a first stage already narrowed down. It never replaces a first-stage
retriever on its own: :class:`~translator.retrieval.hybrid.HybridRetriever` is what puts one in
front of it.

Stdlib + httpx only; no numpy here (the reranker sends text, not vectors, so there is no vector
math to do client-side). Behind the same injectable-``httpx.Client`` seam as
:class:`~translator.retrieval.embedding.EmbeddingRetriever`, so it is testable with an in-memory
transport and no server.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import httpx

DEFAULT_RERANK_MIN_SCORE = 0.30
"""A starting floor for the reranker's relevance score (after :func:`math.sigmoid`-style
normalisation to ``[0, 1]`` by the caller -- the raw cross-encoder logit has no fixed scale).
Tune per corpus and reranker model, the same way the lexical and embedding floors are tuned."""


class RerankError(Exception):
    """The rerank endpoint is unreachable or returned something the client cannot use.

    Structured like the rest of the engine's retrieval errors (:class:`~transunit.reference.
    ReferenceError`, :class:`~translator.retrieval.embedding.EmbeddingError`) so a caller can
    catch it and fail clearly, naming the endpoint, rather than let a raw HTTP or parsing
    exception escape.
    """

    def __init__(self, reason: str, *, url: str | None = None) -> None:
        super().__init__(f"{reason} (endpoint {url})" if url else reason)
        self.reason = reason
        self.url = url


class RerankClient:
    """A thin client for one ``/v1/rerank`` endpoint.

    Does no I/O at construction (unlike :class:`~translator.retrieval.embedding.
    EmbeddingRetriever`, which embeds its whole corpus up front) -- a reranker has no index to
    build; every call is a fresh request over the candidates it is given. ``client`` is
    injectable so a test can drive an in-memory transport with no server.
    """

    def __init__(self, *, base_url: str, model: str = "local", timeout_seconds: float = 120.0,
                 client: httpx.Client | None = None) -> None:
        self._url = base_url.rstrip("/") + "/rerank"
        self._model = model
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None

    def rerank(self, query: str, documents: Sequence[str]) -> list[tuple[int, float]]:
        """Score ``documents`` against ``query``, best first.

        Returns ``(index, relevance_score)`` pairs, ``index`` into ``documents`` and
        ``relevance_score`` the endpoint's raw (unnormalised) score -- the caller decides how to
        map that to ``[0, 1]``. The indices are guaranteed to be a *permutation* of
        ``range(len(documents))`` -- exactly one result per document, no repeats -- and every
        score is guaranteed non-NaN; anything else is a :class:`RerankError` naming the
        endpoint, so a caller may index by them without defending itself.
        Empty ``documents`` short-circuits to ``[]`` with no HTTP call:
        there is nothing to rank, and a request with an empty ``documents`` array is not a
        request this API is meant to answer.
        """
        if not documents:
            return []
        try:
            response = self._client.post(
                self._url,
                json={"model": self._model, "query": query, "documents": list(documents)})
            response.raise_for_status()
            results = response.json()["results"]
        except httpx.HTTPError as exc:
            raise RerankError(f"rerank request failed: {exc}", url=self._url) from exc
        except (KeyError, ValueError) as exc:
            raise RerankError(f"malformed rerank response: {exc}", url=self._url) from exc
        if len(results) != len(documents):
            raise RerankError(
                f"endpoint returned {len(results)} results for {len(documents)} documents",
                url=self._url)
        try:
            scored = [(int(item["index"]), float(item["relevance_score"])) for item in results]
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankError(
                f"malformed rerank response, a result is missing 'index' or "
                f"'relevance_score': {exc}", url=self._url) from exc
        for index, score in scored:
            if not 0 <= index < len(documents):
                raise RerankError(
                    f"rerank result index {index} is out of range for {len(documents)} "
                    f"documents", url=self._url)
            if math.isnan(score):
                # NaN is unorderable: it would corrupt the sort below and, downstream, be
                # ranked wherever a comparison happened to leave it rather than by relevance.
                # ±inf is left alone -- it orders correctly and squashes to a sane 1.0/0.0.
                raise RerankError(
                    f"rerank result for document {index} has a NaN relevance_score",
                    url=self._url)
        # Distinctness is not implied by "right count, all in range": [(0, .9), (0, .8)] passes
        # both. It has to be checked here because the damage surfaces far away -- a caller
        # keying its candidates by index silently loses one and then raises a bare KeyError
        # from inside MMR, with nothing left pointing at the endpoint that actually misbehaved.
        duplicates = sorted(index for index, count in Counter(
            index for index, _ in scored).items() if count > 1)
        if duplicates:
            raise RerankError(
                f"endpoint returned duplicate result indices {duplicates} for "
                f"{len(documents)} documents", url=self._url)
        # Defensively re-sorted: the server conventionally returns results best-first already,
        # but the API contract does not require it, and a caller must be able to trust the order.
        scored.sort(key=lambda pair: -pair[1])
        return scored

    def close(self) -> None:
        """Release the HTTP client **if this object opened it**; an injected one is untouched.

        One ownership rule across the retrieval layer (:class:`~translator.retrieval.embedding.
        EmbeddingRetriever` and :class:`~translator.retrieval.hybrid.HybridRetriever` honour the
        same one): an object closes only what it owns. A client passed in belongs to whoever
        passed it -- they may still be using it, or sharing it with another retriever -- so
        closing it here would break a connection this object was merely lent.
        """
        if self._owns_client:
            self._client.close()
