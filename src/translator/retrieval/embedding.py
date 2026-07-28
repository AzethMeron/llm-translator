"""Optional embedding (semantic) retriever, behind the :class:`transunit.reference.Retriever` seam.

The stdlib-only lexical default (:class:`transunit.reference.LexicalRetriever`) matches on the
*surface* of a line -- shared character n-grams. This matches on *meaning*: cosine similarity over
sentence embeddings from a local, OpenAI-compatible ``/v1/embeddings`` endpoint (llama.cpp,
ollama, vLLM). It finds paraphrases and cross-script equivalents the lexical retriever is blind to
-- Japanese 猫 (kanji) and ネコ (katakana) share no characters but embed to nearly the same vector.

It lives on the ``translator`` side, not in the stdlib-only ``transunit`` contract, and is
**optional**, because it needs two things the lexical default deliberately avoids: an embedding
server, and ``numpy`` (imported lazily, so importing this module -- and the rest of the engine --
never requires it; only *constructing* a retriever does). Measured on Japanese game text it lifted
top-1 relevance from ~37% to ~49% and recovered a relevant match on ~40% of lines the lexical
floor rejected, at that cost.

**Read-only and shareable.** It embeds its corpus once at construction and is immutable
thereafter, so one instance is safely queried by every worker of a run with no lock -- exactly
like the lexical retriever. Self-population (learning within a run) is the lexical retriever's
job; a growing embedding index is out of scope here and selecting learning with this retriever is
refused by the CLI rather than silently ignored.

Determinism: a fixed embedding model returns the same vector for the same input, so retrieval is
reproducible across runs and workers, the same guarantee the lexical retriever gives.

Also exposes :meth:`EmbeddingRetriever.vectors_for`, an accessor onto the already-computed,
L2-normalised rows for a set of indexed entries. It exists for
:class:`translator.retrieval.hybrid.HybridRetriever`'s MMR redundancy term (a cosine between two
already-embedded candidates), so the hybrid never has to re-embed or duplicate this retriever's
index to measure how similar two of its own hits are to each other.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import httpx

from transunit.reference import ReferenceEntry, Retrieved

if TYPE_CHECKING:  # keep numpy out of the import path; only a constructed retriever needs it
    import numpy as np

_PLACEHOLDER = re.compile(r"\[\[\d+\]\]")

DEFAULT_EMBEDDING_MIN_SCORE = 0.55
"""A sensible starting floor for cosine over embeddings. It is on a different scale from the
lexical floor: embedding cosines cluster high (a loosely related pair still scores ~0.5), so the
lexical default of 0.30 would admit almost everything. Tune per corpus, as with the lexical one."""


class EmbeddingError(Exception):
    """The embedding endpoint is unreachable or misbehaving, or numpy is unavailable.

    Structured like the rest of the engine's errors so a caller can catch it and fail clearly
    rather than crash: an embedding retriever that cannot embed is unusable, and the run should
    stop with a message naming the cause, not proceed on empty vectors.
    """

    def __init__(self, reason: str, *, url: str | None = None) -> None:
        super().__init__(f"{reason} (endpoint {url})" if url else reason)
        self.reason = reason
        self.url = url


def _require_numpy() -> Any:
    # Imported here, not at module top, on purpose: the core engine must import without numpy.
    try:
        import numpy
    except ImportError as exc:
        raise EmbeddingError(
            "the embedding retriever needs numpy, which is not installed. Install it (pip install "
            "numpy; it ships in requirements.txt for this optional feature), or use the default "
            "lexical retriever (drop --embedding-url).") from exc
    return numpy


def _clean(text: str) -> str:
    """Strip placeholders before embedding, mirroring the lexical retriever, so a shared ``[[0]]``
    cannot pull unrelated lines together. Never empty -- an all-placeholder line embeds a space."""
    return _PLACEHOLDER.sub(" ", text).strip() or " "


class EmbeddingRetriever:
    """Semantic :class:`~transunit.reference.Retriever` over a local embedding endpoint.

    Only the directions asked for are embedded (``index_source`` / ``index_target``); querying a
    direction that was not built raises rather than returning nothing, so a misconfiguration is
    loud, not silent -- the same contract the lexical retriever honours. The source index covers
    only entries that have a source; the target index covers every entry.
    """

    def __init__(self, entries: Iterable[ReferenceEntry], *, base_url: str,
                 model: str = "local", index_source: bool = True, index_target: bool = False,
                 timeout_seconds: float = 120.0, batch_size: int = 64,
                 client: httpx.Client | None = None) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._np = _require_numpy()
        self._entries = tuple(entries)
        self._url = base_url.rstrip("/") + "/embeddings"
        self._model = model
        self._batch = batch_size
        # ``client`` is injectable so a test can drive an in-memory transport with no server; a
        # normal caller lets it own a real one. A client we opened is ours to release if the corpus
        # embedding below then fails; an injected one stays the caller's to close.
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None
        self._source_ids: tuple[int, ...] | None = None
        self._source_matrix = None
        self._target_ids: tuple[int, ...] | None = None
        self._target_matrix = None
        # entry -> its row in the matching matrix, for vectors_for's MMR-redundancy lookup.
        # Frozen ReferenceEntry is hashable; a duplicate entry's row is simply overwritten by its
        # last occurrence, which is harmless (the vectors are identical for identical text).
        self._row_source: dict[ReferenceEntry, int] = {}
        self._row_target: dict[ReferenceEntry, int] = {}
        try:
            if index_source:
                ids = tuple(i for i, e in enumerate(self._entries) if e.source)
                self._source_ids = ids
                self._source_matrix = self._embed([self._entries[i].source for i in ids])
                self._row_source = {self._entries[i]: row for row, i in enumerate(ids)}
            if index_target:
                ids = tuple(range(len(self._entries)))
                self._target_ids = ids
                self._target_matrix = self._embed([e.target for e in self._entries])
                self._row_target = {self._entries[i]: row for row, i in enumerate(ids)}
        except BaseException:  # release our client then re-raise; cleanup must never swallow
            if self._owns_client:
                self._client.close()
            raise

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` (batched), L2-normalised so a later dot product is the cosine."""
        np = self._np
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            chunk = [_clean(t) for t in texts[start:start + self._batch]]
            try:
                response = self._client.post(
                    self._url, json={"model": self._model, "input": chunk})
                response.raise_for_status()
                data = response.json()["data"]
            except httpx.HTTPError as exc:
                raise EmbeddingError(f"embedding request failed: {exc}", url=self._url) from exc
            except (KeyError, ValueError) as exc:
                raise EmbeddingError(f"malformed embedding response: {exc}",
                                     url=self._url) from exc
            if len(data) != len(chunk):
                raise EmbeddingError(
                    f"endpoint returned {len(data)} embeddings for {len(chunk)} inputs",
                    url=self._url)
            try:
                vectors.extend(item["embedding"] for item in data)
            except (KeyError, TypeError) as exc:
                # A data item is not a dict, or lacks the "embedding" field: malformed like a
                # missing "data" key, so it fails the same structured way, not as a raw error.
                raise EmbeddingError(f"malformed embedding response, an item has no 'embedding': "
                                     f"{exc}", url=self._url) from exc
        try:
            matrix = np.asarray(vectors, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            # Ragged or non-numeric vectors -- an endpoint returning inconsistent dimensions or a
            # non-list "embedding". Fail as a clear EmbeddingError, not a raw numpy exception.
            raise EmbeddingError(f"embeddings are not a uniform numeric matrix: {exc}",
                                 url=self._url) from exc
        if matrix.ndim != 2:
            raise EmbeddingError(
                f"expected 2-D embeddings, got shape {matrix.shape}", url=self._url)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
        return matrix

    def by_source(self, query: str, *, k: int,
                  min_score: float = 0.0) -> tuple[Retrieved, ...]:
        if self._source_matrix is None or self._source_ids is None:
            raise EmbeddingError(
                "this retriever was built without a source index; construct it with "
                "index_source=True to query by source")
        return self._query(self._source_matrix, self._source_ids, query, k, min_score)

    def by_target(self, query: str, *, k: int,
                  min_score: float = 0.0) -> tuple[Retrieved, ...]:
        if self._target_matrix is None or self._target_ids is None:
            raise EmbeddingError(
                "this retriever was built without a target index; construct it with "
                "index_target=True to query by target")
        return self._query(self._target_matrix, self._target_ids, query, k, min_score)

    def _query(self, matrix: np.ndarray, ids: tuple[int, ...], query: str, k: int,
               min_score: float) -> tuple[Retrieved, ...]:
        np = self._np
        if k <= 0 or matrix.shape[0] == 0:
            return ()
        vector = self._embed([query])[0]
        scores = matrix @ vector  # cosine, since both sides are L2-normalised
        # Descending score; ties broken on ascending corpus position (argsort is stable), so the
        # result is reproducible and matches the lexical retriever's ordering rule.
        order = np.argsort(-scores, kind="stable")[:k]
        # Clamped at BOTH ends: Retrieved.score is documented as being in [0, 1], and a cosine
        # runs [-1, 1]. The upper clamp guards float drift on a near-parallel pair; the lower
        # one matters whenever min_score < 0, which admits genuinely anti-parallel hits that
        # would otherwise leave a negative score in a field nothing downstream expects to be
        # negative. Ranking is unaffected -- the sort is on the raw cosine, and so is the
        # min_score gate, so the caller's floor still means what it says.
        return tuple(
            Retrieved(entry=self._entries[ids[int(j)]],
                      score=min(1.0, max(0.0, float(scores[int(j)]))))
            for j in order if scores[int(j)] >= min_score)

    def vectors_for(self, entries: Sequence[ReferenceEntry], *, target: bool) -> np.ndarray:
        """The stored, L2-normalised rows for ``entries``, aligned to their order.

        For :class:`~translator.retrieval.hybrid.HybridRetriever`'s MMR redundancy term: a
        cosine between two candidates it already retrieved from this index, without a second
        embedding round-trip. Raises :class:`EmbeddingError` if the requested direction was
        never indexed, or if an entry given is not one this retriever actually indexed.
        """
        row_map = self._row_target if target else self._row_source
        matrix = self._target_matrix if target else self._source_matrix
        if matrix is None:
            direction = "target" if target else "source"
            raise EmbeddingError(
                f"this retriever was built without a {direction} index; construct it with "
                f"index_{direction}=True")
        try:
            rows = [row_map[entry] for entry in entries]
        except KeyError as exc:
            raise EmbeddingError(f"entry not indexed by this retriever: {exc}") from exc
        return matrix[rows]

    def pairwise_similarity(self, a: ReferenceEntry, b: ReferenceEntry, *, target: bool) -> float:
        """Cosine between two already-indexed entries, as a plain ``float``.

        Built on :meth:`vectors_for` (both rows already L2-normalised, so their dot product IS
        the cosine). This is the one place a caller needs numpy-flavoured similarity math without
        needing numpy itself: a plain Python float in, float out, so
        :class:`~translator.retrieval.hybrid.HybridRetriever` can hand this straight to
        :func:`~translator.retrieval.fusion.mmr_order` as its ``similarity`` callable, and numpy
        stays confined to this one module.
        """
        vectors = self.vectors_for((a, b), target=target)
        return float(vectors[0] @ vectors[1])

    def close(self) -> None:
        """Release the HTTP client **if this retriever opened it**; an injected one is untouched.

        The same ownership rule the construction-failure path already follows, and the one
        :class:`~translator.retrieval.rerank.RerankClient` and
        :class:`~translator.retrieval.hybrid.HybridRetriever` follow: an object closes only what
        it owns. A client passed in belongs to its injector, who may still be using it.
        """
        if self._owns_client:
            self._client.close()

    @property
    def source_entries(self) -> int:
        return sum(1 for entry in self._entries if entry.source)

    def __len__(self) -> int:
        return len(self._entries)
