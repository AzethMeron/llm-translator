"""Reference-retrieval implementations behind the :class:`transunit.reference.Retriever` seam.

The stdlib-only lexical default (:class:`transunit.reference.LexicalRetriever`) needs nothing
beyond the interpreter. Everything in this package is *optional* and lives on the ``translator``
side because it needs a local server (and, for the dense arm, ``numpy``):
:class:`~translator.retrieval.embedding.EmbeddingRetriever` matches by meaning over a
``/v1/embeddings`` endpoint, :class:`~translator.retrieval.rerank.RerankClient` re-scores a
candidate pool over a ``/v1/rerank`` endpoint, and
:class:`~translator.retrieval.hybrid.HybridRetriever` composes the lexical and dense arms via
Reciprocal Rank Fusion, with optional reranking and MMR diversity.
"""
from __future__ import annotations

from .embedding import DEFAULT_EMBEDDING_MIN_SCORE, EmbeddingError, EmbeddingRetriever
from .hybrid import HybridRetriever
from .rerank import DEFAULT_RERANK_MIN_SCORE, RerankClient, RerankError

__all__ = [
    "DEFAULT_EMBEDDING_MIN_SCORE",
    "DEFAULT_RERANK_MIN_SCORE",
    "EmbeddingError",
    "EmbeddingRetriever",
    "HybridRetriever",
    "RerankClient",
    "RerankError",
]
