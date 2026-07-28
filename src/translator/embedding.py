"""Back-compat re-export: the embedding retriever now lives in
:mod:`translator.retrieval.embedding`.

Kept so ``from translator.embedding import EmbeddingRetriever`` (an existing caller, or a script
written against the earlier layout) keeps working. New code should import from
:mod:`translator.retrieval` directly.
"""
from __future__ import annotations

from .retrieval.embedding import (
    DEFAULT_EMBEDDING_MIN_SCORE,
    EmbeddingError,
    EmbeddingRetriever,
)

__all__ = [
    "DEFAULT_EMBEDDING_MIN_SCORE",
    "EmbeddingError",
    "EmbeddingRetriever",
]
