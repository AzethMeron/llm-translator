"""The model-facing layer of the translator: the client and its per-model backends.

This is the boundary the translator talks *through* to reach a language model, kept
strictly separate from the *carrier adapter* (:mod:`transunit.adapter`) it talks through
to reach text. The word "adapter" is reserved for the carrier side; the model side is a
**backend**.

Public surface:

* :class:`LlmClient`, :class:`ServerConfig`, :class:`UsageStats`, :class:`Message` --
  the transport, model-agnostic.
* :class:`LlmError` and its subclasses -- the error taxonomy, organised by blast radius.
* :class:`Backend` and the registry (:func:`get_backend`, :func:`resolve_backend`,
  :func:`available_backends`, :func:`register_backend`) -- selecting and shaping requests
  per model.

The built-in profiles (Qwen, Bielik, EuroLLM, and the default) register themselves the first
time the registry is used -- ``get_backend``, ``resolve_backend``, ``available_backends`` --
so they are always available without an eager import here.
"""
from __future__ import annotations

from .client import (
    LlmClient,
    LlmContentError,
    LlmError,
    LlmIncompleteJsonError,
    LlmRefusalError,
    LlmTruncationError,
    Message,
    ServerConfig,
    UsageStats,
)
from .core import (
    DEFAULT_BACKEND,
    Backend,
    BackendError,
    JsonObjectBackend,
    SchemaBackend,
    StructuredRequest,
    available_backends,
    get_backend,
    register_backend,
    resolve_backend,
    suggest_backend,
)

# The built-in profiles are registered lazily on first registry use (see core
# _ensure_profiles_loaded), so there is deliberately no eager `from . import profiles` here --
# one loading mechanism, not two.

__all__ = [
    # transport
    "LlmClient", "ServerConfig", "UsageStats", "Message",
    "LlmError", "LlmContentError", "LlmRefusalError", "LlmTruncationError",
    "LlmIncompleteJsonError",
    # backends
    "Backend", "BackendError", "SchemaBackend", "JsonObjectBackend", "StructuredRequest",
    "DEFAULT_BACKEND", "get_backend", "resolve_backend", "available_backends",
    "register_backend", "suggest_backend",
]
