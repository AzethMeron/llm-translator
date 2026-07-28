"""The default backend: grammar-constrained JSON for any OpenAI-compatible server.

Selected when a caller names no backend. Correct for Qwen and for most llama.cpp, vLLM
and ollama builds, which honour a strict ``json_schema`` response format. A model whose
server cannot compile that grammar selects a ``json_object`` backend instead -- see
``bielik``.
"""
from __future__ import annotations

from ..core import SchemaBackend, register_backend

register_backend(SchemaBackend("generic", aliases=("openai", "json_schema")))
