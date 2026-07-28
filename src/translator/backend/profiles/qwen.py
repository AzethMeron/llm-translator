"""Qwen: grammar-constrained JSON.

Behaviourally identical to ``generic`` -- Qwen honours strict ``json_schema`` -- but
named so that ``--backend auto`` selects it from a served model id containing "qwen", and
so the choice reads explicitly in logs. This is the reference example of a per-model
profile that needs no special request shaping: a name and a model hint over the default
strategy.
"""
from __future__ import annotations

from ..core import SchemaBackend, register_backend

register_backend(SchemaBackend("qwen", model_hints=("qwen",)))
