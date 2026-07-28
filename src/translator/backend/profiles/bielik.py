"""Bielik: valid-JSON mode, because its llama.cpp build rejects strict schema grammars.

Bielik (SpeakLeash's Polish-specialised model) fails a strict ``json_schema`` request
with "Failed to initialize samplers", but honours ``json_object`` -- valid JSON with the
shape left to the prompt -- perfectly. This profile is the whole fix: selecting it, by
name or via ``--backend auto`` on a model id containing "bielik", shapes every structured
request the way Bielik accepts, and the translator itself does not change.

The reply is still validated against the schema by the client, so choosing this backend
weakens the *constraint* on the model, not the *check* on its output.
"""
from __future__ import annotations

from ..core import JsonObjectBackend, register_backend

register_backend(JsonObjectBackend("bielik", model_hints=("bielik",)))
