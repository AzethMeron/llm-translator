"""EuroLLM: valid-JSON mode, like Bielik.

utter-project's EuroLLM, specialised for the EU languages, is a community GGUF built the
same way as Bielik and, like it, does not accept strict ``json_schema`` grammars. It uses
the same ``json_object`` strategy. Kept a separate profile rather than an alias of
``bielik`` because the two are different models with different hints, and collapsing them
would make ``--backend auto`` and the logs lie about which is in use.
"""
from __future__ import annotations

from ..core import JsonObjectBackend, register_backend

register_backend(JsonObjectBackend("eurollm", model_hints=("eurollm",)))
