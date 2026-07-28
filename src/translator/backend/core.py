"""The intermediate layer between the translator and a local language model.

This is the model-facing boundary, and it is deliberately called a **backend**, not an
"LLM adapter": the word "adapter" already names the *carrier* side of the pipeline
(:mod:`transunit.adapter`), and using it for both is exactly the ambiguity this naming
avoids. A backend sits between the translator and one family of model; a carrier adapter
sits between the translator and one source of text. They never meet.

A backend exists because "the same request" does not mean "the same bytes on the wire"
across models served over an OpenAI-compatible endpoint. The sharpest difference, and the
one that motivated this layer, is how a model will honour a request for *structured*
(JSON) output:

* Grammar-constrained decoding (``response_format`` = ``json_schema``, strict) is the
  strongest guarantee and the default. Qwen and most llama.cpp builds honour it.
* Some builds cannot compile that grammar. Bielik rejects it outright ("Failed to
  initialize samplers"), but honours ``json_object`` (valid JSON, shape unconstrained)
  perfectly, provided the shape is described in the prompt.

Without this layer, switching from Qwen to Bielik "broke something that should not have
broken", because the request shaping was scattered through the client as conditionals on
a config string. Here it is one object per model family. The translator never changes
when the model does; only the selected backend does. Adding a new model is a new small
module in :mod:`translator.backend.profiles`, and nothing else.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_BACKEND = "generic"
"""Backend used when a caller names none. Grammar-constrained decoding, which is correct
for Qwen and most local servers; a model that rejects it selects another by name."""


class BackendError(Exception):
    """A backend was requested that does not exist, or two were registered under one name."""


@dataclass(frozen=True, slots=True)
class Message:
    """One chat turn. The fundamental unit both the client and a backend operate on."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class StructuredRequest:
    """How to ask a specific model for schema-conforming JSON.

    Carries the (possibly rewritten) messages and the ``response_format`` fragment to put
    in the request body. A backend that must describe the schema in the prompt returns
    rewritten messages; one that constrains decoding returns the messages unchanged.
    """

    messages: tuple[Message, ...]
    response_format: dict[str, Any]


class Backend(ABC):
    """Request-shaping for one family of local model.

    Stateless and shared: one instance is registered under its name (and any aliases) and
    reused for every request. Subclasses implement :meth:`structured_request`; everything
    else about talking to the server lives in the client, which is model-agnostic.
    """

    def __init__(self, name: str, *, aliases: Sequence[str] = (),
                 model_hints: Sequence[str] = ()) -> None:
        if not name.strip():
            raise BackendError("a backend needs a non-empty name")
        self.name = name
        self.aliases = tuple(aliases)
        # Substrings of a served model id that point to this backend, used by the `auto`
        # selection and by the diagnostic hint when a structured request is rejected.
        self.model_hints = tuple(hint.lower() for hint in model_hints)

    @abstractmethod
    def structured_request(self, messages: Sequence[Message],
                           schema: dict[str, Any]) -> StructuredRequest:
        """Shape a request for output conforming to ``schema`` for this model."""

    def matches_model(self, model: str) -> bool:
        """Whether a served model id looks like one this backend is meant for."""
        lowered = model.lower()
        return any(hint in lowered for hint in self.model_hints)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"


class SchemaBackend(Backend):
    """Grammar-constrained decoding: ``response_format`` = ``json_schema``, strict.

    The strongest guarantee -- the server builds a GBNF grammar from the schema and
    decoding cannot leave it -- and the default. The messages are sent unchanged, because
    the model is already forced to produce the right shape.
    """

    def structured_request(self, messages: Sequence[Message],
                           schema: dict[str, Any]) -> StructuredRequest:
        return StructuredRequest(
            messages=tuple(messages),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": schema},
            })


class JsonObjectBackend(Backend):
    """Valid-JSON-only mode: ``response_format`` = ``json_object``, shape described in prose.

    For models whose llama.cpp build rejects a strict schema grammar. ``json_object``
    guarantees the output parses as JSON but not that it has the right fields, so the
    schema's shape is appended to the prompt. The reply is still validated against the
    schema by the client, so this is a weaker *constraint*, not a weaker *check*: a
    malformed shape is caught and retried rather than shipped.
    """

    def structured_request(self, messages: Sequence[Message],
                           schema: dict[str, Any]) -> StructuredRequest:
        return StructuredRequest(
            messages=_append_json_instruction(messages, schema),
            response_format={"type": "json_object"})


def _append_json_instruction(messages: Sequence[Message],
                             schema: dict[str, Any]) -> tuple[Message, ...]:
    """Append an instruction to answer as a JSON object with the schema's fields.

    A minimal, literal description -- the field names and their JSON types -- rather than
    the full schema, because the model needs to know the shape to produce, not to reason
    about a specification. Appended to the final user turn so it is the last thing read.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields = []
    for name, spec in properties.items():
        kind = spec.get("type", "string")
        # A union type is a list in the schema; render it as "string|null" rather than letting
        # a Python list repr ("['string', 'null']") leak into the prompt the model reads.
        rendered_kind = "|".join(kind) if isinstance(kind, list) else kind
        marker = "" if name in required else " (optional)"
        fields.append(f'"{name}": <{rendered_kind}>{marker}')
    shape = "{" + ", ".join(fields) + "}"
    instruction = (
        "Respond with a single JSON object and nothing else, of the form "
        f"{shape}. Put it all on one line, escape any double quote inside a value as "
        '\\", use no literal line breaks inside a value, and close every string and '
        "brace. For an empty value use [] for a list and \"\" for a string; use null only "
        "for a field whose type is shown as |null. Output only the JSON.")

    result = list(messages)
    for index in range(len(result) - 1, -1, -1):
        if result[index].role == "user":
            result[index] = Message("user", f"{result[index].content}\n\n{instruction}")
            return tuple(result)
    result.append(Message("user", instruction))
    return tuple(result)


# --- registry -----------------------------------------------------------------

# A module-level registry is global state, which CLAUDE.md tells us to avoid -- so its use
# here is deliberate and bounded. It is written only at import time (each profile module
# registers on import, under the interpreter's import lock, so registration never races), it
# is effectively immutable thereafter, registration is idempotent for a given instance, and a
# name collision with a *different* backend is refused rather than silently overwritten. It
# holds no per-run mutable state; a run's state lives on its LlmClient. This is the codec-
# registry pattern, and the alternative -- threading a backend table through every call site --
# would be worse without buying any real isolation.
_REGISTRY: dict[str, Backend] = {}


def register_backend(backend: Backend) -> None:
    """Register ``backend`` under its name and every alias.

    A name collision with a *different* backend is an error rather than a silent
    overwrite: two models answering to one name would make selection depend on import
    order. Re-registering the same instance is a no-op, so importing a profile twice is
    harmless.
    """
    for key in (backend.name, *backend.aliases):
        lowered = key.lower()
        existing = _REGISTRY.get(lowered)
        if existing is not None and existing is not backend:
            raise BackendError(
                f"backend name {key!r} is already registered to {existing!r}; "
                f"cannot also register {backend!r}")
        _REGISTRY[lowered] = backend


_profiles_loaded = False


def _ensure_profiles_loaded() -> None:
    """Import the built-in profiles once, so they self-register on first use.

    Lazy rather than a top-level import, to keep the import graph acyclic: the profiles depend
    on this module, so this module must not import them while it is still being defined. By the
    time any registry function runs, definition is complete.

    Guarded by a dedicated ``_loaded`` flag rather than ``if not _REGISTRY``: emptiness and
    "not yet loaded" are different questions, and conflating them would skip the built-ins the
    moment a caller registers a custom backend before touching the registry. Import is
    idempotent and runs under the interpreter's import lock, so a concurrent first touch is
    safe.
    """
    global _profiles_loaded
    if _profiles_loaded:
        return
    from . import profiles  # noqa: F401  -- importing registers the built-ins
    _profiles_loaded = True


def get_backend(name: str) -> Backend:
    """Look up a backend by name or alias, listing what is available when it is missing."""
    _ensure_profiles_loaded()
    backend = _REGISTRY.get(name.lower())
    if backend is None:
        raise BackendError(
            f"unknown backend {name!r}; available backends are "
            f"{', '.join(available_backends())}")
    return backend


def available_backends() -> list[str]:
    """The distinct registered backend names, sorted."""
    _ensure_profiles_loaded()
    return sorted({backend.name for backend in _REGISTRY.values()})


def suggest_backend(model: str) -> str | None:
    """The backend whose model hints match ``model``, or ``None`` if none do."""
    _ensure_profiles_loaded()
    for backend in _unique_backends():
        if backend.matches_model(model):
            return backend.name
    return None


def resolve_backend(name: str, model: str) -> Backend:
    """Select a backend by ``name``, or by the model id when ``name`` is ``"auto"``.

    ``auto`` is what hardens a model switch: it picks the backend whose hints match the
    served model, falling back to the default when none do. The caller is expected to
    report which backend was chosen, so the selection is never silent.
    """
    if name == "auto":
        guess = suggest_backend(model)
        return get_backend(guess) if guess is not None else get_backend(DEFAULT_BACKEND)
    return get_backend(name)


def _unique_backends() -> list[Backend]:
    seen: dict[int, Backend] = {}
    for backend in _REGISTRY.values():
        seen.setdefault(id(backend), backend)
    return list(seen.values())
