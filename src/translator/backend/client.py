"""Synchronous client for a local, OpenAI-compatible inference server.

Model-agnostic on purpose: everything that varies between models -- above all how a
structured request must be shaped -- lives in a :class:`~translator.backend.core.Backend`,
which this client consults rather than deciding for itself. The client owns only the
transport: the HTTP connection, retry with backoff, the error taxonomy, and usage
accounting.

Every agent role in the pipeline talks to the *same* loaded model and differs only in its
system prompt. That is a deliberate constraint of the target hardware -- a single 16 GB
GPU cannot host several specialist models at once -- and it keeps the server's prefix
cache warm, which is the dominant throughput factor for a large job. Prompts are assembled
most-static-part-first (system rules, then glossary, then the unit) so the shared prefix
is reused across requests rather than recomputed.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import httpx

from .core import (
    DEFAULT_BACKEND,
    Backend,
    Message,
    SchemaBackend,
    get_backend,
    suggest_backend,
)

__all__ = [
    "Message", "ServerConfig", "UsageStats", "LlmClient",
    "LlmError", "LlmContentError", "LlmRefusalError", "LlmTruncationError",
    "LlmIncompleteJsonError",
]

logger = logging.getLogger(__name__)

CONTEXT_WARN_FRACTION = 0.8
"""How full the context window may get before a proactive warning. The prompt plus the output
budget crossing this fraction means the run is approaching truncation; the server's own check
(``_server_error_reason``) is the hard stop, this is the early warning before it."""


def _estimate_tokens(text: str) -> int:
    """A rough, tokenizer-free token estimate, for a budget *warning* -- not a gate.

    No tokenizer is a deliberate non-dependency, so this approximates: CJK and other
    full-width characters count as ~1 token each (a real tokenizer emits roughly one token per
    such character), and every other script runs at ~4 characters per token. It deliberately
    errs toward over-counting, so the warning fires a little before the server's true limit
    rather than after it. The server's own context check remains authoritative.
    """
    wide = sum(1 for char in text if unicodedata.east_asian_width(char) in ("W", "F"))
    return wide + (len(text) - wide + 3) // 4


class LlmError(Exception):
    """A request to the inference server failed or returned unusable output."""

    def __init__(self, reason: str, *, role: str = "<none>", status: int | None = None,
                 body: str = "") -> None:
        super().__init__(
            f"role={role}: {reason}" + (f" (HTTP {status})" if status else "")
            + (f": {body[:400]}" if body else ""))
        self.reason = reason
        self.role = role
        self.status = status
        self.body = body


class LlmContentError(LlmError):
    """The server answered, but this request's output is unusable.

    Distinguished from its base class by *blast radius*, which is what the caller needs in
    order to react correctly. A bare LlmError means the server or transport is broken, so
    every remaining unit would fail the same way and the run should stop. This means the
    model produced garbage for one particular input -- a repetition loop that overruns
    max_tokens, unparseable JSON, a missing field -- which says nothing about the next
    unit, so the caller should record the failure and carry on.
    """


class LlmRefusalError(LlmContentError):
    """The model declined to produce output.

    Worth distinguishing so a caller can count refusals separately from other malformed
    output, rather than recording them as translations. A heavily-aligned model asked to
    translate sensitive source may refuse or silently bowdlerise, and that is a different
    event from a repetition loop.
    """


class LlmTruncationError(LlmContentError):
    """Generation hit the token ceiling before the model finished.

    Its own type because the diagnosis and the fix differ from ordinary malformed output:
    the text is not wrong, it is unfinished, which usually means the model fell into a
    repetition loop on an input it could not handle.
    """


class LlmIncompleteJsonError(LlmContentError):
    """The reply's JSON envelope was cut short, but its content parses once closed.

    Raised instead of the base :class:`LlmContentError` only when
    :func:`_close_truncated_json_object` recovers a schema-conformant object from an otherwise-
    unparseable reply -- a narrow, verified
    failure shape (see ``docs/feature-requests/json-envelope-truncation-repair.md``): the model
    emitted end-of-sequence somewhere inside the ``"translation"`` value or just after it, before
    the object's closing brace. The envelope is repaired; **the content may still be cut off
    mid-clause**, and nothing at this layer can tell the two apart. Carries ``.recovered``, the
    parsed object, so a caller that opts in can treat it as an unverified *candidate* -- to be
    re-checked and re-reviewed, never shipped as certified. Distinct from a garbled reply because
    the response is *usable but unverified* --
    this type exists so a caller can choose to re-check and re-review it rather than discard it, but
    a caller that does nothing special still gets today's safe behaviour, because this is still an
    :class:`LlmContentError` and every existing ``except LlmContentError`` keeps catching it.
    """

    def __init__(self, reason: str, *, role: str = "<none>", body: str = "",
                 recovered: dict[str, Any]) -> None:
        super().__init__(reason, role=role, body=body)
        self.recovered = recovered


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Connection and decoding settings for the local server.

    Purely about the transport: how to reach the server and how hard to retry. How a
    model wants a structured request shaped is a :class:`~translator.backend.core.Backend`
    concern, not a field here.
    """

    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "local"
    timeout_seconds: float = 300.0
    max_retries: int = 4
    retry_backoff_seconds: float = 2.0
    context_window: int = 0
    """The server's context size in tokens, for the proactive prompt-budget warning.

    ``0`` -- the default -- disables the warning (the engine cannot know the window unless
    told; the server's own overflow rejection still catches an oversized prompt). Set it to the
    value the server was launched with (llama.cpp ``-c`` / ``--ctx-size``, per slot) and the
    client warns once when an assembled prompt plus its output budget crosses
    :data:`CONTEXT_WARN_FRACTION` of it, before the wall rather than at it."""
    enable_reasoning: bool = False
    """Whether to let a reasoning model emit its chain-of-thought.

    Off by default, and deliberately so: for a bounded translation task the hidden reasoning
    tokens a model like Qwen3 emits are billed against ``max_tokens`` and routinely truncate
    the JSON answer before it closes, which the harness then rejects as unusable output --
    a silent, temperature-flaky failure that looks like a bad translation but is not. With
    reasoning suppressed the model returns the answer directly and the source-scaled budget
    holds. Enabling it is discouraged; if you do, raise the translate/review token budgets in
    the agent config to leave room for the reasoning trace. Enacted through the chat-template
    kwarg the reasoning model families honour, which a non-reasoning template simply ignores.
    """

    def __post_init__(self) -> None:
        if self.max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {self.max_retries}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")
        if self.retry_backoff_seconds < 0:
            # A negative backoff would make time.sleep raise on the first retry -- a latent
            # crash on the error path, exactly where it must not.
            raise ValueError(
                f"retry_backoff_seconds must be >= 0, got {self.retry_backoff_seconds}")
        if self.context_window < 0:
            raise ValueError(f"context_window must be >= 0, got {self.context_window}")


@dataclass
class UsageStats:
    """Cumulative token and latency accounting, for throughput reporting.

    One client is shared by every worker when the runner translates units concurrently, so
    all mutation is guarded: a bare ``+=`` loses updates under concurrent completion.

    ``seconds`` totals time spent inside requests, not wall-clock. With N workers in flight
    it exceeds elapsed time, so the derived rate is per-stream throughput, not aggregate.
    """

    requests: int = 0
    prompt_tokens: int = 0
    peak_prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    retries: int = 0
    refusals: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, usage: dict[str, Any], elapsed: float) -> None:
        with self._lock:
            self.requests += 1
            # The usage block comes from the server, so its shape is not ours to trust: a
            # null or non-numeric token count must not sink a translation that otherwise
            # succeeded. Accounting is diagnostic; degrade it to zero rather than raise.
            prompt = _token_count(usage.get("prompt_tokens"))
            self.prompt_tokens += prompt
            # The largest single prompt the server actually measured -- the real headroom against
            # the window, reported at the end even when no proactive warning ever fired.
            self.peak_prompt_tokens = max(self.peak_prompt_tokens, prompt)
            self.completion_tokens += _token_count(usage.get("completion_tokens"))
            self.seconds += elapsed

    def record_retry(self) -> None:
        with self._lock:
            self.retries += 1

    def record_refusal(self) -> None:
        with self._lock:
            self.refusals += 1

    @property
    def completion_tokens_per_second(self) -> float:
        return self.completion_tokens / self.seconds if self.seconds else 0.0


class LlmClient:
    """Synchronous chat client with retry and backend-directed structured output.

    Owns its HTTP connection; use as a context manager so the socket is released
    deterministically. The backend is fixed for the client's lifetime -- one loaded model,
    one request shape.
    """

    def __init__(self, config: ServerConfig | None = None, *,
                 backend: Backend | None = None) -> None:
        self.config = config or ServerConfig()
        self.backend = backend or get_backend(DEFAULT_BACKEND)
        self.stats = UsageStats()
        self._client = httpx.Client(timeout=self.config.timeout_seconds)
        # The prompt-budget warning fires once per run, not per unit: a systemic overrun would
        # otherwise repeat for thousands of units. One instance is shared by every worker, so
        # the check-and-set is guarded.
        self._context_warned = False
        self._context_lock = threading.Lock()

    def __enter__(self) -> LlmClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def health(self) -> bool:
        """True when the server answers a models listing."""
        try:
            response = self._client.get(f"{self.config.base_url}/models", timeout=10.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def complete(
        self,
        messages: Sequence[Message],
        *,
        role: str = "<none>",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        schema: dict[str, Any] | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """Run one chat completion, retrying transient failures.

        When ``schema`` is given, the active backend decides how the request asks for
        conforming JSON -- constrained decoding, or a prompt-described shape.
        """
        if schema is not None:
            request = self.backend.structured_request(messages, schema)
            sent = [{"role": m.role, "content": m.content} for m in request.messages]
            response_format: dict[str, Any] | None = request.response_format
        else:
            sent = [{"role": m.role, "content": m.content} for m in messages]
            response_format = None

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": sent,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        if response_format is not None:
            payload["response_format"] = response_format
        if not self.config.enable_reasoning:
            # Suppress chain-of-thought. A reasoning model (Qwen3, ...) otherwise spends
            # hidden reasoning tokens that count against max_tokens and truncate the answer;
            # for translation the reasoning buys nothing. Sent as the chat-template kwarg the
            # reasoning families read (requires the server's jinja templating); a template
            # that does not declare it ignores the kwarg, so this is inert for other models.
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        # Estimate the assembled request against the window before sending, so an operator is
        # warned as it approaches the limit rather than only when the server rejects it. Once
        # per request, before the retry loop, on the actual messages the backend will send.
        self._warn_if_context_tight(sent, max_tokens, role)

        last: Exception | None = None
        for attempt in range(self.config.max_retries):
            if attempt:
                self.stats.record_retry()
                time.sleep(self.config.retry_backoff_seconds * (2 ** (attempt - 1)))
            try:
                started = time.monotonic()
                response = self._client.post(
                    f"{self.config.base_url}/chat/completions", json=payload)
                elapsed = time.monotonic() - started
            except httpx.HTTPError as exc:
                last = LlmError(f"transport failure: {exc}", role=role)
                continue

            if response.status_code >= 500:
                last = LlmError(self._server_error_reason(schema, response.text),
                                role=role, status=response.status_code, body=response.text)
                continue
            if response.status_code != 200:
                # 4xx indicates a malformed request; retrying cannot help.
                raise LlmError(self._server_error_reason(schema, response.text),
                               role=role, status=response.status_code, body=response.text)

            try:
                body = response.json()
                choice = body["choices"][0]
                content = choice["message"]["content"]
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                last = LlmError(f"malformed response: {exc}", role=role, body=response.text)
                continue

            if content is None:
                self.stats.record_refusal()
                raise LlmRefusalError("model returned no content", role=role)

            self.stats.record(body.get("usage", {}), elapsed)
            # Truncated output is reported here rather than left to surface downstream as a
            # confusing parse error, and is not retried: hitting the ceiling is a property
            # of this prompt, so another attempt loops the same way.
            if choice.get("finish_reason") == "length":
                raise LlmTruncationError(
                    f"generation stopped at the {max_tokens}-token ceiling; the model did "
                    f"not finish (usually a repetition loop)",
                    role=role, body=content[-200:])
            return content

        raise last or LlmError("exhausted retries with no recorded error", role=role)

    def _warn_if_context_tight(
            self, sent: list[dict[str, str]], max_tokens: int, role: str) -> None:
        """Warn once when the estimated prompt plus its output budget nears the context window.

        A no-op unless ``config.context_window`` is set (the engine cannot know the window
        otherwise). The estimate is rough and errs high; the server's own overflow rejection
        stays the hard stop. The warning names the same knobs the reactive overflow message
        does, so both point at the same fix.
        """
        window = self.config.context_window
        if window <= 0:
            return
        estimated = sum(_estimate_tokens(message["content"]) for message in sent)
        projected = estimated + max_tokens
        if projected < window * CONTEXT_WARN_FRACTION:
            return
        with self._context_lock:
            if self._context_warned:
                return
            self._context_warned = True
        logger.warning(
            "prompt budget is tight: role %r is an estimated ~%d input tokens + up to %d "
            "output = ~%d, about %.0f%% of the %d-token context window. Approaching truncation "
            "-- lower context.glossary_terms / before_units / after_units / reference_examples, "
            "shorten the instructions, or raise the server's context size (and --context-window "
            "to match). Further budget warnings this run are suppressed.",
            role, estimated, max_tokens, projected, 100 * projected / window, window)

    def _server_error_reason(self, schema: dict[str, Any] | None, body: str) -> str:
        """A request-rejection message that names the likely cause when it is known.

        The single most common rejection in practice is a model whose build cannot compile
        a strict schema grammar (Bielik: "Failed to initialize samplers"). Rather than
        degrade silently to another strategy -- which would hide the mismatch -- the client
        fails clearly and points at the fix: a ``json_object`` backend. Diagnosis only; no
        behaviour changes.
        """
        base = "request rejected"
        lowered = body.lower()
        # Context overflow is a prompt-size problem, independent of the decoding strategy, so
        # it is diagnosed before the schema-specific branch and for every backend. The prompt
        # is assembled from configuration, so the fix is a configuration one: point at the
        # knobs that shrink it, or the server setting that enlarges the window.
        if ("exceed_context" in lowered or "context size" in lowered
                or "context window" in lowered or "n_ctx" in lowered):
            return (f"{base}: the prompt is larger than the server's context window. Shorten "
                    f"the translator/reviewer instructions in the agent config, lower "
                    f"context.glossary_terms or context.before_units / after_units, or raise "
                    f"the server's context size (llama.cpp -c / --ctx-size, and match "
                    f"--parallel so each slot still fits)")
        # The schema hint applies whenever the request was schema-constrained -- to every
        # SchemaBackend (generic, qwen, ...), not just the one named "generic", since any of
        # them can hit the grammar rejection.
        if schema is None or not isinstance(self.backend, SchemaBackend):
            return base
        if "sampler" in lowered or "grammar" in lowered or "json_schema" in lowered:
            hint = suggest_backend(self.config.model) or "bielik"
            return (f"{base}: the server rejected schema-constrained decoding, which some "
                    f"model builds cannot compile. Retry with a json_object backend "
                    f"(--backend {hint})")
        return base

    def complete_json(
        self,
        messages: Sequence[Message],
        schema: dict[str, Any],
        *,
        role: str = "<none>",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Chat completion that must yield a JSON object matching ``schema``.

        The reply is validated against the schema's top-level shape -- it is a JSON object, it
        carries every required field, and each present field has the declared JSON type -- so a
        ``json_object`` backend (whose server does not constrain decoding to the schema) is held
        to the same shape as a grammar-constrained one. A reply that fails is raised as an
        :class:`LlmContentError`, which the harness records against this unit rather than
        shipping. Nested structure beyond the top level is not checked; the schemas in use are
        flat.
        """
        text = self.complete(messages, role=role, temperature=temperature,
                             max_tokens=max_tokens, schema=schema)
        stripped = _strip_code_fence(text)
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            recovered = _close_truncated_json_object(stripped)
            if recovered is not None:
                try:
                    _check_schema_shape(recovered, schema, role=role, body=text)
                except LlmContentError:
                    pass  # Doesn't satisfy the schema either -- fall through to the plain error.
                else:
                    raise LlmIncompleteJsonError(
                        f"response JSON envelope was truncated: {exc}", role=role, body=text,
                        recovered=recovered) from exc
            raise LlmContentError(f"response was not valid JSON: {exc}",
                                  role=role, body=text) from exc
        _check_schema_shape(parsed, schema, role=role, body=text)
        return parsed


def _token_count(value: Any) -> int:
    """A token count from an untrusted usage block, or 0 if it is not a plain integer.

    bool is excluded because it is an int subclass and a ``true`` in the field would
    otherwise read as 1.
    """
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


_JSON_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def _json_type_matches(value: Any, declared: Any) -> bool:
    """Whether ``value`` has the JSON type ``declared`` names.

    ``declared`` may be a single type name or a JSON Schema union -- a list of names -- in
    which case the value matches if it matches any one of them (so ``["string", "null"]``
    accepts a string or null but still rejects a number). A type name this does not recognise
    is accepted rather than rejected, so the check tightens the common flat case without
    reimplementing JSON Schema.
    """
    if isinstance(declared, list):
        return any(_json_type_matches(value, item) for item in declared)
    check = _JSON_TYPE_CHECKS.get(declared) if isinstance(declared, str) else None
    return check is None or check(value)


def _check_schema_shape(parsed: Any, schema: dict[str, Any], *, role: str, body: str) -> None:
    """Raise :class:`LlmContentError` unless ``parsed`` is an object matching ``schema``'s top-level
    shape. Shared by the ordinary parse path and the truncated-envelope recovery path, so a
    recovered object is held to exactly the same standard as one that parsed cleanly."""
    if not isinstance(parsed, dict):
        raise LlmContentError(f"expected a JSON object, got {type(parsed).__name__}",
                              role=role, body=body)
    missing = set(schema.get("required", [])) - parsed.keys()
    if missing:
        raise LlmContentError(f"response missing required fields {sorted(missing)}",
                              role=role, body=body)
    mistyped = [
        f"{name!r} should be {spec['type']}, got {type(parsed[name]).__name__}"
        for name, spec in schema.get("properties", {}).items()
        if name in parsed and "type" in spec
        and not _json_type_matches(parsed[name], spec["type"])
    ]
    if mistyped:
        raise LlmContentError(f"response fields have the wrong type: {'; '.join(mistyped)}",
                              role=role, body=body)


def _close_truncated_json_object(text: str) -> dict[str, Any] | None:
    """Recover a flat JSON object from a reply that stopped one or two characters short of a
    valid envelope. **The recovered value is not known to be complete** -- see the warning below.

    Two suffixes are tried, in this order, and ``None`` is returned unless one of them closes the
    text into a strictly valid JSON object:

    - ``}`` -- the value string closed with its own ``"``, only the object's brace is missing.
    - ``"}`` -- the string's closing quote is missing too, so the value ends wherever the model
      stopped emitting characters.

    **Neither suffix tells you the content is finished, and the second one actively hides that it
    may not be.** Appending ``"`` closes the string at an arbitrary cut point, so a value severed
    mid-clause (``{"translation": "The cat sat on the``) yields exactly as valid an object as one
    the model genuinely finished -- both were observed live, on the same unit, in the findings
    recorded in ``docs/feature-requests/json-envelope-truncation-repair.md`` (one reply ended on a
    proper sentence, another on ``"...żebyś mógł"`` with no completing verb). At this layer the two
    are *indistinguishable*: nothing in the text says which happened, and no syntactic heuristic
    can tell them apart (the same findings record a reply ending in a literal ``"..."`` the model
    itself wrote, which defeats both an "ends in punctuation" and an "ends in an ellipsis" test).

    That indistinguishability is precisely why the caller must never trust the result. Recovery is
    reported by raising :class:`LlmIncompleteJsonError`, never by returning a value into the normal
    path, and ``agents.py::_generate`` is obliged to treat ``.recovered`` as an unverified
    *candidate*: re-check it mechanically, re-review it, spend a repair round trying to get an
    ordinary reply first, and refuse to certify it as ``VERIFIED``. Removing any part of that
    handling on the belief that this function only recovers complete content would ship silently
    truncated translations.

    Deliberately not a general "make this JSON parse" repair: unbalanced brackets, a truncated
    nested container, or a trailing comma all still fail both attempts (closing one level cannot
    repair a break more than one token short), and this function returns ``None`` for every one of
    them rather than guessing further.
    """
    for suffix in ("}", '"}'):
        try:
            candidate = json.loads(text + suffix)
        except json.JSONDecodeError:
            continue
        # No isinstance narrowing: both suffixes end in "}", and the only JSON document that
        # can end in "}" is an object -- so a successful parse here is always a dict.
        return cast("dict[str, Any]", candidate)
    return None


def _strip_code_fence(text: str) -> str:
    """Tolerate models that wrap JSON in a markdown fence despite instructions.

    An opening ``` is unambiguous on its own, so the opening line is always dropped and the
    closing fence only when it is actually there. Requiring the closing fence would strand the
    fenced-*and*-truncated reply -- an unclosed fence is exactly what one looks like -- leaving
    the fence attached, the payload unparseable for a reason that has nothing to do with the
    envelope, and :func:`_close_truncated_json_object` never reached.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) == 1:
        # A one-line fence, e.g. ```{...}``` -- drop the opening and any trailing fence.
        inner = stripped[3:]
        return (inner[:-3] if inner.endswith("```") else inner).strip()
    body = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
    return "\n".join(body)
