"""Tests for LlmClient, driven by an in-memory httpx transport (never a real server).

The client owns the transport concerns -- retry, the error taxonomy organised by blast
radius, and routing a structured request through the active backend -- so these tests pin
those without a GPU or a network. Retry backoff is set to zero so the suite stays fast.
"""
from __future__ import annotations

import json
import logging

import httpx
import pytest

from translator.backend import (
    LlmClient,
    LlmContentError,
    LlmError,
    LlmIncompleteJsonError,
    LlmRefusalError,
    LlmTruncationError,
    Message,
    ServerConfig,
    UsageStats,
    get_backend,
)
from translator.backend.client import (
    _close_truncated_json_object,
    _estimate_tokens,
    _strip_code_fence,
)

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["translation"],
    "properties": {"translation": {"type": "string"}},
}

MESSAGES = [Message("system", "you translate"), Message("user", "hello")]


def completion(content, *, finish_reason="stop", usage=None):
    """A well-formed chat-completion response body."""
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 5, "completion_tokens": 7},
    }


def make_client(handler, *, config=None, backend=None):
    """An LlmClient whose socket is replaced by a MockTransport running ``handler``."""
    client = LlmClient(config or ServerConfig(retry_backoff_seconds=0.0, max_retries=3),
                       backend=backend)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


class TestHealth:
    def test_health_true_when_models_endpoint_answers_200(self) -> None:
        client = make_client(lambda request: httpx.Response(200, json={"data": []}))
        assert client.health() is True
        client.close()

    def test_health_false_on_transport_error(self) -> None:
        def handler(request):
            raise httpx.ConnectError("no server", request=request)

        client = make_client(handler)
        assert client.health() is False
        client.close()

    def test_health_false_on_non_200(self) -> None:
        client = make_client(lambda request: httpx.Response(503))
        assert client.health() is False
        client.close()


class TestHappyPath:
    def test_returns_the_message_content(self) -> None:
        client = make_client(lambda request: httpx.Response(200, json=completion("cześć")))
        assert client.complete(MESSAGES, role="translate") == "cześć"
        client.close()

    def test_usage_is_accumulated(self) -> None:
        client = make_client(lambda request: httpx.Response(200, json=completion("x")))
        client.complete(MESSAGES)
        client.complete(MESSAGES)
        assert client.stats.requests == 2
        assert client.stats.completion_tokens == 14
        client.close()

    def test_a_null_usage_field_does_not_sink_a_successful_completion(self) -> None:
        # Regression: int(None) on a server's null token count used to raise an uncaught
        # TypeError, crashing a translation that had actually succeeded. Accounting is
        # diagnostic; it degrades to zero, it does not throw.
        body = completion("cześć", usage={"prompt_tokens": None, "completion_tokens": "x"})
        client = make_client(lambda request: httpx.Response(200, json=body))
        assert client.complete(MESSAGES) == "cześć"
        assert client.stats.completion_tokens == 0
        assert client.stats.requests == 1
        client.close()

    def test_stop_sequences_are_sent_when_given(self) -> None:
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=completion("x"))

        client = make_client(handler)
        client.complete(MESSAGES, stop=["</s>"])
        assert seen["stop"] == ["</s>"]
        client.close()


class TestReasoningSuppression:
    """Chain-of-thought is suppressed by default so a reasoning model's hidden reasoning
    tokens cannot overrun the per-call token budget and truncate the answer."""

    @staticmethod
    def _seen_body(config):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=completion("x"))

        client = make_client(handler, config=config)
        client.complete(MESSAGES, role="translate")
        client.close()
        return seen

    def test_reasoning_is_suppressed_by_default(self) -> None:
        body = self._seen_body(ServerConfig(retry_backoff_seconds=0.0))
        assert body["chat_template_kwargs"] == {"enable_thinking": False}

    def test_enable_reasoning_omits_the_suppression_kwarg(self) -> None:
        body = self._seen_body(
            ServerConfig(retry_backoff_seconds=0.0, enable_reasoning=True))
        assert "chat_template_kwargs" not in body

    def test_suppression_applies_to_structured_requests_too(self) -> None:
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=completion('{"translation": "x"}'))

        client = make_client(handler, config=ServerConfig(retry_backoff_seconds=0.0),
                             backend=get_backend("qwen"))
        client.complete_json(MESSAGES, SCHEMA, role="translate")
        client.close()
        assert seen["chat_template_kwargs"] == {"enable_thinking": False}
        assert seen["response_format"]["type"] == "json_schema"


class TestStructuredRequestRouting:
    """The backend, not the client, decides how structured output is requested."""

    def _sent_payload(self, backend_name):
        captured = {}

        def handler(request):
            captured.update(json.loads(request.content))
            return httpx.Response(200, json=completion('{"translation": "ok"}'))

        client = make_client(handler, backend=get_backend(backend_name))
        client.complete(MESSAGES, schema=SCHEMA)
        client.close()
        return captured

    def test_schema_backend_sends_strict_json_schema_and_untouched_messages(self) -> None:
        payload = self._sent_payload("generic")
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert payload["messages"][-1]["content"] == "hello"

    def test_json_object_backend_sends_json_object_and_appends_the_shape(self) -> None:
        payload = self._sent_payload("bielik")
        assert payload["response_format"] == {"type": "json_object"}
        assert "JSON object" in payload["messages"][-1]["content"]

    def test_no_response_format_without_a_schema(self) -> None:
        captured = {}

        def handler(request):
            captured.update(json.loads(request.content))
            return httpx.Response(200, json=completion("plain"))

        client = make_client(handler)
        client.complete(MESSAGES)
        assert "response_format" not in captured
        client.close()


class TestRetry:
    def test_a_5xx_is_retried_and_can_then_succeed(self) -> None:
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(500, text="overloaded")
            return httpx.Response(200, json=completion("recovered"))

        client = make_client(handler)
        assert client.complete(MESSAGES) == "recovered"
        assert calls["n"] == 2
        assert client.stats.retries == 1
        client.close()

    def test_a_transport_failure_is_retried_then_surfaces_as_llmerror(self) -> None:
        def handler(request):
            raise httpx.ConnectError("boom", request=request)

        client = make_client(handler, config=ServerConfig(retry_backoff_seconds=0.0,
                                                          max_retries=2))
        with pytest.raises(LlmError, match="transport failure"):
            client.complete(MESSAGES, role="translate")
        client.close()

    def test_a_malformed_body_is_retried_then_surfaces(self) -> None:
        client = make_client(lambda request: httpx.Response(200, text="not json at all"))
        with pytest.raises(LlmError, match="malformed response"):
            client.complete(MESSAGES)
        client.close()


class TestErrorTaxonomy:
    """The type carries the blast radius the caller reacts to."""

    def test_a_4xx_is_not_retried_and_raises_immediately(self) -> None:
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(400, text="bad request")

        client = make_client(handler)
        with pytest.raises(LlmError) as info:
            client.complete(MESSAGES, role="translate")
        assert info.value.status == 400
        assert calls["n"] == 1  # a client error cannot be fixed by retrying
        client.close()

    def test_a_schema_rejection_points_at_the_json_object_backend(self) -> None:
        # Bielik's failure: the default backend asks for a grammar the build cannot compile.
        # The client must name the fix, not degrade silently.
        def handler(request):
            return httpx.Response(400, text="Failed to initialize samplers")

        client = make_client(handler, config=ServerConfig(retry_backoff_seconds=0.0,
                                                          model="bielik-11b"))
        with pytest.raises(LlmError, match="--backend"):
            client.complete(MESSAGES, schema=SCHEMA, role="translate")
        client.close()

    def test_context_overflow_points_at_the_config_knobs(self) -> None:
        # A prompt larger than the window is a configuration problem, not a decoding one, so
        # the client names the knobs that shrink it (or the server setting that enlarges the
        # window) rather than passing back the raw server text alone.
        overflow = ('{"error":{"code":400,"message":"request (25295 tokens) exceeds the '
                    'available context size (4096 tokens)","type":"exceed_context_size_error",'
                    '"n_ctx":4096}}')
        client = make_client(lambda request: httpx.Response(400, text=overflow),
                             config=ServerConfig(retry_backoff_seconds=0.0))
        with pytest.raises(LlmError, match="context window") as info:
            client.complete(MESSAGES, schema=SCHEMA, role="translate")
        assert "glossary_terms" in str(info.value)
        assert info.value.status == 400
        client.close()

    def test_context_overflow_is_diagnosed_even_without_a_schema(self) -> None:
        # Overflow is independent of structured output, so the hint fires for a plain request
        # too -- unlike the schema-rejection hint, which is schema-specific.
        client = make_client(
            lambda request: httpx.Response(400, text="exceeds the available context size"),
            config=ServerConfig(retry_backoff_seconds=0.0))
        with pytest.raises(LlmError, match="context window"):
            client.complete(MESSAGES, role="translate")
        client.close()

    def test_truncation_is_its_own_error(self) -> None:
        client = make_client(
            lambda request: httpx.Response(200, json=completion("half a sen",
                                                                finish_reason="length")))
        with pytest.raises(LlmTruncationError, match="ceiling"):
            client.complete(MESSAGES, max_tokens=16)
        client.close()

    def test_null_content_is_a_refusal_and_is_counted(self) -> None:
        client = make_client(lambda request: httpx.Response(200, json=completion(None)))
        with pytest.raises(LlmRefusalError):
            client.complete(MESSAGES)
        assert client.stats.refusals == 1
        client.close()

    def test_refusal_is_a_content_error_by_type(self) -> None:
        # A refusal is per-input, so the caller should be able to catch it as LlmContentError
        # and carry on rather than aborting the whole run.
        assert issubclass(LlmRefusalError, LlmContentError)
        assert issubclass(LlmTruncationError, LlmContentError)
        assert issubclass(LlmContentError, LlmError)


class TestCompleteJson:
    def test_parses_a_valid_object(self) -> None:
        client = make_client(
            lambda request: httpx.Response(200, json=completion('{"translation": "cześć"}')))
        assert client.complete_json(MESSAGES, SCHEMA) == {"translation": "cześć"}
        client.close()

    def test_a_markdown_fence_is_tolerated(self) -> None:
        fenced = "```json\n{\"translation\": \"ok\"}\n```"
        client = make_client(lambda request: httpx.Response(200, json=completion(fenced)))
        assert client.complete_json(MESSAGES, SCHEMA) == {"translation": "ok"}
        client.close()

    def test_invalid_json_is_a_content_error(self) -> None:
        client = make_client(
            lambda request: httpx.Response(200, json=completion("{not valid")))
        with pytest.raises(LlmContentError, match="not valid JSON"):
            client.complete_json(MESSAGES, SCHEMA)
        client.close()

    def test_a_json_array_is_rejected_as_not_an_object(self) -> None:
        client = make_client(lambda request: httpx.Response(200, json=completion("[1, 2]")))
        with pytest.raises(LlmContentError, match="expected a JSON object"):
            client.complete_json(MESSAGES, SCHEMA)
        client.close()

    def test_a_missing_required_field_is_a_content_error(self) -> None:
        client = make_client(
            lambda request: httpx.Response(200, json=completion('{"other": "x"}')))
        with pytest.raises(LlmContentError, match="missing required fields"):
            client.complete_json(MESSAGES, SCHEMA)
        client.close()

    def test_a_nullable_field_accepts_null(self) -> None:
        # Regression: a reviewer that accepts naturally returns improved_translation: null.
        # The real REVIEW_SCHEMA types it as ["string", "null"], so the review must validate
        # (and complete) rather than be rejected as malformed and silently skip the panel.
        from translator.agents import REVIEW_SCHEMA

        body = '{"acceptable": true, "issues": [], "improved_translation": null}'
        client = make_client(lambda request: httpx.Response(200, json=completion(body)),
                             backend=get_backend("bielik"))
        result = client.complete_json(MESSAGES, REVIEW_SCHEMA, role="accuracy")
        assert result["acceptable"] is True
        assert result["improved_translation"] is None
        client.close()

    def test_null_issues_is_accepted_by_the_review_schema(self) -> None:
        # Regression: a reviewer that accepts returns issues: null (json_object encoding of an
        # empty list). REVIEW_SCHEMA now types issues as ["array", "null"], so the reply
        # validates instead of failing and (via the panel) skipping the other reviewers.
        from translator.agents import REVIEW_SCHEMA

        body = '{"acceptable": true, "issues": null}'
        client = make_client(lambda request: httpx.Response(200, json=completion(body)),
                             backend=get_backend("bielik"))
        result = client.complete_json(MESSAGES, REVIEW_SCHEMA, role="accuracy")
        assert result["issues"] is None
        client.close()

    def test_a_union_typed_field_still_rejects_a_wrong_type(self) -> None:
        # The union stays meaningful: string-or-null accepts those two, not a number.
        from translator.agents import REVIEW_SCHEMA

        body = '{"acceptable": true, "issues": [], "improved_translation": 7}'
        client = make_client(lambda request: httpx.Response(200, json=completion(body)),
                             backend=get_backend("bielik"))
        with pytest.raises(LlmContentError, match="wrong type"):
            client.complete_json(MESSAGES, REVIEW_SCHEMA, role="accuracy")
        client.close()


class TestCloseTruncatedJsonObject:
    """Unit tests for the pure recovery function -- see
    docs/feature-requests/json-envelope-truncation-repair.md for the verified failure shapes this
    targets."""

    def test_recovers_a_string_closed_but_object_not_closed(self) -> None:
        # Shape A (the report's own evidence): the value string closed correctly, only the
        # object's closing brace never arrived.
        assert _close_truncated_json_object('{"translation": "cześć"') == {"translation": "cześć"}

    def test_recovers_a_string_and_object_both_unclosed(self) -> None:
        # Shape B (found live in this session's reproduction, not in the original report): the
        # model's own nested quoting inside the value swallowed the string's own closing quote,
        # so a bare "}" alone still leaves the string unterminated -- only appending '"}' closes it.
        assert (_close_truncated_json_object("{\"translation\": \"abc 'def'}")
               == {"translation": "abc 'def'}"})

    def test_brace_only_is_tried_before_quote_and_brace(self) -> None:
        # A body that already happens to close strictly with a bare "}" must not additionally
        # have a quote appended -- that would corrupt an otherwise-correct recovery.
        assert _close_truncated_json_object('{"translation": "ok"') == {"translation": "ok"}

    def test_a_truncated_nested_array_is_not_recoverable(self) -> None:
        # Closing one level cannot repair a break more than one token short -- this needs two
        # closing tokens (`]` and `}`), which is outside the narrow shape this function targets.
        assert _close_truncated_json_object('{"translation": "x", "other": [1, 2') is None

    def test_a_trailing_comma_is_not_recoverable(self) -> None:
        assert _close_truncated_json_object('{"translation": "x",') is None

    def test_completely_unbalanced_garbage_is_not_recoverable(self) -> None:
        assert _close_truncated_json_object("{not valid") is None

    def test_a_truncated_array_at_top_level_is_not_recoverable(self) -> None:
        # Only a flat object is recovered -- a top-level array, even a cleanly truncated one, is
        # out of scope.
        assert _close_truncated_json_object("[1, 2") is None

    def test_empty_text_is_not_recoverable(self) -> None:
        assert _close_truncated_json_object("") is None


class TestCompleteJsonEnvelopeRepair:
    """complete_json's integration of the recovery path: raised (not returned), schema-checked
    exactly like an ordinary reply, and never touched when the reply already parses."""

    def test_brace_only_truncation_raises_incomplete_json_with_the_recovered_object(self) -> None:
        client = make_client(
            lambda request: httpx.Response(200, json=completion('{"translation": "cześć"')))
        with pytest.raises(LlmIncompleteJsonError) as excinfo:
            client.complete_json(MESSAGES, SCHEMA)
        assert excinfo.value.recovered == {"translation": "cześć"}
        client.close()

    def test_quote_and_brace_truncation_also_recovers(self) -> None:
        body = "{\"translation\": \"abc 'def'}"
        client = make_client(lambda request: httpx.Response(200, json=completion(body)))
        with pytest.raises(LlmIncompleteJsonError) as excinfo:
            client.complete_json(MESSAGES, SCHEMA)
        assert excinfo.value.recovered == {"translation": "abc 'def'}"}
        client.close()

    def test_llm_incomplete_json_error_is_a_content_error(self) -> None:
        # A caller with a bare `except LlmContentError` (today's every caller) must keep catching
        # this without any change -- opting into the recovered payload is additive, not required.
        client = make_client(
            lambda request: httpx.Response(200, json=completion('{"translation": "cześć"')))
        with pytest.raises(LlmContentError):
            client.complete_json(MESSAGES, SCHEMA)
        client.close()

    def test_a_recovered_object_still_fails_a_missing_required_field(self) -> None:
        # The recovered candidate is held to the same schema as an ordinary reply: closing the
        # brace does not manufacture a required field the model never produced. Falls through to
        # the plain "not valid JSON" error rather than a schema-specific one, since the envelope
        # genuinely was never usable.
        client = make_client(
            lambda request: httpx.Response(200, json=completion('{"other": "x"')))
        with pytest.raises(LlmContentError, match="not valid JSON") as excinfo:
            client.complete_json(MESSAGES, SCHEMA)
        assert not isinstance(excinfo.value, LlmIncompleteJsonError)
        client.close()

    def test_a_recovered_object_still_fails_a_wrong_typed_field(self) -> None:
        client = make_client(
            lambda request: httpx.Response(200, json=completion('{"translation": 7')))
        with pytest.raises(LlmContentError, match="not valid JSON") as excinfo:
            client.complete_json(MESSAGES, SCHEMA)
        assert not isinstance(excinfo.value, LlmIncompleteJsonError)
        client.close()

    def test_an_unrecoverable_body_stays_a_plain_content_error(self) -> None:
        # Nested truncation is outside the recovered shape -- unaffected by this change.
        client = make_client(
            lambda request: httpx.Response(
                200, json=completion('{"translation": "x", "other": [1, 2')))
        with pytest.raises(LlmContentError, match="not valid JSON") as excinfo:
            client.complete_json(MESSAGES, SCHEMA)
        assert not isinstance(excinfo.value, LlmIncompleteJsonError)
        client.close()

    def test_an_ordinary_valid_reply_never_touches_the_recovery_path(self) -> None:
        client = make_client(
            lambda request: httpx.Response(200, json=completion('{"translation": "cześć"}')))
        assert client.complete_json(MESSAGES, SCHEMA) == {"translation": "cześć"}
        client.close()

    def test_a_value_severed_mid_clause_recovers_just_as_cleanly_as_a_finished_one(self) -> None:
        # THE DANGER, pinned rather than left to prose. The '"}' suffix closes the string at
        # whatever character the model stopped on, so a value cut mid-clause -- the report's live
        # Polish case ended "...żebyś mógł" with no completing verb -- produces an object every
        # bit as valid as a finished sentence does. The recovered text is EXACTLY the truncated
        # text: nothing here completes it, flags it, or can tell it from a clean stop.
        #
        # This is why agents.py::_generate must keep treating .recovered as an unverified
        # candidate (re-checked, re-reviewed, never silently VERIFIED). Its own coverage lives in
        # tests/translator/test_agents.py::TestEnvelopeTruncationRepair; what this pins is that
        # the client layer hands that caller something it CANNOT trust.
        severed = '{"translation": "The cat sat on the'
        client = make_client(lambda request: httpx.Response(200, json=completion(severed)))
        with pytest.raises(LlmIncompleteJsonError) as excinfo:
            client.complete_json(MESSAGES, SCHEMA)
        assert excinfo.value.recovered == {"translation": "The cat sat on the"}
        client.close()

    def test_mid_clause_recovery_is_indistinguishable_from_a_finished_value(self) -> None:
        # The two live shapes side by side at the pure-function layer: same suffix, same success,
        # no signal separating them. Any future "is it complete?" heuristic must be built
        # downstream, not here -- and the doc's third live case (a model-authored trailing "...")
        # is evidence it cannot be built from syntax at all.
        assert (_close_truncated_json_object('{"translation": "...żebyś mógł')
                == {"translation": "...żebyś mógł"})
        assert (_close_truncated_json_object('{"translation": "Czy to jasne?')
                == {"translation": "Czy to jasne?"})


class TestFencedAndTruncated:
    """A markdown fence and a truncated envelope in both orders.

    Regression: an unclosed opening fence -- which is exactly what a truncated fenced reply looks
    like -- used to be returned with the fence still attached, so the payload never parsed, the
    envelope-repair path never fired, and the unit was lost as if the feature did not exist.
    """

    def test_a_fenced_truncated_reply_is_unfenced_and_then_repaired(self) -> None:
        body = '```json\n{"translation": "cześć"'
        client = make_client(lambda request: httpx.Response(200, json=completion(body)))
        with pytest.raises(LlmIncompleteJsonError) as excinfo:
            client.complete_json(MESSAGES, SCHEMA)
        assert excinfo.value.recovered == {"translation": "cześć"}
        client.close()

    def test_a_fenced_reply_truncated_mid_string_is_also_repaired(self) -> None:
        # Fence plus the quote+brace shape: both repairs compose, and the recovered value is
        # again exactly the truncated text.
        body = '```\n{"translation": "urwane w pół'
        client = make_client(lambda request: httpx.Response(200, json=completion(body)))
        with pytest.raises(LlmIncompleteJsonError) as excinfo:
            client.complete_json(MESSAGES, SCHEMA)
        assert excinfo.value.recovered == {"translation": "urwane w pół"}
        client.close()

    def test_a_closed_fence_around_truncated_json_is_repaired(self) -> None:
        # The other order: the model closed its fence but not its JSON.
        body = '```json\n{"translation": "ok"\n```'
        client = make_client(lambda request: httpx.Response(200, json=completion(body)))
        with pytest.raises(LlmIncompleteJsonError) as excinfo:
            client.complete_json(MESSAGES, SCHEMA)
        assert excinfo.value.recovered == {"translation": "ok"}
        client.close()

    def test_an_unclosed_fence_around_complete_json_parses_normally(self) -> None:
        body = '```json\n{"translation": "ok"}'
        client = make_client(lambda request: httpx.Response(200, json=completion(body)))
        assert client.complete_json(MESSAGES, SCHEMA) == {"translation": "ok"}
        client.close()

    @pytest.mark.parametrize("body", ['```json\n{"translation": "ok"}\n```',
                                      '{"translation": "ok"}'])
    def test_a_closed_fence_and_a_bare_reply_are_unaffected(self, body) -> None:
        client = make_client(lambda request: httpx.Response(200, json=completion(body)))
        assert client.complete_json(MESSAGES, SCHEMA) == {"translation": "ok"}
        client.close()

    def test_fenced_garbage_is_still_a_plain_content_error(self) -> None:
        # Unfencing an unclosed fence does not make unrecoverable content recoverable.
        body = '```json\n{"translation": "x", "other": [1, 2'
        client = make_client(lambda request: httpx.Response(200, json=completion(body)))
        with pytest.raises(LlmContentError, match="not valid JSON") as excinfo:
            client.complete_json(MESSAGES, SCHEMA)
        assert not isinstance(excinfo.value, LlmIncompleteJsonError)
        client.close()


class TestContextManager:
    def test_use_as_a_context_manager_closes_the_socket(self) -> None:
        with make_client(lambda request: httpx.Response(200, json=completion("x"))) as client:
            assert client.complete(MESSAGES) == "x"


class TestServerConfigValidation:
    def test_negative_backoff_is_refused(self) -> None:
        # Regression: a negative backoff makes time.sleep raise on the first retry -- a crash
        # on the error path. It is rejected at construction instead.
        with pytest.raises(ValueError, match="retry_backoff_seconds"):
            ServerConfig(retry_backoff_seconds=-1.0)

    def test_zero_backoff_is_allowed(self) -> None:
        assert ServerConfig(retry_backoff_seconds=0.0).retry_backoff_seconds == 0.0

    def test_max_retries_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            ServerConfig(max_retries=0)

    def test_non_positive_timeout_is_refused(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds"):
            ServerConfig(timeout_seconds=0)


class TestAuditFixes:
    """Regressions for the deep-audit round."""

    def test_schema_rejection_hint_also_fires_for_a_named_schema_backend(self) -> None:
        # Regression: the hint was gated on backend.name == "generic", so a qwen user (also a
        # SchemaBackend that sends json_schema) got no actionable message on the same rejection.
        def handler(request):
            return httpx.Response(400, text="Failed to initialize samplers")

        client = make_client(handler, config=ServerConfig(retry_backoff_seconds=0.0,
                                                          model="qwen3-14b"),
                             backend=get_backend("qwen"))
        with pytest.raises(LlmError, match="--backend"):
            client.complete(MESSAGES, schema=SCHEMA, role="translate")
        client.close()

    def test_complete_json_rejects_a_wrong_typed_field(self) -> None:
        # Regression: for a json_object backend the shape check is the ONLY guard, and it used
        # to check only that required keys were PRESENT. A translation returned as a number
        # would ship. Now the top-level type is validated.
        body = completion('{"translation": 123}')
        client = make_client(lambda request: httpx.Response(200, json=body))
        with pytest.raises(LlmContentError, match="wrong type"):
            client.complete_json(MESSAGES, SCHEMA)
        client.close()

    def test_complete_json_accepts_a_correctly_typed_field(self) -> None:
        body = completion('{"translation": "ok"}')
        client = make_client(lambda request: httpx.Response(200, json=body))
        assert client.complete_json(MESSAGES, SCHEMA) == {"translation": "ok"}
        client.close()

    def test_a_single_line_code_fence_is_unwrapped(self) -> None:
        body = completion('```{"translation": "ok"}```')
        client = make_client(lambda request: httpx.Response(200, json=body))
        assert client.complete_json(MESSAGES, SCHEMA) == {"translation": "ok"}
        client.close()


class TestTokenEstimate:
    def test_latin_runs_about_four_characters_per_token(self) -> None:
        assert _estimate_tokens("hello world") == pytest.approx(3, abs=1)

    def test_cjk_counts_about_one_token_per_character(self) -> None:
        # Full-width characters are ~1 token each, so the estimate must not divide them by four.
        assert _estimate_tokens("你好世界你好世界") == 8

    def test_empty_text_is_zero(self) -> None:
        assert _estimate_tokens("") == 0

    def test_the_estimate_errs_high_not_low(self) -> None:
        # A warning must fire before the wall, so a mixed string is counted at least as its
        # character-quarter, never under.
        text = "a mix of 拉丁 and 漢字 characters"
        assert _estimate_tokens(text) >= len(text) // 4


class TestPeakPromptTokens:
    def test_peak_tracks_the_largest_single_prompt(self) -> None:
        stats = UsageStats()
        for prompt in (100, 250, 80):
            stats.record({"prompt_tokens": prompt, "completion_tokens": 5}, 0.1)
        assert stats.peak_prompt_tokens == 250
        assert stats.prompt_tokens == 430


class TestContextBudgetWarning:
    def _ok(self):
        return lambda request: httpx.Response(200, json=completion('{"translation": "ok"}'))

    def _big_messages(self):
        return [Message("system", "s " * 400), Message("user", "u " * 400)]

    def test_negative_window_is_refused(self) -> None:
        with pytest.raises(ValueError):
            ServerConfig(context_window=-1)

    def test_warns_once_when_the_prompt_nears_the_window(self, caplog) -> None:
        logging.getLogger("translator").propagate = True
        client = make_client(
            self._ok(),
            config=ServerConfig(retry_backoff_seconds=0.0, context_window=400),
            backend=get_backend("generic"))
        with caplog.at_level(logging.WARNING, logger="translator.backend.client"):
            client.complete_json(self._big_messages(), SCHEMA, role="translate")
            client.complete_json(self._big_messages(), SCHEMA, role="accuracy")
        budget = [r for r in caplog.records if "prompt budget is tight" in r.getMessage()]
        assert len(budget) == 1  # once per run, not once per call
        assert "context window" in budget[0].getMessage()
        client.close()

    def test_no_warning_when_the_window_is_unset(self, caplog) -> None:
        logging.getLogger("translator").propagate = True
        client = make_client(
            self._ok(),
            config=ServerConfig(retry_backoff_seconds=0.0, context_window=0),
            backend=get_backend("generic"))
        with caplog.at_level(logging.WARNING, logger="translator.backend.client"):
            client.complete_json(self._big_messages(), SCHEMA, role="translate")
        assert not [r for r in caplog.records if "prompt budget" in r.getMessage()]
        client.close()

    def test_no_warning_comfortably_under_the_threshold(self, caplog) -> None:
        logging.getLogger("translator").propagate = True
        client = make_client(
            self._ok(),
            config=ServerConfig(retry_backoff_seconds=0.0, context_window=100_000),
            backend=get_backend("generic"))
        with caplog.at_level(logging.WARNING, logger="translator.backend.client"):
            client.complete_json([Message("user", "short")], SCHEMA, role="translate",
                                 max_tokens=64)
        assert not [r for r in caplog.records if "prompt budget" in r.getMessage()]
        client.close()


class TestServerErrorReasonAndFence:
    def _client(self):
        return make_client(lambda request: httpx.Response(200, json={"data": []}),
                           backend=get_backend("generic"))

    def test_error_reason_names_the_json_object_fix_on_a_grammar_rejection(self) -> None:
        client = self._client()
        message = client._server_error_reason(SCHEMA, "Failed to initialize samplers")
        assert "json_object" in message
        client.close()

    def test_error_reason_falls_back_to_base_when_the_cause_is_unknown(self) -> None:
        client = self._client()
        assert client._server_error_reason(SCHEMA, "some unrelated failure") == "request rejected"
        client.close()

    def test_strip_code_fence_removes_a_multiline_closed_fence(self) -> None:
        assert _strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_strip_code_fence_unwraps_an_unclosed_fence(self) -> None:
        # Regression: this used to return the text with the fence still attached, which meant a
        # reply that was both fenced and truncated (an unclosed fence is what that looks like)
        # never reached the envelope repair at all.
        assert _strip_code_fence('```\n{"a": 1}') == '{"a": 1}'
        assert _strip_code_fence('```json\n{"translation": "x"') == '{"translation": "x"'

    def test_strip_code_fence_keeps_a_multiline_payload_intact(self) -> None:
        assert _strip_code_fence('```json\n{\n"a": 1\n}\n```') == '{\n"a": 1\n}'
        assert _strip_code_fence('```json\n{\n"a": 1') == '{\n"a": 1'

    def test_strip_code_fence_leaves_unfenced_text_alone(self) -> None:
        assert _strip_code_fence('  {"a": 1}  ') == '{"a": 1}'
