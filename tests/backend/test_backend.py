"""Tests for the model-facing backend layer: request shaping and the registry.

The backend is the renamed, hardened "LLM adapter". Its whole reason to exist is that the
same structured request must be shaped differently per model -- grammar-constrained JSON for
Qwen, prose-described JSON for Bielik -- and that switching between them is a backend choice,
not a code change. These tests pin exactly that.
"""
from __future__ import annotations

import pytest

from translator.backend import (
    Backend,
    BackendError,
    JsonObjectBackend,
    Message,
    SchemaBackend,
    available_backends,
    get_backend,
    register_backend,
    resolve_backend,
    suggest_backend,
)
from translator.backend.core import _append_json_instruction

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["translation"],
    "properties": {
        "translation": {"type": "string"},
        "confidence": {"type": "number"},
    },
}


class TestBuiltInRegistry:
    def test_the_shipped_profiles_are_registered(self) -> None:
        assert set(available_backends()) == {"generic", "qwen", "bielik", "eurollm"}

    def test_generic_is_the_default_and_uses_schema_grammar(self) -> None:
        backend = get_backend("generic")
        assert isinstance(backend, SchemaBackend)

    def test_bielik_and_eurollm_use_json_object(self) -> None:
        assert isinstance(get_backend("bielik"), JsonObjectBackend)
        assert isinstance(get_backend("eurollm"), JsonObjectBackend)

    def test_aliases_resolve_to_the_same_instance(self) -> None:
        generic = get_backend("generic")
        assert get_backend("openai") is generic
        assert get_backend("json_schema") is generic

    def test_lookup_is_case_insensitive(self) -> None:
        assert get_backend("BIELIK") is get_backend("bielik")

    def test_unknown_backend_names_what_is_available(self) -> None:
        with pytest.raises(BackendError, match="unknown backend 'llama'"):
            get_backend("llama")


class TestSchemaBackend:
    """Grammar-constrained decoding: strongest constraint, messages untouched."""

    def test_response_format_is_strict_json_schema(self) -> None:
        request = SchemaBackend("x").structured_request(
            [Message("system", "s"), Message("user", "u")], SCHEMA)
        assert request.response_format == {
            "type": "json_schema",
            "json_schema": {"name": "response", "strict": True, "schema": SCHEMA},
        }

    def test_messages_pass_through_unchanged(self) -> None:
        messages = [Message("system", "s"), Message("user", "u")]
        request = SchemaBackend("x").structured_request(messages, SCHEMA)
        assert request.messages == tuple(messages)


class TestJsonObjectBackend:
    """Valid-JSON mode: shape described in the prompt because the grammar is unavailable."""

    def test_response_format_is_json_object(self) -> None:
        request = JsonObjectBackend("b").structured_request([Message("user", "u")], SCHEMA)
        assert request.response_format == {"type": "json_object"}

    def test_the_schema_shape_is_appended_to_the_last_user_turn(self) -> None:
        request = JsonObjectBackend("b").structured_request(
            [Message("system", "sys"), Message("user", "translate this")], SCHEMA)
        # System turn is left alone; the instruction rides on the final user turn.
        assert request.messages[0] == Message("system", "sys")
        last = request.messages[-1].content
        assert last.startswith("translate this")
        assert '"translation": <string>' in last
        # An optional field is marked so, so the model does not treat it as required.
        assert '"confidence": <number> (optional)' in last

    def test_a_union_type_renders_readably_not_as_a_python_list(self) -> None:
        # A nullable field is a union (a list) in the schema; the prompt must read it as
        # "string|null", not the Python repr "['string', 'null']" the model would puzzle over.
        from translator.agents import REVIEW_SCHEMA

        last = JsonObjectBackend("b").structured_request(
            [Message("user", "review this")], REVIEW_SCHEMA).messages[-1].content
        assert '"improved_translation": <string|null> (optional)' in last
        assert "['string'" not in last

    def test_the_prompt_steers_empty_values_away_from_null(self) -> None:
        # json_object models freely emit null for an empty field, which then fails the type
        # check; the instruction tells the model how to encode empties so it does not.
        last = JsonObjectBackend("b").structured_request(
            [Message("user", "u")], SCHEMA).messages[-1].content
        assert "[] for a list" in last
        assert '"" for a string' in last
        assert "use null only" in last

    def test_a_user_instruction_is_added_when_there_is_no_user_turn(self) -> None:
        request = JsonObjectBackend("b").structured_request([Message("system", "only")], SCHEMA)
        assert request.messages[-1].role == "user"
        assert "JSON object" in request.messages[-1].content

    def test_append_targets_the_last_user_turn_not_an_earlier_one(self) -> None:
        messages = [Message("user", "first"), Message("assistant", "a"),
                    Message("user", "second")]
        result = _append_json_instruction(messages, SCHEMA)
        assert result[0] == Message("user", "first")
        assert result[2].content.startswith("second")
        assert "JSON object" in result[2].content


class TestAutoResolution:
    """`auto` hardens a model switch: the request shape follows the model id."""

    def test_auto_picks_bielik_from_the_model_id(self) -> None:
        assert resolve_backend("auto", "Bielik-11B-v2.3-Instruct-Q5_K_M").name == "bielik"

    def test_auto_picks_qwen_from_the_model_id(self) -> None:
        assert resolve_backend("auto", "qwen3-14b").name == "qwen"

    def test_auto_picks_eurollm_from_the_model_id(self) -> None:
        assert resolve_backend("auto", "EuroLLM-9B-Instruct").name == "eurollm"

    def test_auto_falls_back_to_generic_when_nothing_matches(self) -> None:
        assert resolve_backend("auto", "some-unknown-model").name == "generic"

    def test_an_explicit_name_overrides_the_model_id(self) -> None:
        # Naming a backend beats guessing: a Qwen served under a generic alias is honoured.
        assert resolve_backend("bielik", "qwen3-14b").name == "bielik"

    def test_suggest_backend_returns_none_when_no_hint_matches(self) -> None:
        assert suggest_backend("mystery-model") is None


class TestRegistration:
    def test_a_custom_backend_registers_and_resolves(self) -> None:
        backend = SchemaBackend("mistral-test", model_hints=("mistral-test",))
        register_backend(backend)
        try:
            assert get_backend("mistral-test") is backend
            assert resolve_backend("auto", "mistral-test-7b").name == "mistral-test"
        finally:
            # Keep the shared registry clean for other tests.
            from translator.backend import core
            core._REGISTRY.pop("mistral-test", None)

    def test_reregistering_the_same_instance_is_a_noop(self) -> None:
        backend = get_backend("generic")
        register_backend(backend)  # must not raise
        assert get_backend("generic") is backend

    def test_a_name_collision_with_a_different_backend_is_refused(self) -> None:
        with pytest.raises(BackendError, match="already registered"):
            register_backend(SchemaBackend("generic"))

    def test_a_backend_needs_a_nonempty_name(self) -> None:
        with pytest.raises(BackendError):
            SchemaBackend("  ")


class TestBackendIsAbstract:
    def test_backend_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError):
            Backend("x")  # type: ignore[abstract]


class TestRegistryLoadingGuard:
    """The load guard is a `_loaded` flag, not registry-emptiness.

    Regression: the old `if not _REGISTRY` guard conflated "empty" with "not yet loaded", so
    registering a custom backend before the built-ins loaded (non-empty registry) would skip
    them, and `get_backend("generic")` would then fail in a fresh process. The fix keys the
    guard on a dedicated flag. This test pins that the flag -- not emptiness -- is consulted:
    with the flag already set, an empty registry does NOT trigger a (re)load, which is the
    exact opposite of the old behaviour.
    """

    def test_the_flag_short_circuits_even_with_an_empty_registry(self) -> None:
        from translator.backend import core
        saved_registry = dict(core._REGISTRY)
        saved_flag = core._profiles_loaded
        core._REGISTRY.clear()
        core._profiles_loaded = True  # "already loaded"
        try:
            core._ensure_profiles_loaded()  # old code would have reloaded on the empty registry
            assert core._REGISTRY == {}, "load must be gated on the flag, not on emptiness"
        finally:
            core._REGISTRY.clear()
            core._REGISTRY.update(saved_registry)
            core._profiles_loaded = saved_flag
