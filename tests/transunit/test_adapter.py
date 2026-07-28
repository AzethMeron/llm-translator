"""Tests for :mod:`transunit.adapter`.

A carrier adapter is located by import, not a registry, so any importable package exposing
``<name>.adapter`` with ``name`` and ``sanitize_payload`` works. These tests use the
``fakecarrier`` fixture (on ``sys.path`` via ``tests/conftest.py``) for the happy path and
cover the two failure modes -- the package cannot be imported, or it imports but lacks the
required ``sanitize_payload`` -- both of which must surface as an actionable
:class:`AdapterError`.
"""
from __future__ import annotations

import types

import pytest

from transunit import adapter as adapter_module
from transunit.adapter import Adapter, AdapterError, load_adapter, passthrough


class TestLoadAdapter:
    def test_loads_the_fake_carrier_as_an_adapter(self) -> None:
        module = load_adapter("fakecarrier")
        assert isinstance(module, Adapter)
        assert module.name == "fakecarrier"

    def test_loaded_adapter_sanitises_payloads(self) -> None:
        """The same load that satisfies the protocol must expose a working sanitiser: the
        fake carrier collapses a raw newline to the literal two characters it can hold."""
        module = load_adapter("fakecarrier")
        assert module.sanitize_payload("a\nb") == "a\\nb"

    def test_missing_package_raises_naming_the_module(self) -> None:
        """The error has to point at the exact import that failed, so an operator knows
        which package to provide rather than guessing."""
        with pytest.raises(AdapterError) as excinfo:
            load_adapter("definitely_not_a_real_carrier")
        message = str(excinfo.value)
        assert "definitely_not_a_real_carrier" in message

    def test_package_without_sanitize_payload_raises(self) -> None:
        """A stdlib name with no ``adapter`` submodule exercises the failure path: an
        importable name that does not resolve to a usable adapter must be rejected, not
        returned half-formed."""
        with pytest.raises(AdapterError):
            load_adapter("json")

    def test_module_lacking_sanitize_payload_is_rejected(self, monkeypatch) -> None:
        """Directly cover the 'imports but is not an adapter' branch: a resolved module
        missing ``sanitize_payload`` cannot tell a translator what text the carrier holds,
        so loading it must fail with an actionable message rather than succeed."""
        incomplete = types.SimpleNamespace(name="incomplete")
        monkeypatch.setattr(adapter_module.importlib, "import_module",
                            lambda _name: incomplete)
        with pytest.raises(AdapterError) as excinfo:
            load_adapter("incomplete")
        assert "sanitize_payload" in str(excinfo.value)


class TestPassthrough:
    def test_is_the_identity_function(self) -> None:
        assert passthrough("unchanged text") == "unchanged text"

    def test_empty_string_passes_through(self) -> None:
        assert passthrough("") == ""

    def test_preserves_newlines_that_a_real_sanitiser_would_touch(self) -> None:
        """passthrough is a convenience for a run with no adapter; it deliberately does
        NOT normalise, so a raw newline survives it untouched."""
        assert passthrough("a\nb") == "a\nb"
