"""What a *carrier adapter* must provide to a translator, and how one is located.

A carrier adapter turns some source -- a subtitle track, a game's script files, a
document -- into translation units and writes translated units back. This is the
translator's *downstream* boundary, and it is deliberately not called an "LLM adapter":
the model-facing side lives in :mod:`translator.backend` and is a *backend*, a separate
concept. See ``docs/writing-a-carrier-adapter.md``.

Almost nothing crosses this boundary. A translator reads units and writes them back; it
never learns what a placeholder contained or where the text came from. The one thing it
genuinely needs from the adapter is *normalisation*, because "what text can this carrier
actually hold" is knowledge only the adapter has -- a subtitle cue cannot carry a line
break the layout has not chosen, an engine line cannot carry a raw newline, and another
carrier draws that line elsewhere.

Passing the sanitiser in rather than importing it is what keeps the translator
carrier-agnostic. Importing one adapter's sanitiser directly would tie the translator to
a single medium and defeat the split.
"""
from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class Adapter(Protocol):
    """A carrier-specific normaliser, seen from the translator's side."""

    name: str

    def sanitize_payload(self, text: str) -> str:
        """Normalise a translation into text this carrier can hold.

        Must be idempotent and information-preserving: it corrects defects that have
        exactly one right answer, and leaves anything requiring judgement alone.
        """


class AdapterError(Exception):
    """The named adapter could not be located or does not satisfy the protocol."""


def load_adapter(name: str) -> Adapter:
    """Import the adapter package ``name`` and return its ``adapter`` module.

    Located by import rather than a registry so a new carrier needs no edit here: any
    importable package exposing ``<name>.adapter`` with ``name`` and ``sanitize_payload``
    works.
    """
    try:
        module = importlib.import_module(f"{name}.adapter")
    except ImportError as exc:
        raise AdapterError(
            f"no adapter named {name!r}: could not import {name}.adapter ({exc}). "
            f"An adapter is a package exposing an `adapter` module with `name` and "
            f"`sanitize_payload`.") from exc
    if not hasattr(module, "sanitize_payload"):
        raise AdapterError(
            f"{name}.adapter does not define sanitize_payload, so it cannot tell a "
            f"translator what text this carrier can hold")
    return module  # type: ignore[return-value]


def passthrough(text: str) -> str:
    """A sanitiser that changes nothing.

    For a translator run with no adapter -- a plain corpus, or a test. Rendering is where
    correctness is enforced, so this is a convenience, never a licence to skip the
    adapter's own checks.
    """
    return text
