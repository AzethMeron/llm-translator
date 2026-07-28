"""Translations already produced, looked up by masked source text.

Two things need this. Duplicate text is translated once and shared, and a neighbouring line
that has already been translated is shown to the model *with* its translation, not only in the
source it was extracted as.

The second matters because the reader sees the target language. A unit's ``context_before``
and ``context_after`` are the *source* text of its neighbours, captured at extraction, so a
model asked to keep a scene consistent was being shown the one language the reader will
never see. Terminology, tense and pronouns drift between adjacent lines as a direct
result, which is exactly what the continuity rules are supposed to prevent.

Where that translation appears in the prompt is :mod:`translator.context_packing`'s business,
not this module's: a neighbour's source is part of the continuous surrounding-context passage,
while the pairs looked up here are listed separately as established renderings. This store only
answers "has this exact masked source already been translated, and to what".

Source text is a sound key here for the same reason it is sound for grouping duplicates:
it is the masked payload, so two units share a key only when they present the model with
the identical problem.
"""
from __future__ import annotations

import threading
from collections.abc import Iterable

from transunit.units import Status, Unit


class TranslationMemory:
    """Accepted translations, keyed by the masked source that produced them.

    Safe to share across worker threads: the runner records from its own thread while
    workers read, and both go through the lock.
    """

    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self._by_source: dict[str, str] = dict(initial or {})
        self._lock = threading.Lock()

    @classmethod
    def from_units(cls, units: Iterable[Unit]) -> TranslationMemory:
        """Build from previously journalled results.

        Only VERIFIED and TRANSLATED units contribute. A REJECTED unit's text failed its
        checks, so offering it as context would spread a known-bad rendering.
        """
        memory = cls()
        for unit in units:
            if unit.target and unit.status in (Status.VERIFIED, Status.TRANSLATED):
                memory.record(unit.source, unit.target)
        return memory

    def get(self, source: str) -> str | None:
        with self._lock:
            return self._by_source.get(source)

    def record(self, source: str, target: str) -> None:
        with self._lock:
            self._by_source[source] = target

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_source)
