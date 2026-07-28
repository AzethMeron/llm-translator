"""The adapter module for the fake carrier, satisfying :class:`transunit.adapter.Adapter`.

Its ``sanitize_payload`` collapses a raw newline to the two-character escape ``\\n`` and is
idempotent and information-preserving -- the properties a real sanitiser must have -- so the
same tests that check the protocol also check that a translator wired to an adapter applies
its normalisation.
"""
from __future__ import annotations

name = "fakecarrier"


def sanitize_payload(text: str) -> str:
    """Normalise a translation into text this fake carrier can hold.

    A raw newline would split this carrier's record, so it becomes the literal two
    characters backslash-n. Idempotent: a payload already carrying ``\\n`` is unchanged.
    """
    return text.replace("\r\n", "\n").replace("\n", "\\n")
