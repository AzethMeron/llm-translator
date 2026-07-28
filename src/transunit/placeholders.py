"""The placeholder convention shared by every adapter and the translator.

An adapter replaces the parts of a payload a translator must not touch -- engine
expressions, conditionals, escapes, inline markup -- with ``[[0]]``, ``[[1]]`` and so
on, and restores them afterwards. The translator only has to preserve the set; it never
learns what any of them contained, which is what lets one translator serve carriers
whose syntaxes have nothing in common.
"""
from __future__ import annotations

import re

PLACEHOLDER = re.compile(r"\[\[(\d+)\]\]")


def placeholder_indices(text: str) -> list[int]:
    """Every placeholder index referenced by ``text``, in order of appearance."""
    return [int(match.group(1)) for match in PLACEHOLDER.finditer(text)]
