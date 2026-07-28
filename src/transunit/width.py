"""How wide a piece of text is when a carrier draws it.

In the contract because a length constraint is meaningless unless both sides measure it
the same way. An adapter that knows a message box fits twenty columns and a translator
that counts characters will disagree the moment CJK is involved, and the disagreement is
invisible until text overflows in the running program.

Measured in half-width columns, the unit every fixed-cell renderer draws in: an ASCII
glyph occupies one, an East-Asian wide or full-width glyph two, and anything that draws no
cell of its own none. The width class is decided by :func:`unicodedata.east_asian_width`,
which covers the whole of Unicode -- including the supplementary-plane CJK extensions a
hand-rolled range list silently misses, exactly where a rare or historical ideograph in a
name would otherwise slip past a hard width budget.

The model is deliberately *context-free*: each character costs the same wherever it
appears, so width is additive over concatenation. See :func:`display_columns` for what that
buys, and for the one rendering effect it cannot express.
"""
from __future__ import annotations

import unicodedata

_WIDE = frozenset(("W", "F"))
"""East-Asian width classes that occupy two columns: Wide and Fullwidth."""

_ZERO_WIDTH_CATEGORIES = frozenset(("Mn", "Me", "Cf"))
"""General categories that draw no cell of their own.

``Mn``/``Me`` are the nonspacing and enclosing marks -- by definition they render onto the
preceding glyph, which also covers the ones with canonical combining class 0 (the variation
selectors U+FE0E/U+FE0F above all) that a ``unicodedata.combining`` test misses.

``Cf`` is the format class: the joiners and invisible markers -- ZWSP U+200B, ZWNJ U+200C,
ZWJ U+200D, WORD JOINER U+2060, SOFT HYPHEN U+00AD, the bidi marks and overrides
U+200E/U+200F/U+202A-U+202E, and BOM/ZWNBSP U+FEFF. They instruct the shaper; they draw
nothing. Counting them as one column each over-measured any line carrying them, and because
width feeds a *blocking* rule an over-measured line is rejected outright and the
untranslated source shown in its place.
"""

_ZERO_WIDTH_JAMO_RANGES = ((0x1160, 0x11FF), (0xD7B0, 0xD7FF))
"""Conjoining Hangul jamo that compose onto a preceding leading consonant.

A decomposed Korean syllable is L + V (+ T): the leading consonant carries the syllable's
cell, while the vowel and the trailing consonant are drawn into that same cell. They are
letters (category ``Lo``) with canonical combining class 0, so no general mark test sees
them, and NFD Korean measured DOUBLE its rendered width -- NFD ``'한'`` came to 4 columns
against NFC's 2, so a corpus arriving NFD was rejected wholesale as over-length.

The ranges cover the vowels (V) and trailing consonants (T) only: U+1160-U+11FF from the
base block, and U+D7B0-U+D7FF from Extended-B. The leading consonants -- U+1100-U+115F and
Extended-A U+A960-U+A97F -- are left alone, so they keep the two columns
:func:`unicodedata.east_asian_width` already gives them and the syllable still costs 2.
"""


def _is_composing_jamo(char: str) -> bool:
    """Whether ``char`` is a Hangul vowel or trailing consonant, drawn into the syllable
    started by the leading consonant before it."""
    code = ord(char)
    return any(low <= code <= high for low, high in _ZERO_WIDTH_JAMO_RANGES)


def _character_columns(char: str) -> int:
    """Columns a single character costs, independent of what surrounds it."""
    if unicodedata.category(char) in _ZERO_WIDTH_CATEGORIES or _is_composing_jamo(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in _WIDE else 1


def display_columns(text: str) -> int:
    """Width of ``text`` in half-width display columns.

    Marks, format characters and composing Hangul jamo add nothing (they render into a
    neighbouring cell, or nothing at all); East-Asian wide and full-width characters count
    two; everything else, including East-Asian *ambiguous*, counts one -- the conservative
    reading for a non-CJK terminal.

    Because every character's cost is fixed, the total is additive:
    ``display_columns(a + b) == display_columns(a) + display_columns(b)`` for all strings.
    Callers may therefore measure a payload in pieces and add up, and a line's remaining
    budget can be tracked incrementally.

    KNOWN OVER-COUNT, deliberately not fixed: a sequence joined by ZWJ renders as one
    grapheme but is measured per code point. The family emoji
    ``U+1F468 ZWJ U+1F469 ZWJ U+1F466`` costs 6 here (the joiners are free, the three
    pictographs are wide) against roughly 2 as drawn. Fixing it needs grapheme-cluster
    segmentation, which the standard library does not provide and which no available
    approximation makes safe:

    * A targeted "a character preceded by ZWJ is free" rule does give 2 for the family
      emoji, but ZWJ is not emoji-specific -- it also requests a joining form in Arabic and
      a conjunct in Indic scripts, and between ordinary letters it is simply ignored by the
      shaper. There the rule *under*-counts, without limit: ``a ZWJ b ZWJ c`` measures 1
      and draws 3. An under-count is the worse failure for a budget whose job is to prevent
      overflow, and it makes the budget evadable by interleaving invisible characters.
    * It also breaks additivity for every caller, in exchange for a partial answer: skin
      tone modifiers, keycaps and regional-indicator flags would still be counted per code
      point.

    So the model stays context-free and honestly over-counts ZWJ sequences, rather than
    degrading silently on text that merely looks similar. Emoji-dense payloads under a tight
    column budget are the case to revisit this for, with a real segmentation library.
    """
    return sum(_character_columns(char) for char in text)
