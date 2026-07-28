"""Property-based tests for the pure text functions.

These functions are small, total, and sit on the blocking path: display width decides whether a
translation is REJECTED (and the untranslated source shown in its place), the token estimate
decides whether an operator is warned before a prompt overflows, and the similarity functions
decide what the model is shown as an example. A wrong answer from any of them is silent -- there
is no exception to notice -- so they are exactly what property testing is for.

Each test states the invariant it is defending, not the implementation it happens to match.
"""
from __future__ import annotations

import unicodedata

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from transunit.width import display_columns
from translator.backend.client import _estimate_tokens
from translator.context_packing import trigram_similarity

# Deliberately includes the categories that break naive width maths: format/control characters
# (Cf), combining marks (Mn), full-width CJK (W), and astral-plane emoji.
TEXT = st.text(
    alphabet=st.one_of(
        st.characters(min_codepoint=1, max_codepoint=0x2FFF),
        st.sampled_from("​‌‍⁠­‎‏‪‮﻿"),
        st.sampled_from("あア亜漢한글日本語"),
        st.sampled_from("̧́̈"),          # combining acute, diaeresis, cedilla
        st.sampled_from("👨👩👦🇵🇱😀"),
    ),
    max_size=120,
)


class TestDisplayColumns:
    """Width feeds a BLOCKING rule (rules.check_mechanical's line_width), so an over-count does
    not merely misreport -- it rejects a sound translation and ships the source instead."""

    @given(TEXT)
    def test_never_raises_and_is_never_negative(self, text: str) -> None:
        assert display_columns(text) >= 0

    @given(TEXT, TEXT)
    def test_is_additive_over_concatenation(self, left: str, right: str) -> None:
        """Width is a sum of per-character widths, so splitting a string must not change it.

        Additivity is what lets the engine reason about a line's budget incrementally. It also
        pins the absence of any cross-character rule (ligatures, grapheme clustering) -- if one
        is ever added, this fails and the budget arithmetic has to be revisited deliberately.
        That is exactly the trade refused for ZWJ sequences below.
        """
        assert display_columns(left + right) == display_columns(left) + display_columns(right)

    @given(TEXT)
    def test_is_bounded_by_twice_the_character_count(self, text: str) -> None:
        """No character is wider than a full-width cell."""
        assert display_columns(text) <= 2 * len(text)

    @given(st.text(alphabet=st.characters(min_codepoint=0xAC00, max_codepoint=0xD7A3),
                   max_size=40))
    def test_hangul_normalisation_does_not_change_width(self, text: str) -> None:
        """The same Korean spelled NFC or NFD must occupy the same columns.

        A corpus mixes both spellings (macOS filenames arrive NFD, most editors write NFC), and a
        budget that depends on which one a line happens to carry is a budget that rejects
        arbitrary lines. Decomposed Korean used to measure DOUBLE -- NFD '한' was 4 columns
        against NFC's 2 -- because conjoining jamo have canonical combining class 0 and the
        'combining marks are free' rule missed them.
        """
        assert display_columns(unicodedata.normalize("NFC", text)) == \
            display_columns(unicodedata.normalize("NFD", text)) == 2 * len(text)

    @pytest.mark.parametrize("char, name", [
        ("ো", "BENGALI VOWEL SIGN O"),
        ("ୋ", "ORIYA VOWEL SIGN O"),
        ("ஔ", "TAMIL LETTER AU"),
        ("ൊ", "MALAYALAM VOWEL SIGN O"),
    ])
    @pytest.mark.xfail(strict=True,
                       reason="genuine limit of a per-code-point width model, not a pending fix: "
                              "the two-part Indic vowel signs (about 30 of them) decompose into "
                              "two SPACING marks (Mc), so NFD costs 2 where NFC costs 1. Neither "
                              "spelling can be zero-rated -- U+09BE alone genuinely draws a cell "
                              "-- so only grapheme-cluster segmentation could reconcile them. "
                              "Stated as a fixed list rather than a hypothesis property because "
                              "the search rarely finds these few code points, which made the "
                              "expected failure flaky. Hangul, the case the property was "
                              "originally written for, is fixed and tested above.")
    def test_normalisation_does_not_change_width(self, char: str, name: str) -> None:
        assert display_columns(unicodedata.normalize("NFC", char)) == \
            display_columns(unicodedata.normalize("NFD", char))

    @pytest.mark.parametrize("char, name", [
        ("​", "ZERO WIDTH SPACE"),
        ("‌", "ZERO WIDTH NON-JOINER"),
        ("‍", "ZERO WIDTH JOINER"),
        ("⁠", "WORD JOINER"),
        ("­", "SOFT HYPHEN"),
        ("‎", "LEFT-TO-RIGHT MARK"),
        ("‮", "RIGHT-TO-LEFT OVERRIDE"),
        ("﻿", "ZERO WIDTH NO-BREAK SPACE"),
    ])
    def test_zero_width_characters_occupy_no_columns(self, char: str, name: str) -> None:
        assert display_columns(char) == 0

    @pytest.mark.xfail(strict=True,
                       reason="deliberate over-count, documented in display_columns: a ZWJ "
                              "sequence renders as one grapheme but is measured per code point "
                              "(6 columns for a ~2-column family emoji, down from 8 now the "
                              "joiners are free). Correcting it needs grapheme segmentation the "
                              "stdlib lacks; the available approximation ('a character after ZWJ "
                              "is free') UNDER-counts ordinary ZWJ use -- 'a ZWJ b ZWJ c' would "
                              "measure 1 and draw 3 -- which makes a budget meant to prevent "
                              "overflow evadable, and breaks additivity for every caller. Kept "
                              "as an expected failure so the gap stays visible.")
    def test_a_zwj_emoji_sequence_costs_one_grapheme_not_five(self) -> None:
        family = "\U0001f468‍\U0001f469‍\U0001f466"
        assert display_columns(family) == 2


class TestEstimateTokens:
    """A warning-only estimate, but one that must never under-report: its whole purpose is to
    fire BEFORE the server's hard context limit, not after."""

    @given(TEXT)
    def test_never_raises_and_is_never_negative(self, text: str) -> None:
        assert _estimate_tokens(text) >= 0

    @given(TEXT, TEXT)
    def test_is_monotone_under_appending(self, base: str, extra: str) -> None:
        """Adding text can never lower the estimate, or the warning could be dodged by growing
        the prompt."""
        assert _estimate_tokens(base + extra) >= _estimate_tokens(base)

    @given(TEXT)
    def test_errs_high_never_low(self, text: str) -> None:
        """The docstring promises over-counting; a real tokenizer emits at most one token per
        character, and the estimate must not fall below the ~4-chars-per-token floor it claims."""
        assert _estimate_tokens(text) >= len(text) // 4

    def test_empty_text_costs_nothing(self) -> None:
        assert _estimate_tokens("") == 0


class TestTrigramSimilarity:
    """Drives MMR diversity and neighbour/reference dedup in context packing: a wrong score
    silently changes which examples the model is shown."""

    @given(TEXT, TEXT)
    def test_is_symmetric(self, left: str, right: str) -> None:
        assert trigram_similarity(left, right) == trigram_similarity(right, left)

    @given(TEXT, TEXT)
    def test_is_a_fraction(self, left: str, right: str) -> None:
        assert 0.0 <= trigram_similarity(left, right) <= 1.0

    @given(TEXT)
    def test_a_string_is_maximally_similar_to_itself(self, text: str) -> None:
        assume(text.strip())
        assert trigram_similarity(text, text) == pytest.approx(1.0)

    @given(TEXT)
    def test_nothing_is_similar_to_the_empty_string(self, text: str) -> None:
        assume(text.strip())
        assert trigram_similarity(text, "") == 0.0
