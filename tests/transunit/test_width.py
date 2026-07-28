"""Tests for :mod:`transunit.width`.

Display width is part of the adapter contract: a fixed-cell carrier and the translator
must agree on how many columns text occupies, and that agreement only holds if a
full-width glyph is counted as two and an ASCII glyph as one. These tests pin that
mapping so a regression in the width table surfaces here rather than as overflow in a
running program.
"""
from __future__ import annotations

import unicodedata

import pytest

from translator.rules import RuleSet, Severity, Violation, check_mechanical
from transunit.width import display_columns


class TestAsciiWidth:
    def test_single_ascii_letter_is_one_column(self) -> None:
        assert display_columns("A") == 1

    def test_ascii_word_is_its_character_count(self) -> None:
        assert display_columns("Hello") == 5

    def test_ascii_space_and_punctuation_are_one_each(self) -> None:
        assert display_columns("a, b!") == 5

    def test_empty_string_is_zero_columns(self) -> None:
        """The base case: no text draws nothing, so a budget check on it is trivially
        satisfied rather than erroring."""
        assert display_columns("") == 0


class TestFullWidth:
    @pytest.mark.parametrize(
        "char",
        [
            "日",  # CJK ideograph
            "あ",  # hiragana
            "カ",  # katakana
            "가",  # Hangul syllable
            "Ａ",  # full-width Latin 'A' (U+FF21)
        ],
    )
    def test_full_width_glyph_is_two_columns(self, char: str) -> None:
        assert display_columns(char) == 2

    def test_cjk_run_doubles(self) -> None:
        assert display_columns("日本語") == 6


class TestMixed:
    def test_mixed_ascii_and_cjk_sums_per_glyph(self) -> None:
        """The case the module exists for: counting characters would report 3 for a
        string that actually draws 5 columns once CJK is involved."""
        assert display_columns("日本A") == 5

    def test_leading_and_trailing_ascii_around_cjk(self) -> None:
        assert display_columns("[日]") == 4


class TestWidthAuditFixes:
    """Regressions for the east_asian_width rewrite."""

    def test_supplementary_plane_cjk_counts_as_wide(self) -> None:
        # Regression: a hand-rolled range list stopped below the supplementary planes, so a
        # rare/historical ideograph (in a name, say) was counted 1 and slipped past a hard
        # width budget. It is East-Asian Wide and must count 2.
        assert display_columns(chr(0x20000) * 5) == 10

    def test_combining_marks_add_no_width(self) -> None:
        # A combining accent renders onto the preceding glyph; "e" + U+0301 is one column.
        assert display_columns("é") == 1
        assert display_columns("mañana") == 6  # mañana, decomposed

    def test_ascii_and_cjk_baseline_unchanged(self) -> None:
        assert display_columns("hello") == 5
        assert display_columns("日本語") == 6
        assert display_columns("") == 0


class TestZeroWidthCharacters:
    """Regression: format characters and marks that draw no cell were counted one column
    each, so a line carrying them was over-measured and rejected by the blocking
    ``line_width`` rule -- shipping the untranslated source in place of a sound
    translation."""

    @pytest.mark.parametrize("char, name", [
        ("​", "ZERO WIDTH SPACE"),
        ("‌", "ZERO WIDTH NON-JOINER"),
        ("‍", "ZERO WIDTH JOINER"),
        ("⁠", "WORD JOINER"),
        ("­", "SOFT HYPHEN"),
        ("‎", "LEFT-TO-RIGHT MARK"),
        ("‏", "RIGHT-TO-LEFT MARK"),
        ("‮", "RIGHT-TO-LEFT OVERRIDE"),
        ("﻿", "ZERO WIDTH NO-BREAK SPACE (BOM)"),
        ("️", "VARIATION SELECTOR-16"),
        ("́", "COMBINING ACUTE ACCENT"),
        ("⃝", "COMBINING ENCLOSING CIRCLE"),
    ])
    def test_costs_no_columns(self, char: str, name: str) -> None:
        assert display_columns(char) == 0, name

    def test_interleaving_them_does_not_change_a_word(self) -> None:
        """The point of the fix: soft hyphens and joiners are hyphenation and shaping hints
        a corpus is full of, and they must not consume a line's budget."""
        assert display_columns("hy­phen​ated") == len("hyphenated")


class TestComposingHangulJamo:
    """Regression: conjoining jamo are letters (``Lo``) with canonical combining class 0, so
    the mark test missed them and decomposed Korean measured double its rendered width."""

    @pytest.mark.parametrize("syllable", ["한", "글", "안", "녕", "가", "뷁"])
    def test_decomposed_syllable_measures_the_same_as_composed(self, syllable: str) -> None:
        assert display_columns(unicodedata.normalize("NFD", syllable)) == \
            display_columns(unicodedata.normalize("NFC", syllable)) == 2

    @pytest.mark.parametrize("code_point, columns, role", [
        (0x1100, 2, "leading consonant (L), carries the syllable's cell"),
        (0x115F, 2, "choseong filler, occupies an L slot"),
        (0xA960, 2, "leading consonant, Extended-A"),
        (0x1160, 0, "vowel filler (V), drawn into the L cell"),
        (0x1161, 0, "vowel (V)"),
        (0x11AB, 0, "trailing consonant (T)"),
        (0x11FF, 0, "trailing consonant, end of the base block"),
        (0xD7B0, 0, "vowel, Extended-B"),
        (0xD7CB, 0, "trailing consonant, Extended-B"),
        (0xAC00, 2, "precomposed syllable, unaffected"),
    ])
    def test_jamo_class_costs(self, code_point: int, columns: int, role: str) -> None:
        assert display_columns(chr(code_point)) == columns, role

    def test_a_korean_line_arriving_nfd_fits_the_same_budget_as_nfc(self) -> None:
        # The realistic failure: a corpus normalised NFD (macOS filenames and exports arrive
        # that way) measured 37 columns for a line that draws 22, and every unit of it was
        # rejected as over-length.
        line = "안녕하세요, 반갑습니다"
        decomposed = unicodedata.normalize("NFD", line)
        assert decomposed != line                      # genuinely decomposed
        assert display_columns(decomposed) == display_columns(line) == 22
        assert display_columns(decomposed) <= 24       # the carrier's budget


class TestZwjSequencesAreDeliberatelyOverCounted:
    """The documented limit of the context-free model, pinned so it cannot change silently.

    A ZWJ sequence draws one grapheme but is measured per code point. Fixing it needs
    grapheme segmentation the standard library does not have; the available approximation
    ("a character after ZWJ is free") *under*-counts ordinary text -- the worse error for a
    budget meant to prevent overflow. See :func:`transunit.width.display_columns`.
    """

    def test_a_family_emoji_costs_the_three_pictographs_not_the_joiners(self) -> None:
        family = "\U0001f468‍\U0001f469‍\U0001f466"
        assert display_columns(family) == 6      # was 8; renders as roughly 2

    def test_ordinary_letters_joined_by_zwj_keep_their_columns(self) -> None:
        """The case the rejected approximation would have got wrong: the shaper ignores a
        ZWJ between Latin letters, so all three still draw."""
        assert display_columns("a‍b‍c") == 3


class TestBlockingRuleConsequence:
    """The defect was user-visible through :func:`translator.rules.check_mechanical`: an
    over-measured line raises a blocking ``line_width`` violation, the unit is rejected and
    the carrier shows the untranslated source. This measures that end, not just the width.

    Lives here rather than in the rules tests because it defends the width fix; the rule
    itself is unchanged and is only read.
    """

    @staticmethod
    def _line_width_violations(target: str, max_columns: int) -> list[Violation]:
        return [v for v in check_mechanical("SRC", target, RuleSet(), expected_placeholders=0,
                                            max_columns=max_columns)
                if v.rule_id == "line_width"]

    def test_a_korean_nfd_translation_is_no_longer_rejected(self) -> None:
        decomposed = unicodedata.normalize("NFD", "안녕하세요, 반갑습니다")
        # 27 code points for 22 columns: the old per-code-point count reached 37, past even
        # the 12% tolerance on a 24-column budget, so this was an ERROR.
        assert len(decomposed) == 27
        assert self._line_width_violations(decomposed, 24) == []

    def test_a_line_of_invisible_hints_is_no_longer_rejected(self) -> None:
        target = "Wcze­śniej​zapisz​plik"
        assert self._line_width_violations(target, display_columns(target)) == []

    def test_a_genuinely_too_long_line_is_still_rejected(self) -> None:
        """The fix must not make the budget evadable: real glyphs still cost their cells."""
        violations = self._line_width_violations("한국어" * 20, 24)
        assert [v.severity for v in violations] == [Severity.ERROR]
