"""Unit tests for :mod:`translator.rules`.

The heart of the module is :func:`check_mechanical`, so most tests build a minimal
:class:`RuleSet` and assert on the exact set of :class:`Violation` objects a given
``(source, target)`` pair produces. Rules are exercised one at a time by constructing an
input that can only trip the rule under test, so a failure names the offending rule.
"""
from __future__ import annotations

import unicodedata

import pytest

from translator.rules import (
    NAME_PREFIX,
    NAME_SIMILARITY,
    RuleConfigError,
    RuleSet,
    Severity,
    Violation,
    _common_prefix,
    blocking,
    check_mechanical,
    check_names,
)


def _ids(violations: list[Violation]) -> list[str]:
    return [violation.rule_id for violation in violations]


def _only(violations: list[Violation], rule_id: str) -> Violation:
    """The single violation with ``rule_id``; asserts it is the only one present."""
    assert _ids(violations) == [rule_id], _ids(violations)
    return violations[0]


class TestNonempty:
    """The require_nonempty gate: an empty rendering of non-empty source is a rejection."""

    def test_empty_target_for_nonempty_source_is_error(self):
        violations = check_mechanical("Hello", "", RuleSet(), expected_placeholders=0)
        violation = _only(violations, "nonempty")
        assert violation.severity is Severity.ERROR

    def test_whitespace_only_target_counts_as_empty(self):
        """`.strip()` decides emptiness, so a blank-but-present target still fails."""
        violations = check_mechanical("Hello", "   ", RuleSet(), expected_placeholders=0)
        assert _only(violations, "nonempty").severity is Severity.ERROR

    def test_empty_source_and_empty_target_is_clean(self):
        """Nothing to translate, nothing rendered -- not a defect."""
        assert check_mechanical("", "", RuleSet(), expected_placeholders=0) == []

    def test_whitespace_source_is_treated_as_empty_source(self):
        assert check_mechanical("   ", "", RuleSet(), expected_placeholders=0) == []

    def test_require_nonempty_false_disables_the_check(self):
        ruleset = RuleSet(require_nonempty=False)
        assert check_mechanical("Hello", "", ruleset, expected_placeholders=0) == []


class TestPlaceholders:
    """The placeholder set must survive the round trip exactly -- same indices, no dupes."""

    def test_correct_set_passes(self):
        violations = check_mechanical(
            "a [[0]] b [[1]]", "x [[0]] y [[1]]", RuleSet(), expected_placeholders=2)
        assert "placeholders" not in _ids(violations)

    def test_missing_placeholder_is_error(self):
        violations = check_mechanical(
            "a [[0]] b [[1]]", "only [[0]] here", RuleSet(), expected_placeholders=2)
        assert _only(violations, "placeholders").severity is Severity.ERROR

    def test_extra_placeholder_is_error(self):
        violations = check_mechanical(
            "a [[0]]", "x [[0]] y [[1]]", RuleSet(), expected_placeholders=1)
        _only(violations, "placeholders")

    def test_renumbered_placeholder_is_error(self):
        """Right count, wrong index: the model changed which slot it referenced."""
        violations = check_mechanical(
            "a [[0]]", "x [[5]] y", RuleSet(), expected_placeholders=1)
        _only(violations, "placeholders")

    def test_repeated_placeholder_is_error(self):
        """Set matches expected {0} yet the index appears twice -- flagged as repeated."""
        violations = check_mechanical(
            "a [[0]]", "x [[0]] y [[0]]", RuleSet(), expected_placeholders=1)
        violation = _only(violations, "placeholders")
        assert violation.severity is Severity.ERROR
        assert "repeated" in violation.message

    def test_zero_expected_and_none_present_passes(self):
        assert check_mechanical(
            "plain", "plain out", RuleSet(), expected_placeholders=0) == []


class TestUntranslated:
    """The detector is injected; absent, the check cannot be evaluated and must not fire."""

    def test_detector_true_is_error(self):
        violations = check_mechanical(
            "Hello", "Hello", RuleSet(), expected_placeholders=0,
            is_untranslated=lambda source, target: True)
        assert _only(violations, "untranslated").severity is Severity.ERROR

    def test_detector_false_is_clean(self):
        violations = check_mechanical(
            "Hello", "Bonjour", RuleSet(), expected_placeholders=0,
            is_untranslated=lambda source, target: False)
        assert violations == []

    def test_no_detector_skips_even_a_verbatim_echo(self):
        """None means 'cannot tell the languages apart' -- silence, not a pass claim."""
        violations = check_mechanical(
            "Hello", "Hello", RuleSet(), expected_placeholders=0, is_untranslated=None)
        assert violations == []

    def test_allow_untranslated_disables_the_check(self):
        ruleset = RuleSet(allow_untranslated=True)
        violations = check_mechanical(
            "Hello", "Hello", ruleset, expected_placeholders=0,
            is_untranslated=lambda source, target: True)
        assert violations == []

    def test_detector_receives_source_and_target(self):
        seen: list[tuple[str, str]] = []

        def spy(source: str, target: str) -> bool:
            seen.append((source, target))
            return False

        check_mechanical("SRC", "TGT", RuleSet(), expected_placeholders=0,
                         is_untranslated=spy)
        assert seen == [("SRC", "TGT")]


class TestControlCharacters:
    """A single-line carrier cannot hold a raw control byte; the break is the carrier's."""

    def test_newline_is_error_with_readable_name(self):
        violations = check_mechanical(
            "Hi", "Hello\nWorld", RuleSet(), expected_placeholders=0)
        violation = _only(violations, "control_character")
        assert violation.severity is Severity.ERROR
        assert "LINE FEED" in violation.message

    def test_nul_is_error_with_readable_name(self):
        violations = check_mechanical(
            "Hi", "Hello\x00World", RuleSet(), expected_placeholders=0)
        assert "NUL" in _only(violations, "control_character").message

    def test_delete_is_error_with_readable_name(self):
        violations = check_mechanical(
            "Hi", "Hello\x7fWorld", RuleSet(), expected_placeholders=0)
        assert "DELETE" in _only(violations, "control_character").message

    def test_tab_is_permitted(self):
        """The class deliberately splits its range around 0x09, so horizontal tab is
        allowed where line-breaking control bytes are not."""
        violations = check_mechanical(
            "Hi", "Hello\tWorld", RuleSet(), expected_placeholders=0)
        assert "control_character" not in _ids(violations)


class TestLineWidthGuideline:
    """Without a hard budget the project guideline only warns, and splits on literal \\n."""

    def test_over_the_guideline_warns_and_does_not_block(self):
        ruleset = RuleSet(max_line_columns=20)
        violations = check_mechanical(
            "src", "a" * 25, ruleset, expected_placeholders=0)
        violation = _only(violations, "line_width")
        assert violation.severity is Severity.WARNING
        assert not violation.blocking

    def test_within_the_guideline_is_clean(self):
        ruleset = RuleSet(max_line_columns=20)
        assert check_mechanical(
            "src", "a" * 20, ruleset, expected_placeholders=0) == []

    def test_split_on_literal_backslash_n_measures_each_segment(self):
        """Splits on the two-character sequence backslash-n, not a real newline (which
        would be rejected as a control character instead)."""
        ruleset = RuleSet(max_line_columns=20)
        target = "a" * 25 + "\\n" + "b" * 3
        violations = check_mechanical("src", target, ruleset, expected_placeholders=0)
        violation = _only(violations, "line_width")
        assert violation.severity is Severity.WARNING
        # Only the first, over-long segment is flagged; the message names line 1.
        assert "line 1" in violation.message


class TestLineWidthHardBudget:
    """With max_columns the carrier imposes a real limit; overshoot past tolerance blocks."""

    def test_within_budget_passes(self):
        ruleset = RuleSet(max_columns_tolerance=0.2)
        assert check_mechanical(
            "src", "a" * 10, ruleset, expected_placeholders=0, max_columns=10) == []

    def test_over_budget_within_tolerance_warns(self):
        """max_columns=10, tolerance 0.2 -> allowed 12; a width of 12 is a tolerated warn."""
        ruleset = RuleSet(max_columns_tolerance=0.2)
        violations = check_mechanical(
            "src", "a" * 12, ruleset, expected_placeholders=0, max_columns=10)
        violation = _only(violations, "line_width")
        assert violation.severity is Severity.WARNING
        assert not violation.blocking

    def test_past_tolerance_is_error(self):
        ruleset = RuleSet(max_columns_tolerance=0.2)
        violations = check_mechanical(
            "src", "a" * 13, ruleset, expected_placeholders=0, max_columns=10)
        assert _only(violations, "line_width").severity is Severity.ERROR

    def test_full_width_glyphs_count_two_columns(self):
        """Three full-width glyphs are 6 display columns, tripping a 5-column budget the
        same text would clear if characters were counted naively."""
        ruleset = RuleSet(max_columns_tolerance=0.0)
        target = "Ａ" * 3  # fullwidth Latin A, two columns each
        violations = check_mechanical(
            "src", target, ruleset, expected_placeholders=0, max_columns=5)
        violation = _only(violations, "line_width")
        assert violation.severity is Severity.ERROR
        assert "6 characters" in violation.message

    def test_hard_budget_bypasses_the_guideline_path(self):
        """A wide line under max_columns must not also draw a guideline warning."""
        ruleset = RuleSet(max_line_columns=20, max_columns_tolerance=0.2)
        violations = check_mechanical(
            "src", "a" * 30, ruleset, expected_placeholders=0, max_columns=40)
        assert violations == []


class TestForbiddenPatterns:
    def test_match_is_error_including_reason(self):
        ruleset = RuleSet(forbidden_patterns=(("foo", "no foos allowed"),))
        violations = check_mechanical(
            "src", "this has foo", ruleset, expected_placeholders=0)
        violation = _only(violations, "forbidden")
        assert violation.severity is Severity.ERROR
        assert "no foos allowed" in violation.message
        assert "'foo'" in violation.message

    def test_reasonless_pattern_omits_the_colon(self):
        ruleset = RuleSet(forbidden_patterns=(("bar", ""),))
        violation = _only(
            check_mechanical("src", "a bar b", ruleset, expected_placeholders=0),
            "forbidden")
        assert violation.message == "matches forbidden pattern 'bar'"

    def test_non_matching_target_is_clean(self):
        ruleset = RuleSet(forbidden_patterns=(("foo", "why"),))
        assert check_mechanical("src", "clean text", ruleset, expected_placeholders=0) == []

    def test_regex_is_honoured_not_treated_literally(self):
        ruleset = RuleSet(forbidden_patterns=((r"\d{4}", "no year digits"),))
        assert _ids(check_mechanical(
            "src", "in 2026", ruleset, expected_placeholders=0)) == ["forbidden"]


class TestRequiredTerms:
    """Glossary presence uses the configurable severity, WARNING by default."""

    def test_absent_term_uses_glossary_severity(self):
        ruleset = RuleSet()
        violations = check_mechanical(
            "src", "no blade here", ruleset,
            expected_placeholders=0, required_terms=("Sword",))
        violation = _only(violations, "glossary")
        assert violation.severity is ruleset.glossary_severity is Severity.WARNING

    def test_error_severity_is_configurable(self):
        ruleset = RuleSet(glossary_severity=Severity.ERROR)
        violations = check_mechanical(
            "src", "no blade here", ruleset,
            expected_placeholders=0, required_terms=("Sword",))
        assert _only(violations, "glossary").severity is Severity.ERROR

    def test_present_term_case_insensitive_is_clean(self):
        assert check_mechanical(
            "src", "he drew his SWORD", RuleSet(),
            expected_placeholders=0, required_terms=("sword",)) == []

    def test_empty_term_is_ignored(self):
        assert check_mechanical(
            "src", "anything", RuleSet(),
            expected_placeholders=0, required_terms=("",)) == []


class TestCheckNames:
    """Consistency of a proper noun is decided in code: respelling blocks, inflection does
    not, and a coincidental resemblance is left alone."""

    def test_respelled_name_is_error(self):
        violations = check_names("Basil spoke", "Bazyl went home", ["Basil"])
        violation = _only(violations, "name_respelled")
        assert violation.severity is Severity.ERROR
        assert "'Basil'" in violation.message and "'Bazyl'" in violation.message

    def test_legitimate_inflection_keeps_the_stem_and_passes(self):
        """The genitive Basila keeps 'Basil' as a prefix, so it is not a respelling."""
        assert check_names("Basil spoke", "Basila mowil", ["Basil"]) == []

    def test_unrelated_word_without_shared_prefix_passes(self):
        """No common opening -> not an attempt at the name (dropped for a pronoun, say)."""
        assert check_names("Basil spoke", "Dream flowed on", ["Basil"]) == []

    def test_shared_prefix_but_low_similarity_passes(self):
        """'Barometer' shares 'Ba' (>= NAME_PREFIX) yet scores below NAME_SIMILARITY, so
        it reads as a different word rather than a botched name."""
        assert check_names("Basil spoke", "Barometer readings", ["Basil"]) == []

    def test_lowercase_resemblance_is_ignored(self):
        """Only capitalised tokens can be a name attempt; a lowercase near-match is not."""
        assert check_names("Basil spoke", "bazyl went home", ["Basil"]) == []

    def test_name_absent_from_source_is_never_flagged(self):
        """If the source never named Basil, a target token resembling it is irrelevant."""
        assert check_names("Hello world", "Bazyl appears", ["Basil"]) == []

    def test_exact_name_present_passes(self):
        assert check_names("Basil spoke", "Basil replied", ["Basil"]) == []

    def test_thresholds_have_expected_values(self):
        """Guard the constants the separation of corruption from coincidence depends on."""
        assert NAME_PREFIX == 2
        assert NAME_SIMILARITY == 0.5

    def test_respelling_surfaces_through_check_mechanical(self):
        violations = check_mechanical(
            "Basil spoke", "Bazyl went", RuleSet(),
            expected_placeholders=0, names=["Basil"])
        assert "name_respelled" in _ids(violations)


class TestBlocking:
    def test_returns_only_error_severity(self):
        warn = Violation("a", Severity.WARNING, "w")
        err = Violation("b", Severity.ERROR, "e")
        assert blocking([warn, err, warn]) == [err]

    def test_empty_when_no_errors(self):
        assert blocking([Violation("a", Severity.WARNING, "w")]) == []

    def test_empty_input(self):
        assert blocking([]) == []


class TestRuleSetLoad:
    """RuleSet.load: a valid file round-trips; every mistyped or inconsistent field is
    refused with RuleConfigError rather than coerced."""

    def _write(self, tmp_path, text: str):
        path = tmp_path / "rules.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_valid_file_round_trips_every_field(self, tmp_path):
        path = self._write(tmp_path, """
            [limits]
            max_line_columns = 100
            max_columns_tolerance = 0.15
            allow_untranslated = true
            require_nonempty = false

            [style]
            directives = ["one", "two"]

            [[forbidden]]
            pattern = "foo"
            reason = "no foo"

            [[advisory]]
            id = "register"
            description = "keep formal register"

            [glossary]
            severity = "error"
        """)
        ruleset = RuleSet.load(path)
        assert ruleset.max_line_columns == 100
        assert ruleset.max_columns_tolerance == pytest.approx(0.15)
        assert ruleset.allow_untranslated is True
        assert ruleset.require_nonempty is False
        assert ruleset.style_directives == ("one", "two")
        assert ruleset.forbidden_patterns == (("foo", "no foo"),)
        assert ruleset.advisory_rules == (("register", "keep formal register"),)
        assert ruleset.glossary_severity is Severity.ERROR

    def test_defaults_apply_to_an_empty_but_valid_file(self, tmp_path):
        path = self._write(tmp_path, """
            [[forbidden]]
            pattern = "x"
        """)
        ruleset = RuleSet.load(path)
        assert ruleset.max_line_columns == 110
        assert ruleset.max_columns_tolerance == pytest.approx(0.12)
        assert ruleset.allow_untranslated is False
        assert ruleset.require_nonempty is True
        assert ruleset.glossary_severity is Severity.WARNING
        # A [[forbidden]] with no reason defaults the explanation to empty.
        assert ruleset.forbidden_patterns == (("x", ""),)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(RuleConfigError):
            RuleSet.load(tmp_path / "absent.toml")

    def test_invalid_toml_raises(self, tmp_path):
        path = self._write(tmp_path, "this is = = not toml")
        with pytest.raises(RuleConfigError):
            RuleSet.load(path)

    def test_boolean_max_line_columns_rejected(self, tmp_path):
        """bool is an int subclass; `= true` must not read as one column."""
        path = self._write(tmp_path, "[limits]\nmax_line_columns = true\n")
        with pytest.raises(RuleConfigError):
            RuleSet.load(path)

    def test_string_max_line_columns_rejected(self, tmp_path):
        path = self._write(tmp_path, '[limits]\nmax_line_columns = "wide"\n')
        with pytest.raises(RuleConfigError):
            RuleSet.load(path)

    def test_max_line_columns_below_floor_rejected(self, tmp_path):
        path = self._write(tmp_path, "[limits]\nmax_line_columns = 19\n")
        with pytest.raises(RuleConfigError):
            RuleSet.load(path)

    def test_max_line_columns_at_floor_accepted(self, tmp_path):
        path = self._write(tmp_path, "[limits]\nmax_line_columns = 20\n")
        assert RuleSet.load(path).max_line_columns == 20

    def test_string_allow_untranslated_rejected(self, tmp_path):
        """Coercing bool('false') would silently invert a safety gate, so a string fails."""
        path = self._write(tmp_path, '[limits]\nallow_untranslated = "false"\n')
        with pytest.raises(RuleConfigError):
            RuleSet.load(path)

    def test_negative_tolerance_rejected(self, tmp_path):
        path = self._write(tmp_path, "[limits]\nmax_columns_tolerance = -0.1\n")
        with pytest.raises(RuleConfigError):
            RuleSet.load(path)

    def test_boolean_tolerance_rejected(self, tmp_path):
        path = self._write(tmp_path, "[limits]\nmax_columns_tolerance = true\n")
        with pytest.raises(RuleConfigError):
            RuleSet.load(path)

    def test_invalid_forbidden_regex_rejected(self, tmp_path):
        path = self._write(tmp_path, '[[forbidden]]\npattern = "(unclosed"\n')
        with pytest.raises(RuleConfigError):
            RuleSet.load(path)

    def test_forbidden_without_pattern_rejected(self, tmp_path):
        path = self._write(tmp_path, '[[forbidden]]\nreason = "orphan"\n')
        with pytest.raises(RuleConfigError):
            RuleSet.load(path)

    def test_advisory_missing_id_rejected(self, tmp_path):
        path = self._write(tmp_path, '[[advisory]]\ndescription = "no id"\n')
        with pytest.raises(RuleConfigError):
            RuleSet.load(path)

    def test_advisory_missing_description_rejected(self, tmp_path):
        path = self._write(tmp_path, '[[advisory]]\nid = "lonely"\n')
        with pytest.raises(RuleConfigError):
            RuleSet.load(path)

    def test_bad_glossary_severity_rejected(self, tmp_path):
        path = self._write(tmp_path, '[glossary]\nseverity = "fatal"\n')
        with pytest.raises(RuleConfigError):
            RuleSet.load(path)

    def test_style_directives_must_be_string_list(self, tmp_path):
        """A bare string would otherwise split into one-character directives."""
        path = self._write(tmp_path, '[style]\ndirectives = "not a list"\n')
        with pytest.raises(RuleConfigError):
            RuleSet.load(path)

    def test_style_directives_with_non_string_member_rejected(self, tmp_path):
        path = self._write(tmp_path, "[style]\ndirectives = [\"ok\", 3]\n")
        with pytest.raises(RuleConfigError):
            RuleSet.load(path)

    def test_tolerance_above_one_rejected(self, tmp_path):
        """A tolerance is a fraction: 5 would mean 500% over budget is fine, which is not a
        tolerance at all. Previously accepted, because only the negative side was checked."""
        path = self._write(tmp_path, "[limits]\nmax_columns_tolerance = 5.0\n")
        with pytest.raises(RuleConfigError, match=r"max_columns_tolerance"):
            RuleSet.load(path)

    @pytest.mark.parametrize("value", ["0.0", "1.0"])
    def test_both_ends_of_the_tolerance_range_accepted(self, tmp_path, value):
        path = self._write(tmp_path, f"[limits]\nmax_columns_tolerance = {value}\n")
        assert RuleSet.load(path).max_columns_tolerance == pytest.approx(float(value))


class TestRuleFileSectionShape:
    """A section written with the wrong TOML *shape* must be diagnosed, not crash.

    Each case below used to escape :class:`RuleConfigError` and reach the CLI as a raw
    ``TypeError``, or -- for an array of tables written as one table -- produce a nonsense
    complaint about unknown single-character keys.
    """

    def _write(self, tmp_path, text: str):
        path = tmp_path / "rules.toml"
        path.write_text(text, encoding="utf-8")
        return path

    @pytest.mark.parametrize("section", ["limits", "style", "glossary"])
    def test_a_scalar_where_a_table_belongs_names_the_section(self, tmp_path, section):
        with pytest.raises(RuleConfigError,
                           match=rf"\[{section}\] must be a table, got int"):
            RuleSet.load(self._write(tmp_path, f"{section} = 5\n"))

    @pytest.mark.parametrize("section", ["forbidden", "advisory"])
    def test_an_array_of_tables_written_as_one_table_says_to_double_the_brackets(
            self, tmp_path, section):
        body = ('[forbidden]\npattern = "x"\n' if section == "forbidden"
                else '[advisory]\nid = "x"\ndescription = "y"\n')
        with pytest.raises(RuleConfigError,
                           match=rf"write \[\[{section}\]\], not \[{section}\]"):
            RuleSet.load(self._write(tmp_path, body))

    @pytest.mark.parametrize("section", ["forbidden", "advisory"])
    def test_a_scalar_where_an_array_of_tables_belongs_is_diagnosed(self, tmp_path, section):
        with pytest.raises(RuleConfigError, match=rf"\[\[{section}\]\] must be an array"):
            RuleSet.load(self._write(tmp_path, f"{section} = 5\n"))

    @pytest.mark.parametrize("section", ["forbidden", "advisory"])
    def test_an_array_holding_a_non_table_names_its_position(self, tmp_path, section):
        with pytest.raises(RuleConfigError,
                           match=rf"\[\[{section}\]\] entry 0 must be a table, got int"):
            RuleSet.load(self._write(tmp_path, f"{section} = [5]\n"))


class TestRuleSetInvariants:
    """The bounds live in the type, so a directly-constructed rule set cannot be invalid --
    the loader only adds the file and section to the diagnosis."""

    def test_an_implausibly_narrow_line_budget_cannot_be_constructed(self):
        with pytest.raises(ValueError, match="max_line_columns=19 is implausibly small"):
            RuleSet(max_line_columns=19)

    def test_the_floor_itself_is_valid(self):
        assert RuleSet(max_line_columns=20).max_line_columns == 20

    @pytest.mark.parametrize("bad", [-0.01, 1.01])
    def test_the_tolerance_is_confined_to_the_unit_interval(self, bad):
        with pytest.raises(ValueError, match=r"max_columns_tolerance must be a fraction"):
            RuleSet(max_columns_tolerance=bad)

    @pytest.mark.parametrize("good", [0.0, 1.0])
    def test_both_ends_of_the_unit_interval_are_valid(self, good):
        assert RuleSet(max_columns_tolerance=good).max_columns_tolerance == good


class TestCheckNamesInflection:
    """Regression: stem-changing inflections (the norm for Polish/Russian genitives) must
    NOT be flagged as respellings. The old check demanded the whole name survive as a
    substring, so it flagged every declined form and pushed the model back to the
    ungrammatical base form."""

    @pytest.mark.parametrize("name, target", [
        ("Praga", "poszli do Pragi"),      # genitive, a -> i
        ("Anna", "nie ma Anny"),           # genitive, a -> y
        ("Maria", "z Marii"),              # genitive, ia -> ii
        ("Warszawa", "w Warszawy"),        # a -> y
        ("Kraków", "w Krakowie"),          # locative, ó -> o + ie
        ("Basil", "Basila tutaj"),         # suffixing genitive (the case the old check handled)
    ])
    def test_legitimate_inflection_is_not_flagged(self, name: str, target: str) -> None:
        assert check_names(f"about {name}", target, [name]) == []

    @pytest.mark.parametrize("name, target", [
        ("Basil", "imię Bazyl"),           # stem respelled s->z, i->y
        ("Basil", "to Bazyli"),            # target-language substitution
        ("Basil", "z Basyl"),              # interior i->y on a 5-letter name
        ("Adam", "widzę Adem"),            # interior a->e on a 4-letter name (short-name case)
        ("Anna", "to Anka"),               # interior n->k on a 4-letter name
        ("Whitecliff", "w Whajtkliff"),    # deep corruption
    ])
    def test_stem_respelling_is_still_flagged(self, name: str, target: str) -> None:
        # The fraction-based stem test must keep working for SHORT names: an absolute margin
        # left names of length <= 4 with no detection window at all.
        violations = check_names(f"at {name}", target, [name])
        assert any(v.rule_id == "name_respelled" for v in violations)

    def test_the_exact_name_present_is_never_flagged(self) -> None:
        assert check_names("about Anna", "Anna is here", ["Anna"]) == []

    def test_a_token_sharing_too_little_opening_is_not_a_name_attempt(self) -> None:
        # Only one shared leading character ("K") -- below NAME_PREFIX -- reads as a different
        # word, not a botched name, in both the old and new logic.
        assert check_names("about Kate", "meet Kurt", ["Kate"]) == []


class TestCommonPrefix:
    """Direct tests for the shared-opening helper behind the name-respelling heuristic.

    The whole heuristic is "a corrupted name keeps its beginning", so this length is what
    separates a genuine corruption from a coincidental resemblance. It is total: for any two
    strings it returns a count in ``[0, min(len))`` and never raises, including on the
    exhausted-loop cases (one string a prefix of the other) that the rule reaches whenever a
    name appears in an unsuffixed or truncated form.
    """

    @pytest.mark.parametrize("first, second, expected", [
        ("Basil", "Basil", 5),        # equal: the loop runs out with every character matched
        ("Basil", "Basilio", 5),      # strict prefix, shorter first -- zip stops at the shorter
        ("Basilio", "Basil", 5),      # same pair reversed: the count must be symmetric
        ("", "Basil", 0),             # empty left: no iterations at all, not an error
        ("Basil", "", 0),             # empty right, same
        ("", "", 0),                  # both empty
        ("Kate", "Kurt", 1),          # shares only the opening character
        ("Anna", "Bella", 0),         # nothing in common: the loop breaks immediately
    ])
    def test_prefix_length(self, first: str, second: str, expected: int) -> None:
        assert _common_prefix(first, second) == expected

    @pytest.mark.parametrize("first, second, expected", [
        ("Kraków", "Krakówek", 6),   # non-ASCII in the shared opening; also a prefix pair
        ("Zoë", "Zoe", 2),           # diverges on a precomposed character
        ("𝄞clef", "𝄞claim", 3),      # non-BMP: a surrogate pair must not be split or crash
        ("𝄞", "𝄞", 1),               # a lone astral character counts as one, not two
        ("日本語", "日本人", 2),        # CJK
        # OHM SIGN vs GREEK CAPITAL OMEGA -- one glyph, two code points, written as escapes so
        # the distinction cannot be lost to an editor.
        ("\u2126", "\u03a9", 0),
    ])
    def test_unicode_is_compared_by_code_point_without_raising(
            self, first: str, second: str, expected: int) -> None:
        # Comparison is per code point deliberately: the caller feeds it raw model output, so
        # the helper must tolerate any encoding form rather than assume normalised text.
        assert _common_prefix(first, second) == expected

    def test_differently_normalised_spellings_share_no_prefix(self) -> None:
        """Built rather than written literally, so an editor normalising this file cannot
        silently turn this into a comparison of two identical strings. Composed and decomposed
        forms of the same word look alike but differ from the first code point, which is why
        the caller must normalise before trusting a zero here."""
        composed = unicodedata.normalize("NFC", "école")
        decomposed = unicodedata.normalize("NFD", "école")
        assert composed != decomposed
        assert _common_prefix(composed, decomposed) == 0
        assert _common_prefix(decomposed, composed) == 0
