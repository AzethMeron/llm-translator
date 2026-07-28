"""Unit tests for :mod:`translator.roles`.

Covers :meth:`AgentSet.load` (structure validation and ``{placeholder}`` substitution) and
:meth:`Limits.translate_budget` (the clamped scaling of a translation's token ceiling).
"""
from __future__ import annotations

import pytest

from translator.roles import (
    DEFAULT_SUBSTITUTIONS,
    AgentConfigError,
    AgentSet,
    Context,
    Leniency,
    Limits,
    Reviewer,
)

_VALID_TRANSLATE = (
    "[translate]\n"
    'instructions = "Translate from {source_language} into {target_language}."\n'
)
_VALID_REVIEWER = "[[reviewer]]\nid = \"style\"\ninstructions = \"Check the register.\"\n"


def _write(tmp_path, text: str):
    path = tmp_path / "agents.toml"
    path.write_text(text, encoding="utf-8")
    return path


class TestValidLoad:
    def test_minimal_valid_file_loads(self, tmp_path):
        path = _write(tmp_path, _VALID_TRANSLATE + _VALID_REVIEWER)
        agents = AgentSet.load(path)
        assert len(agents.reviewers) == 1
        assert agents.reviewers[0].id == "style"
        assert agents.reviewers[0].instructions == "Check the register."
        assert agents.reviewers[0].from_rules is False

    def test_translator_instructions_are_stripped(self, tmp_path):
        path = _write(tmp_path, (
            "[translate]\n"
            'instructions = "   Translate faithfully.   "\n'
        ) + _VALID_REVIEWER)
        assert AgentSet.load(path).translator_instructions == "Translate faithfully."

    def test_multiple_reviewers_preserve_declaration_order(self, tmp_path):
        path = _write(tmp_path, _VALID_TRANSLATE + (
            '[[reviewer]]\nid = "first"\ninstructions = "a"\n'
            '[[reviewer]]\nid = "second"\ninstructions = "b"\n'
        ))
        assert [r.id for r in AgentSet.load(path).reviewers] == ["first", "second"]

    def test_default_limits_when_section_absent(self, tmp_path):
        path = _write(tmp_path, _VALID_TRANSLATE + _VALID_REVIEWER)
        assert AgentSet.load(path).limits == Limits()


class TestLeniency:
    def test_default_leniency_when_section_absent(self, tmp_path):
        path = _write(tmp_path, _VALID_TRANSLATE + _VALID_REVIEWER)
        agents = AgentSet.load(path)
        assert agents.leniency == Leniency()
        assert agents.reviewers[0].leniency == Leniency()  # resolved onto the reviewer

    def test_file_wide_leniency_is_read_and_applied_to_reviewers(self, tmp_path):
        path = _write(tmp_path, _VALID_TRANSLATE + _VALID_REVIEWER
                      + "[leniency]\nwindow = 8\nmax_bad = 1\n")
        agents = AgentSet.load(path)
        assert agents.leniency == Leniency(window=8, max_bad=1)
        assert agents.reviewers[0].leniency == Leniency(window=8, max_bad=1)

    def test_a_reviewer_overrides_the_file_wide_default(self, tmp_path):
        path = _write(tmp_path, _VALID_TRANSLATE
                      + '[[reviewer]]\nid = "strict"\ninstructions = "x"\n'
                        "leniency = { max_bad = 0 }\n"
                      + '[[reviewer]]\nid = "lax"\ninstructions = "y"\n'
                      + "[leniency]\nwindow = 30\nmax_bad = 5\n")
        by_id = {r.id: r for r in AgentSet.load(path).reviewers}
        # override inherits window from the default and replaces only max_bad
        assert by_id["strict"].leniency == Leniency(window=30, max_bad=0)
        assert by_id["lax"].leniency == Leniency(window=30, max_bad=5)

    def test_a_bad_window_is_rejected(self, tmp_path):
        path = _write(tmp_path, _VALID_TRANSLATE + _VALID_REVIEWER
                      + "[leniency]\nwindow = 0\n")
        with pytest.raises(AgentConfigError, match=r"leniency.*window.*>= 1"):
            AgentSet.load(path)

    def test_an_unknown_leniency_key_is_rejected(self, tmp_path):
        path = _write(tmp_path, _VALID_TRANSLATE + _VALID_REVIEWER
                      + "[leniency]\nwndow = 5\n")
        with pytest.raises(AgentConfigError, match="unknown key"):
            AgentSet.load(path)

    def test_surfaces_only_past_the_allowance(self):
        policy = Leniency(window=5, max_bad=2)
        assert policy.surfaces(1) is False
        assert policy.surfaces(2) is False
        assert policy.surfaces(3) is True


class TestSubstitution:
    def test_caller_substitutions_replace_tokens(self, tmp_path):
        path = _write(tmp_path, _VALID_TRANSLATE + _VALID_REVIEWER)
        agents = AgentSet.load(
            path, substitutions={"source_language": "Polish",
                                 "target_language": "English"})
        assert "Polish" in agents.translator_instructions
        assert "English" in agents.translator_instructions
        assert "{" not in agents.translator_instructions

    def test_defaults_fill_in_when_no_substitutions(self, tmp_path):
        """The file stays loadable without a caller that knows the language pair."""
        path = _write(tmp_path, _VALID_TRANSLATE + _VALID_REVIEWER)
        agents = AgentSet.load(path)
        assert DEFAULT_SUBSTITUTIONS["source_language"] in agents.translator_instructions
        assert DEFAULT_SUBSTITUTIONS["target_language"] in agents.translator_instructions

    def test_substitution_applies_to_reviewer_prompts(self, tmp_path):
        path = _write(tmp_path, _VALID_TRANSLATE + (
            '[[reviewer]]\nid = "lang"\n'
            'instructions = "Judge the {target_language} phrasing."\n'
        ))
        agents = AgentSet.load(path, substitutions={"target_language": "English"})
        assert agents.reviewers[0].instructions == "Judge the English phrasing."

    def test_unknown_token_in_translate_raises_listing_it(self, tmp_path):
        path = _write(tmp_path, (
            "[translate]\n"
            'instructions = "Translate the {mystery_token} carefully."\n'
        ) + _VALID_REVIEWER)
        with pytest.raises(AgentConfigError) as excinfo:
            AgentSet.load(path)
        assert "mystery_token" in str(excinfo.value)

    def test_unknown_token_in_reviewer_raises(self, tmp_path):
        path = _write(tmp_path, _VALID_TRANSLATE + (
            '[[reviewer]]\nid = "r"\ninstructions = "Judge {unknown_token} strictly."\n'
        ))
        with pytest.raises(AgentConfigError) as excinfo:
            AgentSet.load(path)
        assert "unknown_token" in str(excinfo.value)

    def test_prose_braces_are_left_alone(self, tmp_path):
        """The token pattern is narrow (lowercase identifiers), so a JSON brace example in
        a prompt is prose, not an unfilled placeholder."""
        path = _write(tmp_path, _VALID_TRANSLATE + (
            '[[reviewer]]\nid = "json"\n'
            "instructions = 'Reply with {\"verdict\": \"ok\"}.'\n"
        ))
        agents = AgentSet.load(path)
        assert agents.reviewers[0].instructions == 'Reply with {"verdict": "ok"}.'


class TestReviewerValidation:
    def test_from_rules_without_instructions_is_ok(self, tmp_path):
        path = _write(tmp_path, _VALID_TRANSLATE + (
            '[[reviewer]]\nid = "compliance"\nfrom_rules = true\n'
        ))
        reviewer = AgentSet.load(path).reviewers[0]
        assert reviewer.from_rules is True
        assert reviewer.instructions == ""

    def test_from_rules_with_instructions_raises(self, tmp_path):
        """One of the two would be silently ignored, so the conflict is refused."""
        path = _write(tmp_path, _VALID_TRANSLATE + (
            '[[reviewer]]\nid = "both"\nfrom_rules = true\ninstructions = "also this"\n'
        ))
        with pytest.raises(AgentConfigError):
            AgentSet.load(path)

    def test_neither_from_rules_nor_instructions_raises(self, tmp_path):
        path = _write(tmp_path, _VALID_TRANSLATE + (
            '[[reviewer]]\nid = "empty"\n'
        ))
        with pytest.raises(AgentConfigError):
            AgentSet.load(path)

    def test_whitespace_only_instructions_count_as_none(self, tmp_path):
        path = _write(tmp_path, _VALID_TRANSLATE + (
            '[[reviewer]]\nid = "blank"\ninstructions = "   "\n'
        ))
        with pytest.raises(AgentConfigError):
            AgentSet.load(path)

    def test_duplicate_reviewer_id_raises(self, tmp_path):
        path = _write(tmp_path, _VALID_TRANSLATE + (
            '[[reviewer]]\nid = "dup"\ninstructions = "a"\n'
            '[[reviewer]]\nid = "dup"\ninstructions = "b"\n'
        ))
        with pytest.raises(AgentConfigError):
            AgentSet.load(path)

    def test_reviewer_without_id_raises(self, tmp_path):
        path = _write(tmp_path, _VALID_TRANSLATE + (
            '[[reviewer]]\ninstructions = "no id here"\n'
        ))
        with pytest.raises(AgentConfigError):
            AgentSet.load(path)

    def test_non_boolean_from_rules_raises(self, tmp_path):
        path = _write(tmp_path, _VALID_TRANSLATE + (
            '[[reviewer]]\nid = "r"\nfrom_rules = "yes"\n'
        ))
        with pytest.raises(AgentConfigError):
            AgentSet.load(path)

    def test_zero_reviewers_raises(self, tmp_path):
        """No panel means every translation is accepted unread -- refused as a footgun."""
        path = _write(tmp_path, _VALID_TRANSLATE)
        with pytest.raises(AgentConfigError):
            AgentSet.load(path)


class TestTranslateSectionValidation:
    def test_missing_translate_section_raises(self, tmp_path):
        path = _write(tmp_path, _VALID_REVIEWER)
        with pytest.raises(AgentConfigError):
            AgentSet.load(path)

    def test_empty_translate_instructions_raises(self, tmp_path):
        path = _write(tmp_path, '[translate]\ninstructions = "   "\n' + _VALID_REVIEWER)
        with pytest.raises(AgentConfigError):
            AgentSet.load(path)

    def test_non_string_translate_instructions_raises(self, tmp_path):
        path = _write(tmp_path, "[translate]\ninstructions = 42\n" + _VALID_REVIEWER)
        with pytest.raises(AgentConfigError):
            AgentSet.load(path)


class TestFileErrors:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(AgentConfigError):
            AgentSet.load(tmp_path / "absent.toml")

    def test_invalid_toml_raises(self, tmp_path):
        path = _write(tmp_path, "not = = valid = toml")
        with pytest.raises(AgentConfigError):
            AgentSet.load(path)


class TestTranslateBudget:
    """The budget scales with source length but is clamped at both ends."""

    def test_short_source_clamps_up_to_the_floor(self):
        limits = Limits(translate_tokens_per_source_char=8,
                        translate_tokens_floor=256, translate_tokens_ceiling=1024)
        assert limits.translate_budget(0) == 256
        assert limits.translate_budget(10) == 256  # 80 scaled, still below floor

    def test_long_source_clamps_down_to_the_ceiling(self):
        limits = Limits(translate_tokens_per_source_char=8,
                        translate_tokens_floor=256, translate_tokens_ceiling=1024)
        assert limits.translate_budget(1000) == 1024  # 8000 scaled, capped

    def test_mid_range_source_scales_linearly(self):
        limits = Limits(translate_tokens_per_source_char=8,
                        translate_tokens_floor=256, translate_tokens_ceiling=1024)
        assert limits.translate_budget(100) == 800

    def test_floor_boundary_is_inclusive(self):
        limits = Limits(translate_tokens_per_source_char=1,
                        translate_tokens_floor=100, translate_tokens_ceiling=1000)
        assert limits.translate_budget(100) == 100

    def test_ceiling_boundary_is_inclusive(self):
        limits = Limits(translate_tokens_per_source_char=1,
                        translate_tokens_floor=100, translate_tokens_ceiling=1000)
        assert limits.translate_budget(1000) == 1000


class TestLimitsValidation:
    def _with_limits(self, tmp_path, limits_body: str):
        return _write(tmp_path, _VALID_TRANSLATE + _VALID_REVIEWER + "[limits]\n" + limits_body)

    def test_valid_limits_load(self, tmp_path):
        path = self._with_limits(tmp_path, (
            "translate_tokens_per_source_char = 4\n"
            "translate_tokens_floor = 128\n"
            "translate_tokens_ceiling = 512\n"
            "review_tokens = 300\n"
        ))
        limits = AgentSet.load(path).limits
        assert limits.translate_tokens_per_source_char == 4
        assert limits.translate_tokens_floor == 128
        assert limits.translate_tokens_ceiling == 512
        assert limits.review_tokens == 300

    def test_boolean_limit_rejected(self, tmp_path):
        """bool is an int subclass; `= true` must not read as a budget of 1."""
        path = self._with_limits(tmp_path, "translate_tokens_floor = true\n")
        with pytest.raises(AgentConfigError):
            AgentSet.load(path)

    def test_zero_limit_rejected(self, tmp_path):
        path = self._with_limits(tmp_path, "review_tokens = 0\n")
        with pytest.raises(AgentConfigError):
            AgentSet.load(path)

    def test_negative_limit_rejected(self, tmp_path):
        path = self._with_limits(tmp_path, "translate_tokens_ceiling = -5\n")
        with pytest.raises(AgentConfigError):
            AgentSet.load(path)

    def test_floor_above_ceiling_rejected(self, tmp_path):
        """If the floor exceeds the ceiling, every budget would just be the ceiling."""
        path = self._with_limits(tmp_path, (
            "translate_tokens_floor = 2000\n"
            "translate_tokens_ceiling = 1000\n"
        ))
        with pytest.raises(AgentConfigError):
            AgentSet.load(path)


class TestSectionShape:
    """A section written with the wrong TOML *shape* must be diagnosed, not crash.

    Every case here used to escape :class:`AgentConfigError` entirely and reach the CLI as a
    raw ``TypeError``/``AttributeError`` traceback, because the loaders type-checked some
    sections and not others.
    """

    @pytest.mark.parametrize("section", ["translate", "limits", "revision", "context",
                                         "leniency"])
    def test_a_scalar_where_a_table_belongs_names_the_section(self, tmp_path, section):
        # `limits = 5` instead of `[limits]`: an ordinary slip that used to reach set(section).
        body = f"{section} = 5\n" + _VALID_TRANSLATE + _VALID_REVIEWER
        if section == "translate":  # the valid [translate] would collide with the scalar
            body = "translate = 5\n" + _VALID_REVIEWER
        with pytest.raises(AgentConfigError, match=rf"\[{section}\] must be a table, got int"):
            AgentSet.load(_write(tmp_path, body))

    def test_reviewer_as_a_single_table_says_to_double_the_brackets(self, tmp_path):
        """`[reviewer]` for `[[reviewer]]` -- the likeliest mistake in the file. Iterating a
        table yields its keys, so this used to die on ``'str' object has no attribute 'get'``."""
        path = _write(tmp_path, _VALID_TRANSLATE
                      + '[reviewer]\nid = "style"\ninstructions = "Check."\n')
        with pytest.raises(AgentConfigError, match=r"write \[\[reviewer\]\], not \[reviewer\]"):
            AgentSet.load(path)

    def test_a_reviewer_array_holding_a_non_table_names_its_position(self, tmp_path):
        path = _write(tmp_path, "reviewer = [5]\n" + _VALID_TRANSLATE)
        with pytest.raises(AgentConfigError,
                           match=r"\[\[reviewer\]\] entry 0 must be a table, got int"):
            AgentSet.load(path)

    def test_a_per_reviewer_leniency_scalar_is_diagnosed(self, tmp_path):
        path = _write(tmp_path, _VALID_TRANSLATE
                      + '[[reviewer]]\nid = "r"\ninstructions = "x"\nleniency = 5\n')
        with pytest.raises(AgentConfigError, match="leniency must be a table, got int"):
            AgentSet.load(path)


class TestTypeInvariants:
    """The bounds live in the types, so a directly-constructed object cannot be invalid.

    Regression: ``Limits(translate_tokens_floor=5000, translate_tokens_ceiling=10)`` used to
    construct happily and quietly return the *floor* from every ``translate_budget`` call,
    because the check lived in the file parser rather than in the type.
    """

    def test_a_floor_above_the_ceiling_cannot_be_constructed(self):
        with pytest.raises(ValueError, match="translate_tokens_floor"):
            Limits(translate_tokens_floor=5000, translate_tokens_ceiling=10)

    @pytest.mark.parametrize("field", ["translate_tokens_per_source_char",
                                       "translate_tokens_floor", "translate_tokens_ceiling",
                                       "review_tokens"])
    def test_every_limit_must_be_at_least_one(self, field):
        with pytest.raises(ValueError, match=rf"{field} must be >= 1"):
            Limits(**{field: 0})

    def test_equal_floor_and_ceiling_is_valid(self):
        assert Limits(translate_tokens_floor=64,
                      translate_tokens_ceiling=64).translate_budget(1000) == 64

    @pytest.mark.parametrize("field", ["before_units", "after_units", "context_char_budget",
                                       "glossary_terms", "reference_examples",
                                       "reference_revision_examples"])
    def test_every_context_count_must_be_non_negative(self, field):
        assert Context(**{field: 0})  # the boundary is valid
        with pytest.raises(ValueError, match=rf"{field} must be >= 0"):
            Context(**{field: -1})

    def test_the_candidate_pool_must_hold_at_least_one_candidate(self):
        with pytest.raises(ValueError, match="reference_candidate_pool must be >= 1"):
            Context(reference_candidate_pool=0)

    @pytest.mark.parametrize("field", ["reference_min_score", "reference_mmr_lambda",
                                       "reference_lexical_min_score",
                                       "reference_embedding_min_score"])
    @pytest.mark.parametrize("bad", [-0.01, 1.01])
    def test_every_fraction_is_confined_to_the_unit_interval(self, field, bad):
        with pytest.raises(ValueError, match=rf"{field} must be in \[0, 1\]"):
            Context(**{field: bad})

    @pytest.mark.parametrize("field", ["reference_min_score", "reference_mmr_lambda",
                                       "reference_lexical_min_score",
                                       "reference_embedding_min_score"])
    @pytest.mark.parametrize("good", [0.0, 1.0])
    def test_both_ends_of_the_unit_interval_are_valid(self, field, good):
        assert getattr(Context(**{field: good}), field) == good

    def test_a_reviewer_budget_of_zero_tokens_is_refused(self):
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            Reviewer("r", instructions="x", max_tokens=0)

    def test_an_unset_reviewer_budget_is_allowed(self):
        assert Reviewer("r", instructions="x").max_tokens is None

    @pytest.mark.parametrize("field", ["max_revisions", "max_repairs"])
    def test_revision_counts_must_be_non_negative(self, field):
        base = dict(translator_instructions="t", reviewers=(Reviewer("r", instructions="x"),))
        assert AgentSet(**base, **{field: 0})
        with pytest.raises(ValueError, match=rf"{field} must be >= 0"):
            AgentSet(**base, **{field: -1})
