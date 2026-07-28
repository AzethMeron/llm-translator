"""The configurable capabilities: per-reviewer budgets, revision counts, and the context
window -- both that the agent file parses them, and that the harness acts on them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from translator.agents import TranslationAgents
from translator.roles import AgentConfigError, AgentSet, Context, Leniency, Limits, Reviewer
from translator.memory import TranslationMemory
from translator.rules import RuleSet
from transunit.glossary import Term
from transunit.units import Status, Unit


class RecordingClient:
    """Captures the (role, max_tokens, last-user-message) of every call."""

    def __init__(self, translation: str = "ok") -> None:
        self.translation = translation
        self.calls: list[tuple[str, int | None, str]] = []

    def complete_json(self, messages, schema, *, role="<none>", max_tokens=None, **_):
        self.calls.append((role, max_tokens, messages[-1].content))
        if role == "translate":
            return {"translation": self.translation}
        return {"acceptable": True, "issues": []}

    def close(self) -> None:
        pass

    def translate_prompt(self) -> str:
        return next(content for role, _, content in self.calls if role == "translate")

    def tokens(self) -> dict[str, int | None]:
        return {role: mt for role, mt, _ in self.calls}


def unit(**overrides) -> Unit:
    fields = dict(unit_id="u", rel_path="f", line_no=1, span_start=0, span_end=1,
                  command="", kind="T", source="hello", placeholders=())
    fields.update(overrides)
    return Unit(**fields)


def agent_set(*, reviewers=None, limits=None, context=None,
              max_revisions=1, max_repairs=1, repair_truncated_json=True) -> AgentSet:
    return AgentSet(
        translator_instructions="Translate.",
        reviewers=reviewers or (Reviewer("accuracy", instructions="check"),),
        limits=limits or Limits(),
        context=context or Context(),
        max_revisions=max_revisions, max_repairs=max_repairs,
        repair_truncated_json=repair_truncated_json)


# -- loading -------------------------------------------------------------------


def write_agents(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "agents.toml"
    path.write_text(body, encoding="utf-8")
    return path


BASE = """
[translate]
instructions = "Translate."
[[reviewer]]
id = "accuracy"
instructions = "Check."
"""


class TestLoadingPowerSections:
    def test_context_section_is_parsed(self, tmp_path: Path) -> None:
        path = write_agents(tmp_path, BASE + """
[context]
before_units = 5
after_units = 1
include_translations = false
glossary_terms = 3
anonymous_subject = "on"
""")
        ctx = AgentSet.load(path).context
        assert (ctx.before_units, ctx.after_units, ctx.include_translations,
                ctx.glossary_terms, ctx.anonymous_subject) == (5, 1, False, 3, "on")

    def test_revision_section_is_parsed(self, tmp_path: Path) -> None:
        path = write_agents(tmp_path, BASE + "\n[revision]\nmax_revisions = 4\nmax_repairs = 0\n")
        loaded = AgentSet.load(path)
        assert (loaded.max_revisions, loaded.max_repairs) == (4, 0)

    def test_repair_truncated_json_is_parsed(self, tmp_path: Path) -> None:
        path = write_agents(tmp_path, BASE + "\n[revision]\nrepair_truncated_json = false\n")
        assert AgentSet.load(path).repair_truncated_json is False

    def test_reference_knobs_are_parsed(self, tmp_path: Path) -> None:
        path = write_agents(tmp_path, BASE + """
[context]
reference_examples = 4
reference_min_score = 0.3
reference_revision_examples = 2
""")
        ctx = AgentSet.load(path).context
        assert (ctx.reference_examples, ctx.reference_min_score,
                ctx.reference_revision_examples) == (4, 0.3, 2)

    def test_reference_defaults_are_off(self, tmp_path: Path) -> None:
        # The whole feature is opt-in: absent config leaves retrieval disabled and the floor at
        # its documented default.
        ctx = AgentSet.load(write_agents(tmp_path, BASE)).context
        assert ctx.reference_examples == 0
        assert ctx.reference_revision_examples == 0
        assert ctx.reference_min_score == 0.30

    def test_an_integer_min_score_is_accepted(self, tmp_path: Path) -> None:
        # 0 and 1 are natural fractions; an int must widen rather than be rejected as non-float.
        path = write_agents(tmp_path, BASE + "\n[context]\nreference_min_score = 1\n")
        assert AgentSet.load(path).context.reference_min_score == 1.0

    def test_reference_learn_statuses_are_parsed(self, tmp_path: Path) -> None:
        path = write_agents(tmp_path, BASE +
                            '\n[context]\nreference_learn_statuses = ["verified", "translated"]\n')
        assert AgentSet.load(path).context.reference_learn_statuses == \
            (Status.VERIFIED, Status.TRANSLATED)

    def test_reference_learn_defaults_to_empty_off(self, tmp_path: Path) -> None:
        assert AgentSet.load(write_agents(tmp_path, BASE)).context.reference_learn_statuses == ()

    def test_duplicate_learn_statuses_are_collapsed(self, tmp_path: Path) -> None:
        path = write_agents(tmp_path, BASE +
                            '\n[context]\nreference_learn_statuses = ["verified", "verified"]\n')
        assert AgentSet.load(path).context.reference_learn_statuses == (Status.VERIFIED,)

    @pytest.mark.parametrize("bad", [
        '["rejected"]',   # has no usable target
        '["skipped"]',
        '["pending"]',
        '["verified", "rejected"]',
        '["nonsense"]',
        '"verified"',     # not an array
        '[5]',            # not a string
    ])
    def test_bad_learn_statuses_are_rejected(self, tmp_path: Path, bad: str) -> None:
        path = write_agents(tmp_path, BASE + f"\n[context]\nreference_learn_statuses = {bad}\n")
        with pytest.raises(AgentConfigError):
            AgentSet.load(path)

    def test_defaults_apply_when_sections_absent(self, tmp_path: Path) -> None:
        loaded = AgentSet.load(write_agents(tmp_path, BASE))
        assert loaded.context.before_units == 3 and loaded.context.after_units == 2
        assert loaded.max_revisions == 2 and loaded.max_repairs == 2
        assert loaded.repair_truncated_json is True

    def test_per_reviewer_max_tokens_is_parsed(self, tmp_path: Path) -> None:
        path = write_agents(tmp_path, """
[translate]
instructions = "Translate."
[[reviewer]]
id = "quick"
instructions = "Check punctuation."
max_tokens = 128
[[reviewer]]
id = "full"
instructions = "Check meaning."
""")
        reviewers = {r.id: r for r in AgentSet.load(path).reviewers}
        assert reviewers["quick"].max_tokens == 128
        assert reviewers["full"].max_tokens is None

    @pytest.mark.parametrize("bad", [
        "[context]\nbefore_units = -1",
        "[context]\nbefore_units = true",
        "[context]\ninclude_translations = 1",
        "[context]\nanonymous_subject = 5",
        "[revision]\nmax_revisions = -1",
        "[revision]\nmax_repairs = 1.5",
        "[revision]\nrepair_truncated_json = 1",       # not a bool
        "[revision]\ntypo_key = 3",                    # unknown key must not pass silently
        "[context]\nreference_examples = -1",
        "[context]\nreference_examples = true",
        "[context]\nreference_revision_examples = 2.5",
        "[context]\nreference_min_score = 1.5",     # above the [0, 1] range
        "[context]\nreference_min_score = -0.1",    # below it
        "[context]\nreference_min_score = true",     # bool is not a fraction
        "[context]\nreference_min_score = \"high\"",  # not a number
        "[context]\nreference_typo = 3",             # unknown key must not pass silently
    ])
    def test_bad_values_are_rejected(self, tmp_path: Path, bad: str) -> None:
        with pytest.raises(AgentConfigError):
            AgentSet.load(write_agents(tmp_path, BASE + "\n" + bad + "\n"))

    def test_zero_reviewer_max_tokens_is_rejected(self, tmp_path: Path) -> None:
        path = write_agents(tmp_path, """
[translate]
instructions = "Translate."
[[reviewer]]
id = "r"
instructions = "Check."
max_tokens = 0
""")
        with pytest.raises(AgentConfigError, match="max_tokens"):
            AgentSet.load(path)


# -- budgets in effect ---------------------------------------------------------


class TestPerReviewerBudgets:
    def test_review_budget_prefers_the_reviewers_own_value(self) -> None:
        limits = Limits(review_tokens=999)
        assert limits.review_budget(Reviewer("a", instructions="x", max_tokens=200)) == 200
        assert limits.review_budget(Reviewer("b", instructions="y")) == 999

    def test_each_reviewer_gets_its_own_token_budget_at_call_time(self) -> None:
        reviewers = (Reviewer("small", instructions="x", max_tokens=200),
                     Reviewer("big", instructions="y"))
        client = RecordingClient()
        agents = TranslationAgents(client, RuleSet(), [],
                                   agent_set=agent_set(reviewers=reviewers,
                                                       limits=Limits(review_tokens=999)))
        agents.process(unit())
        tokens = client.tokens()
        assert tokens["small"] == 200
        assert tokens["big"] == 999


class TestRevisionBudgetFromConfig:
    def test_config_revision_count_is_used_when_not_overridden(self) -> None:
        client = RecordingClient(translation="ok")
        agents = TranslationAgents(client, RuleSet(), [],
                                   agent_set=agent_set(max_revisions=0, max_repairs=0))
        assert agents.max_revisions == 0 and agents.max_repairs == 0

    def test_an_explicit_argument_overrides_the_config(self) -> None:
        agents = TranslationAgents(RecordingClient(), RuleSet(), [],
                                   agent_set=agent_set(max_revisions=5), max_revisions=1)
        assert agents.max_revisions == 1

    def test_a_negative_effective_budget_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            TranslationAgents(RecordingClient(), RuleSet(), [],
                              agent_set=agent_set(), max_revisions=-1)

    def test_config_repair_truncated_json_flag_is_used_when_not_overridden(self) -> None:
        agents = TranslationAgents(RecordingClient(), RuleSet(), [],
                                   agent_set=agent_set(repair_truncated_json=False))
        assert agents.repair_truncated_json is False

    def test_an_explicit_repair_truncated_json_argument_overrides_the_config(self) -> None:
        agents = TranslationAgents(RecordingClient(), RuleSet(), [],
                                   agent_set=agent_set(repair_truncated_json=True),
                                   repair_truncated_json=False)
        assert agents.repair_truncated_json is False


# -- context window ------------------------------------------------------------


class TestContextWindow:
    def _prompt(self, ctx: Context, **unit_kwargs) -> str:
        client = RecordingClient()
        agents = TranslationAgents(client, RuleSet(), [], agent_set=agent_set(context=ctx),
                                   memory=TranslationMemory())
        agents.process(unit(**unit_kwargs))
        return client.translate_prompt()

    def test_before_is_limited_to_the_nearest_n(self) -> None:
        prompt = self._prompt(Context(before_units=2),
                              context_before=("far", "mid", "near"))
        head = prompt.split("Line to translate")[0]
        assert "near" in head and "mid" in head and "far" not in head

    def test_after_is_limited_to_the_nearest_n(self) -> None:
        prompt = self._prompt(Context(after_units=1),
                              context_after=("next", "later"))
        head = prompt.split("Line to translate")[0]
        assert "next" in head and "later" not in head

    def test_before_units_zero_shows_no_preceding_context(self) -> None:
        # Regression guard: slicing [-0:] returns everything, which would defeat a 0 setting.
        prompt = self._prompt(Context(before_units=0, after_units=0),
                              context_before=("far", "mid", "near"))
        assert "Surrounding context" not in prompt
        head = prompt.split("Line to translate")[0]
        assert "far" not in head and "mid" not in head and "near" not in head

    def test_glossary_terms_are_capped(self) -> None:
        glossary = [Term(source=f"term{i}", target=f"t{i}", category="item") for i in range(20)]
        source = " ".join(f"term{i}" for i in range(20))
        client = RecordingClient()
        agents = TranslationAgents(client, RuleSet(), glossary,
                                   agent_set=agent_set(context=Context(glossary_terms=4)))
        agents.process(unit(source=source))
        block = client.translate_prompt().split("Line to translate")[0]
        assert block.count(" -> ") == 4


class TestContextTranslations:
    """Both sides show a neighbour's translation when memory has one; this was the fix."""

    def _prompt(self, ctx: Context, memory: TranslationMemory, **unit_kwargs) -> str:
        client = RecordingClient()
        agents = TranslationAgents(client, RuleSet(), [], agent_set=agent_set(context=ctx),
                                   memory=memory)
        agents.process(unit(**unit_kwargs))
        return client.translate_prompt()

    def test_following_context_shows_a_translation_when_available(self) -> None:
        # The key correction: a *following* neighbour that is already translated (a duplicate,
        # a resumed run, a caller-chosen order) shows its target, not just its source.
        memory = TranslationMemory({"the next line": "następna linia"})
        prompt = self._prompt(Context(), memory, context_after=("the next line",))
        assert "następna linia" in prompt

    def test_preceding_context_shows_a_translation_when_available(self) -> None:
        memory = TranslationMemory({"the line before": "poprzednia linia"})
        prompt = self._prompt(Context(), memory, context_before=("the line before",))
        assert "poprzednia linia" in prompt

    def test_untranslated_neighbour_is_shown_but_not_claimed_as_established(self) -> None:
        # An untranslated neighbour still appears in the surrounding-context passage (in the
        # source language, like every neighbour), but must not appear in the "already
        # translated" list -- there is no per-neighbour "[not yet translated]" marker to leak as
        # a null-value field; the model tells the two states apart by which list a line is in.
        prompt = self._prompt(Context(), TranslationMemory(), context_after=("unseen",))
        assert "unseen" in prompt
        assert "not yet translated" not in prompt
        assert "Already-translated neighbouring lines" not in prompt

    def test_include_translations_false_shows_source_only(self) -> None:
        memory = TranslationMemory({"the next line": "następna linia"})
        prompt = self._prompt(Context(include_translations=False), memory,
                              context_after=("the next line",))
        assert "następna linia" not in prompt
        assert "the next line" in prompt


class TestAnonymousSubject:
    def test_the_configured_stand_in_replaces_a_context_placeholder(self) -> None:
        # A placeholder in a neighbour's line, speaker unknown, is replaced with the configured
        # word rather than left raw (which would teach the model to emit one).
        client = RecordingClient()
        agents = TranslationAgents(
            client, RuleSet(), [],
            agent_set=agent_set(context=Context(anonymous_subject="ON-DUTY")),
            memory=TranslationMemory())
        agents.process(unit(speaker=None, context_before=("[[0]] arrives.",)))
        head = client.translate_prompt().split("Line to translate")[0]
        assert "ON-DUTY arrives." in head
        assert "[[0]]" not in head


class TestValidationEdges:
    def test_leniency_window_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError):
            Leniency(window=0)

    def test_leniency_max_bad_below_zero_is_refused(self) -> None:
        with pytest.raises(ValueError):
            Leniency(max_bad=-1)

    def test_a_non_table_leniency_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(AgentConfigError):
            AgentSet.load(write_agents(tmp_path, BASE + '\nleniency = "loose"\n'))

    def test_reviewer_instructions_must_be_a_string(self, tmp_path: Path) -> None:
        body = ('[translate]\ninstructions = "Translate."\n'
                '[[reviewer]]\nid = "r"\ninstructions = 5\n')
        with pytest.raises(AgentConfigError):
            AgentSet.load(write_agents(tmp_path, body))
