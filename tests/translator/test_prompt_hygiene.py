"""An optional prompt section is included only when it has content.

The requirement this pins: nothing should leak a dangling header for an option that is not
in use -- no "Style policy:" with nothing under it, no "Speaker:" for narration, no
"Established terminology" for an empty glossary. Empty sections are churn that change the
prompt (and so the model's output) while conveying nothing.
"""
from __future__ import annotations

from translator.agents import TranslationAgents
from translator.roles import AgentSet, Reviewer
from translator.rules import RuleSet
from transunit.glossary import Term
from transunit.units import Unit

AGENTS = AgentSet(
    translator_instructions="Translate.",
    reviewers=(Reviewer(id="accuracy", instructions="Check fidelity."),),
)


def build(rules: RuleSet, glossary=()) -> TranslationAgents:
    return TranslationAgents(_NoClient(), rules, list(glossary), agent_set=AGENTS)


class _NoClient:
    """Stands in for a client; these tests only build prompts, never call the model."""

    def close(self) -> None:
        pass


def plain_unit(**overrides) -> Unit:
    fields = dict(unit_id="u", rel_path="f", line_no=1, span_start=0, span_end=1,
                  command="", kind="T", source="Good morning.", placeholders=())
    fields.update(overrides)
    return Unit(**fields)


class TestSystemPrompt:
    def test_no_style_header_when_there_are_no_directives(self) -> None:
        prompt = build(RuleSet(style_directives=()))._system_prompt("Translate.",
                                                                    placeholders=False)
        assert "Style policy" not in prompt

    def test_style_header_present_when_directives_exist(self) -> None:
        prompt = build(RuleSet(style_directives=("Be terse.",)))._system_prompt(
            "Translate.", placeholders=False)
        assert "Style policy:" in prompt
        assert "- Be terse." in prompt

    def test_no_placeholder_contract_for_a_line_without_placeholders(self) -> None:
        prompt = build(RuleSet())._system_prompt("Translate.", placeholders=False)
        assert "placeholders written as" not in prompt

    def test_placeholder_contract_present_for_a_line_with_placeholders(self) -> None:
        prompt = build(RuleSet())._system_prompt("Translate.", placeholders=True)
        assert "placeholders written as [[0]]" in prompt

    def test_no_section_is_ever_an_empty_header(self) -> None:
        # A section is header + body joined; a blank line followed immediately by a colon-
        # terminated line with nothing after it would be a leak. Assert no "\n\n" splits into
        # a fragment that is only a header.
        prompt = build(RuleSet(style_directives=()))._system_prompt("Translate.",
                                                                    placeholders=False)
        for section in prompt.split("\n\n"):
            assert section.strip(), "empty section leaked into the system prompt"


class TestUnitBlock:
    def test_no_speaker_line_for_a_speakerless_unit(self) -> None:
        block = build(RuleSet())._unit_block(plain_unit(speaker=None))
        assert "Speaker" not in block

    def test_speaker_line_present_when_a_speaker_is_known(self) -> None:
        block = build(RuleSet())._unit_block(plain_unit(speaker="narrator"))
        assert "Speaker: narrator" in block

    def test_no_terminology_header_for_an_empty_glossary(self) -> None:
        block = build(RuleSet())._unit_block(plain_unit())
        assert "Established terminology" not in block

    def test_terminology_header_present_when_a_term_matches(self) -> None:
        glossary = [Term(source="door", target="drzwi", category="item")]
        block = build(RuleSet(), glossary)._unit_block(plain_unit(source="the door"))
        assert "Established terminology" in block
        assert "door -> drzwi" in block

    def test_no_context_headers_when_there_is_no_context(self) -> None:
        block = build(RuleSet())._unit_block(plain_unit())
        assert "Surrounding context" not in block

    def test_no_length_limit_line_when_the_unit_has_no_budget(self) -> None:
        block = build(RuleSet())._unit_block(plain_unit(max_columns=None))
        assert "Length limit" not in block

    def test_length_limit_line_present_when_the_unit_has_a_budget(self) -> None:
        block = build(RuleSet())._unit_block(plain_unit(max_columns=42))
        assert "Length limit: 42" in block

    def test_no_unit_block_section_is_empty(self) -> None:
        block = build(RuleSet())._unit_block(plain_unit(speaker=None))
        for section in block.split("\n\n"):
            assert section.strip(), "empty section leaked into the unit block"
