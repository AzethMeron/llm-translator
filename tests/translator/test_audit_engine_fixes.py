"""Regressions for the deep-audit round on the translator engine.

Each test pins a specific defect the audit found: a disputed translation being salvaged, a
dangling revision header, a from_rules reviewer with nothing to judge, first-match name
resolution, an unvalidated report_every, a CLI budget stacktrace, and silently-ignored
config typos.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from translator.agents import TranslationAgents
from translator.backend import LlmContentError
from translator.roles import AgentConfigError, AgentSet, Reviewer
from translator.rules import RuleConfigError, RuleSet
from translator.runner import RunnerError, run_batch
from transunit.glossary import Term
from transunit.units import Status, Unit

RULES = RuleSet(advisory_rules=(("register", "keeps register"),))
PANEL = AgentSet(translator_instructions="Translate.",
                 reviewers=(Reviewer("accuracy", instructions="check fidelity"),))


class FakeClient:
    def __init__(self, responses):
        self.responses = {r: list(v) for r, v in responses.items()}
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, messages, schema, *, role="<none>", **_):
        self.calls.append((role, messages[-1].content))
        queue = self.responses.get(role)
        if queue:
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return {"acceptable": True, "issues": []}

    def close(self):
        pass


def plain_unit(**over) -> Unit:
    fields = dict(unit_id="u", rel_path="f", line_no=1, span_start=0, span_end=1,
                  command="", kind="T", source="hello", placeholders=())
    fields.update(over)
    return Unit(**fields)


def agents(client, glossary=(), rules=RULES, **kw) -> TranslationAgents:
    kw.setdefault("agent_set", PANEL)
    return TranslationAgents(client, rules, list(glossary), **kw)


class TestSalvageDoesNotShipDisputedText:
    def test_a_content_error_after_an_objection_rejects_rather_than_salvaging(self) -> None:
        # round 0 passes mechanics but the reviewer objects; round 1's generation then errors
        # repeatedly, exhausting round 1's own repair budget (max_repairs=2, so 3 calls). The
        # round-0 text was rejected by review, so it must NOT be shipped as TRANSLATED with a
        # "review did not complete" message -- the run must record REJECTED.
        client = FakeClient({
            "translate": [{"translation": "round zero"},
                          LlmContentError("garbage", role="translate"),
                          LlmContentError("garbage", role="translate"),
                          LlmContentError("garbage", role="translate")],
            "accuracy": [{"acceptable": False, "issues": ["too literal"]}],
        })
        outcome = agents(client, max_revisions=1).process(plain_unit())
        assert outcome.status is Status.REJECTED
        assert outcome.target is None
        # 1 translate + 1 review (round 0) + 3 translate (round 1's exhausted repair budget)
        assert len(client.calls) == 5

    def test_a_content_error_during_review_still_salvages_a_sound_translation(self) -> None:
        # The legitimate salvage path: review itself errors (no objection recorded), so the
        # mechanically-sound translation survives as TRANSLATED.
        client = FakeClient({
            "translate": [{"translation": "round zero"}],
            "accuracy": [LlmContentError("reviewer garbage", role="accuracy")],
        })
        outcome = agents(client).process(plain_unit())
        assert outcome.status is Status.TRANSLATED
        assert outcome.target == "round zero"


class TestObjectionAlwaysCarriesAReason:
    def test_an_empty_objection_is_given_a_synthetic_reason(self) -> None:
        client = FakeClient({
            "translate": [{"translation": "a"}, {"translation": "b"}],
            "accuracy": [{"acceptable": False, "issues": []},  # objects with no reason
                         {"acceptable": True, "issues": []}],
        })
        agents(client, max_revisions=1).process(plain_unit())
        revision = [p for r, p in client.calls if r == "translate"][1]
        assert "marked unacceptable" in revision
        # And no dangling "reasons:" header with nothing under it.
        assert "that already works:\n\n" not in revision
        assert "that already works:\n-" in revision


class TestFromRulesNeedsAdvisory:
    def test_from_rules_reviewer_with_empty_advisory_is_refused(self) -> None:
        panel = AgentSet(translator_instructions="t", reviewers=(Reviewer("r", from_rules=True),))
        with pytest.raises(AgentConfigError, match="no \\[\\[advisory\\]\\]"):
            TranslationAgents(FakeClient({}), RuleSet(advisory_rules=()), [], agent_set=panel)

    def test_from_rules_reviewer_with_advisory_is_fine(self) -> None:
        panel = AgentSet(translator_instructions="t", reviewers=(Reviewer("r", from_rules=True),))
        TranslationAgents(FakeClient({}), RULES, [], agent_set=panel)  # must not raise


class TestLongestNameMatch:
    def test_the_exact_longer_name_is_not_shadowed_by_a_shorter_prefix(self) -> None:
        glossary = [Term(source="Ann", target="Ann", category="character", entity_id=1),
                    Term(source="Anna", target="Anna", category="character", entity_id=2)]
        harness = agents(FakeClient({"translate": [{"translation": "x"}]}), glossary=glossary)
        assert harness._speaker_label(plain_unit(speaker="7:Anna")) == "Anna (character 7)"


class TestRunBatchValidatesReportEvery:
    class _Stub:
        memory = None

        def process(self, unit):  # pragma: no cover - never reached
            raise AssertionError

    def test_zero_report_every_is_a_structured_error(self, tmp_path: Path) -> None:
        with pytest.raises(RunnerError, match="report_every"):
            run_batch(self._Stub(), [plain_unit()], tmp_path / "j.jsonl", report_every=0)


class TestConfigLoadersRejectUnknownKeys:
    def _agents(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "agents.toml"
        path.write_text('[translate]\ninstructions="t"\n[[reviewer]]\nid="a"\n'
                        'instructions="c"\n' + body, encoding="utf-8")
        return path

    def test_unknown_top_level_agent_key_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(AgentConfigError, match="unknown key"):
            AgentSet.load(self._agents(tmp_path, "[revison]\nmax_revisions=1\n"))

    def test_unknown_context_key_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(AgentConfigError, match="unknown key"):
            AgentSet.load(self._agents(tmp_path, "[context]\nbefor_units=2\n"))

    def test_unknown_reviewer_key_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "agents.toml"
        path.write_text('[translate]\ninstructions="t"\n[[reviewer]]\nid="a"\n'
                        'instructions="c"\nmax_token=5\n', encoding="utf-8")
        with pytest.raises(AgentConfigError, match="unknown key"):
            AgentSet.load(path)

    def test_unknown_rule_key_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "rules.toml"
        path.write_text("[limits]\nmax_line_colums = 90\n", encoding="utf-8")  # typo
        with pytest.raises(RuleConfigError, match="unknown key"):
            RuleSet.load(path)
