"""Tests for the sequential agent harness, driven by a scripted fake model.

Self-contained: the agent set, rules and glossary are built here rather than loaded from the
shipped config, so the harness's behaviour is pinned independently of what any particular
prompt file happens to say. Inputs are English->Polish, but nothing here depends on the pair
-- the harness is language-agnostic.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from translator.agents import Attempt, TranslationAgents
from translator.backend import (
    LlmClient,
    LlmContentError,
    LlmError,
    LlmIncompleteJsonError,
    LlmRefusalError,
    LlmTruncationError,
)
from translator.roles import AgentSet, Context, Leniency, Limits, Reviewer
from translator.rules import RuleSet
from transunit.glossary import Term
from transunit.reference import GrowableLexicalRetriever, LexicalRetriever, ReferenceEntry
from transunit.units import Status, Unit

GLOSSARY = [
    Term(source="Anna", target="Anna", category="character", entity_id=9),
    Term(source="Whitecliff", target="Whitecliff", category="name", entity_id=1),
    Term(source="mithril", target="mithryl", category="item", entity_id=2),
]

RULES = RuleSet(
    max_line_columns=110,
    style_directives=("Write natural Polish.",),
    advisory_rules=(("register", "Keeps the speaker's register."),),
)

AGENTS = AgentSet(
    translator_instructions="You translate from English into Polish.",
    reviewers=(
        Reviewer(id="accuracy", instructions="Check fidelity."),
        Reviewer(id="fluency", instructions="Check it reads as native Polish."),
        Reviewer(id="rules", from_rules=True),
    ),
    limits=Limits(),
)


def make_unit(source: str = "[[0]] opened the door.", placeholders: int = 1,
              **overrides) -> Unit:
    fields = dict(
        unit_id="u1",
        rel_path="chapter/01.txt",
        line_no=10,
        span_start=0,
        span_end=1,
        command="TEXT",
        kind="LINE",
        source=source,
        placeholders=tuple(f"%V{i}%" for i in range(placeholders)),
        speaker="9:Anna",
    )
    fields.update(overrides)
    return Unit(**fields)


class FakeClient(LlmClient):
    """An LlmClient that replays scripted responses instead of calling a server."""

    def __init__(self, responses: dict[str, list]) -> None:
        self.responses = {role: list(items) for role, items in responses.items()}
        self.calls: list[tuple[str, str]] = []
        self.default_review = {"acceptable": True, "issues": []}

    def complete_json(self, messages, schema, *, role="<none>", **_):
        self.calls.append((role, messages[-1].content))
        queue = self.responses.get(role)
        if queue:
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if role == "translate":
            raise AssertionError("fake client ran out of scripted translations")
        return dict(self.default_review)

    def close(self) -> None:
        pass


def agents(client: FakeClient, **kwargs) -> TranslationAgents:
    kwargs.setdefault("agent_set", AGENTS)
    return TranslationAgents(client, RULES, GLOSSARY, **kwargs)


class TestHappyPath:
    def test_accepted_translation_is_verified(self) -> None:
        client = FakeClient({"translate": [{"translation": "[[0]] otworzył drzwi."}]})
        outcome = agents(client).process(make_unit())
        assert outcome.status is Status.VERIFIED
        assert outcome.target == "[[0]] otworzył drzwi."
        assert outcome.rounds == 1

    def test_skipped_units_never_reach_the_model(self) -> None:
        client = FakeClient({})
        outcome = agents(client).process(replace(make_unit(), status=Status.SKIPPED))
        assert outcome.status is Status.SKIPPED
        assert client.calls == []

    @pytest.mark.parametrize("blank", ["", "   ", "\t\n "])
    def test_blank_source_is_skipped_without_reaching_the_model(self, blank: str) -> None:
        # Regression: a blank source used to be sent to the model, which -- having nothing to
        # translate -- rendered the prompt scaffolding into the target (an injectable
        # non-translation), or produced an empty target that passed as verified because the
        # non-empty rule cannot fire on a blank source. A blank payload must be skipped.
        client = FakeClient({})
        outcome = agents(client).process(make_unit(source=blank, placeholders=0))
        assert outcome.status is Status.SKIPPED
        assert outcome.target is None
        assert client.calls == []

    def test_whole_review_panel_runs_when_everything_passes(self) -> None:
        client = FakeClient({"translate": [{"translation": "[[0]] otworzył drzwi."}]})
        agents(client).process(make_unit())
        assert [role for role, _ in client.calls] == \
            ["translate", "accuracy", "fluency", "rules"]

    def test_glossary_terms_are_offered_to_the_translator(self) -> None:
        client = FakeClient({"translate": [{"translation": "Anna niesie [[0]]"}]})
        agents(client).process(make_unit(source="Anna carries the [[0]]", placeholders=1))
        _, prompt = client.calls[0]
        assert "Anna" in prompt

    def test_speaker_is_resolved_to_a_named_character(self) -> None:
        client = FakeClient({"translate": [{"translation": "[[0]] otworzył drzwi."}]})
        agents(client).process(make_unit())
        _, prompt = client.calls[0]
        assert "Anna (character 9)" in prompt


class TestMechanicalRepair:
    def test_dropped_placeholder_triggers_a_repair_attempt(self) -> None:
        client = FakeClient({"translate": [
            {"translation": "otworzył drzwi."},        # placeholder dropped
            {"translation": "[[0]] otworzył drzwi."},  # repaired
        ]})
        outcome = agents(client).process(make_unit())
        assert outcome.status is Status.VERIFIED
        assert outcome.target == "[[0]] otworzył drzwi."

    def test_repair_feedback_names_the_actual_problem(self) -> None:
        client = FakeClient({"translate": [
            {"translation": "otworzył drzwi."},
            {"translation": "[[0]] otworzył drzwi."},
        ]})
        agents(client).process(make_unit())
        _, second_prompt = client.calls[1]
        assert "placeholder set changed" in second_prompt

    def test_unrepairable_translation_is_rejected_not_shipped(self) -> None:
        client = FakeClient({"translate": [{"translation": "no placeholder here"}] * 5})
        outcome = agents(client).process(make_unit())
        assert outcome.status is Status.REJECTED
        assert outcome.target is not None
        assert any(v.rule_id == "placeholders" for v in outcome.violations)

    def test_repair_budget_is_respected(self) -> None:
        client = FakeClient({"translate": [{"translation": "broken"}] * 10})
        agents(client, max_repairs=1, max_revisions=0).process(make_unit())
        assert len([r for r, _ in client.calls if r == "translate"]) == 2

    def test_injected_untranslated_detector_forces_a_repair(self) -> None:
        # The detector is injected, so the harness can catch an echoed source without knowing
        # the language pair itself.
        client = FakeClient({"translate": [
            {"translation": "[[0]] UNTRANSLATED"},
            {"translation": "[[0]] otworzył drzwi."},
        ]})
        outcome = agents(client,
                         is_untranslated=lambda s, t: "UNTRANSLATED" in t).process(make_unit())
        assert outcome.status is Status.VERIFIED
        assert outcome.target == "[[0]] otworzył drzwi."


class TestReviewLoop:
    def test_review_objection_triggers_revision(self) -> None:
        client = FakeClient({
            "translate": [
                {"translation": "[[0]] zrobił otwarcie drzwi"},
                {"translation": "[[0]] otworzył drzwi."},
            ],
            "accuracy": [
                {"acceptable": False, "issues": ["awkward phrasing"]},
                {"acceptable": True, "issues": []},
            ],
        })
        outcome = agents(client).process(make_unit())
        assert outcome.status is Status.VERIFIED
        assert outcome.rounds == 2

    def test_panel_stops_at_the_first_objection(self) -> None:
        client = FakeClient({
            "translate": [{"translation": "[[0]] x"}] * 5,
            "accuracy": [{"acceptable": False, "issues": ["typo"]}] * 5,
        })
        agents(client, max_revisions=1).process(make_unit())
        roles = [role for role, _ in client.calls]
        assert "fluency" not in roles and "rules" not in roles

    def test_objections_are_fed_back_to_the_translator(self) -> None:
        client = FakeClient({
            "translate": [{"translation": "[[0]] a"}, {"translation": "[[0]] b"}],
            "accuracy": [{"acceptable": False, "issues": ["missing full stop"]},
                          {"acceptable": True, "issues": []}],
        })
        agents(client).process(make_unit())
        translate_prompts = [p for r, p in client.calls if r == "translate"]
        assert "missing full stop" in translate_prompts[1]

    def test_unresolved_dispute_is_flagged_not_silently_accepted(self) -> None:
        client = FakeClient({
            "translate": [{"translation": "[[0]] x"}] * 10,
            "accuracy": [{"acceptable": False, "issues": ["still wrong"]}] * 10,
        })
        outcome = agents(client, max_revisions=1).process(make_unit())
        assert outcome.status is Status.TRANSLATED
        assert outcome.error is not None
        assert outcome.status is not Status.VERIFIED

    def test_revision_loop_terminates(self) -> None:
        client = FakeClient({
            "translate": [{"translation": "[[0]] x"}] * 50,
            "accuracy": [{"acceptable": False, "issues": ["no"]}] * 50,
        })
        outcome = agents(client, max_revisions=3).process(make_unit())
        assert outcome.rounds == 4


class TestFailureHandling:
    def test_refusal_is_recorded_distinctly(self) -> None:
        client = FakeClient({"translate": [LlmRefusalError("declined", role="translate")]})
        outcome = agents(client).process(make_unit())
        assert outcome.status is Status.REJECTED
        assert outcome.target is None
        assert "refused" in (outcome.error or "")

    def test_unusable_output_rejects_this_unit_but_lets_the_run_continue(self) -> None:
        # Garbage for one input says nothing about the next unit, so it is a terminal outcome
        # for this one, not an aborted run -- but it is no longer a FREE terminal outcome: a
        # content error on translate now spends the repair budget re-asking (like every other
        # translate-time defect) before finally giving up. AGENTS' default max_repairs is 2, so
        # exhausting it takes 3 calls.
        client = FakeClient({"translate": [
            LlmContentError("response was not valid JSON", role="translate"),
            LlmContentError("response was not valid JSON", role="translate"),
            LlmContentError("response was not valid JSON", role="translate"),
        ]})
        outcome = agents(client).process(make_unit())
        assert outcome.status is Status.REJECTED
        assert "not valid JSON" in (outcome.error or "")
        assert len(client.calls) == 3

    def test_truncated_generation_rejects_this_unit(self) -> None:
        client = FakeClient({"translate": [
            LlmTruncationError("generation stopped at the ceiling", role="translate")]})
        outcome = agents(client).process(make_unit())
        assert outcome.status is Status.REJECTED
        assert "ceiling" in (outcome.error or "")

    def test_infrastructure_error_propagates_instead_of_burning_the_unit(self) -> None:
        # An LlmError must NOT become a terminal REJECTED Outcome: run_batch journals every
        # Outcome as complete, and a server outage would then destroy every remaining unit
        # while exiting 0.
        client = FakeClient({"translate": [LlmError("server exploded", role="translate")]})
        with pytest.raises(LlmError, match="server exploded"):
            agents(client).process(make_unit())

    def test_a_reviewer_content_error_abstains_without_aborting_the_panel(self) -> None:
        # A reviewer whose reply cannot be parsed must not discard the mechanically-sound
        # translation, and must not silence the reviewers after it: it abstains, they still
        # run, and the unit is kept for human review because the panel did not fully complete.
        client = FakeClient({
            "translate": [{"translation": "[[0]] otworzył drzwi."}],
            "accuracy": [LlmContentError("reviewer produced garbage", role="accuracy")],
        })
        outcome = agents(client).process(make_unit())
        assert outcome.status is Status.TRANSLATED
        assert outcome.target == "[[0]] otworzył drzwi."
        assert "review incomplete" in (outcome.error or "")
        # fluency and rules ran after accuracy abstained, instead of the panel being bypassed
        assert {role for role, _ in client.calls} >= {"accuracy", "fluency", "rules"}

    def test_a_reviewer_abstention_does_not_suppress_a_later_objection(self) -> None:
        # The core of the fix: a first reviewer that cannot be parsed must not veto a later
        # reviewer's real objection. accuracy abstains; fluency objects; the objection must
        # still drive a revision (a second translate call).
        client = FakeClient({
            "translate": [{"translation": "[[0]] otworzył drzwi."},
                          {"translation": "[[0]] rozwarł drzwi."}],
            "accuracy": [LlmContentError("garbage", role="accuracy"),
                         LlmContentError("garbage", role="accuracy")],
            "fluency": [{"acceptable": False, "issues": ["too stiff"]},
                        {"acceptable": True, "issues": []}],
        })
        outcome = agents(client).process(make_unit())
        assert sum(1 for role, _ in client.calls if role == "translate") == 2  # a revision ran
        assert outcome.status is Status.TRANSLATED  # kept for review: accuracy never evaluated

    def test_a_bad_reviewer_reply_is_logged_as_a_warning(self, caplog) -> None:
        # An abstention silently degrades the review, so it must be warned about at runtime,
        # not only recorded in the journal note.
        import logging

        client = FakeClient({
            "translate": [{"translation": "[[0]] otworzył drzwi."}],
            "accuracy": [LlmContentError("reviewer produced garbage", role="accuracy")],
        })
        with caplog.at_level(logging.WARNING, logger="translator.agents"):
            agents(client).process(make_unit())
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("accuracy" in r.getMessage() and "could not evaluate" in r.getMessage()
                   and "model reply:" in r.getMessage() for r in warnings)

    def test_leniency_gates_the_console_warning_but_the_log_gets_every_one(self, caplog) -> None:
        # Within a window of 3, up to 1 unusable reply from a reviewer stays off the console;
        # the 2nd and 3rd surface. The log (every record) always gets all three.
        import logging
        from dataclasses import replace as dc_replace

        strict = dc_replace(AGENTS.reviewers[0], leniency=Leniency(window=3, max_bad=1))
        aset = dc_replace(AGENTS, reviewers=(strict, *AGENTS.reviewers[1:]))
        client = FakeClient({
            "translate": [{"translation": "[[0]] otworzył drzwi."} for _ in range(3)],
            "accuracy": [LlmContentError("garbage", role="accuracy") for _ in range(3)],
        })
        ag = agents(client, agent_set=aset)
        with caplog.at_level(logging.WARNING, logger="translator.agents"):
            for _ in range(3):
                ag.process(make_unit())
        records = [r for r in caplog.records if r.name == "translator.agents"]
        assert len(records) == 3  # the log always receives every unusable reply
        suppressed = [getattr(r, "leniency_suppress_console", False) for r in records]
        assert suppressed == [True, False, False]  # 1st tolerated on console, 2nd/3rd surface

    def test_a_reviewer_null_issues_list_is_treated_as_no_issues(self) -> None:
        # json_object models encode "no issues" as null; the consumer must read it as empty
        # rather than crashing on `tuple(str(i) for i in None)`.
        client = FakeClient({
            "translate": [{"translation": "[[0]] otworzył drzwi."}],
            "accuracy": [{"acceptable": True, "issues": None}],
        })
        outcome = agents(client).process(make_unit())
        assert outcome.status is Status.VERIFIED

    def test_malformed_reviewer_reply_over_the_wire_abstains_warns_and_runs_the_rest(
            self, caplog) -> None:
        # End-to-end through the REAL client validation (not the scripted fake): a reviewer
        # reply that fails the schema raises LlmContentError inside complete_json; review_panel
        # must isolate it -- warn, abstain, and still run every later reviewer -- and the unit
        # is kept for human review rather than aborting the panel.
        import logging

        import httpx

        from translator.backend import ServerConfig, get_backend

        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            content = ('{"translation": "[[0]] otworzył drzwi."}' if calls["n"] == 1
                       else '{"acceptable": true, "issues": 5}')  # issues:5 fails the schema
            return httpx.Response(200, json={
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

        client = LlmClient(ServerConfig(retry_backoff_seconds=0.0), backend=get_backend("qwen"))
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        with caplog.at_level(logging.WARNING, logger="translator.agents"):
            outcome = agents(client).process(make_unit())
        client.close()

        assert outcome.status is Status.TRANSLATED           # kept for review, panel not aborted
        assert outcome.target == "[[0]] otworzył drzwi."
        assert calls["n"] == 1 + len(AGENTS.reviewers)       # translate + every reviewer ran
        assert sum(1 for r in caplog.records
                   if r.levelno == logging.WARNING) == len(AGENTS.reviewers)

    def test_a_string_encoded_acceptable_abstains_rather_than_being_misread(self) -> None:
        # 'acceptable' is left strict on purpose (F2 of the report): a json_object model that
        # writes "true"/"false" as a STRING must NOT be coerced, because a stray "false" read as
        # True would ship a rejected line as accepted. It fails the schema, so it degrades to a
        # safe abstention (kept for review) via the same isolation path -- not a silent accept.
        import httpx

        from translator.backend import ServerConfig, get_backend

        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            content = ('{"translation": "[[0]] otworzył drzwi."}' if calls["n"] == 1
                       else '{"acceptable": "true", "issues": []}')  # boolean as a string
            return httpx.Response(200, json={
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {}})

        client = LlmClient(ServerConfig(retry_backoff_seconds=0.0), backend=get_backend("qwen"))
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        outcome = agents(client).process(make_unit())
        client.close()
        assert outcome.status is Status.TRANSLATED           # abstention, not a silent VERIFIED
        assert "review incomplete" in (outcome.error or "")

    def test_outcome_applies_cleanly_onto_the_unit(self) -> None:
        client = FakeClient({"translate": [{"translation": "[[0]] otworzył drzwi."}]})
        applied = agents(client).process(make_unit()).applied()
        assert applied.status is Status.VERIFIED
        assert applied.target == "[[0]] otworzył drzwi."
        assert applied.unit_id == "u1"

    def test_rejected_outcome_records_diagnostic_notes(self) -> None:
        client = FakeClient({"translate": [LlmRefusalError("boom", role="translate")]})
        applied = agents(client).process(make_unit()).applied()
        assert any("error" in note for note in applied.notes)

    def test_negative_budgets_are_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError):
            agents(FakeClient({}), max_revisions=-1)


class TestEnvelopeTruncationRepair:
    """LlmIncompleteJsonError (a truncated-but-recoverable JSON envelope) now spends the same
    repair budget as any other translate-time defect, instead of unwinding straight to REJECTED
    with zero attempts spent -- see docs/feature-requests/json-envelope-truncation-repair.md."""

    def test_a_recovered_envelope_is_not_accepted_as_is_when_budget_remains(self) -> None:
        # The loop must not short-circuit on a merely-recovered candidate: it spends another
        # round trying for an ordinary reply before ever falling back to the repaired one.
        client = FakeClient({"translate": [
            LlmIncompleteJsonError("cut off", role="translate",
                                   recovered={"translation": "[[0]] otworzył"}),
            {"translation": "[[0]] otworzył drzwi."},
        ]})
        outcome = agents(client).process(make_unit())
        assert outcome.status is Status.VERIFIED
        assert outcome.target == "[[0]] otworzył drzwi."
        assert len([r for r, _ in client.calls if r == "translate"]) == 2

    def test_the_repair_instruction_names_the_truncation(self) -> None:
        client = FakeClient({"translate": [
            LlmIncompleteJsonError("cut off", role="translate",
                                   recovered={"translation": "[[0]] otworzył"}),
            {"translation": "[[0]] otworzył drzwi."},
        ]})
        agents(client).process(make_unit())
        _, second_prompt = client.calls[1]
        assert "JSON envelope was cut off" in second_prompt

    def test_recovered_content_is_mechanically_checked_like_any_candidate(self) -> None:
        # The recovered translation is missing the placeholder -- check_mechanical must still
        # catch that, exactly as it would for an ordinary reply. Exhausting the repair budget on
        # nothing but placeholder-dropping recoveries is REJECTED, the existing precedent for an
        # unrepairable mechanical defect.
        client = FakeClient({"translate": [
            LlmIncompleteJsonError("cut off", role="translate",
                                   recovered={"translation": "otworzył drzwi"})
        ] * 5})
        outcome = agents(client).process(make_unit())
        assert outcome.status is Status.REJECTED
        assert any(v.rule_id == "placeholders" for v in outcome.violations)

    def test_exhausting_the_budget_on_a_repaired_candidate_never_reaches_verified(self) -> None:
        # Every round is a recovered envelope, so the budget runs out still holding one. The
        # panel would accept it (no objections scripted), but it must NOT become VERIFIED --
        # the loop never independently confirmed an ordinary, complete reply.
        client = FakeClient({"translate": [
            LlmIncompleteJsonError("cut off", role="translate",
                                   recovered={"translation": "[[0]] otworzył drzwi."})
        ] * 3})  # AGENTS' default max_repairs=2 -> 3 attempts
        outcome = agents(client).process(make_unit())
        assert outcome.status is Status.TRANSLATED
        assert outcome.target == "[[0]] otworzył drzwi."
        assert outcome.error is not None
        assert "truncated JSON envelope" in outcome.error
        assert len([r for r, _ in client.calls if r == "translate"]) == 3

    def test_a_single_attempt_budget_still_recovers_one_envelope(self) -> None:
        # max_repairs=0: the loop's one and only round is the recovery itself, spent (not
        # wasted) rather than raising immediately as an ordinary content error would.
        client = FakeClient({"translate": [
            LlmIncompleteJsonError("cut off", role="translate",
                                   recovered={"translation": "[[0]] otworzył drzwi."})
        ]})
        outcome = agents(client, max_repairs=0).process(make_unit())
        assert outcome.status is Status.TRANSLATED
        assert outcome.target == "[[0]] otworzył drzwi."
        assert "truncated JSON envelope" in (outcome.error or "")

    def test_a_later_clean_round_is_not_penalised_by_an_earlier_repair(self) -> None:
        # Round 0 needs an envelope repair and then a review objection sends it to round 1;
        # round 1's generation is an ordinary, unrepaired reply. from_repair must reset per
        # _generate call, so this round-1 result is eligible for VERIFIED like any other.
        client = FakeClient({
            "translate": [
                LlmIncompleteJsonError("cut off", role="translate",
                                       recovered={"translation": "[[0]] zrobił otwarcie"}),
                {"translation": "[[0]] zrobił otwarcie drzwi"},
                {"translation": "[[0]] otworzył drzwi."},
            ],
            "accuracy": [
                {"acceptable": False, "issues": ["awkward phrasing"]},
                {"acceptable": True, "issues": []},
            ],
        })
        outcome = agents(client).process(make_unit())
        assert outcome.status is Status.VERIFIED
        assert outcome.target == "[[0]] otworzył drzwi."

    def test_repair_truncated_json_false_never_accepts_a_recovered_envelope(self) -> None:
        # A project that wants strictness: even with budget remaining and a mechanically-clean
        # recovered candidate, an envelope-truncation failure is treated exactly like any other
        # unusable reply -- re-asked, and REJECTED once the budget is exhausted, never kept.
        client = FakeClient({"translate": [
            LlmIncompleteJsonError("cut off", role="translate",
                                   recovered={"translation": "[[0]] otworzył drzwi."})
        ] * 3})  # AGENTS' default max_repairs=2 -> 3 attempts
        outcome = agents(client, repair_truncated_json=False).process(make_unit())
        assert outcome.status is Status.REJECTED
        assert outcome.target is None
        assert len([r for r, _ in client.calls if r == "translate"]) == 3

    def test_repair_truncated_json_false_still_spends_the_repair_budget_first(self) -> None:
        # Strictness only governs whether a recovered candidate may ever be ACCEPTED -- it does
        # not skip the retries: an ordinary reply on a later round still succeeds normally.
        client = FakeClient({"translate": [
            LlmIncompleteJsonError("cut off", role="translate",
                                   recovered={"translation": "[[0]] otworzył"}),
            {"translation": "[[0]] otworzył drzwi."},
        ]})
        outcome = agents(client, repair_truncated_json=False).process(make_unit())
        assert outcome.status is Status.VERIFIED
        assert outcome.target == "[[0]] otworzył drzwi."


class TestLengthIsKeptNotRejected:
    """Over-length after repairs is TRANSLATED, not REJECTED: showing the source is worse."""

    def test_overlong_translation_is_kept_and_flagged(self) -> None:
        long_text = "a" * 200
        client = FakeClient({"translate": [{"translation": long_text}] * 6})
        outcome = agents(client, max_repairs=1, max_revisions=0).process(
            make_unit(source="short", placeholders=0, max_columns=50))
        assert outcome.status is Status.TRANSLATED
        assert outcome.target == long_text
        assert "length" in (outcome.error or "")


class TestNameRespelling:
    def test_a_respelled_glossary_name_is_rejected(self) -> None:
        # "Whitecliff" is a glossary name; the model spelling it "Whitecliffe"/"Whajtklif"
        # is caught mechanically, because instruction alone does not hold a spelling.
        client = FakeClient({"translate": [{"translation": "[[0]] w Whajtkliff"}] * 6})
        outcome = agents(client, max_repairs=1, max_revisions=0).process(
            make_unit(source="[[0]] at Whitecliff", placeholders=1))
        assert any(v.rule_id == "name_respelled" for v in outcome.violations)


class TestUnitsWithoutPlaceholders:
    def test_plain_line_translates(self) -> None:
        client = FakeClient({"translate": [{"translation": "Dzień dobry."}]})
        outcome = agents(client).process(make_unit(source="Good morning.", placeholders=0))
        assert outcome.status is Status.VERIFIED
        assert outcome.target == "Dzień dobry."

    def test_adding_a_placeholder_to_a_plain_line_is_rejected(self) -> None:
        client = FakeClient({"translate": [{"translation": "Dzień [[0]] dobry."}] * 5})
        outcome = agents(client).process(make_unit(source="Good morning.", placeholders=0))
        assert outcome.status is Status.REJECTED


class TestPlaceholderExpectation:
    """The prompt must state this line's placeholder budget, not just the general rule."""

    def test_line_without_placeholders_is_told_not_to_invent_one(self) -> None:
        client = FakeClient({"translate": [{"translation": "Odprowadzono ją."}]})
        agents(client).process(make_unit(source="She was led away.", placeholders=0))
        _, prompt = client.calls[0]
        # Stated as a direct instruction, not a report of an absent field: "field: none" style
        # phrasing is banned prompt-wide (see context_packing.py's module docstring), and this is
        # the one place outside that module the same rule applies.
        assert "has none" not in prompt.lower()
        assert "this line has no placeholders" in prompt.lower()
        assert "never invent a placeholder" in prompt

    def test_line_with_placeholders_is_told_how_many(self) -> None:
        client = FakeClient({"translate": [{"translation": "[[0]] i [[1]] przyszli"}]})
        agents(client).process(make_unit(source="[[0]] and [[1]] arrive", placeholders=2))
        _, prompt = client.calls[0]
        assert "exactly 2 ([[0]], [[1]])" in prompt

    def test_invented_placeholder_is_still_caught_mechanically(self) -> None:
        client = FakeClient({"translate": [{"translation": "[[0]] odprowadzono."}] * 5})
        outcome = agents(client).process(make_unit(source="She was led away.", placeholders=0))
        assert outcome.status is Status.REJECTED
        assert any(v.rule_id == "placeholders" for v in outcome.violations)


class TestRevisionCarriesContext:
    """A revision must see what it produced, why it failed, and every reviewer."""

    def _revision_prompt(self, client: FakeClient) -> str:
        prompts = [p for role, p in client.calls if role == "translate"]
        assert len(prompts) >= 2, "expected a revision round"
        return prompts[1]

    def test_previous_attempt_is_shown_verbatim(self) -> None:
        client = FakeClient({
            "translate": [{"translation": "[[0]] otworzył drzwi słabo"},
                          {"translation": "[[0]] otworzył drzwi."}],
            "accuracy": [{"acceptable": False, "issues": ["too weak"]},
                          {"acceptable": True, "issues": []}],
        })
        agents(client).process(make_unit())
        assert "[[0]] otworzył drzwi słabo" in self._revision_prompt(client)

    def test_an_attempt_with_no_issues_shows_the_draft_but_no_empty_reasons_header(self) -> None:
        """The guard on ``previous.issues``: an Attempt can legitimately carry no reasons, and
        emitting the "sent back for these reasons" header with nothing under it would tell the
        model it was rejected for reasons it cannot see -- worse than saying nothing. The draft
        itself must still be carried, or the revision re-translates from scratch."""
        client = FakeClient({"translate": [{"translation": "[[0]] otworzył drzwi."}]})
        agents(client).translate(make_unit(), Attempt(target="[[0]] draft", issues=()))
        prompt = client.calls[0][1]
        assert "[[0]] draft" in prompt
        assert "sent back for these reasons" not in prompt

    def test_objections_are_attributed_to_the_reviewer_that_raised_them(self) -> None:
        client = FakeClient({
            "translate": [{"translation": "[[0]] a"}, {"translation": "[[0]] b"}],
            "accuracy": [{"acceptable": True, "issues": []}] * 3,
            "fluency": [{"acceptable": False, "issues": ["far too stiff"]},
                        {"acceptable": True, "issues": []}],
        })
        agents(client).process(make_unit())
        assert "fluency: far too stiff" in self._revision_prompt(client)

    def test_reviewer_rewrite_is_offered_not_discarded(self) -> None:
        client = FakeClient({
            "translate": [{"translation": "[[0]] a"}, {"translation": "[[0]] b"}],
            "accuracy": [{"acceptable": False, "issues": ["weak"],
                           "improved_translation": "[[0]] rozwarł drzwi"},
                          {"acceptable": True, "issues": []}],
        })
        agents(client).process(make_unit())
        assert "[[0]] rozwarł drzwi" in self._revision_prompt(client)

    def test_review_objections_survive_a_mechanical_repair(self) -> None:
        # A repair round must not drop the reviewers' wording complaint, or the model fixes
        # the placeholder and silently reverts the change they asked for.
        client = FakeClient({
            "translate": [{"translation": "[[0]] a"},
                          {"translation": "no placeholder"},
                          {"translation": "[[0]] b"}],
            "accuracy": [{"acceptable": False, "issues": ["far too stiff"]},
                          {"acceptable": True, "issues": []}],
        })
        agents(client).process(make_unit())
        prompts = [p for role, p in client.calls if role == "translate"]
        assert "far too stiff" in prompts[2]
        assert "placeholder set changed" in prompts[2]


class TestContextDoesNotDemonstratePlaceholders:
    """Surrounding lines must not show the model a device its own line cannot use."""

    def _prompt(self, client: FakeClient, **unit_kwargs) -> str:
        agents(client).process(make_unit(**unit_kwargs))
        return client.calls[0][1]

    def test_context_placeholders_are_replaced_with_a_readable_name(self) -> None:
        client = FakeClient({"translate": [{"translation": "Odprowadzono ją."}]})
        prompt = self._prompt(client, source="She was led away.", placeholders=0,
                              context_before=("[[0]] is chained up.", "[[0]] weeps."))
        assert "[[0]]" not in prompt.split("Line to translate")[0]
        assert "Anna is chained up." in prompt

    def test_unknown_speaker_falls_back_to_a_neutral_pronoun(self) -> None:
        client = FakeClient({"translate": [{"translation": "Odprowadzono ją."}]})
        agents(client).process(replace(
            make_unit(source="She was led away.", placeholders=0), speaker=None,
            context_before=("[[0]] is chained up.",)))
        prompt = client.calls[0][1]
        assert "they is chained up." in prompt
        assert "[[0]]" not in prompt.split("Line to translate")[0]

    def test_the_line_being_translated_keeps_its_own_placeholders(self) -> None:
        client = FakeClient({"translate": [{"translation": "[[0]] otworzył drzwi."}]})
        prompt = self._prompt(client, source="[[0]] opened the door.", placeholders=1)
        assert "[[0]]" in prompt.split("Line to translate")[1]

    def test_a_neighbour_and_the_current_lines_own_placeholder_are_masked_independently(
            self) -> None:
        # Both a neighbour and the current line carry a [[0]]: only the neighbour's is masked,
        # the current line's own survives untouched, wherever it appears (embedded in the blob
        # or repeated in "Line to translate:").
        client = FakeClient({"translate": [{"translation": "[[0]] otworzył."}]})
        prompt = self._prompt(client, source="[[0]] opened it.", placeholders=1,
                              context_before=("[[0]] arrived first.",))
        assert "Anna arrived first." in prompt
        assert "[[0]] opened it." in prompt  # unmasked, both in the blob and the repeat


class TestUnifiedContextBlob:
    """The blob format replaced the old two-section Preceding/Following rendering."""

    def _prompt(self, client: FakeClient, **unit_kwargs) -> str:
        agents(client).process(make_unit(**unit_kwargs))
        return client.calls[0][1]

    def test_the_old_two_section_labels_are_gone_even_with_neighbours_shown(self) -> None:
        client = FakeClient({"translate": [{"translation": "Coś."}]})
        prompt = self._prompt(client, source="Something happened.", placeholders=0,
                              context_before=("Before this.",), context_after=("After this.",))
        assert "Preceding sentences" not in prompt
        assert "What is said next" not in prompt
        assert "Surrounding context" in prompt
        assert "Before this." in prompt and "After this." in prompt

    def test_before_and_after_and_current_read_as_one_continuous_passage(self) -> None:
        client = FakeClient({"translate": [{"translation": "she pushed it open."}]})
        prompt = self._prompt(client, source="she pushed it open.", placeholders=0,
                              context_before=("The old door creaked as",))
        block = prompt.split("Surrounding context")[1].split("\n\n")[0]
        assert "The old door creaked as\nshe pushed it open." in block

    def test_the_line_to_translate_is_repeated_verbatim_and_last(self) -> None:
        client = FakeClient({"translate": [{"translation": "Coś się stało."}]})
        prompt = self._prompt(client, source="Something happened.", placeholders=0,
                              context_before=("Before this.",))
        assert prompt.rstrip().endswith("Line to translate:\nSomething happened.")
        assert prompt.count("Something happened.") >= 2  # once in the blob, once repeated


REFERENCE = [
    ReferenceEntry("[[0]] opened the door.", "[[0]] otworzył drzwi."),
    ReferenceEntry("[[0]] closed the window.", "[[0]] zamknął okno."),
    ReferenceEntry("", "Gra zapisana."),  # target-only
]

REF_HEADER = "Reference translations of similar lines"
REVISION_HEADER = "Established target-language renderings similar to your draft"


class TestReferenceRetrieval:
    """External reference translations, retrieved into the prompt, off unless configured."""

    def _retriever(self) -> LexicalRetriever:
        return LexicalRetriever(REFERENCE, index_source=True, index_target=True)

    def test_a_custom_retriever_is_driven_purely_through_the_protocol(self) -> None:
        # The seam's promise: a non-lexical Retriever (e.g. the optional embedding one) drops in
        # without the engine knowing its type. This minimal one records the calls the engine makes
        # and returns a canned hit; the engine must reach it only through by_source / by_target.
        from transunit.reference import ReferenceEntry as _Entry
        from transunit.reference import Retrieved, Retriever

        calls: list[str] = []

        class CustomRetriever:
            def by_source(self, query, *, k, min_score=0.0):
                calls.append("by_source")
                return (Retrieved(_Entry("similar source", "custom rendering"), 0.9),)

            def by_target(self, query, *, k, min_score=0.0):
                calls.append("by_target")
                return ()

        retriever = CustomRetriever()
        assert isinstance(retriever, Retriever)  # structurally a Retriever, no inheritance
        aset = replace(AGENTS, context=Context(reference_examples=2, reference_revision_examples=1,
                                               reference_min_score=0.1))
        client = FakeClient({"translate": [{"translation": "[[0]] otworzył drzwi."}]})
        agents(client, reference=retriever, agent_set=aset).process(make_unit())
        prompt = client.calls[0][1]
        assert REF_HEADER in prompt and "custom rendering" in prompt  # its hit reached the prompt
        assert "by_source" in calls                                   # driven via the protocol
        assert set(calls) <= {"by_source", "by_target"}               # and only via the protocol

    def test_off_by_default_even_with_a_corpus_present(self) -> None:
        # A retriever supplied but reference_examples left at its default 0 must change nothing:
        # the feature is opt-in, and an accidental corpus never leaks into the prompt.
        client = FakeClient({"translate": [{"translation": "[[0]] otworzył drzwi."}]})
        agents(client, reference=self._retriever()).process(make_unit())
        assert REF_HEADER not in client.calls[0][1]

    def test_no_reference_argument_is_harmless(self) -> None:
        # Enabling the knob without a corpus must not crash; there is simply nothing to add.
        aset = replace(AGENTS, context=Context(reference_examples=3))
        client = FakeClient({"translate": [{"translation": "[[0]] otworzył drzwi."}]})
        outcome = agents(client, agent_set=aset).process(make_unit())
        assert outcome.status is Status.VERIFIED
        assert REF_HEADER not in client.calls[0][1]

    def test_block_appears_when_enabled_and_a_match_exists(self) -> None:
        aset = replace(AGENTS, context=Context(reference_examples=2, reference_min_score=0.1))
        client = FakeClient({"translate": [{"translation": "[[0]] otworzył drzwi."}]})
        agents(client, reference=self._retriever(), agent_set=aset).process(make_unit())
        prompt = client.calls[0][1]
        assert REF_HEADER in prompt
        assert "otworzył drzwi" in prompt

    def test_reference_example_placeholders_are_neutralised(self) -> None:
        # An example must never teach the model to emit a placeholder; its [[n]] are replaced with
        # the same stand-in neighbour context uses (here the speaker, Anna).
        aset = replace(AGENTS, context=Context(reference_examples=2, reference_min_score=0.1))
        client = FakeClient({"translate": [{"translation": "[[0]] otworzył drzwi."}]})
        agents(client, reference=self._retriever(), agent_set=aset).process(make_unit())
        # Isolate just the reference section (up to the next blank-line part separator), so the
        # legitimate [[0]] in the later placeholder-expectation section is not what we test.
        block = client.calls[0][1].split(REF_HEADER)[1].split("\n\n")[0]
        assert "[[0]]" not in block
        assert "Anna otworzył drzwi." in block

    def test_no_boilerplate_when_nothing_clears_the_floor(self) -> None:
        aset = replace(AGENTS, context=Context(reference_examples=2, reference_min_score=0.95))
        client = FakeClient({"translate": [{"translation": "Nic."}]})
        agents(client, reference=self._retriever(), agent_set=aset).process(
            make_unit(source="Totally unrelated astrophysics.", placeholders=0))
        assert REF_HEADER not in client.calls[0][1]

    def test_revision_examples_are_off_by_default(self) -> None:
        aset = replace(AGENTS, context=Context(reference_examples=2, reference_min_score=0.1))
        client = FakeClient({
            "translate": [{"translation": "[[0]] otworzyl drzwi."},
                          {"translation": "[[0]] otworzył drzwi."}],
            "accuracy": [{"acceptable": False, "issues": ["diacritics"]}],
        })
        agents(client, reference=self._retriever(), agent_set=aset).process(make_unit())
        revision = [content for role, content in client.calls if role == "translate"][1]
        assert REVISION_HEADER not in revision

    def test_revision_carries_target_keyed_examples_when_enabled(self) -> None:
        # The by-target path fires only on a revision (there is a draft to key on) and pulls in
        # established target renderings -- the answer to "reference text with no source".
        aset = replace(AGENTS, context=Context(reference_revision_examples=2,
                                               reference_min_score=0.1))
        client = FakeClient({
            "translate": [{"translation": "[[0]] otworzyl drzwi."},
                          {"translation": "[[0]] otworzył drzwi."}],
            "accuracy": [{"acceptable": False, "issues": ["diacritics"]}],
        })
        agents(client, reference=self._retriever(), agent_set=aset).process(make_unit())
        calls = [content for role, content in client.calls if role == "translate"]
        assert REVISION_HEADER not in calls[0]     # first pass has no draft to key on
        assert REVISION_HEADER in calls[1]         # revision does

    def test_a_purely_target_only_corpus_still_helps_on_revision(self) -> None:
        # Material we have translated but whose originals we lack. It cannot seed a translation
        # (no source to match), but by-target retrieval keyed on the draft still surfaces it
        # during revision -- the answer to "target data, no original".
        corpus = [ReferenceEntry("", "[[0]] otworzył wielkie wrota.")]
        retriever = LexicalRetriever(corpus, index_source=False, index_target=True)
        aset = replace(AGENTS, context=Context(reference_revision_examples=2,
                                               reference_min_score=0.1))
        client = FakeClient({
            "translate": [{"translation": "[[0]] otworzyl drzwi."},
                          {"translation": "[[0]] otworzył drzwi."}],
            "accuracy": [{"acceptable": False, "issues": ["diacritics"]}],
        })
        agents(client, reference=retriever, agent_set=aset).process(make_unit())
        revision = [content for role, content in client.calls if role == "translate"][1]
        assert REVISION_HEADER in revision
        assert "otworzył wielkie wrota" in revision

    def test_target_only_reference_does_not_reach_a_first_pass_acceptance(self) -> None:
        # The documented limitation: a unit accepted on the first pass has no draft, so a purely
        # target-only corpus never reaches it. (Bilingual reference would, via by-source.)
        corpus = [ReferenceEntry("", "[[0]] otworzył wielkie wrota.")]
        retriever = LexicalRetriever(corpus, index_source=False, index_target=True)
        aset = replace(AGENTS, context=Context(reference_revision_examples=2,
                                               reference_min_score=0.1))
        client = FakeClient({"translate": [{"translation": "[[0]] otworzył drzwi."}]})
        agents(client, reference=retriever, agent_set=aset).process(make_unit())
        assert all(REVISION_HEADER not in content for _, content in client.calls)


class TestReferenceLearning:
    """learn_reference: feed accepted output into a writable reference, gated by status."""

    def _agents(self, reference, statuses):
        aset = replace(AGENTS, context=Context(reference_examples=1, reference_min_score=0.1,
                                               reference_learn_statuses=statuses))
        return agents(FakeClient({}), reference=reference, agent_set=aset)

    def test_a_configured_status_is_added(self) -> None:
        reference = GrowableLexicalRetriever([], index_source=True)
        harness = self._agents(reference, (Status.VERIFIED,))
        harness.learn_reference("the dragon breathes fire", "smok zieje ogniem", Status.VERIFIED)
        hit = reference.by_source("the dragon breathes fire", k=1, min_score=0.1)[0]
        assert hit.entry.target == "smok zieje ogniem"

    def test_an_unconfigured_status_is_ignored(self) -> None:
        reference = GrowableLexicalRetriever([], index_source=True)
        harness = self._agents(reference, (Status.VERIFIED,))
        harness.learn_reference("x", "y", Status.TRANSLATED)  # not in the configured set
        assert len(reference) == 0

    def test_a_read_only_reference_is_never_written(self) -> None:
        # A non-writable retriever supplied with learning configured must simply not be written.
        reference = LexicalRetriever([ReferenceEntry("seed", "seed")], index_source=True)
        harness = self._agents(reference, (Status.VERIFIED,))
        harness.learn_reference("a", "b", Status.VERIFIED)  # no-op, must not raise
        assert len(reference) == 1

    def test_learning_off_by_default_adds_nothing(self) -> None:
        reference = GrowableLexicalRetriever([], index_source=True)
        aset = replace(AGENTS, context=Context(reference_examples=1))  # no learn statuses
        harness = agents(FakeClient({}), reference=reference, agent_set=aset)
        harness.learn_reference("a", "b", Status.VERIFIED)
        assert len(reference) == 0

    def test_a_none_target_is_ignored(self) -> None:
        reference = GrowableLexicalRetriever([], index_source=True)
        harness = self._agents(reference, (Status.VERIFIED,))
        harness.learn_reference("a", None, Status.VERIFIED)
        assert len(reference) == 0


class TestUncoveredEnginePaths:
    def test_speaker_id_with_an_unknown_name_falls_back_to_character_n(self) -> None:
        # The name does not resolve to a glossary character, so both the speaker label and the
        # context stand-in fall back to the bare "character N" / anonymous form.
        client = FakeClient({"translate": [{"translation": "Ktoś wszedł."}]})
        agents(client).process(make_unit(source="Someone entered.", placeholders=0,
                                         speaker="7:Nobody"))
        assert "character 7" in client.calls[0][1]

    def test_a_content_error_on_translate_rejects_the_unit(self) -> None:
        # Repair budget (max_repairs=2) is spent re-asking before this is terminal.
        client = FakeClient({"translate": [
            LlmContentError("unparseable", role="translate"),
            LlmContentError("unparseable", role="translate"),
            LlmContentError("unparseable", role="translate"),
        ]})
        outcome = agents(client).process(make_unit())
        assert outcome.status is Status.REJECTED
        assert outcome.target is None
        assert len(client.calls) == 3

    def test_revision_reference_block_is_absent_when_nothing_matches(self) -> None:
        # by_target is enabled but the draft clears nothing above the floor: the None path in
        # _reference_revision, so no dangling header.
        retriever = LexicalRetriever([ReferenceEntry("", "zupełnie inny niepowiązany tekst")],
                                     index_source=False, index_target=True)
        aset = replace(AGENTS, context=Context(reference_revision_examples=2,
                                               reference_min_score=0.9))
        client = FakeClient({
            "translate": [{"translation": "[[0]] otworzyl drzwi."},
                          {"translation": "[[0]] otworzył drzwi."}],
            "accuracy": [{"acceptable": False, "issues": ["diacritics"]}],
        })
        agents(client, reference=retriever, agent_set=aset).process(make_unit())
        assert all(REVISION_HEADER not in content for _, content in client.calls)
