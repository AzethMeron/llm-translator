"""End-to-end and stress behaviour of the real pipeline, at the sizes that break things.

`test_cli.py` proves the CLI wiring and `test_agents.py` the harness logic, both on ordinary
one-line inputs. This file covers what neither does: payloads at the extremes of length, a
catalogue large enough for concurrency to interleave, and interrupted runs resumed. Every model
call is still a scripted fake, so this needs no server.

The invariant running through all of it: whatever happens, the journal is a complete and readable
record. A run that loses or corrupts a result is worse than one that fails loudly, because the
caller treats every journalled outcome as final.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from translator.agents import TranslationAgents
from translator.backend import LlmClient
from translator.roles import AgentSet, Limits, Reviewer
from translator.rules import RuleSet
from translator.runner import run_batch
from transunit.units import Status, Unit

RULES = RuleSet(max_line_columns=10_000, advisory_rules=(("register", "Keeps register."),))
AGENTS = AgentSet(
    translator_instructions="Translate.",
    reviewers=(Reviewer(id="accuracy", instructions="Check."),),
    limits=Limits(),
    max_revisions=0,
    max_repairs=0,
)


class EchoClient(LlmClient):
    """Returns a translation derived from the source, so a result can be traced to its unit.

    Deliberately not a fixed string: at concurrency the point of the test is that unit N's
    translation lands against unit N, which a constant reply could not detect.
    """

    def __init__(self) -> None:
        self.translate_calls = 0
        self.max_tokens_seen: list[int] = []

    def complete_json(self, messages, schema, *, role="<none>", max_tokens=1024, **_):
        if role != "translate":
            return {"acceptable": True, "issues": []}
        self.translate_calls += 1
        self.max_tokens_seen.append(max_tokens)
        source = messages[-1].content.rsplit("\n", 1)[-1]
        return {"translation": f"T:{source[:200]}"}

    def close(self) -> None:
        pass


def unit(index: int, source: str) -> Unit:
    return Unit(unit_id=f"u{index}", rel_path="f.txt", line_no=index, span_start=index,
                span_end=index + 1, command="TEXT", kind="LINE", source=source)


def agents_for(client: LlmClient) -> TranslationAgents:
    return TranslationAgents(client, RULES, [], agent_set=AGENTS)


def journal_rows(path: Path) -> list[dict]:
    """Every line of the journal, insisting each one parses.

    Reading with a plain loop rather than the engine's own reader on purpose: this asserts the
    bytes on disk are well-formed, which is what a resume and any downstream consumer depend on.
    A torn or interleaved line from concurrent writers would fail here.
    """
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover -- only on a real corruption bug
            raise AssertionError(f"journal line {number} is not valid JSON: {exc}") from exc
    return rows


class TestExtremeLengths:
    """A payload's length drives the token budget and the prompt size, and both ends misbehave
    differently: a huge line can ask for an unbounded response, a tiny one can be handed a budget
    too small to answer in."""

    def test_a_very_long_source_is_translated_and_its_budget_stays_capped(self) -> None:
        """The ceiling exists so one enormous line cannot ask for an unbounded response.

        Without it a repetition loop on a long input runs until the server gives up, which is
        both slow and, because the reply is then truncated, indistinguishable from bad output.
        """
        client = EchoClient()
        outcome = agents_for(client).process(unit(1, "very long sentence. " * 500))
        assert outcome.status in (Status.VERIFIED, Status.TRANSLATED)
        assert client.max_tokens_seen[0] == AGENTS.limits.translate_tokens_ceiling

    @pytest.mark.parametrize("source", ["a", "。", "?", "1", "\U0001f600"])
    def test_a_single_character_source_still_gets_a_usable_budget(self, source: str) -> None:
        """The floor exists so a one-character line can still produce a whole sentence plus its
        JSON wrapper -- a budget proportional to a 1-character source would be a few tokens."""
        client = EchoClient()
        outcome = agents_for(client).process(unit(2, source))
        assert outcome.status in (Status.VERIFIED, Status.TRANSLATED)
        assert client.max_tokens_seen[0] == AGENTS.limits.translate_tokens_floor

    @pytest.mark.parametrize("blank", ["", " ", "\t", "\n", "   \n\t "])
    def test_a_blank_source_never_reaches_the_model(self, blank: str) -> None:
        """Asking a model to translate nothing makes it render the prompt's own scaffolding into
        the target -- an injectable non-translation the mechanical checks cannot catch, because
        the non-empty rule cannot fire when the source is itself blank."""
        client = EchoClient()
        outcome = agents_for(client).process(unit(3, blank))
        assert outcome.status is Status.SKIPPED
        assert client.translate_calls == 0

    def test_a_source_of_astral_and_combining_characters_round_trips_through_the_journal(
            self, tmp_path: Path) -> None:
        """Non-BMP and combining characters must survive the journal write/read unchanged: the
        catalogue is UTF-8 JSON, and a mangled character here reaches the carrier verbatim."""
        source = "𝄞 á \U0001f469‍\U0001f467 กำ ᛗᚨᚾ"
        journal = tmp_path / "j.jsonl"
        run_batch(agents_for(EchoClient()), [unit(4, source)], journal)
        rows = journal_rows(journal)
        assert len(rows) == 1
        assert rows[0]["source"] == source


class TestScale:
    """Concurrency is the supported path, so the journal must stay intact under it."""

    def test_five_hundred_units_at_concurrency_eight_all_land_exactly_once(
            self, tmp_path: Path) -> None:
        """Every unit is journalled exactly once, with its own translation.

        The per-unit translation is derived from the source, so a result landing against the
        wrong unit is detectable -- a constant reply would hide it.
        """
        journal = tmp_path / "j.jsonl"
        units = [unit(i, f"line {i}") for i in range(500)]
        progress = run_batch(agents_for(EchoClient()), units, journal, concurrency=8)

        rows = journal_rows(journal)
        assert len(rows) == 500
        assert {row["unit_id"] for row in rows} == {f"u{i}" for i in range(500)}
        for row in rows:
            assert row["source"] in str(row["target"]), "a result landed against the wrong unit"
        assert progress.done == 500
        assert progress.verified + progress.translated + progress.rejected + progress.skipped \
            == progress.done, "the status counters must account for every completed unit"

    def test_duplicate_units_are_translated_once_and_journalled_for_every_occurrence(
            self, tmp_path: Path) -> None:
        """Duplicate grouping is a real cost saving, but every occurrence still needs its own
        journal row -- each injects into its own location in the carrier."""
        journal = tmp_path / "j.jsonl"
        units = [unit(i, "the same line") for i in range(20)]
        client = EchoClient()
        run_batch(agents_for(client), units, journal, concurrency=4)

        rows = journal_rows(journal)
        assert len(rows) == 20
        assert {row["unit_id"] for row in rows} == {f"u{i}" for i in range(20)}
        assert client.translate_calls == 1, "duplicates should share one generation"


class TestResume:
    """A run that stops must lose nothing and repeat nothing."""

    def test_a_resumed_run_translates_exactly_the_remainder(self, tmp_path: Path) -> None:
        journal = tmp_path / "j.jsonl"
        units = [unit(i, f"line {i}") for i in range(30)]

        first = EchoClient()
        run_batch(agents_for(first), units[:10], journal)
        assert first.translate_calls == 10

        # The runner is handed only what is still pending, exactly as the CLI computes it from
        # the journal -- so a resumed run must not re-translate what is already recorded.
        done = {row["unit_id"] for row in journal_rows(journal)}
        second = EchoClient()
        run_batch(agents_for(second), [u for u in units if u.unit_id not in done], journal,
                  already_done=len(done))

        assert second.translate_calls == 20
        rows = journal_rows(journal)
        assert len(rows) == 30
        assert len({row["unit_id"] for row in rows}) == 30, "no unit journalled twice"
