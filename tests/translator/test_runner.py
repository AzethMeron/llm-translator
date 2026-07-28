"""Tests for resumable, concurrent batch execution.

The runner is where durability lives: a result is journalled the instant it completes, an
interrupted run resumes without re-translating, and an infrastructure failure never discards
work already in flight. Work order is injectable, so that is pinned too. The model is a
stub -- the runner's contract is with the journal and the agents interface, not the GPU.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import pytest

from translator.agents import Outcome
from translator.backend import LlmError
from translator.memory import TranslationMemory
from translator.runner import (
    Progress,
    RunnerError,
    catalog_order,
    completed_ids,
    group_duplicates,
    pending_units,
    run_batch,
)
from transunit.reference import GrowableLexicalRetriever, ReferenceEntry
from transunit.units import Status, Unit, read_journal, write_catalog


def unit(uid: str, source: str, *, rel_path="a.txt", line_no=1, speaker=None,
         status=Status.PENDING) -> Unit:
    # An injectable status (VERIFIED/TRANSLATED) must carry a target, so supply one.
    target = f"T:{source}" if status.is_injectable else None
    return Unit(unit_id=uid, rel_path=rel_path, line_no=line_no, span_start=0, span_end=1,
                command="", kind="T", source=source, speaker=speaker, status=status,
                target=target)


class FakeAgents:
    """A stand-in for TranslationAgents: deterministic outcomes, optional failures."""

    def __init__(self, *, fail_on=(), memory=None, reference=None, learn_statuses=(),
                 status=Status.VERIFIED) -> None:
        self.memory = memory
        self.reference = reference
        self._learn_statuses = frozenset(learn_statuses)
        self._fail_on = set(fail_on)
        self._status = status
        self.processed: list[str] = []
        self.learned: list[tuple[str, str | None, Status]] = []

    def process(self, u: Unit) -> Outcome:
        self.processed.append(u.source)
        if u.source in self._fail_on:
            raise LlmError("server exploded", role="translate")
        return Outcome(unit=u, status=self._status, target=f"T:{u.source}", rounds=1)

    def learn_reference(self, source: str, target: str | None, status: Status) -> None:
        # Mirror the real gate so a runner test can observe self-population end to end.
        self.learned.append((source, target, status))
        if target and status in self._learn_statuses and self.reference is not None:
            self.reference.add(ReferenceEntry(source=source, target=target))


class TestPendingUnits:
    def test_excludes_journalled_and_non_pending(self, tmp_path: Path) -> None:
        catalog = tmp_path / "units.jsonl"
        write_catalog([unit("a", "one"), unit("b", "two"),
                       unit("c", "three", status=Status.VERIFIED)], catalog)
        journal = tmp_path / "journal.jsonl"
        journal.write_text(unit("a", "one").with_target("T", Status.VERIFIED).to_json() + "\n")
        pending = pending_units(catalog, journal)
        assert [u.unit_id for u in pending] == ["b"]

    def test_default_order_is_catalogue_position(self, tmp_path: Path) -> None:
        catalog = tmp_path / "units.jsonl"
        write_catalog([unit("a", "x", rel_path="b.txt", line_no=2),
                       unit("b", "y", rel_path="a.txt", line_no=9),
                       unit("c", "z", rel_path="a.txt", line_no=1)], catalog)
        pending = pending_units(catalog, tmp_path / "none.jsonl")
        assert [u.unit_id for u in pending] == ["c", "b", "a"]

    def test_a_custom_priority_key_reorders(self, tmp_path: Path) -> None:
        # The generalisation: work order is a caller concern. Here, longer sources first.
        catalog = tmp_path / "units.jsonl"
        write_catalog([unit("a", "short"), unit("b", "a much longer source line"),
                       unit("c", "mid length")], catalog)
        pending = pending_units(catalog, tmp_path / "none.jsonl",
                                key=lambda u: -len(u.source))
        assert [u.unit_id for u in pending] == ["b", "c", "a"]

    def test_catalog_order_is_the_documented_default(self) -> None:
        assert catalog_order(unit("a", "x", rel_path="p", line_no=3)) == ("p", 3)


class TestGroupDuplicates:
    def test_identical_source_and_speaker_group_together(self) -> None:
        units = [unit("a", "yes"), unit("b", "yes"), unit("c", "no")]
        groups = group_duplicates(units)
        representatives = {rep.unit_id: [m.unit_id for m in members] for rep, members in groups}
        assert representatives == {"a": ["a", "b"], "c": ["c"]}

    def test_same_text_different_speaker_stays_separate(self) -> None:
        units = [unit("a", "well", speaker="1"), unit("b", "well", speaker="2")]
        groups = group_duplicates(units)
        assert len(groups) == 2

    def test_representative_is_the_first_occurrence(self) -> None:
        units = [unit("a", "hi"), unit("b", "hi")]
        (rep, members), = group_duplicates(units)
        assert rep.unit_id == "a" and [m.unit_id for m in members] == ["a", "b"]


class TestRunBatch:
    def test_every_member_is_journalled_even_when_deduplicated(self, tmp_path: Path) -> None:
        journal = tmp_path / "journal.jsonl"
        units = [unit("a", "same"), unit("b", "same"), unit("c", "other")]
        run_batch(FakeAgents(), units, journal, concurrency=1)
        recorded = {u.unit_id: u for u in read_journal(journal)}
        assert set(recorded) == {"a", "b", "c"}
        # Duplicates share the group's translation but keep their own ids.
        assert recorded["a"].target == recorded["b"].target == "T:same"

    def test_progress_counts_by_status(self, tmp_path: Path) -> None:
        journal = tmp_path / "j.jsonl"
        progress = run_batch(FakeAgents(), [unit("a", "x"), unit("b", "y")], journal)
        assert progress.done == 2
        assert progress.verified == 2

    def test_concurrency_processes_every_unit(self, tmp_path: Path) -> None:
        journal = tmp_path / "j.jsonl"
        units = [unit(str(i), f"line {i}") for i in range(20)]
        run_batch(FakeAgents(), units, journal, concurrency=4)
        assert len({u.unit_id for u in read_journal(journal)}) == 20

    def test_zero_concurrency_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(RunnerError, match="concurrency"):
            run_batch(FakeAgents(), [unit("a", "x")], tmp_path / "j.jsonl", concurrency=0)

    def test_an_infrastructure_failure_propagates_but_keeps_finished_work(
            self, tmp_path: Path) -> None:
        # The failing unit aborts the run, but units already journalled must survive: resume
        # depends on them.
        journal = tmp_path / "j.jsonl"
        units = [unit("a", "ok"), unit("b", "boom"), unit("c", "also-ok")]
        with pytest.raises(LlmError, match="server exploded"):
            run_batch(FakeAgents(fail_on=["boom"]), units, journal, concurrency=1)
        done = {u.unit_id for u in read_journal(journal)}
        assert "a" in done  # journalled before the failure, and not lost

    def test_a_resumed_run_skips_journalled_units(self, tmp_path: Path) -> None:
        catalog = tmp_path / "units.jsonl"
        journal = tmp_path / "j.jsonl"
        units = [unit("a", "one"), unit("b", "two"), unit("c", "three")]
        write_catalog(units, catalog)
        run_batch(FakeAgents(), pending_units(catalog, journal), journal)
        # Second pass: everything is journalled, so nothing is pending.
        assert pending_units(catalog, journal) == []
        again = run_batch(FakeAgents(), pending_units(catalog, journal), journal)
        assert again.done == 0

    def test_verified_results_feed_translation_memory(self, tmp_path: Path) -> None:
        memory = TranslationMemory()
        run_batch(FakeAgents(memory=memory), [unit("a", "hello")], tmp_path / "j.jsonl")
        assert memory.get("hello") == "T:hello"

    def test_the_runner_offers_every_result_to_reference_learning(self, tmp_path: Path) -> None:
        # The runner hands each result to learn_reference (which applies its own status gate),
        # rather than pre-filtering, so the "what may seed" decision lives in one place.
        agents = FakeAgents()
        run_batch(agents, [unit("a", "one"), unit("b", "two")], tmp_path / "j.jsonl")
        assert [source for source, _, _ in agents.learned] == ["one", "two"]

    def test_self_population_makes_a_result_retrievable_for_later_units(
            self, tmp_path: Path) -> None:
        reference = GrowableLexicalRetriever([], index_source=True)
        agents = FakeAgents(reference=reference, learn_statuses=(Status.VERIFIED,))
        run_batch(agents, [unit("a", "the dragon breathes fire")], tmp_path / "j.jsonl")
        hits = reference.by_source("the dragon breathes fire and smoke", k=1, min_score=0.1)
        assert hits and hits[0].entry.target == "T:the dragon breathes fire"

    def test_rejected_results_do_not_feed_memory(self, tmp_path: Path) -> None:
        memory = TranslationMemory()
        run_batch(FakeAgents(memory=memory, status=Status.REJECTED),
                  [unit("a", "hello")], tmp_path / "j.jsonl")
        assert memory.get("hello") is None

    def test_progress_callback_fires(self, tmp_path: Path) -> None:
        seen: list[int] = []
        run_batch(FakeAgents(), [unit(str(i), f"l{i}") for i in range(3)],
                  tmp_path / "j.jsonl", on_progress=lambda p: seen.append(p.done),
                  report_every=1)
        assert seen  # at least the final callback

    def test_completed_ids_reads_the_journal(self, tmp_path: Path) -> None:
        journal = tmp_path / "j.jsonl"
        run_batch(FakeAgents(), [unit("a", "x"), unit("b", "y")], journal)
        assert completed_ids(journal) == {"a", "b"}


class TestProgress:
    def test_position_spans_the_whole_job_not_just_this_run(self) -> None:
        progress = Progress(total=100, already_done=40)
        for _ in range(10):
            progress.record(Outcome(unit=unit("x", "s"), status=Status.VERIFIED, target="t"))
        assert progress.completed == 50
        assert progress.remaining == 50

    def test_eta_is_unknown_before_any_throughput(self) -> None:
        progress = Progress(total=10)
        assert progress.eta_seconds == float("inf")
        assert "unknown" in progress.summary()

    def test_summary_reports_each_status_bucket(self) -> None:
        progress = Progress(total=3)
        progress.record(Outcome(unit=unit("a", "s"), status=Status.VERIFIED, target="t"))
        progress.record(Outcome(unit=unit("b", "s"), status=Status.TRANSLATED, target="t"))
        progress.record(Outcome(unit=unit("c", "s"), status=Status.REJECTED, target=None))
        summary = progress.summary()
        assert "verified 1" in summary and "needs review 1" in summary and "rejected 1" in summary


class TestProgressInternals:
    def test_record_counts_rejected_and_skipped(self) -> None:
        progress = Progress(total=2)
        progress.record(Outcome(unit=unit("a", "x", status=Status.REJECTED),
                                status=Status.REJECTED, target=None))
        progress.record(Outcome(unit=unit("b", "y"), status=Status.SKIPPED, target=None))
        assert (progress.rejected, progress.skipped) == (1, 1)

    def test_every_terminal_status_lands_in_a_category_that_sums_to_done(self) -> None:
        """The invariant the counters exist for: each recorded outcome increments exactly one
        category, so the four buckets always account for ``done``. If a status ever stopped
        being counted, the position, rate and ETA of a multi-hour run would all quietly drift."""
        progress = Progress(total=4)
        for uid, status in (("a", Status.VERIFIED), ("b", Status.TRANSLATED),
                            ("c", Status.REJECTED), ("d", Status.SKIPPED)):
            progress.record(Outcome(unit=unit(uid, "s"), status=status, target=None))
        assert (progress.verified, progress.translated, progress.rejected,
                progress.skipped) == (1, 1, 1, 1)
        assert (progress.verified + progress.translated + progress.rejected
                + progress.skipped) == progress.done == 4

    def test_a_non_terminal_status_is_rejected_loudly_and_changes_no_counter(self) -> None:
        """PENDING has no bucket: silently counting it in ``done`` alone would break the sum
        above with no signal. It must raise, naming both the status and the offending unit so
        the cause is debuggable without reproducing a long run -- and leave the counters clean."""
        progress = Progress(total=1)
        with pytest.raises(RunnerError, match="PENDING") as excinfo:
            progress.record(Outcome(unit=unit("stuck", "s"), status=Status.PENDING, target=None))
        assert "stuck" in str(excinfo.value)
        assert (progress.done, progress.verified, progress.translated, progress.rejected,
                progress.skipped) == (0, 0, 0, 0, 0)

    def test_duration_formats_scale_with_length(self) -> None:
        assert Progress._duration(300).endswith("m")        # minutes for a short job
        assert Progress._duration(3 * 3600).endswith("h")   # hours for a shift
        assert Progress._duration(3 * 86400).endswith("d")  # days beyond that

    def test_interrupt_handler_sets_the_stop_flag(self) -> None:
        from translator.runner import _Interruptible
        interruptible = _Interruptible()
        assert interruptible.stop is False
        interruptible._handle()
        assert interruptible.stop is True


class _SimultaneousFailures:
    """Agents whose every unit fails with its own distinguishable exception, all at once.

    The barrier holds each worker at the failure point until every worker has arrived, so the
    N failures are genuinely concurrent and land in the same ``wait(FIRST_COMPLETED)`` batch --
    the exact shape in which ``failure = failure or exc`` has more than one candidate to choose
    between. No sleeps, so this is deterministic rather than timing-dependent.
    """

    memory = None
    reference = None

    def __init__(self, sources: tuple[str, ...]) -> None:
        self._barrier = threading.Barrier(len(sources))
        self._lock = threading.Lock()
        self.attempted: list[str] = []

    def process(self, u: Unit) -> Outcome:
        self._barrier.wait(timeout=30)
        with self._lock:
            self.attempted.append(u.source)
        raise LlmError(f"failure-{u.source}", role="translate")

    def learn_reference(self, source: str, target: str | None, status: Status) -> None:
        raise AssertionError("no unit ever succeeds here")


class TestConcurrentFailureReporting:
    """What happens to N distinct worker failures that occur at the same moment."""

    SOURCES = ("alpha", "beta", "gamma", "delta")

    def _explode(self, tmp_path: Path) -> tuple[_SimultaneousFailures, list[Unit]]:
        return (_SimultaneousFailures(self.SOURCES),
                [unit(source, source) for source in self.SOURCES])

    def test_exactly_one_of_four_simultaneous_failures_propagates(self, tmp_path: Path) -> None:
        """Pins the current behaviour: all four workers really do fail, and exactly one cause
        reaches the caller. Which one is not deterministic (it depends on the order
        ``wait`` returns the completed futures in), so the assertion is on the count, not the
        identity -- pinning the identity would be pinning a race."""
        agents, units = self._explode(tmp_path)
        with pytest.raises(LlmError) as info:
            run_batch(agents, units, tmp_path / "j.jsonl", concurrency=4)
        assert sorted(agents.attempted) == sorted(self.SOURCES)  # all four genuinely failed
        surfaced = [source for source in self.SOURCES if f"failure-{source}" in str(info.value)]
        assert len(surfaced) == 1

    @pytest.mark.xfail(strict=True, reason=(
        "BUG (diagnostics): runner.run_batch keeps only the first worker failure "
        "(`failure = failure or exc`, src/translator/runner.py:269) and drops every later one "
        "on the floor -- the module has no logger at all, so with concurrency=4 and four "
        "distinct simultaneous causes (say a 503, a context overflow, a refusal and a transport "
        "reset) three vanish without a single line anywhere. An operator debugging a run that "
        "died at hour six sees one cause and cannot know the others existed."))
    def test_the_discarded_failures_leave_some_trace(self, tmp_path: Path, caplog) -> None:
        agents, units = self._explode(tmp_path)
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(LlmError):
                run_batch(agents, units, tmp_path / "j.jsonl", concurrency=4)
        recorded = "\n".join(record.getMessage() for record in caplog.records)
        missing = [source for source in self.SOURCES if f"failure-{source}" not in recorded]
        assert not missing, f"failures with no trace anywhere: {missing}"


class _BaseExceptionAgents(FakeAgents):
    """FakeAgents that raises a *BaseException* (not an Exception) for one source.

    ``run_batch`` catches ``BaseException`` from a worker on purpose -- a KeyboardInterrupt
    raised inside a pool thread is not the operator's Ctrl-C (that is handled by
    ``_Interruptible`` on the main thread) and must not be able to skip the journal flush of
    work already finished.
    """

    def __init__(self, *, boom_on: str, error: BaseException) -> None:
        super().__init__()
        self._boom_on = boom_on
        self._error = error

    def process(self, u: Unit) -> Outcome:
        if u.source == self._boom_on:
            self.processed.append(u.source)
            raise self._error
        return super().process(u)


class TestWorkerBaseException:
    @pytest.mark.parametrize("error", [KeyboardInterrupt("ctrl-c in a worker"),
                                       SystemExit("worker called sys.exit")])
    def test_a_base_exception_propagates_and_stops_further_submission(
            self, tmp_path: Path, error: BaseException) -> None:
        """A BaseException from a worker must behave like any other failure: work already
        journalled survives, nothing new is submitted, and the original exception -- not a
        wrapper, and not a silent success -- reaches the caller. Losing it would let a run exit
        0 having translated a fraction of its units."""
        journal = tmp_path / "j.jsonl"
        agents = _BaseExceptionAgents(boom_on="boom", error=error)
        units = [unit("a", "ok"), unit("b", "boom"), unit("c", "never-reached")]
        with pytest.raises(type(error)) as info:
            run_batch(agents, units, journal, concurrency=1)
        assert info.value is error  # the worker's own exception, re-raised unchanged
        assert {u.unit_id for u in read_journal(journal)} == {"a"}
        assert agents.processed == ["ok", "boom"]  # nothing submitted after the failure


class TestProgressReportingWithDuplicates:
    def test_a_duplicate_group_larger_than_report_every_skips_its_report(
            self, tmp_path: Path) -> None:
        """``progress.done`` advances once per group MEMBER but is tested against
        ``report_every`` only once per GROUP, so a group bigger than the interval steps straight
        over the multiple and the periodic report is skipped. Pinned rather than xfailed because
        the final callback still fires, so no total is lost -- but on a catalogue where duplicate
        groups are large (the common case: a game's repeated UI strings) an operator can go a
        long way with no live progress line at all.
        """
        journal = tmp_path / "j.jsonl"
        grouped: list[int] = []
        run_batch(FakeAgents(), [unit(uid, "same") for uid in "abc"], journal,
                  on_progress=lambda p: grouped.append(p.done), report_every=2)
        # done jumps 0 -> 3 in one step; 3 % 2 != 0, so only the unconditional final call fires.
        assert grouped == [3]

        distinct: list[int] = []
        run_batch(FakeAgents(), [unit(uid, uid) for uid in "def"], tmp_path / "j2.jsonl",
                  on_progress=lambda p: distinct.append(p.done), report_every=2)
        assert distinct == [2, 3]  # the same three units, reported at the interval as intended
