"""Contention tests for the state a concurrent run shares between workers.

:class:`~translator.agents.TranslationAgents` documents (class docstring) that one instance is
meant to be shared by every one of :func:`~translator.runner.run_batch`'s workers, and names
the three places that share mutable state: the client's
:class:`~translator.backend.UsageStats`, the translation memory, and the per-reviewer leniency
window. Those claims were documented but never exercised under real contention -- a lost
update there is invisible in a single-threaded test and silently wrong in a multi-hour run.

Every test here runs real threads released together by a :class:`threading.Barrier` (no sleeps,
no timing assumptions) and asserts an *exact* expected value computed sequentially, so a lost
update fails the assertion rather than merely making it flaky. Patterns are chosen so their
expected result is order-independent; anything whose answer would depend on the interleaving
is asserted only on its order-independent invariants.
"""
from __future__ import annotations

import logging
import sys
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

import httpx
import pytest

from translator.agents import TranslationAgents
from translator.backend import LlmClient, Message, ServerConfig, UsageStats, get_backend
from translator.roles import AgentSet, Leniency, Reviewer
from translator.rules import RuleSet

WORKERS = 8
"""Enough threads to interleave on a normal machine without making the suite slow."""


@pytest.fixture(autouse=True)
def _fine_grained_thread_switching() -> Iterator[None]:
    """Shrink the interpreter's thread-switch interval for the duration of each test.

    At the 5 ms default a race window a few bytecodes wide is almost never hit, so a test
    that removed a lock would still pass and prove nothing. At 1 us the interpreter switches
    inside ``x += 1``: measured against this module's UsageStats test, replacing the stats lock
    with a no-op loses ~25% of the recorded completion tokens at 1 us and exactly zero at the
    default. Restored afterwards so no other test's timing is affected.
    """
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


# --- helpers ----------------------------------------------------------------

@contextmanager
def _agents() -> Iterator[TranslationAgents]:
    """A minimal TranslationAgents whose only exercised state is the leniency window.

    The client is opened as a context manager so its socket is released deterministically even
    when a test fails; no request is ever sent through it here.
    """
    agent_set = AgentSet(
        translator_instructions="Translate.",
        reviewers=(Reviewer(id="seed", instructions="check"),))
    with LlmClient(ServerConfig(retry_backoff_seconds=0.0)) as client:
        yield TranslationAgents(client, RuleSet(), [], agent_set=agent_set)


def _in_parallel(worker: Callable[[int], None], *, workers: int = WORKERS) -> None:
    """Run ``worker(index)`` on ``workers`` threads released simultaneously.

    The barrier is what makes the contention real: every thread is parked until the last one
    arrives, so the hammering overlaps instead of accidentally serialising on thread startup.
    A failure inside a thread is collected and re-asserted here, because an exception raised on
    a worker thread would otherwise vanish and leave the test passing.
    """
    ready = threading.Barrier(workers)
    failures: list[BaseException] = []

    def run(index: int) -> None:
        try:
            ready.wait(timeout=30)
            worker(index)
        except BaseException as exc:  # noqa: BLE001 -- surfaced to the test below
            failures.append(exc)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads), "a worker thread hung"
    assert not failures, failures


def _surfaced_sequentially(pattern: Iterable[bool], leniency: Leniency) -> int:
    """Independent, single-threaded reimplementation of ``_register_reply``'s bookkeeping.

    Deliberately a second implementation rather than a call into the module under test: the
    point is to check the concurrent path against something that cannot share its bug.
    """
    history: deque[bool] = deque(maxlen=leniency.window)
    surfaced = 0
    for bad in pattern:
        history.append(bad)
        if bad and leniency.surfaces(sum(history)):
            surfaced += 1
    return surfaced


# --- the per-reviewer leniency window ---------------------------------------

class TestReplyHistoryUnderContention:
    """``TranslationAgents._register_reply`` under the concurrency it claims to support."""

    def test_one_shared_reviewer_hammered_by_every_worker_surfaces_the_exact_count(self) -> None:
        """All-bad replies make the answer order-independent -- once the window holds more than
        ``max_bad`` bad replies it never drops back -- so the surfaced count has one correct
        value no matter how the threads interleave. An append lost to an unguarded deque, or a
        ``sum`` taken over a deque another thread is mutating, moves that count.
        """
        leniency = Leniency(window=4, max_bad=2)
        reviewer = Reviewer(id="shared", instructions="check", leniency=leniency)
        per_worker = 300
        surfaced = 0
        counter_lock = threading.Lock()  # guards the test's own tally, not the code under test

        with _agents() as agents:
            def worker(_: int) -> None:
                nonlocal surfaced
                local = sum(agents._register_reply(reviewer, bad=True)
                            for _ in range(per_worker))
                with counter_lock:
                    surfaced += local

            _in_parallel(worker)

            expected = _surfaced_sequentially([True] * (WORKERS * per_worker), leniency)
            assert surfaced == expected == WORKERS * per_worker - leniency.max_bad
            assert list(agents._reply_history) == ["shared"]
            assert len(agents._reply_history["shared"]) == leniency.window

    def test_each_workers_own_reviewer_matches_its_sequential_reference_exactly(self) -> None:
        """Per-reviewer histories are independent, so a mixed good/bad pattern *is* fully
        deterministic when each thread owns its reviewer id -- which makes this the strict test:
        every worker's surfaced count must equal the sequential reference for its own pattern.
        The contention is on the shared ``_reply_history`` dict, where a lost insertion would
        show up as another worker's history being clobbered.
        """
        leniency = Leniency(window=5, max_bad=1)
        # A different bad/good rhythm per worker, so no two histories evolve alike.
        patterns = {index: [(step % (index + 2)) != 0 for step in range(200)]
                    for index in range(WORKERS)}
        results: dict[int, int] = {}
        results_lock = threading.Lock()

        with _agents() as agents:
            def worker(index: int) -> None:
                reviewer = Reviewer(id=f"reviewer-{index}", instructions="check",
                                    leniency=leniency)
                count = sum(agents._register_reply(reviewer, bad=bad)
                            for bad in patterns[index])
                with results_lock:
                    results[index] = count

            _in_parallel(worker)

            assert results == {index: _surfaced_sequentially(pattern, leniency)
                               for index, pattern in patterns.items()}
            assert len(agents._reply_history) == WORKERS

    def test_a_window_size_change_under_contention_rebuilds_without_corruption(self) -> None:
        """The rebuild branch (``history.maxlen != reviewer.leniency.window``) replaces the deque
        in the shared dict. Two reviewers sharing an id but disagreeing on the window force that
        branch on almost every call, from every thread at once -- the shape that would expose an
        unguarded rebuild as a lost deque or a ``RuntimeError: deque mutated during iteration``.

        Only the invariants that survive any interleaving are asserted: the answer itself
        genuinely depends on which window was in force, so pinning a count here would be pinning
        a race.
        """
        narrow = Reviewer(id="dup", instructions="c", leniency=Leniency(window=2, max_bad=0))
        wide = Reviewer(id="dup", instructions="c", leniency=Leniency(window=32, max_bad=0))

        with _agents() as agents:
            def worker(index: int) -> None:
                for step in range(300):
                    reviewer = narrow if (index + step) % 2 else wide
                    agents._register_reply(reviewer, bad=bool(step % 3))

            _in_parallel(worker)

            assert list(agents._reply_history) == ["dup"]
            history = agents._reply_history["dup"]
            assert history.maxlen in (2, 32)
            assert len(history) <= history.maxlen  # never over-filled by a racing append


# --- usage accounting -------------------------------------------------------

class TestUsageStatsUnderContention:
    """One :class:`UsageStats` is shared by every worker; its docstring promises no lost updates
    (``a bare += loses updates under concurrent completion``)."""

    def test_concurrent_records_lose_no_updates(self) -> None:
        stats = UsageStats()
        per_worker = 400

        def worker(index: int) -> None:
            for _ in range(per_worker):
                stats.record({"prompt_tokens": 3, "completion_tokens": 5}, 0.001)
                stats.record({"prompt_tokens": index + 1, "completion_tokens": 0}, 0.0)

        _in_parallel(worker)

        calls = WORKERS * per_worker
        assert stats.requests == 2 * calls
        assert stats.completion_tokens == 5 * calls
        assert stats.prompt_tokens == 3 * calls + sum(i + 1 for i in range(WORKERS)) * per_worker
        assert stats.peak_prompt_tokens == max(3, WORKERS)
        assert stats.seconds == pytest.approx(0.001 * calls)

    def test_retry_and_refusal_counters_lose_no_updates(self) -> None:
        """These are the counters a long run's summary is read off; a lost increment
        under-reports exactly the symptom an operator is looking for."""
        stats = UsageStats()
        per_worker = 500

        def worker(_: int) -> None:
            for _step in range(per_worker):
                stats.record_retry()
                stats.record_refusal()

        _in_parallel(worker)
        assert stats.retries == stats.refusals == WORKERS * per_worker

    def test_the_derived_rate_is_read_without_the_lock(self) -> None:
        """Documents a real (if benign) torn read: ``completion_tokens_per_second`` touches
        ``seconds`` and ``completion_tokens`` outside ``_lock``, so a concurrent ``record`` can
        land between those reads and the reported rate can mix a pre-update numerator with a
        post-update denominator.

        Proven without any timing assumption: the lock is held by this thread while another
        thread reads the property. A guarded read would block until released; an unguarded one
        returns immediately, which is what is asserted. The consequence is confined to a
        diagnostic figure -- the totals themselves are guarded (tests above) -- so this pins the
        behaviour rather than declaring it broken.
        """
        stats = UsageStats()
        stats.record({"prompt_tokens": 10, "completion_tokens": 20}, 1.0)
        finished = threading.Event()

        def read_the_rate() -> None:
            _ = stats.completion_tokens_per_second
            finished.set()

        reader = threading.Thread(target=read_the_rate, daemon=True)
        with stats._lock:
            reader.start()
            reader.join(timeout=5.0)
            unguarded = finished.is_set()
        reader.join(timeout=5.0)
        assert unguarded, "the rate property now blocks on _lock -- the torn read is fixed"


# --- the one-shot context-window warning ------------------------------------

class TestContextWarningUnderContention:
    """``LlmClient._warn_if_context_tight`` promises one warning per run, not per call. The
    check-and-set is guarded; under concurrency an unguarded one would emit several."""

    @staticmethod
    def _client() -> LlmClient:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7}})

        client = LlmClient(ServerConfig(retry_backoff_seconds=0.0, context_window=400),
                           backend=get_backend("generic"))
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        return client

    def test_the_budget_warning_fires_exactly_once_across_concurrent_calls(self, caplog) -> None:
        logging.getLogger("translator").propagate = True
        messages = [Message("system", "s " * 400), Message("user", "u " * 400)]
        with self._client() as client, \
                caplog.at_level(logging.WARNING, logger="translator.backend.client"):
            def worker(index: int) -> None:
                for _ in range(5):
                    client.complete(messages, role=f"role-{index}")

            _in_parallel(worker)

        warnings = [record for record in caplog.records
                    if "prompt budget is tight" in record.getMessage()]
        assert len(warnings) == 1  # WORKERS * 5 calls all crossed the threshold
