"""Unit tests for :mod:`translator.memory`.

:class:`TranslationMemory` is a lock-guarded source-to-target store. Tests cover the basic
record/get/len contract, the :meth:`from_units` filter (only renderable units contribute),
and a deterministic concurrency smoke test for the lock.
"""
from __future__ import annotations

import threading

from transunit.units import Status, Unit
from translator.memory import TranslationMemory


def _unit(source: str, *, target: str | None = None, status: Status = Status.PENDING,
          unit_id: str = "u") -> Unit:
    """A minimal valid Unit. TRANSLATED requires a target, per the Unit invariant."""
    return Unit(
        unit_id=unit_id, rel_path="f.srt", line_no=1, span_start=0, span_end=0,
        command="", kind="dialogue", source=source, status=status, target=target)


class TestRecordAndGet:
    def test_record_then_get_returns_target(self):
        memory = TranslationMemory()
        memory.record("hello", "bonjour")
        assert memory.get("hello") == "bonjour"

    def test_get_unknown_returns_none(self):
        assert TranslationMemory().get("absent") is None

    def test_record_overwrites_existing_key(self):
        memory = TranslationMemory()
        memory.record("k", "first")
        memory.record("k", "second")
        assert memory.get("k") == "second"
        assert len(memory) == 1

    def test_len_counts_distinct_sources(self):
        memory = TranslationMemory()
        memory.record("a", "1")
        memory.record("b", "2")
        assert len(memory) == 2

    def test_empty_memory_has_zero_len(self):
        assert len(TranslationMemory()) == 0

    def test_empty_string_key_is_distinct_and_storable(self):
        memory = TranslationMemory()
        memory.record("", "empty-source rendering")
        assert memory.get("") == "empty-source rendering"


class TestInitial:
    def test_initial_dict_seeds_the_store(self):
        memory = TranslationMemory({"a": "1", "b": "2"})
        assert memory.get("a") == "1"
        assert len(memory) == 2

    def test_none_initial_is_empty(self):
        assert len(TranslationMemory(None)) == 0

    def test_initial_dict_is_copied_not_aliased(self):
        """Mutating the caller's dict afterwards must not leak into the memory."""
        source_dict = {"a": "1"}
        memory = TranslationMemory(source_dict)
        source_dict["b"] = "2"
        assert memory.get("b") is None
        assert len(memory) == 1


class TestFromUnits:
    def test_verified_unit_with_target_is_included(self):
        memory = TranslationMemory.from_units(
            [_unit("src", target="tgt", status=Status.VERIFIED)])
        assert memory.get("src") == "tgt"

    def test_translated_unit_with_target_is_included(self):
        memory = TranslationMemory.from_units(
            [_unit("src", target="tgt", status=Status.TRANSLATED)])
        assert memory.get("src") == "tgt"

    def test_rejected_unit_is_excluded(self):
        """A rejected rendering failed its checks; offering it as context spreads it."""
        memory = TranslationMemory.from_units(
            [_unit("src", target="bad", status=Status.REJECTED)])
        assert memory.get("src") is None
        assert len(memory) == 0

    def test_pending_unit_is_excluded(self):
        memory = TranslationMemory.from_units(
            [_unit("src", target="draft", status=Status.PENDING)])
        assert len(memory) == 0

    def test_skipped_unit_is_excluded(self):
        memory = TranslationMemory.from_units(
            [_unit("src", target="src", status=Status.SKIPPED)])
        assert len(memory) == 0

    def test_verified_unit_with_empty_target_is_excluded(self):
        """An empty target is falsy, so it is filtered like a missing one."""
        memory = TranslationMemory.from_units(
            [_unit("src", target="", status=Status.VERIFIED)])
        assert len(memory) == 0

    def test_empty_iterable_yields_empty_memory(self):
        assert len(TranslationMemory.from_units([])) == 0

    def test_only_eligible_units_survive_a_mixed_batch(self):
        units = [
            _unit("a", target="ta", status=Status.VERIFIED, unit_id="1"),
            _unit("b", target="tb", status=Status.TRANSLATED, unit_id="2"),
            _unit("c", target="tc", status=Status.REJECTED, unit_id="3"),
            _unit("d", target="td", status=Status.PENDING, unit_id="4"),
            _unit("e", target="te", status=Status.SKIPPED, unit_id="5"),
        ]
        memory = TranslationMemory.from_units(units)
        assert len(memory) == 2
        assert memory.get("a") == "ta"
        assert memory.get("b") == "tb"
        assert memory.get("c") is None

    def test_later_duplicate_source_wins(self):
        """Two eligible units sharing a source collapse to the last recorded target."""
        units = [
            _unit("same", target="first", status=Status.VERIFIED, unit_id="1"),
            _unit("same", target="second", status=Status.VERIFIED, unit_id="2"),
        ]
        memory = TranslationMemory.from_units(units)
        assert memory.get("same") == "second"
        assert len(memory) == 1


class TestConcurrency:
    """The store is shared across worker threads; the lock must make record atomic."""

    def test_distinct_keys_across_threads_have_no_lost_updates(self):
        memory = TranslationMemory()
        thread_count, per_thread = 16, 200
        barrier = threading.Barrier(thread_count)

        def worker(tid: int) -> None:
            barrier.wait()  # release together to maximise contention
            for i in range(per_thread):
                memory.record(f"t{tid}-k{i}", f"v{tid}-{i}")

        threads = [threading.Thread(target=worker, args=(tid,))
                   for tid in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(memory) == thread_count * per_thread
        # Spot-check that every thread's writes are individually retrievable.
        for tid in range(thread_count):
            assert memory.get(f"t{tid}-k0") == f"v{tid}-0"
            assert memory.get(f"t{tid}-k{per_thread - 1}") == f"v{tid}-{per_thread - 1}"

    def test_concurrent_writes_to_one_key_leave_a_consistent_single_entry(self):
        memory = TranslationMemory()
        thread_count = 32
        barrier = threading.Barrier(thread_count)
        allowed = {f"v{tid}" for tid in range(thread_count)}

        def worker(tid: int) -> None:
            barrier.wait()
            for _ in range(100):
                memory.record("shared", f"v{tid}")

        threads = [threading.Thread(target=worker, args=(tid,))
                   for tid in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(memory) == 1
        assert memory.get("shared") in allowed
