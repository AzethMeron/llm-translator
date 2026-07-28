"""The durability guarantees, exercised by injecting the failures they defend against.

A full run spans hours and will be interrupted, so these are correctness properties, not
nice-to-haves: an atomic catalogue write that a crash cannot truncate, a journal that
tolerates the torn final record a killed process leaves, and a resume that re-reads the
journal and continues without repeating finished work.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from transunit.units import (
    CatalogError,
    Status,
    Unit,
    merge_journal,
    read_catalog,
    read_journal,
    write_catalog,
)


def unit(uid: str, source: str, *, status=Status.PENDING, target=None) -> Unit:
    return Unit(unit_id=uid, rel_path="f.txt", line_no=1, span_start=0, span_end=1,
                command="", kind="T", source=source, status=status, target=target)


class TestAtomicCatalogueWrite:
    def test_write_then_read_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        units = [unit("a", "one"), unit("b", "two")]
        assert write_catalog(units, path) == 2
        assert [u.unit_id for u in read_catalog(path)] == ["a", "b"]

    def test_no_partial_file_is_left_behind_on_success(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        write_catalog([unit("a", "x")], path)
        assert not path.with_suffix(".jsonl.partial").exists()

    def test_a_failure_mid_write_leaves_no_partial_and_re_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"

        def exploding():
            yield unit("a", "x")
            raise RuntimeError("source blew up")

        with pytest.raises(RuntimeError, match="source blew up"):
            write_catalog(exploding(), path)
        # The half-written temporary must be cleaned up, not left to masquerade as an
        # interrupted run.
        assert not path.with_suffix(".jsonl.partial").exists()

    def test_a_failed_write_does_not_destroy_an_existing_catalogue(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        write_catalog([unit("good", "kept")], path)

        def exploding():
            yield unit("a", "x")
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            write_catalog(exploding(), path)
        # The rename never happened, so the previous good catalogue is intact.
        assert [u.unit_id for u in read_catalog(path)] == ["good"]


class TestJournalTornRecord:
    def test_a_torn_final_line_is_tolerated_and_dropped(self, tmp_path: Path) -> None:
        # The exact shape a process killed mid-write leaves: a last line with no newline.
        journal = tmp_path / "j.jsonl"
        journal.write_text(
            unit("a", "one").to_json() + "\n" + '{"unit_id": "b", "rel_pa',  # truncated
            encoding="utf-8")
        recovered = list(read_journal(journal))
        assert [u.unit_id for u in recovered] == ["a"]  # 'b' simply stays pending

    def test_a_malformed_line_that_is_not_the_last_raises(self, tmp_path: Path) -> None:
        # Corruption anywhere but the torn tail must not be silently skipped: doing so would
        # discard a completed translation while still reporting success.
        journal = tmp_path / "j.jsonl"
        journal.write_text("{ broken }\n" + unit("b", "two").to_json() + "\n",
                           encoding="utf-8")
        with pytest.raises(CatalogError, match="corrupt journal record"):
            list(read_journal(journal))

    def test_blank_lines_are_ignored(self, tmp_path: Path) -> None:
        journal = tmp_path / "j.jsonl"
        journal.write_text("\n" + unit("a", "x").to_json() + "\n\n", encoding="utf-8")
        assert [u.unit_id for u in read_journal(journal)] == ["a"]

    def test_a_missing_journal_yields_nothing(self, tmp_path: Path) -> None:
        assert list(read_journal(tmp_path / "absent.jsonl")) == []


class TestResume:
    def test_merge_journal_lets_later_results_win(self, tmp_path: Path) -> None:
        catalog = tmp_path / "c.jsonl"
        journal = tmp_path / "j.jsonl"
        output = tmp_path / "merged.jsonl"
        write_catalog([unit("a", "one"), unit("b", "two")], catalog)
        # A later run re-did 'a' and produced a verified result.
        journal.write_text(unit("a", "one", status=Status.VERIFIED, target="ONE").to_json()
                           + "\n", encoding="utf-8")
        written, updated = merge_journal(catalog, journal, output)
        assert (written, updated) == (2, 1)
        merged = {u.unit_id: u for u in read_catalog(output)}
        assert merged["a"].status is Status.VERIFIED and merged["a"].target == "ONE"
        assert merged["b"].status is Status.PENDING

    def test_merge_journal_without_a_journal_is_an_error(self, tmp_path: Path) -> None:
        catalog = tmp_path / "c.jsonl"
        write_catalog([unit("a", "x")], catalog)
        with pytest.raises(CatalogError, match="journal not found"):
            merge_journal(catalog, tmp_path / "absent.jsonl", tmp_path / "out.jsonl")
