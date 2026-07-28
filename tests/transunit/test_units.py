"""Tests for :mod:`transunit.units`.

The translation unit is the pipeline's source of truth, so its invariants, its JSON-Lines
round-trip, and the durability guarantees of its catalogue/journal I/O all have to hold
exactly. The high-value cases here are: construction rejects impossible units, the
serialisation preserves tuples, ``write_catalog`` is atomic (no ``.partial`` survives),
``read_journal`` tolerates a torn final line but not corruption elsewhere, and
``merge_journal`` lets later results win.
"""
from __future__ import annotations

import json

import pytest

from transunit.units import (
    CatalogError,
    Status,
    Unit,
    fsync_directory,
    make_unit_id,
    merge_journal,
    read_catalog,
    read_journal,
    write_catalog,
)


def make_unit(**overrides) -> Unit:
    """A valid Unit with sensible defaults; override only the field under test."""
    base = dict(
        unit_id="u1",
        rel_path="a/b.txt",
        line_no=1,
        span_start=0,
        span_end=10,
        command="say",
        kind="line",
        source="Hello",
    )
    base.update(overrides)
    return Unit(**base)


class TestUnitInvariants:
    def test_span_end_may_equal_span_start(self) -> None:
        """The span is half-open, so an empty span is legitimate; only a reversed one is
        impossible."""
        assert make_unit(span_start=5, span_end=5).span_end == 5

    def test_span_end_before_span_start_raises(self) -> None:
        with pytest.raises(CatalogError) as excinfo:
            make_unit(span_start=10, span_end=3)
        assert "span_end" in excinfo.value.reason and "span_start" in excinfo.value.reason
        assert excinfo.value.path is None
        assert excinfo.value.line_no is None

    def test_zero_max_columns_raises(self) -> None:
        """A carrier budget of zero columns could never hold any text, so it is a
        misconfiguration rather than a very tight limit."""
        with pytest.raises(CatalogError) as excinfo:
            make_unit(max_columns=0)
        assert "max_columns must be positive" in excinfo.value.reason

    def test_negative_max_columns_raises(self) -> None:
        with pytest.raises(CatalogError):
            make_unit(max_columns=-5)

    def test_positive_max_columns_is_accepted(self) -> None:
        assert make_unit(max_columns=1).max_columns == 1

    def test_none_max_columns_means_no_limit(self) -> None:
        assert make_unit(max_columns=None).max_columns is None

    def test_translated_status_requires_a_target(self) -> None:
        """TRANSLATED is rendered like a verified line, so it must not exist without the
        text it would render."""
        with pytest.raises(CatalogError) as excinfo:
            make_unit(status=Status.TRANSLATED, target=None)
        assert "target is None" in excinfo.value.reason

    def test_translated_status_with_target_is_valid(self) -> None:
        unit = make_unit(status=Status.TRANSLATED, target="Cześć")
        assert unit.status is Status.TRANSLATED and unit.target == "Cześć"

    def test_rejected_status_may_have_no_target(self) -> None:
        """Only TRANSLATED carries the target requirement; a rejected unit legitimately
        has none."""
        assert make_unit(status=Status.REJECTED, target=None).target is None


class TestStatusSemantics:
    @pytest.mark.parametrize("status", [Status.VERIFIED, Status.TRANSLATED])
    def test_injectable_states(self, status: Status) -> None:
        assert status.is_injectable

    @pytest.mark.parametrize("status", [Status.PENDING, Status.REJECTED, Status.SKIPPED])
    def test_non_injectable_states(self, status: Status) -> None:
        """REJECTED and SKIPPED must never render: their target may be empty or still in
        the source language, so injecting it would put a defect before the reader."""
        assert not status.is_injectable

    def test_is_done_covers_verified_and_skipped(self) -> None:
        assert make_unit(status=Status.VERIFIED, target="x").is_done
        assert make_unit(status=Status.SKIPPED).is_done
        assert not make_unit(status=Status.PENDING).is_done


class TestWithTarget:
    def test_returns_a_new_unit_leaving_the_original_untouched(self) -> None:
        original = make_unit()
        updated = original.with_target("Cześć", Status.VERIFIED)
        assert updated.target == "Cześć" and updated.status is Status.VERIFIED
        assert original.target is None and original.status is Status.PENDING


class TestJsonRoundTrip:
    def test_full_unit_survives_round_trip_with_tuples_intact(self) -> None:
        """Every collection field must come back a tuple, not a list, or the frozen Unit
        would be silently mutable through its own fields."""
        unit = make_unit(
            placeholders=("[[0]]", "[[1]]"),
            speaker="NARRATOR",
            context_before=("prev",),
            context_after=("next", "after"),
            max_columns=40,
            status=Status.VERIFIED,
            target="Witaj",
            notes=("checked",),
        )
        restored = Unit.from_json(unit.to_json())
        assert restored == unit
        assert isinstance(restored.placeholders, tuple)
        assert isinstance(restored.context_before, tuple)
        assert isinstance(restored.context_after, tuple)
        assert isinstance(restored.notes, tuple)

    def test_status_is_serialised_as_its_string_value(self) -> None:
        payload = json.loads(make_unit(status=Status.SKIPPED).to_json())
        assert payload["status"] == "skipped"

    def test_from_json_reports_invalid_json_with_location(self) -> None:
        with pytest.raises(CatalogError) as excinfo:
            Unit.from_json("{not json", path=None, line_no=4)
        assert "invalid JSON" in excinfo.value.reason
        assert excinfo.value.line_no == 4

    def test_from_json_lists_missing_required_fields(self) -> None:
        with pytest.raises(CatalogError) as excinfo:
            Unit.from_json('{"unit_id": "x"}')
        assert "missing fields" in excinfo.value.reason

    def test_from_json_rejects_an_unknown_status_value(self) -> None:
        """An out-of-vocabulary status is a bad field value, distinct from missing
        fields, and must not be coerced to a default."""
        raw = json.loads(make_unit().to_json())
        raw["status"] = "not-a-status"
        with pytest.raises(CatalogError) as excinfo:
            Unit.from_json(json.dumps(raw))
        assert "bad field value" in excinfo.value.reason

    def test_from_json_propagates_construction_invariant(self) -> None:
        raw = json.loads(make_unit().to_json())
        raw["span_start"], raw["span_end"] = 9, 2
        with pytest.raises(CatalogError):
            Unit.from_json(json.dumps(raw))


class TestMakeUnitId:
    def test_deterministic_for_identical_inputs(self) -> None:
        assert make_unit_id("a.txt", "Hello", 0) == make_unit_id("a.txt", "Hello", 0)

    def test_independent_of_position_in_the_source(self) -> None:
        """The id is keyed on (path, text, occurrence), never on line/byte position, so a
        payload keeps its id -- and its translation -- when recognition moves it."""
        moved = make_unit(line_no=99, span_start=500, span_end=520)
        anchor = make_unit(line_no=1, span_start=0, span_end=10)
        assert make_unit_id(moved.rel_path, moved.source, 0) == make_unit_id(
            anchor.rel_path, anchor.source, 0)

    def test_differs_by_ordinal(self) -> None:
        """The occurrence index is what keeps a verbatim repeat from collapsing to one
        id."""
        assert make_unit_id("a.txt", "Hello", 0) != make_unit_id("a.txt", "Hello", 1)

    def test_differs_by_source_text(self) -> None:
        assert make_unit_id("a.txt", "Hello", 0) != make_unit_id("a.txt", "Goodbye", 0)

    def test_is_sixteen_hex_characters(self) -> None:
        result = make_unit_id("a.txt", "Hello", 0)
        assert len(result) == 16 and all(c in "0123456789abcdef" for c in result)


class TestWriteAndReadCatalog:
    def test_round_trip_streams_units_back(self, tmp_path) -> None:
        units = [make_unit(unit_id="u1"), make_unit(unit_id="u2", source="World")]
        path = tmp_path / "cat.jsonl"
        assert write_catalog(units, path) == 2
        assert list(read_catalog(path)) == units

    def test_write_creates_parent_directories(self, tmp_path) -> None:
        path = tmp_path / "deep" / "nested" / "cat.jsonl"
        write_catalog([make_unit()], path)
        assert path.is_file()

    def test_success_leaves_no_partial_behind(self, tmp_path) -> None:
        """The temporary sibling is the signature of an interrupted run; a clean write
        must not leave one to be mistaken for one."""
        path = tmp_path / "cat.jsonl"
        write_catalog([make_unit()], path)
        assert not (tmp_path / "cat.jsonl.partial").exists()

    def test_read_missing_catalogue_raises_with_path(self, tmp_path) -> None:
        missing = tmp_path / "absent.jsonl"
        with pytest.raises(CatalogError) as excinfo:
            list(read_catalog(missing))
        assert "not found" in excinfo.value.reason
        assert excinfo.value.path == missing

    def test_read_blank_lines_are_skipped(self, tmp_path) -> None:
        path = tmp_path / "cat.jsonl"
        path.write_text(make_unit().to_json() + "\n\n\n")
        assert len(list(read_catalog(path))) == 1

    def test_read_malformed_line_reports_its_number(self, tmp_path) -> None:
        path = tmp_path / "cat.jsonl"
        path.write_text(make_unit().to_json() + "\n" + "garbage\n")
        with pytest.raises(CatalogError) as excinfo:
            list(read_catalog(path))
        assert excinfo.value.line_no == 2
        assert excinfo.value.path == path


class TestWriteCatalogAtomicity:
    def test_mid_iteration_failure_removes_partial_and_reraises(self, tmp_path) -> None:
        """If the source iterable raises part-way through, the half-written temporary must
        be cleaned up so it cannot accumulate as a phantom interrupted run -- and the
        original target must be left untouched."""
        path = tmp_path / "cat.jsonl"

        def failing_units():
            yield make_unit(unit_id="u1")
            raise RuntimeError("source exhausted early")

        with pytest.raises(RuntimeError, match="source exhausted early"):
            write_catalog(failing_units(), path)

        assert not (tmp_path / "cat.jsonl.partial").exists()
        assert not path.exists()

    def test_failure_does_not_clobber_an_existing_catalogue(self, tmp_path) -> None:
        """Atomicity's whole point: a failed rewrite leaves the previous good catalogue
        in place rather than a truncated one."""
        path = tmp_path / "cat.jsonl"
        write_catalog([make_unit(unit_id="good")], path)

        def failing_units():
            yield make_unit(unit_id="u1")
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            write_catalog(failing_units(), path)

        assert [u.unit_id for u in read_catalog(path)] == ["good"]
        assert not (tmp_path / "cat.jsonl.partial").exists()


class TestReadJournal:
    def test_missing_journal_yields_nothing(self, tmp_path) -> None:
        assert list(read_journal(tmp_path / "absent.jsonl")) == []

    def test_reads_records_in_write_order(self, tmp_path) -> None:
        path = tmp_path / "journal.jsonl"
        first = make_unit(unit_id="a")
        second = make_unit(unit_id="b")
        path.write_text(first.to_json() + "\n" + second.to_json() + "\n")
        assert list(read_journal(path)) == [first, second]

    def test_tolerates_a_torn_final_line(self, tmp_path) -> None:
        """A final line with no trailing newline is the signature of a process killed
        mid-write; the completed records before it must still be recovered and the torn
        one dropped."""
        path = tmp_path / "journal.jsonl"
        good = make_unit(unit_id="a")
        path.write_text(good.to_json() + "\n" + '{"partial": tru')
        assert list(read_journal(path)) == [good]

    def test_malformed_line_elsewhere_raises(self, tmp_path) -> None:
        """Corruption before the last line means a completed translation would be silently
        discarded while the run reported success, so it must raise instead."""
        path = tmp_path / "journal.jsonl"
        path.write_text("this is not json\n" + make_unit(unit_id="b").to_json() + "\n")
        with pytest.raises(CatalogError) as excinfo:
            list(read_journal(path))
        assert excinfo.value.line_no == 1
        assert excinfo.value.path == path
        assert "corrupt journal record" in excinfo.value.reason

    def test_malformed_final_line_with_newline_still_raises(self, tmp_path) -> None:
        """The trailing newline is what distinguishes a fully-written (but corrupt) line
        from a torn one; with the newline present it is corruption, not truncation."""
        path = tmp_path / "journal.jsonl"
        path.write_text(make_unit(unit_id="a").to_json() + "\n" + "broken\n")
        with pytest.raises(CatalogError) as excinfo:
            list(read_journal(path))
        assert excinfo.value.line_no == 2

    def test_blank_lines_are_skipped(self, tmp_path) -> None:
        path = tmp_path / "journal.jsonl"
        path.write_text("\n" + make_unit(unit_id="a").to_json() + "\n\n")
        assert len(list(read_journal(path))) == 1


class TestMergeJournal:
    def test_missing_journal_raises(self, tmp_path) -> None:
        catalog = tmp_path / "cat.jsonl"
        write_catalog([make_unit()], catalog)
        with pytest.raises(CatalogError) as excinfo:
            merge_journal(catalog, tmp_path / "absent.jsonl", tmp_path / "out.jsonl")
        assert "journal not found" in excinfo.value.reason

    def test_journalled_result_replaces_the_catalogue_entry(self, tmp_path) -> None:
        catalog = tmp_path / "cat.jsonl"
        journal = tmp_path / "journal.jsonl"
        output = tmp_path / "out.jsonl"
        write_catalog(
            [make_unit(unit_id="a"), make_unit(unit_id="b", source="World")], catalog)
        translated = make_unit(unit_id="a", status=Status.VERIFIED, target="Witaj")
        write_catalog([translated], journal)

        written, updated = merge_journal(catalog, journal, output)

        assert (written, updated) == (2, 1)
        merged = {u.unit_id: u for u in read_catalog(output)}
        assert merged["a"] == translated
        assert merged["b"].status is Status.PENDING

    def test_later_journal_entry_wins(self, tmp_path) -> None:
        """Re-running a unit appends a fresh journal line; the later result must supersede
        the earlier one so a correction is not lost to an earlier attempt."""
        catalog = tmp_path / "cat.jsonl"
        journal = tmp_path / "journal.jsonl"
        output = tmp_path / "out.jsonl"
        write_catalog([make_unit(unit_id="a")], catalog)
        early = make_unit(unit_id="a", status=Status.TRANSLATED, target="first")
        late = make_unit(unit_id="a", status=Status.VERIFIED, target="second")
        journal.write_text(early.to_json() + "\n" + late.to_json() + "\n")

        merge_journal(catalog, journal, output)

        (merged,) = list(read_catalog(output))
        assert merged.target == "second" and merged.status is Status.VERIFIED

    def test_a_journal_entry_for_an_absent_unit_is_refused_not_dropped(self, tmp_path) -> None:
        """A journalled result whose id is not in the catalogue is a real completed
        translation with nowhere to go -- usually a drifted journal/catalogue. Silently
        discarding it while reporting success is the failure this refuses."""
        catalog = tmp_path / "cat.jsonl"
        journal = tmp_path / "journal.jsonl"
        output = tmp_path / "out.jsonl"
        write_catalog([make_unit(unit_id="a")], catalog)
        write_catalog([make_unit(unit_id="orphan", status=Status.VERIFIED, target="x")], journal)

        with pytest.raises(CatalogError, match="no matching unit"):
            merge_journal(catalog, journal, output)


class TestFsyncDirectory:
    def test_existing_directory_is_flushed_without_error(self, tmp_path) -> None:
        fsync_directory(tmp_path)

    def test_missing_directory_is_not_an_error(self, tmp_path) -> None:
        """A filesystem that cannot open a directory makes the rename durable anyway, so a
        refusal (here, a nonexistent path) is swallowed rather than raised."""
        fsync_directory(tmp_path / "does-not-exist")

    def test_a_failing_directory_fsync_is_tolerated(self, tmp_path, monkeypatch) -> None:
        # The directory opens, but fsync on it is unsupported; where that happens the rename is
        # durable anyway, so the refusal is swallowed rather than crashing the write path.
        from transunit import units as units_module

        def boom(descriptor: int) -> None:
            raise OSError("fsync on a directory is unsupported here")

        monkeypatch.setattr(units_module.os, "fsync", boom)
        units_module.fsync_directory(tmp_path)  # must not raise


class TestInjectableInvariant:
    """A unit whose status says it may be rendered must carry the text to render."""

    def test_verified_without_target_is_rejected(self) -> None:
        with pytest.raises(CatalogError, match="injectable but target is None"):
            make_unit(unit_id="v", status=Status.VERIFIED, target=None)

    def test_translated_without_target_is_rejected(self) -> None:
        with pytest.raises(CatalogError, match="injectable but target is None"):
            make_unit(unit_id="t", status=Status.TRANSLATED, target=None)

    def test_rejected_without_target_is_allowed(self) -> None:
        # REJECTED and SKIPPED are not injectable, so they may lack a target.
        assert make_unit(status=Status.REJECTED, target=None).target is None
        assert make_unit(status=Status.SKIPPED, target=None).target is None

    def test_verified_with_empty_target_is_allowed(self) -> None:
        # The invariant is "not None", not "non-empty"; an empty string is a value.
        assert make_unit(status=Status.VERIFIED, target="").target == ""
