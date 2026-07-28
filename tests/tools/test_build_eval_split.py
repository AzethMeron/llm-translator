"""Tests for ``tools/lib/build_eval_split.py``, the shared corpus/query split builder.

Both live evaluations are built on this one script, so a defect here does not crash anything --
it quietly changes what was measured. The cases below pin the properties the *results* depend on:
the split is deterministic (a rerun compares like with like), queries are genuinely held out (a
query still present in the corpus retrieves itself and inflates every score), a shortfall is a
loud error rather than a smaller sample, and the self-tuning oversample loop terminates instead of
spinning on a corpus that can never satisfy the request.

Everything is filesystem-only: no server, no network.
"""
from __future__ import annotations

import json
import random
import re

import pytest

import build_eval_split as split


def row(unit_id: str, source: str, *, target: str = "target", status: str = "translated",
        **extra: object) -> dict:
    return {"unit_id": unit_id, "source": source, "target": target, "status": status, **extra}


def write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def args_for(tmp_path, **overrides):
    """A parsed argument namespace, built through the real parser so defaults stay in one place."""
    argv = [
        "--journal", str(overrides.pop("journal", tmp_path / "journal.jsonl")),
        "--corpus-out", str(overrides.pop("corpus_out", tmp_path / "corpus.jsonl")),
        "--queries-out", str(overrides.pop("queries_out", tmp_path / "queries.jsonl")),
        "--queries", str(overrides.pop("queries", 5)),
    ]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return split.build_parser().parse_args(argv)


class TestReadUnits:
    """A journal is appended to while it is read, so exactly one kind of damage is tolerable."""

    def test_reads_every_object(self, tmp_path):
        path = tmp_path / "j.jsonl"
        write_jsonl(path, [row("a", "one"), row("b", "two")])
        assert [r["unit_id"] for r in split.read_units(path)] == ["a", "b"]

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "j.jsonl"
        path.write_text(json.dumps(row("a", "one")) + "\n\n   \n", encoding="utf-8")
        assert len(list(split.read_units(path))) == 1

    def test_a_torn_final_line_is_tolerated(self, tmp_path):
        """The writer may still be appending; a half-written last record is not corruption."""
        path = tmp_path / "j.jsonl"
        path.write_text(json.dumps(row("a", "one")) + '\n{"unit_id": "b", "sou',
                        encoding="utf-8")
        assert [r["unit_id"] for r in split.read_units(path)] == ["a"]

    def test_a_malformed_line_in_the_middle_raises_naming_the_line(self, tmp_path):
        """Silently skipping a record makes the split stop matching the corpus it describes."""
        path = tmp_path / "j.jsonl"
        path.write_text(json.dumps(row("a", "one")) + "\n{ broken\n"
                        + json.dumps(row("c", "three")) + "\n", encoding="utf-8")
        with pytest.raises(split.SplitError, match="line 2"):
            list(split.read_units(path))

    def test_a_non_object_line_raises(self, tmp_path):
        path = tmp_path / "j.jsonl"
        path.write_text("[1, 2]\n" + json.dumps(row("a", "one")) + "\n", encoding="utf-8")
        with pytest.raises(split.SplitError, match="line 1 is not a JSON object"):
            list(split.read_units(path))

    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(split.SplitError, match="file not found"):
            list(split.read_units(tmp_path / "absent.jsonl"))


class TestTranslatedRows:
    def test_only_rows_with_the_wanted_status_and_both_texts(self, tmp_path):
        path = tmp_path / "j.jsonl"
        write_jsonl(path, [
            row("a", "one"),
            row("b", "two", status="pending"),
            row("c", "  ", target="x"),
            row("d", "four", target="   "),
            row("e", "five", status="verified"),
        ])
        rows = split.translated_rows(path, statuses=frozenset({"translated", "verified"}))
        assert [r["unit_id"] for r in rows] == ["a", "e"]

    def test_no_usable_rows_raises(self, tmp_path):
        """An empty corpus would otherwise produce an empty split and a meaningless evaluation."""
        path = tmp_path / "j.jsonl"
        write_jsonl(path, [row("a", "one", status="pending")])
        with pytest.raises(split.SplitError, match="no rows with status"):
            split.translated_rows(path, statuses=frozenset({"translated"}))


class TestStratified:
    def test_is_deterministic_under_a_fixed_seed(self, tmp_path):
        """A rerun of an evaluation must compare like with like, which starts here."""
        pool = [row(f"u{i}", "x" * (i % 80 + 1)) for i in range(120)]
        first = split.stratified(pool, 20, random.Random(7))
        second = split.stratified(pool, 20, random.Random(7))
        assert [r["unit_id"] for r in first] == [r["unit_id"] for r in second]

    def test_a_different_seed_gives_a_different_sample(self, tmp_path):
        pool = [row(f"u{i}", "x" * (i % 80 + 1)) for i in range(120)]
        a = [r["unit_id"] for r in split.stratified(pool, 20, random.Random(1))]
        b = [r["unit_id"] for r in split.stratified(pool, 20, random.Random(2))]
        assert a != b

    def test_deduplicates_by_unit_id(self, tmp_path):
        """A journal resumed onto itself repeats unit ids; sampling one twice would double-count."""
        pool = [row("dup", "hello there") for _ in range(10)] + [row("other", "hello there")]
        chosen = split.stratified(pool, 5, random.Random(0))
        ids = [r["unit_id"] for r in chosen]
        assert sorted(ids) == ["dup", "other"]

    def test_rows_outside_every_band_are_still_reachable(self, tmp_path):
        """A blank source and a 400+ character monster fit no band; they must not vanish from
        the pool entirely, only from the stratified quota."""
        pool = [row("blank", ""), row("huge", "x" * 500)] + [row(f"u{i}", "abc") for i in range(3)]
        chosen = split.stratified(pool, 5, random.Random(0))
        assert len(chosen) == 5

    def test_a_duplicate_inside_one_band_is_taken_only_once(self, tmp_path):
        """The bands are filled before dedup, so the same id can appear several times in the
        slice a band contributes."""
        pool = [row("dup", "a short line") for _ in range(10)]
        pool += [row(f"u{i}", "another short line") for i in range(10)]
        chosen = split.stratified(pool, 25, random.Random(0))
        assert len(chosen) == len({r["unit_id"] for r in chosen})

    def test_never_returns_more_than_asked(self, tmp_path):
        """per_band has a floor of 1, so five bands can over-fill a request for three."""
        pool = [row(f"u{i}", "x" * length)
                for i, length in enumerate([1, 3, 8, 20, 40, 100, 200])]
        assert len(split.stratified(pool, 3, random.Random(0))) == 3

    def test_a_thin_band_is_topped_up_from_the_remainder(self, tmp_path):
        """A corpus with no long lines must still yield the full count, not silently fewer."""
        pool = [row(f"u{i}", "abc") for i in range(50)]  # every row in the shortest band
        chosen = split.stratified(pool, 12, random.Random(3))
        assert len(chosen) == 12
        assert len({r["unit_id"] for r in chosen}) == 12

    def test_long_lines_survive_a_pool_dominated_by_short_ones(self, tmp_path):
        """The whole point of stratifying: long, hard lines are the informative ones."""
        pool = [row(f"s{i}", "ab") for i in range(200)]
        pool += [row(f"l{i}", "x" * 90) for i in range(5)]
        chosen = split.stratified(pool, 10, random.Random(0))
        assert any(r["unit_id"].startswith("l") for r in chosen)

    def test_returns_what_it_can_when_the_pool_is_smaller_than_asked(self, tmp_path):
        pool = [row("a", "one"), row("b", "two")]
        assert len(split.stratified(pool, 10, random.Random(0))) == 2


class TestEligibleQueries:
    def test_excludes_sources_present_in_the_corpus(self, tmp_path):
        """A query already in the corpus retrieves itself at 1.0 and inflates every score."""
        pool = [row("a", "in corpus"), row("b", "fresh")]
        chosen = split.eligible_queries(pool, corpus_sources={"in corpus"}, script=None)
        assert [r["unit_id"] for r in chosen] == ["b"]

    def test_deduplicates_by_source_text(self, tmp_path):
        """Two ids, one string: judging the same line twice would weight it double."""
        pool = [row("a", "same"), row("b", "same"), row("c", "other")]
        chosen = split.eligible_queries(pool, corpus_sources=set(), script=None)
        assert [r["unit_id"] for r in chosen] == ["a", "c"]

    def test_blank_sources_are_dropped(self, tmp_path):
        pool = [row("a", "   "), row("b", ""), row("c", "real"), {"unit_id": "d"}]
        chosen = split.eligible_queries(pool, corpus_sources=set(), script=None)
        assert [r["unit_id"] for r in chosen] == ["c"]

    def test_the_script_filter_applies(self, tmp_path):
        """Untranslatable rows (numbers, ASCII markup) otherwise waste the whole sample."""
        pool = [row("a", "こんにちは"), row("b", "12345"), row("c", "hello")]
        script = re.compile(split.SCRIPT_PATTERNS["japanese"])
        chosen = split.eligible_queries(pool, corpus_sources=set(), script=script)
        assert [r["unit_id"] for r in chosen] == ["a"]

    def test_comparison_against_the_corpus_ignores_surrounding_whitespace(self, tmp_path):
        """The corpus side is stripped, so the query side must be too or the filter misses."""
        pool = [row("a", "  in corpus  ")]
        assert split.eligible_queries(pool, corpus_sources={"in corpus"}, script=None) == []


class TestToPending:
    def test_preserves_surrounding_context(self, tmp_path):
        """The engine packs context_before/after into the prompt: dropping them here would
        evaluate a different prompt than production sends."""
        source = row("a", "line", context_before=["earlier"], context_after=["later"],
                     speaker="NPC", placeholders=["[[0]]"], max_columns=32, kind="CHOICE",
                     rel_path="a/b.txt", line_no=4, span_start=1, span_end=9, command="say")
        pending = split.to_pending(source)
        assert pending["context_before"] == ["earlier"]
        assert pending["context_after"] == ["later"]
        assert pending["speaker"] == "NPC"
        assert pending["placeholders"] == ["[[0]]"]
        assert pending["max_columns"] == 32
        assert pending["kind"] == "CHOICE"
        assert (pending["rel_path"], pending["line_no"]) == ("a/b.txt", 4)

    def test_clears_the_translation(self, tmp_path):
        """A query that arrives carrying its answer would be resumed, not translated."""
        pending = split.to_pending(row("a", "line", target="already done"))
        assert pending["target"] is None
        assert pending["status"] == "pending"
        assert pending["notes"] == []

    def test_absent_optional_fields_get_neutral_defaults(self, tmp_path):
        pending = split.to_pending({"unit_id": "a", "source": "line"})
        assert pending["context_before"] == [] and pending["context_after"] == []
        assert pending["kind"] == "LINE" and pending["speaker"] is None


class TestSplitError:
    def test_carries_the_shortfall_numbers(self, tmp_path):
        """The predecessor script under-delivered silently; the numbers must survive."""
        error = split.SplitError("too few", path=tmp_path / "j.jsonl", wanted=40, available=24)
        assert error.wanted == 40 and error.available == 24
        assert "wanted 40, 24 available" in str(error)
        assert str(tmp_path / "j.jsonl") in str(error)

    def test_a_bare_reason_needs_no_numbers(self, tmp_path):
        error = split.SplitError("file not found", path=tmp_path / "j.jsonl")
        assert error.wanted is None and "wanted" not in str(error)


class TestBuildWithAPool:
    """A separate pool exists alongside the journal: the corpus keeps every translated row and
    queries come from elsewhere."""

    def test_corpus_keeps_every_row_and_queries_come_from_the_pool(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        pool = tmp_path / "p.jsonl"
        write_jsonl(journal, [row(f"j{i}", f"journal line {i}") for i in range(20)])
        write_jsonl(pool, [row(f"p{i}", f"pool line {i}", status="pending", target=None)
                           for i in range(20)])
        report = split.build(args_for(tmp_path, journal=journal, pool=pool, queries=5))
        assert report["corpus_entries"] == 20
        assert report["held_out_from_corpus"] is False
        queries = read_jsonl(tmp_path / "queries.jsonl")
        assert len(queries) == 5
        assert all(q["source"].startswith("pool line") for q in queries)

    def test_a_pool_row_already_in_the_corpus_is_not_asked(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        pool = tmp_path / "p.jsonl"
        write_jsonl(journal, [row(f"j{i}", f"shared line {i}") for i in range(20)])
        write_jsonl(pool, [row("dup", "shared line 0")]
                    + [row(f"p{i}", f"fresh line {i}") for i in range(10)])
        split.build(args_for(tmp_path, journal=journal, pool=pool, queries=5))
        sources = {q["source"] for q in read_jsonl(tmp_path / "queries.jsonl")}
        assert "shared line 0" not in sources


class TestBuildHeldOut:
    """The journal is all there is: queries are held out of it and the corpus shrinks."""

    def _journal(self, tmp_path, rows):
        path = tmp_path / "j.jsonl"
        write_jsonl(path, rows)
        return path

    def test_no_query_source_survives_in_the_corpus(self, tmp_path):
        journal = self._journal(tmp_path, [row(f"u{i}", f"line {i}") for i in range(60)])
        split.build(args_for(tmp_path, journal=journal, queries=10))
        corpus_sources = {r["source"] for r in read_jsonl(tmp_path / "corpus.jsonl")}
        for query in read_jsonl(tmp_path / "queries.jsonl"):
            assert query["source"] not in corpus_sources

    def test_the_whole_oversampled_holdout_leaves_the_corpus(self, tmp_path):
        """A returned row could re-introduce a chosen query's source verbatim and silently void
        its eligibility, so the entire holdout stays out -- not just the rows actually asked."""
        # Every line appears twice, so holding out exactly `queries` rows leaves each one's twin
        # behind: the loop must grow the holdout, and everything it grew by must stay out.
        rows = [row(f"u{i}", f"line {i // 2}") for i in range(60)]
        journal = self._journal(tmp_path, rows)
        report = split.build(args_for(tmp_path, journal=journal, queries=10))
        held_out = 60 - report["corpus_entries"]
        assert held_out > report["queries"], "the holdout was never oversampled"
        corpus_sources = {r["source"] for r in read_jsonl(tmp_path / "corpus.jsonl")}
        for query in read_jsonl(tmp_path / "queries.jsonl"):
            assert query["source"] not in corpus_sources

    def test_a_corpus_that_can_never_satisfy_the_request_raises_rather_than_spinning(
            self, tmp_path):
        """The loop grows the holdout until enough queries survive the not-already-in-the-corpus
        filter. On a corpus that repeats itself completely, no holdout ever suffices -- it must
        terminate at the full journal and report the shortfall, not loop forever."""
        rows = [row(f"u{i}", "the one and only line") for i in range(40)]
        journal = self._journal(tmp_path, rows)
        with pytest.raises(split.SplitError) as excinfo:
            split.build(args_for(tmp_path, journal=journal, queries=10))
        assert excinfo.value.wanted == 10
        assert excinfo.value.available < 10  # one distinct source, and it stays in the corpus

    def test_the_holdout_never_swallows_the_whole_journal(self, tmp_path):
        """Regression: 40 rows of 20 doubled sources with --queries 15 grew the holdout to the
        full journal, left `corpus_entries: 0`, and REPORTED SUCCESS -- every held-out row looked
        eligible only because there was no corpus left to duplicate it. The evaluation built on
        that split retrieves from nothing. A corpus must always keep at least one row."""
        rows = [row(f"u{i}", f"line {i // 2}") for i in range(40)]
        journal = self._journal(tmp_path, rows)
        report = split.build(args_for(tmp_path, journal=journal, queries=15))
        assert report["corpus_entries"] >= 1
        assert read_jsonl(tmp_path / "corpus.jsonl")

    def test_asking_for_the_whole_journal_is_refused_up_front(self, tmp_path):
        """Holding out every row cannot leave a corpus, whatever the sampling does."""
        journal = self._journal(tmp_path, [row(f"u{i}", f"line {i}") for i in range(10)])
        with pytest.raises(split.SplitError, match="no reference corpus") as excinfo:
            split.build(args_for(tmp_path, journal=journal, queries=10))
        assert (excinfo.value.wanted, excinfo.value.available) == (10, 9)

    def test_one_short_of_the_journal_leaves_exactly_one_corpus_row(self, tmp_path):
        """The boundary the guard is set at: the largest holdout that is still a split."""
        journal = self._journal(tmp_path, [row(f"u{i}", f"line {i}") for i in range(10)])
        report = split.build(args_for(tmp_path, journal=journal, queries=9))
        assert report["corpus_entries"] == 1

    def test_the_shortfall_is_reported_when_the_journal_is_smaller_than_the_request(
            self, tmp_path):
        journal = self._journal(tmp_path, [row(f"u{i}", f"line {i}") for i in range(3)])
        with pytest.raises(split.SplitError) as excinfo:
            split.build(args_for(tmp_path, journal=journal, queries=10))
        assert excinfo.value.wanted == 10

    def test_is_deterministic(self, tmp_path):
        """Same arguments, same split -- including the oversampling the loop had to do."""
        rows = [row(f"u{i}", f"line {i // 2}") for i in range(60)]
        journal = self._journal(tmp_path, rows)
        first_report = split.build(args_for(tmp_path, journal=journal, queries=10))
        first = (tmp_path / "queries.jsonl").read_text(encoding="utf-8")
        first_corpus = (tmp_path / "corpus.jsonl").read_text(encoding="utf-8")
        second_report = split.build(args_for(tmp_path, journal=journal, queries=10))
        assert (tmp_path / "queries.jsonl").read_text(encoding="utf-8") == first
        assert (tmp_path / "corpus.jsonl").read_text(encoding="utf-8") == first_corpus
        assert first_report == second_report

    def test_the_script_filter_reaches_the_held_out_path(self, tmp_path):
        rows = [row(f"j{i}", f"日本語の行 {i}") for i in range(30)]
        rows += [row(f"a{i}", f"ascii line {i}") for i in range(30)]
        journal = self._journal(tmp_path, rows)
        split.build(args_for(tmp_path, journal=journal, queries=5, script_pattern="japanese"))
        for query in read_jsonl(tmp_path / "queries.jsonl"):
            assert re.search(split.SCRIPT_PATTERNS["japanese"], query["source"])

    def test_an_invalid_script_pattern_raises(self, tmp_path):
        journal = self._journal(tmp_path, [row(f"u{i}", f"line {i}") for i in range(20)])
        with pytest.raises(split.SplitError, match="not a valid regex"):
            split.build(args_for(tmp_path, journal=journal, queries=2, script_pattern="[unclosed"))

    def test_the_report_describes_the_split(self, tmp_path):
        journal = self._journal(tmp_path, [row(f"u{i}", "x" * (i + 1)) for i in range(60)])
        report = split.build(args_for(tmp_path, journal=journal, queries=10, seed=11))
        assert report["queries"] == 10
        assert report["held_out_from_corpus"] is True
        assert report["seed"] == 11
        lengths = report["query_source_length"]
        assert lengths["min"] <= lengths["median"] <= lengths["max"]


class TestMain:
    def test_a_successful_run_reports_and_writes_the_json(self, tmp_path, capsys):
        journal = tmp_path / "j.jsonl"
        write_jsonl(journal, [row(f"u{i}", f"line {i}") for i in range(40)])
        code = split.main([
            "--journal", str(journal), "--corpus-out", str(tmp_path / "c.jsonl"),
            "--queries-out", str(tmp_path / "q.jsonl"), "--queries", "5",
            "--out", str(tmp_path / "report" / "split.json"),
        ])
        assert code == 0
        assert json.loads((tmp_path / "report" / "split.json").read_text())["queries"] == 5
        assert "held out of the journal" in capsys.readouterr().out

    def test_a_pool_run_does_not_claim_the_corpus_shrank(self, tmp_path, capsys):
        journal, pool = tmp_path / "j.jsonl", tmp_path / "p.jsonl"
        write_jsonl(journal, [row(f"j{i}", f"journal {i}") for i in range(20)])
        write_jsonl(pool, [row(f"p{i}", f"pool {i}") for i in range(20)])
        code = split.main(["--journal", str(journal), "--pool", str(pool),
                           "--corpus-out", str(tmp_path / "c.jsonl"),
                           "--queries-out", str(tmp_path / "q.jsonl"), "--queries", "5"])
        assert code == 0
        assert "held out of the journal" not in capsys.readouterr().out

    def test_a_split_error_exits_two_without_a_traceback(self, tmp_path, capsys):
        journal = tmp_path / "j.jsonl"
        write_jsonl(journal, [row(f"u{i}", "the one and only line") for i in range(20)])
        code = split.main(["--journal", str(journal), "--corpus-out", str(tmp_path / "c.jsonl"),
                           "--queries-out", str(tmp_path / "q.jsonl"), "--queries", "9"])
        assert code == 2
        assert "too few eligible queries" in capsys.readouterr().err

    def test_zero_queries_is_rejected_before_anything_is_read(self, tmp_path, capsys):
        """--queries 0 would divide by an empty length list when the report is built."""
        code = split.main(["--journal", str(tmp_path / "absent.jsonl"),
                           "--corpus-out", str(tmp_path / "c.jsonl"),
                           "--queries-out", str(tmp_path / "q.jsonl"), "--queries", "0"])
        assert code == 2
        assert "at least 1" in capsys.readouterr().err
