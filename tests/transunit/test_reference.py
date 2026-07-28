"""The reference corpus and the lexical retriever.

Covers the data model and loaders, and the retriever's ranking, determinism, thresholds and
the two retrieval directions -- including the target-only case, which is the whole reason the
source field is optional. The retriever is exercised as a black box (similar-in ranks above
unrelated-in) plus the invariants a caller depends on: bounded results, a similarity floor, a
[0, 1] score, and byte-identical output across repeated and concurrent queries.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from transunit.reference import (
    GrowableLexicalRetriever,
    LexicalRetriever,
    ReferenceEntry,
    ReferenceError,
    Retrieved,
    Retriever,
    WritableRetriever,
    read_reference,
    reference_from_units,
)
from transunit.units import Status, Unit


# -- the entry ---------------------------------------------------------------

class TestReferenceEntry:
    def test_a_bilingual_entry_is_accepted(self) -> None:
        entry = ReferenceEntry(source="hello", target="cześć")
        assert entry.source == "hello" and entry.target == "cześć"

    def test_source_may_be_empty_for_a_target_only_entry(self) -> None:
        assert ReferenceEntry(source="", target="Zapisano.").source == ""

    @pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
    def test_an_empty_target_is_refused(self, bad: str) -> None:
        # A reference with no target is not a translation example and can steer nothing;
        # constructing one must be impossible rather than silently useless.
        with pytest.raises(ReferenceError):
            ReferenceEntry(source="hello", target=bad)


# -- loading -----------------------------------------------------------------

def write_jsonl(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestReadReference:
    def test_minimal_source_target_lines(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "ref.jsonl", [
            '{"source": "the door", "target": "drzwi"}',
            '{"source": "the window", "target": "okno"}',
        ])
        entries = read_reference(path)
        assert [(e.source, e.target) for e in entries] == [
            ("the door", "drzwi"), ("the window", "okno")]

    def test_target_only_line_is_kept_with_empty_source(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "ref.jsonl", ['{"target": "Zapisano grę."}'])
        entries = read_reference(path)
        assert entries == [ReferenceEntry(source="", target="Zapisano grę.")]

    def test_blank_lines_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "ref.jsonl"
        path.write_text('\n\n{"source": "a", "target": "b"}\n\n', encoding="utf-8")
        assert len(read_reference(path)) == 1

    def test_a_journal_line_with_injectable_status_is_used(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "j.jsonl", [
            '{"status": "verified", "source": "a", "target": "A"}',
            '{"status": "translated", "source": "b", "target": "B"}',
        ])
        assert [(e.source, e.target) for e in read_reference(path)] == [("a", "A"), ("b", "B")]

    @pytest.mark.parametrize("status", ["pending", "rejected", "skipped"])
    def test_a_non_injectable_journal_line_is_skipped_not_failed(
            self, tmp_path: Path, status: str) -> None:
        # A journal legitimately holds pending/rejected/skipped rows; pointing --reference at one
        # must drop those quietly, not treat them as corrupt or offer a known-bad target.
        path = write_jsonl(tmp_path / "j.jsonl", [
            f'{{"status": "{status}", "source": "a", "target": null}}',
            '{"status": "verified", "source": "b", "target": "B"}',
        ])
        assert [(e.source, e.target) for e in read_reference(path)] == [("b", "B")]

    def test_a_real_unit_json_line_round_trips(self, tmp_path: Path) -> None:
        unit = Unit(unit_id="u1", rel_path="f", line_no=1, span_start=0, span_end=1,
                    command="T", kind="L", source="the door", target="drzwi",
                    status=Status.VERIFIED)
        path = write_jsonl(tmp_path / "j.jsonl", [unit.to_json()])
        assert [(e.source, e.target) for e in read_reference(path)] == [("the door", "drzwi")]

    def test_a_missing_file_is_an_error_not_an_empty_corpus(self, tmp_path: Path) -> None:
        # Unlike the glossary (whose default path may legitimately be absent), --reference names
        # a file explicitly; a missing one is a mistake to report, not silently nothing.
        with pytest.raises(ReferenceError):
            read_reference(tmp_path / "nope.jsonl")

    def test_invalid_json_raises_with_location(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "bad.jsonl", ['{"source": "a", "target": "b"}', "not json"])
        with pytest.raises(ReferenceError) as info:
            read_reference(path)
        assert info.value.line_no == 2

    def test_a_non_object_line_raises(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "bad.jsonl", ['["a", "b"]'])
        with pytest.raises(ReferenceError):
            read_reference(path)

    @pytest.mark.parametrize("line", [
        '{"source": "a"}',                 # no target
        '{"source": "a", "target": ""}',   # empty target
        '{"source": "a", "target": "  "}', # whitespace target
        '{"source": "a", "target": 5}',    # non-string target
    ])
    def test_a_missing_or_empty_target_raises(self, tmp_path: Path, line: str) -> None:
        with pytest.raises(ReferenceError):
            read_reference(write_jsonl(tmp_path / "bad.jsonl", [line]))

    def test_a_non_string_source_raises(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "bad.jsonl", ['{"source": 5, "target": "b"}'])
        with pytest.raises(ReferenceError):
            read_reference(path)

    def test_an_unknown_status_raises(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "bad.jsonl", ['{"status": "weird", "target": "b"}'])
        with pytest.raises(ReferenceError):
            read_reference(path)


class TestReferenceFromUnits:
    def _unit(self, source: str, target: str | None, status: Status) -> Unit:
        return Unit(unit_id="u", rel_path="f", line_no=1, span_start=0, span_end=1,
                    command="T", kind="L", source=source, target=target, status=status)

    def test_only_injectable_units_contribute(self) -> None:
        units = [
            self._unit("a", "A", Status.VERIFIED),
            self._unit("b", "B", Status.TRANSLATED),
            self._unit("c", None, Status.REJECTED),
            self._unit("d", None, Status.PENDING),
        ]
        assert [(e.source, e.target) for e in reference_from_units(units)] == [
            ("a", "A"), ("b", "B")]


# -- retrieval ---------------------------------------------------------------

CORPUS = [
    ReferenceEntry("The dragon breathes fire.", "Smok zieje ogniem."),
    ReferenceEntry("The dragon breathes ice.", "Smok zieje lodem."),
    ReferenceEntry("A knight draws his sword.", "Rycerz dobywa miecza."),
    ReferenceEntry("", "Gra została zapisana."),  # target-only
]


class TestBySource:
    def test_a_more_similar_source_ranks_higher(self) -> None:
        retriever = LexicalRetriever(CORPUS, index_source=True)
        hits = retriever.by_source("The dragon breathes fire and smoke.", k=3)
        assert [h.entry.target for h in hits[:2]] == ["Smok zieje ogniem.", "Smok zieje lodem."]

    def test_an_exact_match_scores_one(self) -> None:
        retriever = LexicalRetriever(CORPUS, index_source=True)
        top = retriever.by_source("The dragon breathes fire.", k=1)[0]
        assert top.entry.target == "Smok zieje ogniem."
        assert top.score == pytest.approx(1.0)

    def test_scores_stay_within_the_unit_interval(self) -> None:
        retriever = LexicalRetriever(CORPUS, index_source=True)
        for hit in retriever.by_source("dragon fire ice knight sword", k=4, min_score=0.0):
            assert 0.0 <= hit.score <= 1.0

    def test_k_bounds_the_number_returned(self) -> None:
        retriever = LexicalRetriever(CORPUS, index_source=True)
        assert len(retriever.by_source("The dragon breathes fire.", k=1, min_score=0.0)) == 1

    def test_the_floor_drops_weak_matches(self) -> None:
        retriever = LexicalRetriever(CORPUS, index_source=True)
        assert retriever.by_source("Quantum chromodynamics seminar.", k=3, min_score=0.2) == ()

    def test_a_target_only_entry_is_never_returned_by_source(self) -> None:
        # It has no source to match, so source-keyed retrieval must ignore it entirely.
        retriever = LexicalRetriever(CORPUS, index_source=True)
        hits = retriever.by_source("Gra została zapisana.", k=4, min_score=0.0)
        assert all(hit.entry.source for hit in hits)

    def test_placeholders_do_not_dominate_similarity(self) -> None:
        entries = [ReferenceEntry("[[0]] opened the door.", "[[0]] otworzył drzwi.")]
        retriever = LexicalRetriever(entries, index_source=True)
        # The query shares the door wording but a *different* placeholder; it should still match
        # strongly, because placeholders are stripped before scoring.
        assert retriever.by_source("[[3]] opened the door.", k=1, min_score=0.0)[0].score > 0.5


class TestByTarget:
    def test_target_side_retrieval_covers_every_entry(self) -> None:
        retriever = LexicalRetriever(CORPUS, index_target=True)
        hit = retriever.by_target("Gra została zapisana teraz.", k=1, min_score=0.0)[0]
        assert hit.entry.target == "Gra została zapisana."  # the target-only entry is reachable

    def test_target_retrieval_ranks_by_target_similarity(self) -> None:
        retriever = LexicalRetriever(CORPUS, index_target=True)
        hits = retriever.by_target("Smok zieje ogniem i dymem.", k=2)
        assert hits[0].entry.target == "Smok zieje ogniem."


class TestDirectionGuards:
    def test_by_target_without_a_target_index_raises(self) -> None:
        retriever = LexicalRetriever(CORPUS, index_source=True, index_target=False)
        with pytest.raises(ReferenceError):
            retriever.by_target("anything", k=1)

    def test_by_source_without_a_source_index_raises(self) -> None:
        retriever = LexicalRetriever(CORPUS, index_source=False, index_target=True)
        with pytest.raises(ReferenceError):
            retriever.by_source("anything", k=1)


class TestDeterminism:
    def test_repeated_queries_are_identical(self) -> None:
        retriever = LexicalRetriever(CORPUS, index_source=True)
        first = retriever.by_source("The dragon breathes fire.", k=4, min_score=0.0)
        second = retriever.by_source("The dragon breathes fire.", k=4, min_score=0.0)
        assert first == second

    def test_ties_break_on_corpus_position(self) -> None:
        # Two identical sources must both be returned, in corpus order, never dict order.
        entries = [
            ReferenceEntry("same line", "first"),
            ReferenceEntry("same line", "second"),
        ]
        retriever = LexicalRetriever(entries, index_source=True)
        hits = retriever.by_source("same line", k=2, min_score=0.0)
        assert [h.entry.target for h in hits] == ["first", "second"]

    def test_concurrent_queries_match_the_sequential_result(self) -> None:
        # One retriever is shared by every worker in a run; its immutability must make that safe.
        retriever = LexicalRetriever(CORPUS, index_source=True)
        query = "The dragon breathes fire and ice."
        expected = retriever.by_source(query, k=4, min_score=0.0)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(
                lambda _: retriever.by_source(query, k=4, min_score=0.0), range(64)))
        assert all(result == expected for result in results)


class TestScriptAgnostic:
    def test_retrieval_works_for_a_non_space_delimited_script(self) -> None:
        # Character n-grams handle Chinese, where there are no word boundaries, like any script.
        entries = [
            ReferenceEntry("龙喷出火焰。", "The dragon breathes fire."),
            ReferenceEntry("骑士拔出剑。", "A knight draws his sword."),
        ]
        retriever = LexicalRetriever(entries, index_source=True)
        hit = retriever.by_source("龙喷出火焰和烟雾。", k=1, min_score=0.0)[0]
        assert hit.entry.target == "The dragon breathes fire."


class TestEdgeCases:
    def test_an_empty_corpus_returns_nothing(self) -> None:
        retriever = LexicalRetriever([], index_source=True, index_target=True)
        assert len(retriever) == 0
        assert retriever.by_source("anything", k=3) == ()
        assert retriever.by_target("anything", k=3) == ()

    def test_zero_k_returns_nothing(self) -> None:
        retriever = LexicalRetriever(CORPUS, index_source=True)
        assert retriever.by_source("The dragon breathes fire.", k=0) == ()

    def test_source_entries_counts_only_bilingual(self) -> None:
        retriever = LexicalRetriever(CORPUS, index_source=True)
        assert retriever.source_entries == 3  # one of the four is target-only

    def test_a_negative_ngram_is_refused(self) -> None:
        with pytest.raises(ValueError):
            LexicalRetriever(CORPUS, ngram=0)

    def test_retrieved_carries_entry_and_score(self) -> None:
        retriever = LexicalRetriever(CORPUS, index_source=True)
        hit = retriever.by_source("The dragon breathes fire.", k=1)[0]
        assert isinstance(hit, Retrieved)
        assert isinstance(hit.entry, ReferenceEntry)


class TestReferenceFromUnitsStatuses:
    def _unit(self, source, target, status):
        return Unit(unit_id="u", rel_path="f", line_no=1, span_start=0, span_end=1,
                    command="T", kind="L", source=source, target=target, status=status)

    def test_a_narrower_status_filter_is_honoured(self) -> None:
        units = [self._unit("a", "A", Status.VERIFIED),
                 self._unit("b", "B", Status.TRANSLATED)]
        only_verified = reference_from_units(units, statuses=(Status.VERIFIED,))
        assert [(e.source, e.target) for e in only_verified] == [("a", "A")]


GROW_SEED = [
    ReferenceEntry("A knight draws his sword.", "Rycerz dobywa miecza."),
    ReferenceEntry("", "Gra zapisana."),  # target-only
]


class TestGrowableRetriever:
    def test_it_is_a_writable_retriever(self) -> None:
        assert isinstance(GrowableLexicalRetriever(GROW_SEED), WritableRetriever)
        assert isinstance(GrowableLexicalRetriever(GROW_SEED), Retriever)

    def test_a_read_only_retriever_is_not_writable(self) -> None:
        assert not isinstance(LexicalRetriever(GROW_SEED), WritableRetriever)

    def test_a_seed_is_retrievable(self) -> None:
        retriever = GrowableLexicalRetriever(GROW_SEED, index_source=True)
        hit = retriever.by_source("A knight draws his blade and sword.", k=1, min_score=0.0)[0]
        assert hit.entry.target == "Rycerz dobywa miecza."

    def test_an_added_entry_becomes_retrievable(self) -> None:
        retriever = GrowableLexicalRetriever(GROW_SEED, index_source=True)
        retriever.add(ReferenceEntry("The dragon breathes fire.", "Smok zieje ogniem."))
        hit = retriever.by_source("The dragon breathes fire and smoke.", k=1, min_score=0.1)[0]
        assert hit.entry.target == "Smok zieje ogniem."

    def test_learning_from_an_empty_seed_cold_start(self) -> None:
        # No --reference: the store starts empty and learns entirely from added output.
        retriever = GrowableLexicalRetriever([], index_source=True)
        assert len(retriever) == 0
        retriever.add(ReferenceEntry("The dragon breathes fire.", "Smok zieje ogniem."))
        retriever.add(ReferenceEntry("A knight draws his sword.", "Rycerz dobywa miecza."))
        hits = retriever.by_source("The dragon breathes ice.", k=1, min_score=0.05)
        assert hits[0].entry.target == "Smok zieje ogniem."

    def test_added_entry_with_a_novel_shingle_is_still_found(self) -> None:
        # Frozen idf must still index shingles the seed never saw, or a learned entry hinging on
        # new vocabulary would be unreachable.
        retriever = GrowableLexicalRetriever([ReferenceEntry("alpha", "A")], index_source=True)
        retriever.add(ReferenceEntry("zzzqqq unheard vocabulary", "Z"))
        hit = retriever.by_source("zzzqqq unheard vocabulary", k=1, min_score=0.1)[0]
        assert hit.entry.target == "Z"

    def test_a_target_only_added_entry_is_reachable_by_target_only(self) -> None:
        retriever = GrowableLexicalRetriever([], index_source=True, index_target=True)
        retriever.add(ReferenceEntry("", "Zapisano stan gry."))
        assert retriever.by_source("Zapisano stan gry.", k=3, min_score=0.0) == ()
        assert retriever.by_target("Zapisano stan gry teraz.", k=1, min_score=0.1)[0].entry.target \
            == "Zapisano stan gry."

    def test_len_and_source_entries_track_additions(self) -> None:
        retriever = GrowableLexicalRetriever(GROW_SEED, index_source=True)
        assert (len(retriever), retriever.source_entries) == (2, 1)
        retriever.add(ReferenceEntry("A wizard casts a spell.", "Mag rzuca zaklęcie."))
        assert (len(retriever), retriever.source_entries) == (3, 2)

    def test_direction_guard_applies(self) -> None:
        retriever = GrowableLexicalRetriever(GROW_SEED, index_source=True, index_target=False)
        with pytest.raises(ReferenceError):
            retriever.by_target("anything", k=1)

    def test_indexing_no_direction_is_refused(self) -> None:
        with pytest.raises(ValueError):
            GrowableLexicalRetriever(GROW_SEED, index_source=False, index_target=False)

    def test_concurrent_add_and_query_stay_consistent(self) -> None:
        # Single-writer / multi-reader in the harness, but the store must tolerate the stronger
        # case of many concurrent writers and readers without corrupting its indexes.
        retriever = GrowableLexicalRetriever([], index_source=True)
        added = 300

        def writer(i: int) -> None:
            retriever.add(ReferenceEntry(f"line number {i} about dragons", f"target {i}"))

        def reader(_: int) -> None:
            retriever.by_source("line about dragons", k=3, min_score=0.0)

        with ThreadPoolExecutor(max_workers=8) as pool:
            tasks = [pool.submit(writer if n % 2 == 0 else reader, n) for n in range(added * 2)]
            for task in tasks:
                task.result()  # re-raise anything a thread hit

        assert len(retriever) == added
        # Every writer's entry is present and retrievable by its unique wording.
        hit = retriever.by_source("line number 42 about dragons", k=1, min_score=0.2)[0]
        assert hit.entry.target == "target 42"


class TestRetrieverEdgeBranches:
    def test_growable_refuses_a_bad_ngram(self) -> None:
        with pytest.raises(ValueError):
            GrowableLexicalRetriever(GROW_SEED, ngram=0)

    def test_growable_by_source_without_a_source_index_raises(self) -> None:
        retriever = GrowableLexicalRetriever(GROW_SEED, index_source=False, index_target=True)
        with pytest.raises(ReferenceError):
            retriever.by_source("anything", k=1)

    @pytest.mark.parametrize("blank", ["", "   ", "[[0]]"])
    def test_a_query_with_no_shingles_returns_nothing(self, blank: str) -> None:
        # An empty, whitespace-only, or placeholder-only query normalises to nothing to match on.
        retriever = LexicalRetriever(CORPUS, index_source=True)
        assert retriever.by_source(blank, k=3, min_score=0.0) == ()

    def test_a_one_character_entry_and_query_still_match(self) -> None:
        # The short-string shingle path (shorter than the n-gram) must still index and match.
        retriever = LexicalRetriever([ReferenceEntry("q", "Q")], index_source=True)
        assert retriever.by_source("q", k=1, min_score=0.0)[0].entry.target == "Q"
