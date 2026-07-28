"""Adversarial, scale and failure-mode probing for the reference retriever.

Complements ``test_reference.py`` (which pins the contract) by pushing on the edges the
happy-path tests do not: a large corpus, pathological query lengths, unicode/emoji/RTL,
heavier concurrent learning, and the exact-boundary behaviour of the similarity floor. The
theme throughout is that a bad or extreme input yields a bounded, correct, *quiet* result --
never a crash and never silent corruption.
"""
from __future__ import annotations

import threading
import time

import pytest

from transunit.reference import (
    GrowableLexicalRetriever,
    LexicalRetriever,
    ReferenceEntry,
    ReferenceError,
    read_reference,
)


def _bi(source: str, target: str) -> ReferenceEntry:
    return ReferenceEntry(source=source, target=target)


# --- scale ------------------------------------------------------------------

class TestScale:
    def test_a_planted_match_is_found_among_twenty_thousand_entries(self) -> None:
        # The inverted index must find the needle without a linear scan mattering: build 20k
        # and confirm a planted near-duplicate ranks first, and that build+queries are quick.
        entries = [_bi(f"the quantity of item number {i} in the ledger", f"pozycja {i}")
                   for i in range(20_000)]
        entries.append(_bi("a wholly distinctive crimson zeppelin phrase", "sterowiec"))
        start = time.monotonic()
        retriever = LexicalRetriever(entries, index_source=True)
        hits = retriever.by_source("a wholly distinctive crimson zeppelin phrase", k=3,
                                   min_score=0.3)
        elapsed = time.monotonic() - start
        assert hits and hits[0].entry.target == "sterowiec"
        assert hits[0].score == pytest.approx(1.0)
        assert elapsed < 15.0  # generous smoke bound; the index makes this ~sub-second

    def test_k_far_larger_than_the_corpus_returns_at_most_the_corpus(self) -> None:
        retriever = LexicalRetriever([_bi("alpha", "a"), _bi("alpine", "b")], index_source=True)
        hits = retriever.by_source("alpha", k=10_000_000, min_score=0.0)
        assert 0 < len(hits) <= 2

    def test_a_very_long_query_is_handled(self) -> None:
        retriever = LexicalRetriever([_bi("repeat", "r")], index_source=True)
        hits = retriever.by_source("repeat " * 5_000, k=1, min_score=0.0)  # ~35k chars
        assert len(hits) == 1  # bounded, no crash


# --- unicode / scripts ------------------------------------------------------

class TestUnicodeRobustness:
    @pytest.mark.parametrize("text", [
        "I love 🍕🍕🍕 and ☕",                 # emoji (incl. a repeated one)
        "café naïve façade Zoë",                # combining/precomposed diacritics
        "مرحبا بالعالم هذا اختبار",             # Arabic RTL
        "混沌の魔法陣を展開する",                 # CJK, no spaces
        "🇵🇱🇯🇵 flags and ZWJ 👨‍👩‍👧",           # flag + ZWJ sequences (multi-codepoint)
    ])
    def test_retrieval_never_crashes_on_exotic_text(self, text: str) -> None:
        retriever = LexicalRetriever([_bi(text, "t")], index_source=True, index_target=True)
        assert retriever.by_source(text, k=1, min_score=0.0)[0].score == pytest.approx(1.0)
        assert retriever.by_target("t", k=1, min_score=0.0)

    def test_a_diacritic_variant_still_partially_matches(self) -> None:
        retriever = LexicalRetriever([_bi("café", "x")], index_source=True)
        # a query sharing most characters retrieves it; the point is no crash and a sane score
        hits = retriever.by_source("cafe", k=1, min_score=0.0)
        assert hits and 0.0 < hits[0].score <= 1.0


# --- similarity-floor boundary ----------------------------------------------

class TestFloorBoundary:
    def test_the_floor_is_inclusive_at_an_exact_match(self) -> None:
        retriever = LexicalRetriever([_bi("identical line", "x")], index_source=True)
        assert retriever.by_source("identical line", k=1, min_score=1.0)  # 1.0 >= 1.0

    def test_a_floor_of_one_excludes_a_merely_similar_line(self) -> None:
        retriever = LexicalRetriever([_bi("similar but not equal", "x")], index_source=True)
        assert retriever.by_source("similar and not equal", k=1, min_score=1.0) == ()

    def test_a_floor_above_every_score_returns_nothing_quietly(self) -> None:
        retriever = LexicalRetriever([_bi("abc", "x")], index_source=True)
        assert retriever.by_source("abc", k=1, min_score=1.0000001) == ()


# --- duplicates & degenerate entries ----------------------------------------

class TestDegenerateEntries:
    def test_duplicate_entries_are_both_returned_in_corpus_order(self) -> None:
        retriever = LexicalRetriever([_bi("same text", "first"), _bi("same text", "second")],
                                     index_source=True)
        hits = retriever.by_source("same text", k=2, min_score=0.5)
        assert [h.entry.target for h in hits] == ["first", "second"]  # tie broken by position
        assert hits[0].score == pytest.approx(hits[1].score)

    def test_a_placeholder_only_source_is_indexed_but_never_matches(self) -> None:
        # Its normalized form is empty (placeholders stripped), so it has no shingles and can be
        # returned by nothing -- but it must not break indexing or a query for other lines.
        retriever = LexicalRetriever([_bi("[[0]] [[1]]", "ph"), _bi("real content", "rc")],
                                     index_source=True)
        assert retriever.by_source("real content", k=5, min_score=0.1)[0].entry.target == "rc"
        assert retriever.by_source("[[0]]", k=5, min_score=0.0) == ()  # empty query, no shingles

    def test_a_target_that_is_only_placeholders_is_accepted_but_unmatchable(self) -> None:
        retriever = LexicalRetriever([_bi("src", "[[0]]")], index_source=True, index_target=True)
        assert len(retriever) == 1
        assert retriever.by_target("[[0]]", k=1, min_score=0.0) == ()


# --- failure modes: loading -------------------------------------------------

class TestLoadingFailureModes:
    def test_a_journal_line_injectable_but_missing_target_raises_with_location(
            self, tmp_path) -> None:
        path = tmp_path / "ref.jsonl"
        path.write_text('{"status": "verified", "source": "s"}\n', encoding="utf-8")
        with pytest.raises(ReferenceError) as info:
            read_reference(path)
        assert "target" in str(info.value) and "ref.jsonl:1" in str(info.value)

    def test_a_mixed_file_reports_the_first_bad_line_not_a_silent_drop(self, tmp_path) -> None:
        path = tmp_path / "ref.jsonl"
        path.write_text('{"source":"a","target":"x"}\n'      # good
                        '{"source":"b"}\n'                    # bad: no target
                        '{"source":"c","target":"y"}\n', encoding="utf-8")
        with pytest.raises(ReferenceError) as info:
            read_reference(path)
        assert ":2" in str(info.value)  # the exact offending line, not a swallowed skip

    def test_an_entry_constructed_with_a_whitespace_target_is_refused(self) -> None:
        with pytest.raises(ReferenceError):
            ReferenceEntry(source="s", target="   \t ")


# --- growable: frozen-idf & heavier concurrency -----------------------------

class TestGrowableAdversarial:
    def test_a_tiny_seed_then_many_novel_additions_stays_findable(self) -> None:
        retriever = GrowableLexicalRetriever([_bi("seed line", "s")], index_source=True)
        for i in range(1_000):
            retriever.add(_bi(f"novel vocabulary entry number {i} zzz{i}", f"t{i}"))
        hits = retriever.by_source("novel vocabulary entry number 777 zzz777", k=1,
                                   min_score=0.2)
        assert hits and hits[0].entry.target == "t777"
        assert len(retriever) == 1_001

    def test_learning_is_order_dependent_but_reproducible_for_one_order(self) -> None:
        def build():
            r = GrowableLexicalRetriever([_bi("base", "b")], index_source=True)
            for word in ("alpha match", "beta match", "gamma match"):
                r.add(_bi(word, word.split()[0]))
            return tuple((h.entry.target, round(h.score, 6))
                         for h in r.by_source("beta match", k=3, min_score=0.0))
        assert build() == build()  # same insertion order -> identical result

    def test_heavy_concurrent_add_and_query_stays_consistent(self) -> None:
        retriever = GrowableLexicalRetriever([_bi("anchor phrase", "anchor")],
                                             index_source=True, index_target=True)
        errors: list[Exception] = []
        threads_count, per_thread = 8, 200

        def worker(base: int) -> None:
            try:
                for i in range(per_thread):
                    retriever.add(_bi(f"worker {base} item {i}", f"w{base}_{i}"))
                    # the anchor must remain retrievable throughout the mutation
                    assert retriever.by_source("anchor phrase", k=1, min_score=0.3)
                    retriever.by_target(f"w{base}_{i}", k=2, min_score=0.0)
            except Exception as exc:  # noqa: BLE001 -- surface any thread failure to the test
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(b,)) for b in range(threads_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(retriever) == 1 + threads_count * per_thread
        # every entry landed exactly once and remains retrievable
        assert retriever.by_source("worker 3 item 42", k=1, min_score=0.4)[0].entry.target \
            == "w3_42"
