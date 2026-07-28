"""Tests for :mod:`transunit.glossary`.

A glossary is established terminology as a source-to-target mapping. The behaviours that
matter are: an absent glossary is a legitimate empty state (not an error), a *malformed*
one is a corrupt-file error carrying its line number, and :func:`relevant_terms` matches
longest-first so a compound term wins over its constituents.
"""
from __future__ import annotations

import json

import pytest

from transunit.glossary import (
    GlossaryError,
    Term,
    read_glossary,
    relevant_terms,
    write_glossary,
)


class TestTermSerialisation:
    def test_to_json_carries_every_field(self) -> None:
        term = Term(source="Warszawa", target="Warsaw", category="name", entity_id=7)
        assert json.loads(term.to_json()) == {
            "source": "Warszawa",
            "target": "Warsaw",
            "category": "name",
            "entity_id": 7,
        }

    def test_to_json_preserves_non_ascii(self) -> None:
        """``ensure_ascii=False`` keeps the file human-readable; a term written as
        ``\\u...`` escapes would be unreadable to the person maintaining it."""
        assert "łódź" in Term(source="łódź", target="Lodz").to_json()


class TestRoundTrip:
    def test_write_then_read_recovers_the_terms(self, tmp_path) -> None:
        terms = [
            Term(source="New York", target="Nowy Jork", category="name"),
            Term(source="cat", target="kot"),
            Term(source="Alice", target="Alicja", category="character", entity_id=3),
        ]
        path = tmp_path / "glossary.jsonl"
        assert write_glossary(terms, path) == 3
        # Order is not part of the contract (relevant_terms re-sorts), so compare as sets.
        assert set(read_glossary(path)) == set(terms)

    def test_write_creates_missing_parent_directories(self, tmp_path) -> None:
        path = tmp_path / "nested" / "deep" / "glossary.jsonl"
        write_glossary([Term(source="a", target="b")], path)
        assert path.is_file()

    def test_write_orders_longest_source_first(self, tmp_path) -> None:
        """Longest-first on disk is a convenience for a human reader; verify it so the
        documented file shape does not silently drift."""
        path = tmp_path / "g.jsonl"
        write_glossary([Term(source="ab", target="x"), Term(source="abcd", target="y")], path)
        sources = [json.loads(line)["source"] for line in path.read_text().splitlines()]
        assert sources == ["abcd", "ab"]


class TestReadMissingAndInvalid:
    def test_missing_file_is_empty_not_an_error(self, tmp_path) -> None:
        """A project accumulates terminology; a source arriving without any is normal,
        so an absent glossary must read as no terms rather than forcing every caller to
        fabricate an empty file."""
        assert read_glossary(tmp_path / "absent.jsonl") == []

    def test_directory_in_place_of_file_raises(self, tmp_path) -> None:
        directory = tmp_path / "glossary.jsonl"
        directory.mkdir()
        with pytest.raises(GlossaryError) as excinfo:
            read_glossary(directory)
        assert excinfo.value.path == directory
        assert "not a regular file" in excinfo.value.reason

    def test_blank_lines_are_ignored(self, tmp_path) -> None:
        path = tmp_path / "g.jsonl"
        path.write_text('\n{"source": "a", "target": "b"}\n\n')
        assert read_glossary(path) == [Term(source="a", target="b")]

    def test_malformed_json_reports_line_number(self, tmp_path) -> None:
        path = tmp_path / "g.jsonl"
        path.write_text('{"source": "a", "target": "b"}\nnot json at all\n')
        with pytest.raises(GlossaryError) as excinfo:
            read_glossary(path)
        assert excinfo.value.path == path
        assert "line 2" in excinfo.value.reason

    def test_entry_missing_target_raises(self, tmp_path) -> None:
        """source and target are the irreducible minimum of a term; an entry lacking one
        is not a term at all, so it is rejected rather than silently loaded."""
        path = tmp_path / "g.jsonl"
        path.write_text('{"source": "a"}\n')
        with pytest.raises(GlossaryError) as excinfo:
            read_glossary(path)
        assert "'source' and 'target'" in excinfo.value.reason

    def test_entry_missing_source_raises(self, tmp_path) -> None:
        path = tmp_path / "g.jsonl"
        path.write_text('{"target": "b"}\n')
        with pytest.raises(GlossaryError):
            read_glossary(path)

    def test_unknown_field_raises_with_line_number(self, tmp_path) -> None:
        """A stray key means the file was written against a different schema; loading it
        anyway would drop the field the writer believed was in force."""
        path = tmp_path / "g.jsonl"
        path.write_text('{"source": "a", "target": "b", "bogus": 1}\n')
        with pytest.raises(GlossaryError) as excinfo:
            read_glossary(path)
        assert "line 1" in excinfo.value.reason

    def test_non_object_json_line_raises(self, tmp_path) -> None:
        path = tmp_path / "g.jsonl"
        path.write_text('["source", "target"]\n')
        with pytest.raises(GlossaryError):
            read_glossary(path)


class TestRelevantTerms:
    def test_longest_source_wins_over_constituent(self) -> None:
        """A compound term must be offered before its parts so the model sees the
        established rendering of the whole, not just the pieces."""
        terms = [Term(source="York", target="Jork"), Term(source="New York", target="Nowy Jork")]
        hits = relevant_terms("a trip to New York City", terms)
        assert [t.source for t in hits] == ["New York", "York"]

    def test_only_substring_matches_are_returned(self) -> None:
        terms = [
            Term(source="cat", target="kot"),
            Term(source="dog", target="pies"),
        ]
        hits = relevant_terms("the cat sat", terms)
        assert [t.source for t in hits] == ["cat"]

    def test_substring_match_is_deliberately_loose(self) -> None:
        """Over-matching is intentional: an inflected form still contains the lemma, so
        matching mid-word is preferred to missing a real occurrence."""
        hits = relevant_terms("kotem", [Term(source="kot", target="cat")])
        assert len(hits) == 1

    def test_empty_source_term_never_matches(self) -> None:
        """A term with an empty source would substring-match everything; it is filtered
        so it cannot flood every payload."""
        hits = relevant_terms("anything", [Term(source="", target="x")])
        assert hits == []

    def test_no_matches_yields_empty(self) -> None:
        assert relevant_terms("nothing here", [Term(source="zzz", target="q")]) == []

    def test_limit_caps_the_number_returned(self) -> None:
        terms = [Term(source=c * 3, target="t") for c in "abcdef"]
        text = "".join(t.source for t in terms)
        assert len(relevant_terms(text, terms, limit=2)) == 2

    def test_limit_keeps_the_longest(self) -> None:
        terms = [
            Term(source="a", target="1"),
            Term(source="abcd", target="2"),
            Term(source="ab", target="3"),
        ]
        hits = relevant_terms("abcd", terms, limit=1)
        assert [t.source for t in hits] == ["abcd"]
