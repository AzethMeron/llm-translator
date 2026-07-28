"""Tests for :mod:`transunit.placeholders`.

The placeholder convention is what lets one translator serve carriers with unrelated
syntaxes: the translator only has to preserve the *set* of ``[[n]]`` markers. These tests
cover the extraction that a preservation check is built on -- order, repeats, and
multi-digit indices -- and confirm near-misses are not mistaken for placeholders.
"""
from __future__ import annotations

from transunit.placeholders import placeholder_indices


class TestPlaceholderIndices:
    def test_returns_indices_in_order_of_appearance(self) -> None:
        assert placeholder_indices("first [[2]] then [[0]] then [[1]]") == [2, 0, 1]

    def test_no_placeholders_yields_empty_list(self) -> None:
        assert placeholder_indices("plain text with no markers") == []

    def test_empty_string_yields_empty_list(self) -> None:
        assert placeholder_indices("") == []

    def test_repeated_index_is_reported_each_time(self) -> None:
        """A repeat is not deduplicated: a preservation check compares multisets, so a
        translation that dropped one of two identical markers must be detectable."""
        assert placeholder_indices("[[0]] and again [[0]]") == [0, 0]

    def test_multi_digit_index_is_parsed_whole(self) -> None:
        assert placeholder_indices("[[12]] [[7]] [[100]]") == [12, 7, 100]

    def test_single_brackets_are_not_placeholders(self) -> None:
        """Only the doubled-bracket form is a placeholder; a single ``[0]`` is ordinary
        text a carrier may legitimately contain."""
        assert placeholder_indices("[0] (1) {2}") == []

    def test_non_digit_contents_are_not_placeholders(self) -> None:
        assert placeholder_indices("[[a]] [[x1]] [[ ]]") == []

    def test_adjacent_placeholders(self) -> None:
        assert placeholder_indices("[[0]][[1]][[2]]") == [0, 1, 2]
