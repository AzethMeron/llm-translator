"""Tests for the unified context packer: selection (pack) and rendering (render_context) are
tested separately, matching their split responsibilities. Pure: no I/O, no server.
"""
from __future__ import annotations

import pytest

from transunit.reference import ReferenceEntry, Retrieved
from translator.context_packing import ContextPacker, render_context, trigram_similarity


def _packer(**kwargs) -> ContextPacker:
    return ContextPacker(**kwargs)


def _pack(packer, *, before=(), current="current line", after=(), references=(),
          stand_in="them", translate_of=None, similarity=trigram_similarity):
    return packer.pack(before=before, current=current, after=after, references=references,
                       stand_in=stand_in, translate_of=translate_of or (lambda s: None),
                       similarity=similarity)


class TestTrigramSimilarity:
    """The default similarity function pack() uses to diversify references."""

    def test_identical_text_scores_one(self) -> None:
        assert trigram_similarity("the door creaked", "the door creaked") == pytest.approx(1.0)

    def test_completely_different_text_scores_lower_than_similar_text(self) -> None:
        similar = trigram_similarity("the door creaked open", "the door creaked shut")
        different = trigram_similarity("the door creaked open", "xyz totally unrelated stuff")
        assert similar > different

    def test_case_insensitive(self) -> None:
        assert trigram_similarity("Hello World", "hello world") == pytest.approx(1.0)

    def test_empty_strings_score_zero(self) -> None:
        assert trigram_similarity("", "") == 0.0
        assert trigram_similarity("something", "") == 0.0

    def test_very_short_strings_are_handled_without_crashing(self) -> None:
        # Below trigram length (3 chars): falls back to whole-string comparison rather than an
        # empty (and therefore always-zero) shingle set.
        assert trigram_similarity("ab", "ab") == pytest.approx(1.0)
        assert trigram_similarity("a", "b") == 0.0


class TestPackSelection:
    def test_the_innermost_neighbour_on_each_side_is_always_guaranteed(self) -> None:
        # char_budget=1 is as tight as a budget can be without being disabled (0); only the
        # guaranteed innermost neighbour on each side must still appear.
        packer = _packer(char_budget=1, guaranteed_adjacent=1)
        packed = _pack(packer, before=("far before", "near before"), current="X",
                       after=("near after", "far after"))
        assert "near before" in packed.blob and "near after" in packed.blob
        assert "far before" not in packed.blob and "far after" not in packed.blob

    def test_guaranteed_adjacent_zero_makes_nothing_mandatory(self) -> None:
        # A budget of 1 is too tight for any neighbour to fit, and with guaranteed_adjacent=0
        # nothing is exempt from that -- so no neighbour survives on either side. The blob is
        # then empty (not "just the current line"): with zero real neighbours there is no
        # surrounding context to show, and agents.py's separate "Line to translate:" section
        # already shows the current line -- repeating it here under a "Surrounding context"
        # heading would be a dangling, redundant header, exactly what the no-null-fields rule
        # forbids.
        packer = _packer(char_budget=1, guaranteed_adjacent=0)
        packed = _pack(packer, before=("near before",), current="X", after=("near after",))
        assert "near before" not in packed.blob and "near after" not in packed.blob
        assert packed.blob == ""

    def test_a_neighbour_matching_a_reference_source_is_kept_only_as_the_neighbour(self) -> None:
        packer = _packer()
        ref = Retrieved(ReferenceEntry("near before", "translated near before"), 0.9)
        packed = _pack(packer, before=("near before",), current="X", references=(ref,))
        assert "near before" in packed.blob
        assert packed.references == ()  # the duplicate reference was dropped

    def test_dedup_compares_the_masked_form(self) -> None:
        # The neighbour and the reference source differ in their RAW placeholder index, but mask
        # to the same displayed text -- still a visible duplicate, still dropped.
        packer = _packer()
        ref = Retrieved(ReferenceEntry("[[1]] arrived", "target"), 0.9)
        packed = _pack(packer, before=("[[0]] arrived",), current="X", references=(ref,),
                       stand_in="Anna")
        assert packed.references == ()

    def test_budget_trims_optional_neighbours_but_never_the_guaranteed_one(self) -> None:
        packer = _packer(char_budget=len("innermost before") + len("X") + 2,
                         guaranteed_adjacent=1)
        packed = _pack(packer, before=("outer before", "innermost before"), current="X")
        assert "innermost before" in packed.blob
        assert "outer before" not in packed.blob

    def test_an_optional_neighbour_that_fits_the_budget_is_kept(self) -> None:
        # The mirror of the trim test: a budget generous enough for the OPTIONAL neighbour too
        # must include it, not just the guaranteed one.
        packer = _packer(char_budget=1000, guaranteed_adjacent=1)
        packed = _pack(packer, before=("outer before", "innermost before"), current="X",
                       after=("innermost after", "outer after"))
        assert all(text in packed.blob for text in
                   ("outer before", "innermost before", "innermost after", "outer after"))

    def test_a_reference_too_costly_for_the_remainder_is_skipped_but_a_cheaper_one_after_it_fits(
            self) -> None:
        # A budget that fits the guaranteed neighbour plus exactly one small reference, with a
        # large reference ranked ahead of a small one -- the large one must be skipped (not
        # abort the whole pass) and the small one after it still accepted.
        packer = _packer(char_budget=len("X") + 40, mmr_lambda=1.0, guaranteed_adjacent=0)
        big = Retrieved(ReferenceEntry("a" * 100, "b" * 100), 0.9)   # highest relevance, too big
        small = Retrieved(ReferenceEntry("tiny", "maly"), 0.5)        # lower relevance, fits
        packed = _pack(packer, current="X", references=(big, small))
        assert small in packed.references
        assert big not in packed.references

    def test_mmr_diversifies_references_under_a_tight_budget(self) -> None:
        # Two near-duplicate high-score references and one dissimilar lower-score one; a budget
        # tight enough for only one reference must, at a diversity-leaning lambda, still be free
        # to prefer variety over the second near-duplicate -- exercised here via a budget that
        # fits exactly one and a hand-crafted similarity forcing the dissimilar one to win.
        packer = _packer(char_budget=1000, mmr_lambda=0.1, guaranteed_adjacent=0)
        dup_a = Retrieved(ReferenceEntry("alpha", "target a"), 0.9)
        dup_b = Retrieved(ReferenceEntry("beta", "target b"), 0.85)
        distinct = Retrieved(ReferenceEntry("gamma", "target c"), 0.5)

        def similarity(a: str, b: str) -> float:
            return 0.99 if {a, b} == {"alpha", "beta"} else 0.0

        packed = _pack(packer, current="X", references=(dup_a, dup_b, distinct),
                       similarity=similarity)
        sources = [hit.entry.source for hit in packed.references]
        assert sources[0] == "alpha"        # highest raw relevance, no redundancy yet
        assert sources[1] == "gamma"        # beats 'beta' for being dissimilar to 'alpha'

    def test_empty_references_and_neighbours_produce_an_empty_pack(self) -> None:
        packer = _packer()
        packed = _pack(packer, current="X")
        assert packed.blob == "" and packed.established == () and packed.references == ()

    def test_before_units_and_after_units_zero_means_no_neighbours_at_all(self) -> None:
        # Simulates before_units=0/after_units=0 upstream: agents.py passes empty tuples, and
        # the ±1 guarantee has nothing to guarantee -- config wins, nothing is force-added.
        packer = _packer(char_budget=0, guaranteed_adjacent=1)
        packed = _pack(packer, before=(), current="X", after=())
        assert packed.blob == ""

    def test_established_lists_a_translation_for_every_shown_neighbour_that_has_one(self) -> None:
        packer = _packer()
        translations = {"near before": "translated before", "near after": "translated after"}
        packed = _pack(packer, before=("near before",), current="X", after=("near after",),
                       translate_of=translations.get)
        assert set(packed.established) == {("near before", "translated before"),
                                           ("near after", "translated after")}

    def test_established_omits_a_neighbour_with_no_known_translation(self) -> None:
        packer = _packer()
        packed = _pack(packer, before=("untranslated neighbour",), current="X")
        assert packed.established == ()

    def test_construction_validates_its_parameters(self) -> None:
        with pytest.raises(ValueError, match="char_budget"):
            _packer(char_budget=-1)
        with pytest.raises(ValueError, match="guaranteed_adjacent"):
            _packer(guaranteed_adjacent=-1)
        with pytest.raises(ValueError, match="mmr_lambda"):
            _packer(mmr_lambda=1.5)


class TestBlobContinuity:
    def test_a_sentence_split_across_a_neighbour_and_the_current_line_reads_continuously(
            self) -> None:
        packer = _packer()
        packed = _pack(packer, before=("The old door creaked as",), current="she pushed it open.")
        assert packed.blob == "The old door creaked as\nshe pushed it open."
        assert "SOURCE:" not in packed.blob and "TARGET:" not in packed.blob

    def test_neighbour_placeholders_are_masked_but_the_current_lines_own_are_not(self) -> None:
        packer = _packer()
        packed = _pack(packer, before=("[[0]] left the room.",), current="[[0]] followed.",
                       stand_in="Anna")
        before_line, current_line = packed.blob.split("\n")
        assert before_line == "Anna left the room."
        assert current_line == "[[0]] followed."


class TestRenderContext:
    def _render(self, packed, **kwargs):
        kwargs.setdefault("stand_in", "them")
        kwargs.setdefault("source_label", "EN")
        kwargs.setdefault("target_label", "PL")
        return render_context(packed, **kwargs)

    def test_an_all_empty_pack_renders_to_the_empty_string(self) -> None:
        packer = _packer()
        packed = _pack(packer, current="X")
        assert self._render(packed) == ""

    def test_reference_blob_and_established_sections_appear_in_that_order(self) -> None:
        packer = _packer()
        ref = Retrieved(ReferenceEntry("elsewhere", "gdzie indziej"), 0.9)
        packed = _pack(packer, before=("near before",), current="X",
                       references=(ref,), translate_of=lambda s: "translated")
        rendered = self._render(packed)
        ref_i = rendered.index("Reference translations")
        blob_i = rendered.index("Surrounding context")
        established_i = rendered.index("Already-translated")
        assert ref_i < blob_i < established_i

    def test_the_reference_section_appears_only_when_references_are_non_empty(self) -> None:
        packer = _packer()
        assert "Reference translations" not in self._render(_pack(packer, current="X"))
        ref = Retrieved(ReferenceEntry("elsewhere", "gdzie indziej"), 0.9)
        packed = _pack(packer, current="X", references=(ref,))
        assert "Reference translations" in self._render(packed)

    def test_the_blob_section_appears_only_when_there_is_a_neighbour(self) -> None:
        packer = _packer()
        assert "Surrounding context" not in self._render(_pack(packer, current="X"))
        packed = _pack(packer, before=("near before",), current="X")
        assert "Surrounding context" in self._render(packed)

    def test_the_established_section_appears_only_when_non_empty(self) -> None:
        packer = _packer()
        packed_untranslated = _pack(packer, before=("near before",), current="X")
        assert "Already-translated" not in self._render(packed_untranslated)
        packed_translated = _pack(packer, before=("near before",), current="X",
                                  translate_of=lambda s: "translated")
        assert "Already-translated" in self._render(packed_translated)

    def test_reference_examples_are_masked_for_display(self) -> None:
        packer = _packer()
        ref = Retrieved(ReferenceEntry("[[0]] left", "[[0]] wyszedł"), 0.9)
        packed = _pack(packer, current="X", references=(ref,))
        rendered = self._render(packed, stand_in="Anna")
        assert "[[0]]" not in rendered
        assert "Anna left" in rendered and "Anna wyszedł" in rendered

    def test_no_null_value_fields_anywhere(self) -> None:
        # The literal anti-pattern this design forbids: a section header stating an absent value
        # ("field: none") instead of being omitted outright.
        packer = _packer()
        cases = [
            _pack(packer, current="X"),
            _pack(packer, before=("near before",), current="X"),
            _pack(packer, current="X",
                 references=(Retrieved(ReferenceEntry("e", "t"), 0.9),)),
            _pack(packer, before=("near before",), current="X",
                 translate_of=lambda s: "translated"),
        ]
        for packed in cases:
            rendered = self._render(packed)
            assert "none" not in rendered.lower(), rendered
