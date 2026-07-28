"""Property-based tests for lexical reference retrieval and glossary term selection.

Both decide what a translation prompt is allowed to contain. Neither can fail loudly: a
retriever that scores badly returns the wrong examples, and a glossary selector that truncates
in the wrong direction drops the terminology a project spent a run establishing. The output is
a plausible list either way, so only a stated invariant can catch it.

The contracts pinned here are the ones the modules themselves publish -- ``Retrieved``'s "score
in [0, 1]", the ``Retriever`` protocol's "deterministic, safe to share across workers", and
``_NgramIndex``'s frozen-idf promise that a growing corpus does not rescore what is already in
it. Each test states the invariant it defends, not the implementation it happens to match.
"""
from __future__ import annotations

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from transunit.glossary import Term, relevant_terms
from transunit.reference import (
    GrowableLexicalRetriever,
    LexicalRetriever,
    ReferenceEntry,
    _normalize,
    _NgramIndex,
    _shingles,
)

# A mix of scripts, plus placeholders and whitespace runs, because _normalize exists to fold
# exactly those away and the padding in _shingles is what makes word edges count.
TEXT = st.text(
    alphabet=st.one_of(
        st.characters(min_codepoint=32, max_codepoint=0x2FF),
        st.sampled_from("あア亜漢한글日本語"),
        st.sampled_from(" \t\n[]0123456789"),
    ),
    max_size=40,
)
NON_BLANK = TEXT.filter(lambda s: bool(s.strip()))
ENTRY = st.builds(ReferenceEntry, source=TEXT, target=NON_BLANK)
CORPUS = st.lists(ENTRY, min_size=1, max_size=8,
                  unique_by=lambda e: (e.source, e.target))
NGRAM = st.integers(min_value=1, max_value=6)


def _sourced(corpus: list[ReferenceEntry]) -> list[ReferenceEntry]:
    """The entries a source-side index actually covers.

    Normalised content, not mere truthiness: a whitespace-only source normalises to "" and
    produces no shingles, so the index cannot cover it and no query can retrieve it. Treating it
    as "has a source" asserts a property the data cannot carry, which is a bug in the test rather
    than in the retriever.
    """
    return [e for e in corpus if _normalize(e.source)]


class TestNormalize:
    """The single point where two texts are made comparable. Every score downstream is a
    function of its output, so an unstable normalisation is an unstable retriever."""

    @given(TEXT)
    def test_is_idempotent(self, text: str) -> None:
        """Normalising an already-normalised text must be a no-op.

        Indexed keys and queries both go through it, but a caller could reasonably normalise
        once more (or feed back a stored key). If a second pass moved the text, an entry would
        stop matching its own source.
        """
        once = _normalize(text)
        assert _normalize(once) == once

    @given(TEXT)
    def test_leaves_no_placeholder_and_no_whitespace_run(self, text: str) -> None:
        """The two things it exists to remove: a placeholder every line shares would dominate
        similarity, and a whitespace run would make the same sentence two different keys."""
        normalized = _normalize(text)
        assert "[[0]]" not in normalized
        assert "  " not in normalized
        assert normalized == normalized.strip()

    @given(TEXT, st.integers(1, 20))
    def test_placeholder_numbering_does_not_change_the_normal_form(
            self, text: str, index: int) -> None:
        """Which slot an adapter happened to assign is not a semantic difference; two lines
        differing only in placeholder numbering must retrieve each other."""
        assert _normalize(f"a [[0]] b {text}") == _normalize(f"a [[{index}]] b {text}")


class TestShingles:
    """Character n-grams are what make the retriever script-agnostic. The two ends of the
    ngram range are where an off-by-one would live."""

    @given(TEXT, NGRAM)
    def test_never_raises_and_holds_only_ngram_length_pieces_of_the_padded_text(
            self, text: str, ngram: int) -> None:
        shingles = _shingles(text, ngram)
        padded = f" {text} "
        for shingle in shingles:
            assert shingle in padded
        if text and len(padded) > ngram:
            assert all(len(s) == ngram for s in shingles)

    @given(TEXT)
    def test_ngram_of_one_is_the_padded_character_set(self, text: str) -> None:
        """ngram=1 is the documented lower bound (the constructors reject 0), and it must
        degrade to a plain character-set comparison rather than an empty result."""
        assume(text)
        assert _shingles(text, 1) == set(f" {text} ")

    @given(TEXT, st.integers(1, 200))
    def test_an_ngram_longer_than_the_text_yields_the_whole_padded_text(
            self, text: str, ngram: int) -> None:
        """A short line against a long ngram must still be indexable -- a corpus is full of
        one-word lines, and returning nothing would make them permanently unretrievable
        instead of merely coarse."""
        assume(text)
        assume(len(text) + 2 <= ngram)
        assert _shingles(text, ngram) == {f" {text} "}

    def test_the_empty_text_has_no_shingles(self) -> None:
        assert _shingles("", 3) == set()


class TestLexicalRetrieverContract:
    """The published :class:`Retrieved` contract: "its similarity score in [0, 1]". Callers
    compare that score against an absolute ``min_score`` floor, so a score outside the range
    would make every configured floor mean something different."""

    @given(CORPUS, TEXT, NGRAM, st.integers(1, 10))
    def test_every_score_is_a_fraction(self, corpus: list[ReferenceEntry], query: str,
                                       ngram: int, k: int) -> None:
        retriever = LexicalRetriever(corpus, index_source=True, index_target=True, ngram=ngram)
        for hit in retriever.by_source(query, k=k):
            assert 0.0 <= hit.score <= 1.0
        for hit in retriever.by_target(query, k=k):
            assert 0.0 <= hit.score <= 1.0

    @given(CORPUS, TEXT, NGRAM, st.integers(1, 10), st.floats(0.0, 1.0))
    def test_results_are_best_first_capped_at_k_and_above_the_floor(
            self, corpus: list[ReferenceEntry], query: str, ngram: int, k: int,
            min_score: float) -> None:
        """The three promises the Retriever protocol makes to the engine, which trusts them
        without re-checking: at most k, each >= min_score, best first."""
        retriever = LexicalRetriever(corpus, ngram=ngram)
        hits = retriever.by_source(query, k=k, min_score=min_score)
        assert len(hits) <= k
        assert all(hit.score >= min_score for hit in hits)
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    @given(CORPUS, TEXT, NGRAM)
    def test_k_of_zero_or_less_retrieves_nothing(self, corpus: list[ReferenceEntry],
                                                 query: str, ngram: int) -> None:
        retriever = LexicalRetriever(corpus, ngram=ngram)
        assert retriever.by_source(query, k=0) == ()
        assert retriever.by_source(query, k=-3) == ()

    @given(CORPUS, TEXT, NGRAM)
    def test_is_deterministic(self, corpus: list[ReferenceEntry], query: str,
                              ngram: int) -> None:
        """The protocol requires it in writing: one retriever is shared by every worker in a
        run, and a query that answered differently on the second call would make a run's
        prompts -- and so its output -- irreproducible."""
        retriever = LexicalRetriever(corpus, index_source=True, index_target=True, ngram=ngram)
        assert retriever.by_source(query, k=5) == retriever.by_source(query, k=5)
        assert retriever.by_target(query, k=5) == retriever.by_target(query, k=5)

    @given(CORPUS, st.integers(0, 7), NGRAM)
    def test_an_exact_source_match_is_ranked_first(self, corpus: list[ReferenceEntry],
                                                   which: int, ngram: int) -> None:
        """Querying an entry's own source must return that entry at the top.

        The weakest thing a similarity retriever can be asked to do. If it fails, the score is
        not a similarity at all -- and the engine's most common query is a line that is nearly,
        or exactly, one it has seen.
        """
        sourced = _sourced(corpus)
        assume(sourced)
        target_entry = sourced[which % len(sourced)]
        # A cosine of 1.0 means "identical shingle set", which distinct texts can share
        # ("aa" and "aaa" at ngram=2). Rank-first is only well-defined when the match is
        # unique; a genuine tie is settled by corpus order by design.
        wanted = _shingles(_normalize(target_entry.source), ngram)
        assume(sum(1 for e in sourced
                   if _shingles(_normalize(e.source), ngram) == wanted) == 1)
        hits = LexicalRetriever(corpus, ngram=ngram).by_source(target_entry.source,
                                                              k=len(corpus))
        assert hits
        assert hits[0].entry == target_entry
        assert hits[0].score == pytest.approx(1.0)

    @given(CORPUS, TEXT, NGRAM, st.randoms(use_true_random=False))
    def test_scores_do_not_depend_on_the_order_the_corpus_was_loaded_in(
            self, corpus: list[ReferenceEntry], query: str, ngram: int,
            random: object) -> None:
        """Two runs handed the same reference file, shuffled, must retrieve the same things.

        idf is a corpus-wide statistic and norms are per-entry, so neither may depend on
        position. If they did, concatenating two reference corpora in a different order would
        quietly change every score -- the kind of drift that is impossible to attribute later.
        """
        shuffled = list(corpus)
        random.shuffle(shuffled)  # type: ignore[attr-defined]
        first = LexicalRetriever(corpus, ngram=ngram).by_source(query, k=len(corpus))
        second = LexicalRetriever(shuffled, ngram=ngram).by_source(query, k=len(corpus))
        assert (sorted((h.entry.source, h.entry.target, h.score) for h in first)
                == sorted((h.entry.source, h.entry.target, h.score) for h in second))

    @given(CORPUS, NGRAM)
    def test_an_unindexed_direction_raises_instead_of_returning_nothing(
            self, corpus: list[ReferenceEntry], ngram: int) -> None:
        """A misconfigured retriever must not look like an empty corpus; that reads as "no
        reference material matched" and would be tuned around rather than fixed."""
        from transunit.reference import ReferenceError
        source_only = LexicalRetriever(corpus, index_source=True, index_target=False,
                                       ngram=ngram)
        with pytest.raises(ReferenceError, match="without a target index"):
            source_only.by_target("x", k=1)
        target_only = LexicalRetriever(corpus, index_source=False, index_target=True,
                                       ngram=ngram)
        with pytest.raises(ReferenceError, match="without a source index"):
            target_only.by_source("x", k=1)

    @pytest.mark.parametrize("ngram", [0, -1])
    def test_a_degenerate_ngram_is_rejected_at_construction(self, ngram: int) -> None:
        with pytest.raises(ValueError, match="ngram must be >= 1"):
            LexicalRetriever([], ngram=ngram)
        with pytest.raises(ValueError, match="ngram must be >= 1"):
            GrowableLexicalRetriever([], ngram=ngram)

    @given(CORPUS, NGRAM)
    def test_an_empty_corpus_retrieves_nothing_rather_than_raising(
            self, corpus: list[ReferenceEntry], ngram: int) -> None:
        empty = LexicalRetriever([], index_source=True, index_target=True, ngram=ngram)
        assert empty.by_source("anything", k=5) == ()
        assert empty.by_target("anything", k=5) == ()
        assert len(empty) == 0


class TestGrowableLexicalRetriever:
    """The self-populating variant. Its efficiency claim rests on freezing idf at the seed;
    this pins the correctness consequence of that choice."""

    @given(CORPUS, CORPUS, TEXT, NGRAM)
    def test_adding_entries_never_rescores_an_entry_already_in_the_corpus(
            self, seed: list[ReferenceEntry], later: list[ReferenceEntry], query: str,
            ngram: int) -> None:
        """The frozen-idf guarantee, stated as behaviour: a seeded entry's score for a given
        query is the same before and after the corpus grows.

        This is what makes ``add`` O(the entry's shingles) *and* defensible. If idf drifted as
        the run learned, an entry's score -- and so whether it cleared ``min_score`` -- would
        depend on when in the run it was queried, which is a non-reproducibility far worse than
        the documented "what has been added before it" ordering effect.
        """
        retriever = GrowableLexicalRetriever(seed, index_source=True, index_target=True,
                                             ngram=ngram)
        before = {(h.entry.source, h.entry.target): h.score
                  for h in retriever.by_source(query, k=len(seed) + len(later) + 5)}
        for entry in later:
            retriever.add(entry)
        after = {(h.entry.source, h.entry.target): h.score
                 for h in retriever.by_source(query, k=len(seed) + len(later) + 5)}
        for key, score in before.items():
            assert key in after, f"{key} disappeared from the results after growth"
            assert after[key] == score

    @given(CORPUS, ENTRY, TEXT, NGRAM)
    def test_an_added_entry_is_immediately_retrievable_by_its_own_text(
            self, seed: list[ReferenceEntry], new: ReferenceEntry, query: str,
            ngram: int) -> None:
        """Adding must actually index, including shingles the seed never saw -- the case the
        frozen idf has to make room for."""
        # Normalisation, not truthiness: a whitespace-only source normalises to "" and yields no
        # shingles at all, so there is genuinely nothing to match on. Requiring it to be
        # retrievable would assert a property the data cannot carry, not a defect.
        assume(_normalize(new.source))
        retriever = GrowableLexicalRetriever(seed, ngram=ngram)
        retriever.add(new)
        hits = retriever.by_source(new.source, k=len(seed) + 1)
        assert any(h.entry == new for h in hits)

    @given(CORPUS, ENTRY, NGRAM)
    def test_a_target_only_entry_grows_the_target_index_but_not_the_source_index(
            self, seed: list[ReferenceEntry], new: ReferenceEntry, ngram: int) -> None:
        """The documented asymmetry: an entry without a source cannot be retrieved by source,
        and must not be allowed to desynchronise the two index-to-corpus id maps."""
        entry = ReferenceEntry(source="", target=new.target)
        retriever = GrowableLexicalRetriever(seed, index_source=True, index_target=True,
                                             ngram=ngram)
        before = retriever.source_entries
        retriever.add(entry)
        assert retriever.source_entries == before
        assert len(retriever) == len(seed) + 1
        assert any(h.entry == entry for h in retriever.by_target(entry.target, k=len(seed) + 1))

    def test_a_retriever_indexing_neither_direction_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one of source or target"):
            GrowableLexicalRetriever([], index_source=False, index_target=False)

    @given(CORPUS, TEXT, NGRAM)
    def test_scores_match_the_read_only_retriever_for_the_same_corpus(
            self, corpus: list[ReferenceEntry], query: str, ngram: int) -> None:
        """Single source of truth: the growable retriever must not be a second, subtly
        different scoring implementation. Seeded identically, it must answer identically."""
        read_only = LexicalRetriever(corpus, ngram=ngram).by_source(query, k=len(corpus))
        growable = GrowableLexicalRetriever(corpus, ngram=ngram).by_source(query,
                                                                          k=len(corpus))
        assert read_only == growable


class TestNgramIndexDirectly:
    """The scoring core, tested without a retriever wrapped around it."""

    @given(st.lists(TEXT, max_size=8), TEXT, NGRAM, st.integers(1, 10))
    def test_scores_are_fractions_and_positions_are_in_range(
            self, keys: list[str], query: str, ngram: int, k: int) -> None:
        index = _NgramIndex(keys, ngram)
        for position, score in index.query(query, k=k, min_score=0.0):
            assert 0 <= position < len(keys)
            assert 0.0 <= score <= 1.0

    @given(st.lists(TEXT, max_size=8), TEXT, NGRAM)
    def test_a_higher_floor_returns_a_subset(self, keys: list[str], query: str,
                                             ngram: int) -> None:
        """``min_score`` must only ever filter, never reorder or rescore -- callers raise it to
        tighten precision and would otherwise be changing the ranking as a side effect."""
        index = _NgramIndex(keys, ngram)
        loose = index.query(query, k=len(keys) or 1, min_score=0.0)
        strict = index.query(query, k=len(keys) or 1, min_score=0.5)
        assert strict == [pair for pair in loose if pair[1] >= 0.5]


class TestRelevantTerms:
    """Glossary selection: which established terms the prompt carries. Over-matching shows the
    model a term it did not need; under-matching silently drops one it did, and the model then
    invents a rendering that contradicts the project's own terminology."""

    TERMS = st.lists(
        st.builds(Term, source=TEXT, target=NON_BLANK),
        max_size=12, unique_by=lambda t: (t.source, t.target))

    @given(TEXT, TERMS, st.integers(0, 30))
    def test_never_returns_more_than_the_limit(self, text: str, terms: list[Term],
                                               limit: int) -> None:
        """The limit is a prompt-budget guard; exceeding it overflows the context window the
        caller sized for it."""
        assert len(relevant_terms(text, terms, limit=limit)) <= limit

    @given(TEXT, TERMS, st.integers(0, 30))
    def test_returns_only_terms_that_literally_occur_in_the_text(
            self, text: str, terms: list[Term], limit: int) -> None:
        for term in relevant_terms(text, terms, limit=limit):
            assert term.source and term.source in text

    @given(TEXT, TERMS, st.integers(0, 30))
    def test_is_ordered_longest_source_first(self, text: str, terms: list[Term],
                                             limit: int) -> None:
        """Longest-first is what makes a compound term beat its constituents, and -- because
        the list is truncated at the limit -- what decides which terms survive the cut. A
        different order would drop the most specific terms first, exactly backwards.
        """
        lengths = [len(t.source) for t in relevant_terms(text, terms, limit=limit)]
        assert lengths == sorted(lengths, reverse=True)

    @given(TEXT, TERMS)
    def test_a_limit_of_zero_returns_nothing(self, text: str, terms: list[Term]) -> None:
        assert relevant_terms(text, terms, limit=0) == []

    @given(TEXT, TERMS)
    def test_is_deterministic(self, text: str, terms: list[Term]) -> None:
        assert relevant_terms(text, terms) == relevant_terms(text, terms)

    @given(TEXT, TERMS, st.integers(0, 30))
    def test_the_truncation_keeps_the_longest_matches(self, text: str, terms: list[Term],
                                                      limit: int) -> None:
        """Every term dropped by the limit must be no longer than every term kept."""
        kept = relevant_terms(text, terms, limit=limit)
        all_hits = relevant_terms(text, terms, limit=len(terms) + 1)
        dropped = all_hits[len(kept):]
        if kept and dropped:
            assert len(kept[-1].source) >= len(dropped[0].source)

    @pytest.mark.xfail(strict=True,
                       reason="known defect, found by this property suite: relevant_terms "
                              "slices with hits[:limit] and never validates limit, so a "
                              "NEGATIVE limit is interpreted by Python as 'drop that many from "
                              "the END' -- i.e. it silently discards the SHORTEST matches and "
                              "returns the rest, rather than rejecting the call. Minimal "
                              "repro: relevant_terms('ab', [Term('ab','x'), Term('a','y')], "
                              "limit=-1) returns one term instead of raising. A negative limit "
                              "is always a caller bug (an arithmetic slip in a budget "
                              "calculation), and returning a plausible non-empty list hides it "
                              "completely -- the prompt silently carries a different set of "
                              "terms than the caller believes. Fix pending (do not fix in "
                              "src).")
    @pytest.mark.parametrize("limit", [-1, -5])
    def test_a_negative_limit_is_rejected(self, limit: int) -> None:
        with pytest.raises(ValueError):
            relevant_terms("ab", [Term(source="ab", target="x"),
                                  Term(source="a", target="y")], limit=limit)
