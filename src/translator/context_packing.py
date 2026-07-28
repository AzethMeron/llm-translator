"""Unified positional-context + reference-example packing, rendered as one continuous passage.

Two responsibilities, kept separate per the project's single-responsibility rule:

* **Selection** (:meth:`ContextPacker.pack`) -- which neighbouring lines and retrieved reference
  examples survive the ±1 discourse guarantee, deduplication and (if configured) a character
  budget. Pure: no I/O, no globals; a memory lookup and a text-similarity function are injected.
* **Rendering** (:func:`render_context`) -- how the survivors become prompt text. This is where
  the continuous-blob format lives: previous, current and next lines are joined as one passage
  rather than shown as separate bulleted "preceding"/"following" sections, because a game line is
  frequently a fragment -- a sentence, clause, or idiom split across unit boundaries -- and a
  model reads a continuous passage the way it would read the source, resolving a cross-boundary
  idiom correctly, where individually labelled snippets read as unrelated thoughts. The exact line
  being translated is repeated verbatim by the caller immediately after (see ``agents.py``), so
  however continuous the passage reads, there is never doubt about the translation target.

**No null-value fields.** Every section :func:`render_context` can produce is included only when
it has real content -- there is no "field: none" filler anywhere in its output. An empty blob, an
empty established-translations list, or an empty reference list each simply omit their heading.

Positional neighbours are never MMR-reordered or -dropped by content similarity: doing so would
fragment the very continuity the blob exists to preserve. A character budget instead trims from
the *outside in* -- the farthest neighbour on each side goes first -- so the guaranteed innermost
neighbour and everything closer to it survive any budget. Reference examples have no such
adjacency constraint (they are independent lines pulled from elsewhere in the corpus), so *they*
are where :func:`~translator.retrieval.fusion.mmr_order` earns its keep: avoiding a shortlist
dominated by several near-identical examples.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from transunit.reference import Retrieved

from .retrieval.fusion import mmr_order

_PLACEHOLDER = re.compile(r"\[\[\d+\]\]")


def _mask(text: str, stand_in: str) -> str:
    """Replace engine placeholders with a readable stand-in, exactly as neighbour context has
    always been shown: a placeholder in a *neighbour* line refers to that line's own lookup, not
    this unit's, so left raw it teaches the model to emit one on a line that has none."""
    return _PLACEHOLDER.sub(stand_in, text)


def _trigrams(text: str) -> set[str]:
    normalized = text.casefold().strip()
    if len(normalized) < 3:
        return {normalized} if normalized else set()
    return {normalized[i:i + 3] for i in range(len(normalized) - 2)}


def trigram_similarity(a: str, b: str) -> float:
    """Jaccard similarity over character trigrams: a stdlib, embedding-free notion of how alike
    two lines look, used to diversify packed reference examples without an embedding round-trip
    for every unit translated (the retriever that *found* them may already have embedded them;
    this does not need to)."""
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass(frozen=True, slots=True)
class PackedContext:
    """What survived selection, ready for :func:`render_context`."""

    blob: str
    """The continuous surrounding-context passage: masked optional-and-guaranteed ``before``
    lines, then the unmasked current line, then masked optional-and-guaranteed ``after`` lines,
    newline-joined. Already masked -- unlike ``established``/``references`` below -- because it
    is flattened into one string here, mixing masked neighbour text with the current line's own
    (deliberately unmasked) placeholders; there is no way to selectively re-mask part of a joined
    string later. Empty when there are no neighbours to show at all (render_context then omits
    the whole surrounding-context section, per the module's no-null-fields rule)."""
    established: tuple[tuple[str, str], ...]
    """Raw ``(neighbour_source, neighbour_target)`` pairs for every neighbour shown in the blob
    that already has a translation. Not yet masked -- render_context does that, along with the
    reference examples below, so all display-time formatting lives in one place."""
    references: tuple[Retrieved, ...]
    """Reference examples that survived deduplication against the neighbours, MMR diversification
    and the character budget. Raw ``Retrieved`` hits, unmasked, in final display order."""


class ContextPacker:
    """Selects what a unit's positional neighbours and retrieved reference examples contribute
    to the prompt. See the module docstring for the design rationale."""

    def __init__(self, *, char_budget: int = 0, mmr_lambda: float = 0.7,
                 guaranteed_adjacent: int = 1) -> None:
        if char_budget < 0:
            raise ValueError(f"char_budget must be >= 0, got {char_budget}")
        if guaranteed_adjacent < 0:
            raise ValueError(f"guaranteed_adjacent must be >= 0, got {guaranteed_adjacent}")
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError(f"mmr_lambda must be in [0, 1], got {mmr_lambda}")
        self._char_budget = char_budget
        self._mmr_lambda = mmr_lambda
        self._guaranteed_adjacent = guaranteed_adjacent

    def pack(self, *, before: Sequence[str], current: str, after: Sequence[str],
             references: Sequence[Retrieved], stand_in: str,
             translate_of: Callable[[str], str | None],
             similarity: Callable[[str, str], float]) -> PackedContext:
        guaranteed_before, optional_before = self._split(before, from_end=True)
        guaranteed_after, optional_after = self._split(after, from_end=False)

        # Dedup on the MASKED form -- what will actually be shown -- so a reference example that
        # would look identical to a neighbour already in the blob is dropped as the redundancy it
        # visually is, even on the rare case its raw (unmasked) text differs from the neighbour's.
        shown_masked = {_mask(line, stand_in) for line in
                        (*guaranteed_before, *optional_before, *guaranteed_after, *optional_after)}
        deduped = tuple(hit for hit in references
                        if _mask(hit.entry.source, stand_in) not in shown_masked)

        if self._char_budget <= 0:
            # Disabled: reproduce the pre-budget behaviour exactly -- everything before_units /
            # after_units / reference_examples already selected is shown, nothing further trimmed.
            kept_before, kept_after, kept_references = optional_before, optional_after, deduped
        else:
            used = sum(len(_mask(line, stand_in))
                      for line in (*guaranteed_before, *guaranteed_after))
            kept_before, used = self._fit_neighbours(optional_before, from_end=True, used=used,
                                                      stand_in=stand_in)
            kept_after, used = self._fit_neighbours(optional_after, from_end=False, used=used,
                                                     stand_in=stand_in)
            kept_references, used = self._fit_references(deduped, stand_in, similarity, used=used)

        full_before = (*kept_before, *guaranteed_before)
        full_after = (*guaranteed_after, *kept_after)
        blob = ("\n".join((*(_mask(line, stand_in) for line in full_before), current,
                          *(_mask(line, stand_in) for line in full_after)))
               if (full_before or full_after) else "")
        established = tuple((line, target) for line in (*full_before, *full_after)
                            if (target := translate_of(line)) is not None)
        return PackedContext(blob=blob, established=established, references=kept_references)

    def _split(self, lines: Sequence[str],
              *, from_end: bool) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Split into (guaranteed innermost, optional). ``from_end`` selects which end is
        "innermost" -- the end of ``before`` is closest to the current line; the start of
        ``after`` is."""
        n = self._guaranteed_adjacent
        if n <= 0 or not lines:
            return (), tuple(lines)
        if from_end:
            cut = max(0, len(lines) - n)
            return tuple(lines[cut:]), tuple(lines[:cut])
        return tuple(lines[:n]), tuple(lines[n:])

    def _fit_neighbours(self, optional: Sequence[str], *, from_end: bool, used: int,
                        stand_in: str) -> tuple[tuple[str, ...], int]:
        """Greedily accept optional neighbours closest-to-current first, so a tight budget trims
        the farthest lines, preserving contiguity around the guaranteed core. Never MMR-reordered:
        skipping an interior line would fragment the continuous passage the blob exists to give."""
        ordered = list(reversed(optional)) if from_end else list(optional)
        kept: list[str] = []
        for line in ordered:
            cost = len(_mask(line, stand_in))  # masked (displayed) length, like every other cost
            if used + cost > self._char_budget:
                break
            kept.append(line)
            used += cost
        if from_end:
            kept.reverse()  # restore chronological (oldest-of-kept-first) order
        return tuple(kept), used

    def _fit_references(self, deduped: tuple[Retrieved, ...], stand_in: str,
                        similarity: Callable[[str, str], float],
                        *, used: int) -> tuple[tuple[Retrieved, ...], int]:
        """MMR-diversify the deduplicated references, then greedily accept them in that order
        while the budget allows -- unlike neighbours, references have no adjacency to preserve,
        so diversity (not proximity) is the right tie-breaker among them."""
        if not deduped:
            return (), used
        relevance = {hit: hit.score for hit in deduped}

        def entry_similarity(a: Retrieved, b: Retrieved) -> float:
            return similarity(a.entry.source, b.entry.source)

        ordered = mmr_order(list(deduped), relevance, entry_similarity,
                            lambda_=self._mmr_lambda, k=len(deduped))
        kept: list[Retrieved] = []
        for hit in ordered:
            cost = len(_mask(hit.entry.source, stand_in)) + len(_mask(hit.entry.target, stand_in))
            if used + cost > self._char_budget:
                continue  # a later, cheaper reference may still fit
            kept.append(hit)
            used += cost
        return tuple(kept), used


def render_context(packed: PackedContext, *, stand_in: str, source_label: str,
                   target_label: str) -> str:
    """Turn a :class:`PackedContext` into the prompt text a caller appends -- the sole place that
    masks reference/established text for display and formats every section. Sections that have no
    content contribute nothing: no "field: none", no dangling heading."""
    parts: list[str] = []
    if packed.references:
        rendered = "\n".join(
            f"  {source_label}: {_mask(hit.entry.source, stand_in)}\n"
            f"  {target_label}: {_mask(hit.entry.target, stand_in)}"
            for hit in packed.references)
        parts.append(
            "Reference translations of similar lines, from existing translated material. Match "
            "their choices where they apply -- terminology, phrasing, register. They are examples "
            "to follow, not rules, and may not fit this line exactly:\n" + rendered)
    if packed.blob:
        parts.append(
            "Surrounding context (previous and next lines). It may contain partial sentences, "
            "clauses, or idioms that continue across lines -- use it only to understand meaning "
            "and continuity. Do not translate this context; the exact line to translate is given "
            "again below.\n" + packed.blob)
    if packed.established:
        rendered = "\n".join(
            f"  {source_label}: {_mask(source, stand_in)}\n"
            f"  {target_label}: {_mask(target, stand_in)}"
            for source, target in packed.established)
        parts.append("Already-translated neighbouring lines:\n" + rendered)
    return "\n\n".join(parts)
