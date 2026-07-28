"""External reference translations, retrieved per unit.

A *reference corpus* is already-translated material that did not come from the current run: a
previous run, a sister project, a vendor or human reference, or the already-translated half of
the same work. It is kept strictly separate from the run's own output (:class:`translator.
memory.TranslationMemory`): that store does dedup and seeds neighbour context, while this one
is read-only guidance retrieved into the prompt so a new translation matches an existing body
of work.

Because a corpus is far larger than any prompt, the engine does not show all of it; it
retrieves the entries *relevant to the current unit* and shows only those. Retrieval has two
directions, because a reference entry need not have a source:

* **by source** -- match the unit's source against reference sources, to surface prior
  renderings of similar lines before translating (the primary use);
* **by target** -- match a draft translation against reference targets, to surface established
  target-language renderings during revision. This is how material translated *without* its
  original still helps: it cannot seed a translation, but it can pull a draft toward the house
  voice.

The default retriever (:class:`LexicalRetriever`) is stdlib-only, deterministic, and safe to
share across the concurrent workers of a run. A stronger embedding retriever can be dropped in
behind the :class:`Retriever` interface without touching the engine; none is provided here,
because it would need a model or an embedding endpoint to run.
"""
from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .units import Status


class ReferenceError(Exception):
    """A reference corpus could not be read or parsed, or a retriever was misused."""

    def __init__(self, reason: str, *, path: Path | None = None,
                 line_no: int | None = None) -> None:
        location = f"{path}:{line_no}" if path and line_no else (str(path) if path else "")
        super().__init__(f"{location}: {reason}" if location else reason)
        self.reason = reason
        self.path = path
        self.line_no = line_no


@dataclass(frozen=True, slots=True)
class ReferenceEntry:
    """One reference translation.

    ``target`` is required -- an entry with no target is not a translation example and can steer
    nothing. ``source`` is optional: an empty source marks a *target-only* entry, usable only by
    target-side retrieval (see the module docstring).
    """

    source: str
    target: str

    def __post_init__(self) -> None:
        if not self.target or not self.target.strip():
            raise ReferenceError("a reference entry needs a non-empty target")


@dataclass(frozen=True, slots=True)
class Retrieved:
    """A reference entry a query matched, with its similarity score in ``[0, 1]``."""

    entry: ReferenceEntry
    score: float


@runtime_checkable
class Retriever(Protocol):
    """The seam the engine depends on, so a stronger retriever can replace the default.

    Both methods return at most ``k`` matches, each with score ``>= min_score``, best first, and
    are required to be **deterministic** (identical corpus and query -> identical result) and
    safe to call concurrently, because one retriever is shared by every worker in a run.
    """

    def by_source(self, query: str, *, k: int,
                  min_score: float = 0.0) -> tuple[Retrieved, ...]:
        """Entries whose *source* is most similar to ``query``."""
        ...

    def by_target(self, query: str, *, k: int,
                  min_score: float = 0.0) -> tuple[Retrieved, ...]:
        """Entries whose *target* is most similar to ``query``."""
        ...


@runtime_checkable
class WritableRetriever(Retriever, Protocol):
    """A retriever that also accepts new entries during a run (a self-populating memory).

    The engine tests for this at runtime, so a read-only retriever is simply never written to.
    An implementation must make :meth:`add` safe to call from one thread while others query --
    the runner records accepted results from its own thread while workers retrieve.
    """

    def add(self, entry: ReferenceEntry) -> None:
        """Add one accepted translation, so later queries can retrieve it."""
        ...


# -- loading -----------------------------------------------------------------

def read_reference(path: Path) -> list[ReferenceEntry]:
    """Load a reference corpus from JSON Lines.

    Each line is one of two shapes, and the two may be mixed in one file:

    * a minimal reference entry -- ``{"source": "...", "target": "..."}``; omit ``source`` for a
      target-only entry;
    * a ``transunit`` unit/journal record -- the same shape the engine writes, so a prior run's
      ``journal.jsonl`` is a corpus as-is. A record carrying a non-injectable ``status``
      (pending or rejected -- no usable target) is skipped, not an error, because that is the
      expected content of a journal.

    A genuinely malformed line (bad JSON, a non-object, a missing or empty target where one is
    required) raises :class:`ReferenceError` with its location rather than being dropped: a
    silently skipped entry is a silently missing reference.
    """
    if not path.is_file():
        raise ReferenceError("reference corpus not found", path=path)
    entries: list[ReferenceEntry] = []
    for line_no, raw in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        entry = _parse_record(raw, path=path, line_no=line_no)
        if entry is not None:
            entries.append(entry)
    return entries


def _parse_record(line: str, *, path: Path, line_no: int) -> ReferenceEntry | None:
    """Parse one JSONL record into an entry, or ``None`` when it is a skippable journal line."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ReferenceError(f"invalid JSON: {exc}", path=path, line_no=line_no) from exc
    if not isinstance(record, dict):
        raise ReferenceError("a reference record must be a JSON object", path=path,
                             line_no=line_no)
    if "status" in record:
        # A unit/journal record. Only an injectable one carries a usable target; skip the rest,
        # which is what a pending/rejected line is.
        try:
            status = Status(record["status"])
        except ValueError as exc:
            raise ReferenceError(f"unknown status {record['status']!r}: {exc}", path=path,
                                 line_no=line_no) from exc
        if not status.is_injectable:
            return None

    target = record.get("target")
    if not isinstance(target, str) or not target.strip():
        raise ReferenceError("a reference entry needs a non-empty string 'target'", path=path,
                             line_no=line_no)
    source = record.get("source", "")
    if not isinstance(source, str):
        raise ReferenceError(
            f"'source' must be a string, got {type(source).__name__}", path=path,
            line_no=line_no)
    return ReferenceEntry(source=source, target=target)


def reference_from_units(units: Iterable[object],
                         statuses: Iterable[Status] = (Status.VERIFIED, Status.TRANSLATED),
                         ) -> list[ReferenceEntry]:
    """Build a corpus from in-memory ``transunit`` units (e.g. a loaded journal).

    Only units whose status is in ``statuses`` *and* that carry a target contribute. The default
    is the injectable pair (verified and translated), mirroring
    :meth:`translator.memory.TranslationMemory.from_units`: a rejected unit's text failed its
    checks, so offering it as a reference would spread a known-bad rendering. A caller that wants
    to seed only fully-verified material passes ``statuses=(Status.VERIFIED,)``. Typed loosely to
    avoid importing :class:`~transunit.units.Unit` here; it only reads ``status``, ``source`` and
    ``target``.
    """
    wanted = frozenset(statuses)
    entries: list[ReferenceEntry] = []
    for unit in units:
        status = getattr(unit, "status", None)
        target = getattr(unit, "target", None)
        if target and status in wanted:
            entries.append(ReferenceEntry(source=getattr(unit, "source", ""), target=target))
    return entries


# -- lexical retrieval -------------------------------------------------------

_PLACEHOLDER = re.compile(r"\[\[\d+\]\]")
_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Casefold, drop placeholders, and collapse whitespace.

    Placeholders are removed so a token every line shares (``[[0]]``) cannot dominate
    similarity; casefolding makes matching case-insensitive across scripts.
    """
    return _WHITESPACE.sub(" ", _PLACEHOLDER.sub(" ", text)).strip().casefold()


def _shingles(text: str, ngram: int) -> set[str]:
    """The set of character ``ngram``-grams of ``text``, padded so word edges are counted.

    Character n-grams are script-agnostic: they work for space-delimited and non-space-delimited
    writing alike, and reward the partial overlap that whole-word matching misses in inflected
    languages.
    """
    if not text:
        return set()
    padded = f" {text} "
    if len(padded) <= ngram:
        return {padded}
    return {padded[i:i + ngram] for i in range(len(padded) - ngram + 1)}


class _NgramIndex:
    """TF-IDF cosine over character n-grams, with an inverted index.

    Query cost is proportional to the postings of the query's shingles, not to the corpus size,
    which is the point of the inverted index: a large reference corpus stays cheap per unit.
    Norms are precomputed, so scoring a match is one division. Scores accumulate in a fixed order
    (shingles sorted, ties broken on ascending position) so a query yields byte-identical results
    on every run and worker.

    idf is **frozen** at the seed. New keys added later (self-population) reuse it, so norms stay
    precomputable and a growing corpus keeps the same query cost -- the alternative, recomputing
    idf and every norm on each append, is what makes a naive growable index quadratic. Freezing
    is a standard streaming-IR approximation: a shingle unknown to the seed is treated as rare
    (the seed's unseen-idf), which is what a genuinely novel shingle is.

    Not internally locked. :class:`LexicalRetriever` never mutates it (no lock needed);
    :class:`GrowableLexicalRetriever` owns a lock around ``add`` and ``query`` together.
    """

    def __init__(self, keys: Sequence[str], ngram: int) -> None:
        self._ngram = ngram
        shingle_sets = [_shingles(_normalize(key), ngram) for key in keys]
        seed_count = len(keys)
        document_frequency: Counter[str] = Counter()
        for shingles in shingle_sets:
            document_frequency.update(shingles)
        # Smoothed idf, always positive; +1 keeps a shingle in every document from scoring zero.
        self._idf: dict[str, float] = {
            sh: math.log((seed_count + 1) / (freq + 1)) + 1.0
            for sh, freq in document_frequency.items()}
        # The idf a shingle absent from the seed would have (document frequency 0): its rare
        # weight, given to both an off-corpus query term and a novel shingle from a learned entry.
        self._unseen_idf = math.log(seed_count + 1) + 1.0
        self._postings: dict[str, list[int]] = defaultdict(list)
        self._norms: list[float] = []
        for shingles in shingle_sets:
            self._append(shingles)

    def add(self, key: str) -> None:
        """Index one more key, reusing the frozen idf. O(the key's shingles)."""
        self._append(_shingles(_normalize(key), self._ngram))

    def _append(self, shingles: set[str]) -> None:
        index = len(self._norms)
        total = 0.0
        for shingle in sorted(shingles):  # sorted -> deterministic float accumulation
            weight = self._idf.get(shingle)
            if weight is None:
                # A shingle the seed never saw. Freeze it at the rare weight and record it, so a
                # later query for a learned entry that hinges on this shingle can still find it.
                weight = self._unseen_idf
                self._idf[shingle] = weight
            self._postings[shingle].append(index)  # ascending index -> postings stay sorted
            total += weight * weight
        self._norms.append(math.sqrt(total))

    def query(self, text: str, *, k: int, min_score: float) -> list[tuple[int, float]]:
        """The ``(corpus_index, score)`` of the best ``k`` keys matching ``text``."""
        if k <= 0 or not self._norms:
            return []
        shingles = _shingles(_normalize(text), self._ngram)
        if not shingles:
            return []
        query_norm_sq = 0.0
        dot: dict[int, float] = {}
        for shingle in sorted(shingles):  # sorted -> deterministic float accumulation
            idf = self._idf.get(shingle)
            if idf is None:
                query_norm_sq += self._unseen_idf * self._unseen_idf
                continue
            query_norm_sq += idf * idf
            contribution = idf * idf
            for index in self._postings.get(shingle, ()):
                dot[index] = dot.get(index, 0.0) + contribution
        if query_norm_sq <= 0.0:  # pragma: no cover -- unreachable: a non-empty shingle set
            return []             # gives idf >= 1 per term, so the sum is always positive
        query_norm = math.sqrt(query_norm_sq)
        scored: list[tuple[int, float]] = []
        for index, product in dot.items():
            norm = self._norms[index]
            if norm <= 0.0:  # pragma: no cover -- unreachable: a dotted entry shares a shingle,
                continue     # so it has >= 1 shingle and therefore a positive norm
            # Clamp to the cosine's true range: floating-point rounding can nudge an exact match
            # a hair above 1.0, which would break the [0, 1] contract callers rely on.
            score = min(1.0, product / (query_norm * norm))
            if score >= min_score:
                scored.append((index, score))
        # Best first; ascending corpus position breaks ties so the order never depends on dict
        # iteration.
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]


class LexicalRetriever:
    """The default :class:`Retriever`: deterministic, stdlib-only lexical matching.

    Built once from a corpus, then queried read-only, so a single instance is shared by every
    worker in a run. Only the directions asked for are indexed -- a retriever built with
    ``index_target=False`` raises rather than silently returning nothing if queried by target,
    because a silent empty result would hide the misconfiguration.

    The source index covers only entries that *have* a source; the target index covers every
    entry (all have a target), which is what lets a target-only corpus still be retrieved by
    target.
    """

    def __init__(self, entries: Iterable[ReferenceEntry], *, index_source: bool = True,
                 index_target: bool = False, ngram: int = 3) -> None:
        if ngram < 1:
            raise ValueError(f"ngram must be >= 1, got {ngram}")
        self._entries = tuple(entries)
        self._ngram = ngram
        self._source_ids: tuple[int, ...] | None = None
        self._source_index: _NgramIndex | None = None
        if index_source:
            ids = tuple(i for i, entry in enumerate(self._entries) if entry.source)
            self._source_ids = ids
            self._source_index = _NgramIndex([self._entries[i].source for i in ids], ngram)
        self._target_ids: tuple[int, ...] | None = None
        self._target_index: _NgramIndex | None = None
        if index_target:
            ids = tuple(range(len(self._entries)))  # every entry has a target
            self._target_ids = ids
            self._target_index = _NgramIndex([self._entries[i].target for i in ids], ngram)

    def by_source(self, query: str, *, k: int,
                  min_score: float = 0.0) -> tuple[Retrieved, ...]:
        if self._source_index is None or self._source_ids is None:
            raise ReferenceError(
                "this retriever was built without a source index; construct it with "
                "index_source=True to query by source")
        return self._collect(self._source_index, self._source_ids, query, k, min_score)

    def by_target(self, query: str, *, k: int,
                  min_score: float = 0.0) -> tuple[Retrieved, ...]:
        if self._target_index is None or self._target_ids is None:
            raise ReferenceError(
                "this retriever was built without a target index; construct it with "
                "index_target=True to query by target")
        return self._collect(self._target_index, self._target_ids, query, k, min_score)

    def _collect(self, index: _NgramIndex, ids: tuple[int, ...], query: str, k: int,
                 min_score: float) -> tuple[Retrieved, ...]:
        return tuple(
            Retrieved(entry=self._entries[ids[local]], score=score)
            for local, score in index.query(query, k=k, min_score=min_score))

    @property
    def source_entries(self) -> int:
        """How many entries have a source (are usable by source-side retrieval)."""
        return sum(1 for entry in self._entries if entry.source)

    def __len__(self) -> int:
        return len(self._entries)


class GrowableLexicalRetriever:
    """A :class:`WritableRetriever` that learns within the run.

    Seeded like :class:`LexicalRetriever`, but also accepts accepted translations via
    :meth:`add`, so a later unit can retrieve an earlier one by similarity -- not only the exact
    matches dedup already shares, nor only the positional neighbours context already shows.

    **Efficient on a large corpus.** It reuses the frozen-idf :class:`_NgramIndex`, so ``add`` is
    O(the entry's shingles) and a query stays proportional to matching postings, not to how much
    has been learned. A large ``--reference`` seed costs a one-time build and nothing per query
    beyond the read-only retriever.

    **Single-writer / multi-reader.** The runner appends from its own thread; the workers query
    from theirs. One lock guards ``add`` and the queries together, because a query reads state an
    append is midway through mutating. The read-only :class:`LexicalRetriever` -- lock-free
    because it never changes -- remains the default.

    **Not reproducible by design.** What a unit retrieves depends on what has been added before
    it, which under concurrency is completion order. A self-populating memory is adaptive rather
    than deterministic; that is the trade, and it is why it is off unless asked for.
    """

    def __init__(self, entries: Iterable[ReferenceEntry] = (), *, index_source: bool = True,
                 index_target: bool = False, ngram: int = 3) -> None:
        if ngram < 1:
            raise ValueError(f"ngram must be >= 1, got {ngram}")
        if not (index_source or index_target):
            raise ValueError("a retriever must index at least one of source or target")
        seed = list(entries)
        self._lock = threading.Lock()
        self._entries: list[ReferenceEntry] = list(seed)
        self._source_ids: list[int] = [i for i, entry in enumerate(seed) if entry.source]
        self._target_ids: list[int] = list(range(len(seed)))
        self._source_index = (_NgramIndex([seed[i].source for i in self._source_ids], ngram)
                              if index_source else None)
        self._target_index = (_NgramIndex([seed[i].target for i in self._target_ids], ngram)
                              if index_target else None)

    def add(self, entry: ReferenceEntry) -> None:
        with self._lock:
            index = len(self._entries)
            self._entries.append(entry)
            if self._target_index is not None:
                self._target_ids.append(index)
                self._target_index.add(entry.target)
            if self._source_index is not None and entry.source:
                self._source_ids.append(index)
                self._source_index.add(entry.source)

    def by_source(self, query: str, *, k: int,
                  min_score: float = 0.0) -> tuple[Retrieved, ...]:
        with self._lock:
            if self._source_index is None:
                raise ReferenceError(
                    "this retriever was built without a source index; construct it with "
                    "index_source=True to query by source")
            return self._collect(self._source_index, self._source_ids, query, k, min_score)

    def by_target(self, query: str, *, k: int,
                  min_score: float = 0.0) -> tuple[Retrieved, ...]:
        with self._lock:
            if self._target_index is None:
                raise ReferenceError(
                    "this retriever was built without a target index; construct it with "
                    "index_target=True to query by target")
            return self._collect(self._target_index, self._target_ids, query, k, min_score)

    def _collect(self, index: _NgramIndex, ids: list[int], query: str, k: int,
                 min_score: float) -> tuple[Retrieved, ...]:
        return tuple(
            Retrieved(entry=self._entries[ids[local]], score=score)
            for local, score in index.query(query, k=k, min_score=min_score))

    @property
    def source_entries(self) -> int:
        with self._lock:
            return sum(1 for entry in self._entries if entry.source)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
