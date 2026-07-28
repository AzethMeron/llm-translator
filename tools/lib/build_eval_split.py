"""Build a reference corpus and a held-out query set from a real translated journal.

Both live evaluations -- ``tools/eval_retrieval.sh`` (retrieval relevance) and
``tools/eval_translation.sh`` (end-to-end translation A/B) -- need the same thing: a body of
established translations to retrieve *from*, and a set of held-out lines to translate that are
**not already in it**. Sharing one builder is what keeps the two evaluations comparable; building
the split by hand is what made the earlier results irreproducible.

Two split shapes, because real corpora differ in what material is available:

* ``--pool`` given (a separate pool exists alongside the journal): the whole journal becomes the
  corpus, and queries are drawn from that separate, much larger pool of untranslated units. The
  corpus keeps its full size.
* no ``--pool`` (the journal is all there is): queries are **held out of** the journal and the
  corpus is what remains. Shrinks the corpus, which is why the split ratio is reported.

A query is only eligible if its source is non-blank, **not verbatim in the corpus** (otherwise the
retriever is being asked to find a line it already has, which measures nothing), unique within the
query set, and -- with ``--script-pattern`` -- actually written in the expected script.

Sampling is length-stratified and seeded, so the same arguments always produce the same split.

**Shortfall is an error, never a silent under-delivery.** The ad-hoc predecessor of this script
asked for 40 queries, produced 24, and reported neither -- so an evaluation ran at 60% of its
intended sample without anyone noticing. Here, too few eligible queries raises :class:`SplitError`
naming the shortfall and how to fix it.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from transunit.units import Status  # noqa: E402

LENGTH_BANDS: tuple[tuple[int, int], ...] = ((1, 6), (6, 15), (15, 30), (30, 60), (60, 400))
"""Source-length bands (half-open, characters) sampled evenly.

A sample drawn without stratifying is dominated by whatever length the corpus has most of --
usually very short lines -- and short lines are both the easiest to translate and the least
informative about retrieval. Sampling each band evenly keeps long, hard lines in the set.
"""

SCRIPT_PATTERNS = {
    "japanese": r"[぀-ヿ㐀-鿿]",
    "cyrillic": r"[Ѐ-ӿ]",
    "greek": r"[Ͱ-Ͽ]",
    "arabic": r"[؀-ۿ]",
    "han": r"[㐀-鿿]",
}
"""Named presets for ``--script-pattern``, so the common cases need no raw regex."""


class SplitError(Exception):
    """The requested split cannot be built from this data.

    Structured rather than a bare message so a caller can tell "your data is too small" from
    "your arguments are wrong", and so the shortfall numbers survive into the report.
    """

    def __init__(self, reason: str, *, path: Path | None = None,
                 wanted: int | None = None, available: int | None = None) -> None:
        detail = f" (wanted {wanted}, {available} available)" if wanted is not None else ""
        super().__init__(f"{reason}{detail}" + (f" [{path}]" if path else ""))
        self.reason = reason
        self.path = path
        self.wanted = wanted
        self.available = available


def read_units(path: Path) -> Iterator[dict[str, Any]]:
    """Every well-formed JSON object in a JSONL file, by line.

    A torn or truncated final line is tolerated only if it is the *last* one (a journal being
    appended to while this runs); a malformed line anywhere else is an error, because silently
    skipping records is how a split quietly stops matching the corpus it claims to describe.
    """
    if not path.is_file():
        raise SplitError("file not found", path=path)
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            if number == len(lines):
                break  # a half-written trailing line: the writer is still appending
            raise SplitError(f"line {number} is not valid JSON: {exc}", path=path) from exc
        if not isinstance(record, dict):
            raise SplitError(f"line {number} is not a JSON object", path=path)
        yield record


def translated_rows(path: Path, *, statuses: frozenset[str]) -> list[dict[str, Any]]:
    """Journal rows carrying a usable translation, i.e. the material a retriever can learn from."""
    rows = [row for row in read_units(path)
            if row.get("status") in statuses
            and str(row.get("source") or "").strip()
            and str(row.get("target") or "").strip()]
    if not rows:
        raise SplitError(
            f"no rows with status in {sorted(statuses)} carrying both a source and a target",
            path=path)
    return rows


def stratified(pool: Sequence[dict[str, Any]], count: int,
               rng: random.Random) -> list[dict[str, Any]]:
    """``count`` rows sampled evenly across :data:`LENGTH_BANDS`, deterministically.

    Bands that cannot fill their share are topped up from whatever remains, so a corpus with no
    very long lines still yields the full count rather than silently returning fewer.
    """
    banded: list[list[dict[str, Any]]] = [[] for _ in LENGTH_BANDS]
    for row in pool:
        length = len(str(row.get("source") or ""))
        for index, (low, high) in enumerate(LENGTH_BANDS):
            if low <= length < high:
                banded[index].append(row)
                break
    per_band = max(1, count // len(LENGTH_BANDS))
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for band in banded:
        rng.shuffle(band)
        for row in band[:per_band]:
            if row["unit_id"] not in seen:
                seen.add(row["unit_id"])
                chosen.append(row)
    if len(chosen) < count:
        remainder = [row for row in pool if row["unit_id"] not in seen]
        rng.shuffle(remainder)
        for row in remainder[:count - len(chosen)]:
            seen.add(row["unit_id"])
            chosen.append(row)
    rng.shuffle(chosen)
    return chosen[:count]


def to_pending(row: dict[str, Any]) -> dict[str, Any]:
    """A query unit: the same payload and context, with any existing translation stripped.

    The surrounding-context fields are carried over deliberately -- the engine packs them into
    the prompt, so dropping them here would evaluate a different prompt than production sends.
    """
    return {
        "unit_id": row["unit_id"],
        "rel_path": row.get("rel_path", ""),
        "line_no": row.get("line_no", 0),
        "span_start": row.get("span_start", 0),
        "span_end": row.get("span_end", 0),
        "command": row.get("command", ""),
        "kind": row.get("kind", "LINE"),
        "source": row["source"],
        "placeholders": row.get("placeholders", []),
        "speaker": row.get("speaker"),
        "context_before": row.get("context_before", []),
        "context_after": row.get("context_after", []),
        "max_columns": row.get("max_columns"),
        "status": Status.PENDING.value,
        "target": None,
        "notes": [],
    }


def eligible_queries(pool: Sequence[dict[str, Any]], *, corpus_sources: set[str],
                     script: re.Pattern[str] | None) -> list[dict[str, Any]]:
    """Pool rows that may legitimately be asked as queries, deduplicated by source text."""
    chosen: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for row in pool:
        source = str(row.get("source") or "").strip()
        if not source or source in corpus_sources or source in seen_sources:
            continue
        if script is not None and not script.search(source):
            continue
        seen_sources.add(source)
        chosen.append(row)
    return chosen


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    script = None
    if args.script_pattern:
        raw = SCRIPT_PATTERNS.get(args.script_pattern, args.script_pattern)
        try:
            script = re.compile(raw)
        except re.error as exc:
            raise SplitError(f"--script-pattern is not a valid regex: {exc}") from exc

    journal_rows = translated_rows(args.journal, statuses=frozenset(args.corpus_statuses))

    if args.pool is not None:
        # The corpus keeps every translated row; queries come from a separate, untranslated pool.
        corpus_rows = journal_rows
        pool_rows = list(read_units(args.pool))
        corpus_sources = {str(row.get("source") or "").strip() for row in corpus_rows}
        candidates = eligible_queries(pool_rows, corpus_sources=corpus_sources, script=script)
    else:
        # No pool: queries must be held OUT of the journal, so the corpus is what is left after.
        #
        # Holding out exactly `queries` rows is not enough. A corpus with many repeated lines --
        # one measured corpus repeats a third of its own -- leaves most held-out rows still present
        # verbatim in what remains, and those are ineligible: asking the retriever for a line it
        # already has measures nothing. Hold out progressively more until enough SURVIVE that
        # filter, rather than making the caller guess the ratio (that corpus needs ~1.6x; one
        # without repeats needs none).
        #
        # The whole oversampled set stays out of the corpus, not just the chosen queries -- a
        # returned row could re-introduce a chosen query's source and silently void its
        # eligibility. Deterministic: the same seed grows through the same sequence.
        #
        # The holdout is capped BELOW the journal's size, so a corpus always remains. Held out in
        # full, every row would look eligible purely because there is no corpus left to duplicate
        # it, and the build would "succeed" with a zero-entry reference corpus -- a split against
        # which every retriever scores nothing.
        largest_holdout = len(journal_rows) - 1
        if args.queries > largest_holdout:
            raise SplitError(
                "holding out that many queries would leave no reference corpus behind; the "
                "corpus must keep at least one row. Lower --queries or supply a larger journal",
                path=args.journal, wanted=args.queries, available=largest_holdout)
        wanted = args.queries
        while True:
            held_out = stratified(journal_rows, wanted, rng)
            held_ids = {row["unit_id"] for row in held_out}
            corpus_rows = [row for row in journal_rows if row["unit_id"] not in held_ids]
            corpus_sources = {str(row.get("source") or "").strip() for row in corpus_rows}
            candidates = eligible_queries(held_out, corpus_sources=corpus_sources, script=script)
            if len(candidates) >= args.queries or wanted == largest_holdout:
                break
            wanted = min(largest_holdout, max(wanted + 1, int(wanted * 1.5)))
    if len(candidates) < args.queries:
        raise SplitError(
            "too few eligible queries: a query must be non-blank, absent from the corpus, "
            "unique, and match --script-pattern. Lower --queries, widen --script-pattern, or "
            "supply a larger --pool",
            path=args.pool or args.journal, wanted=args.queries, available=len(candidates))
    queries = stratified(candidates, args.queries, rng)

    write_jsonl(args.corpus_out,
                [{"source": row["source"], "target": row["target"]} for row in corpus_rows])
    write_jsonl(args.queries_out, [to_pending(row) for row in queries])

    lengths = sorted(len(str(row["source"])) for row in queries)
    return {
        "journal": str(args.journal),
        "pool": str(args.pool) if args.pool else None,
        "seed": args.seed,
        "corpus_entries": len(corpus_rows),
        "queries": len(queries),
        "eligible_candidates": len(candidates),
        "held_out_from_corpus": args.pool is None,
        "query_source_length": {
            "min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1],
        },
        "corpus_out": str(args.corpus_out),
        "queries_out": str(args.queries_out),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a reference corpus + held-out query set from a translated journal.")
    parser.add_argument("--journal", type=Path, required=True,
                        help="translated journal: the source of established translations")
    parser.add_argument("--pool", type=Path, default=None,
                        help="optional separate pool of units to draw queries from; without it, "
                             "queries are held out of --journal and the corpus shrinks")
    parser.add_argument("--corpus-out", type=Path, required=True)
    parser.add_argument("--queries-out", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=1000,
                        help="how many held-out queries to produce (exact; a shortfall is an "
                             "error, not a smaller set)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--script-pattern", default="",
                        help=f"regex, or one of {sorted(SCRIPT_PATTERNS)}, that a query's source "
                             f"must match -- keeps untranslatable rows out of the sample")
    parser.add_argument("--corpus-statuses", nargs="+", default=["verified", "translated"],
                        help="journal statuses whose rows may enter the reference corpus")
    parser.add_argument("--out", type=Path, default=None, help="write the split report as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.queries < 1:
        print("error: --queries must be at least 1", file=sys.stderr)
        return 2
    try:
        report = build(args)
    except SplitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lengths = report["query_source_length"]
    print(f"corpus  {report['corpus_entries']:>7} entries -> {report['corpus_out']}")
    print(f"queries {report['queries']:>7} of {report['eligible_candidates']} eligible "
          f"-> {report['queries_out']}")
    print(f"        source length min/median/max = "
          f"{lengths['min']}/{lengths['median']}/{lengths['max']}")
    if report["held_out_from_corpus"]:
        total = report["corpus_entries"] + report["queries"]
        share = 100.0 * report["queries"] / total
        print(f"        held out of the journal: the corpus lost {share:.1f}% of its rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
