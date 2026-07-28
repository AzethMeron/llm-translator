"""Sweep ``reference_mmr_lambda`` and measure what it actually costs and buys.

The end-to-end evaluation found the hybrid path reproducing the corpus's established rendering
markedly less often than the embedding arm alone, and blamed MMR diversification: it suppresses
near-identical examples, which are exactly what reproducing an established rendering needs. The
docs therefore suggest raising ``reference_mmr_lambda`` toward ``1.0`` for a consistency-focused
project -- **a named mechanism, never a measurement**. This measures it.

Why a sweep rather than another A/B. ``tools/eval_translation.sh`` compares exactly two arms, and
its headline consistency metric ("closer to the established rendering") is *pairwise* -- it cannot
rank five settings without running every pair. But the same underlying quantity, **mean trigram
overlap with the established rendering**, is *absolute*: it is computed per arm against the corpus,
not against the other arm. So each λ needs translating only once, and the curve falls out. Quality
is deliberately not judged here -- the A/B already established it is a wash, and paying for
pairwise judging across a sweep would buy noise.

Reported per λ: how often the output actually reproduces the established rendering, the mean
overlap behind that, and the rejection rate -- because if diversity is load-bearing for
robustness, turning it off should show up as more rejections, and that trade is the whole point.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "lib"))

from eval_translation import (  # noqa: E402
    CONSISTENCY_FLOOR,
    EvalError,
    _slug,
    injectable,
    jaccard,
    read_journal,
    run_arm,
    status_counts,
)
from transunit.reference import read_reference  # noqa: E402

MATCH_THRESHOLD = 0.5
"""Trigram overlap at which output counts as *reproducing* the established rendering.

A threshold, not just the mean, because the mean hides the shape: a run that half-copies every
line scores the same as one that reproduces half of them exactly, and only the second is what
"consistency with an existing body of translation" means. Reported alongside the mean so a
distribution change is visible rather than averaged away.
"""


def established_targets(corpus: Path, queries: Sequence[dict[str, Any]], *,
                        embedding_url: str, embedding_model: str) -> dict[str, str]:
    """Each query's established rendering, where the corpus has one above the floor.

    Retrieved with the embedding arm regardless of which arm is being scored: the question is
    "was there an established rendering to match", which must not be answered by whichever
    configuration is under test.
    """
    from translator.retrieval.embedding import EmbeddingRetriever

    entries = list(read_reference(corpus))
    if not entries:
        raise EvalError(f"reference corpus {corpus} yielded no entries")
    retriever = EmbeddingRetriever(entries, base_url=embedding_url, model=embedding_model,
                                   index_source=True, index_target=False)
    try:
        found: dict[str, str] = {}
        for row in queries:
            hits = retriever.by_source(str(row.get("source") or ""), k=1,
                                       min_score=CONSISTENCY_FLOOR)
            if hits:
                found[row["unit_id"]] = hits[0].entry.target
    finally:
        retriever.close()
    if not found:
        raise EvalError(
            f"no query had an established rendering clearing {CONSISTENCY_FLOOR}; is {corpus} "
            f"the corpus these queries were held out of?")
    return found


def score(journal: Path, established: dict[str, str]) -> dict[str, Any]:
    rows = read_journal(journal)
    counts = status_counts(rows)
    total = max(1, sum(counts.values()))
    overlaps = [jaccard(established[uid], str(row["target"]))
                for uid, row in rows.items()
                if uid in established and injectable(row)]
    measured = len(overlaps)
    return {
        "measured": measured,
        "mean_overlap": sum(overlaps) / measured if measured else 0.0,
        "reproduced": sum(1 for value in overlaps if value >= MATCH_THRESHOLD),
        "reproduced_rate": (sum(1 for value in overlaps if value >= MATCH_THRESHOLD) / measured
                            if measured else 0.0),
        "rejection_rate": counts.get("rejected", 0) / total,
        "verified_rate": counts.get("verified", 0) / total,
        "statuses": counts,
    }


def sweep(args: argparse.Namespace) -> dict[str, Any]:
    args.work.mkdir(parents=True, exist_ok=True)
    queries = [json.loads(line) for line in
               args.queries.read_text(encoding="utf-8").splitlines() if line.strip()]

    specs = [f"{args.arm}@{value}" for value in args.lambdas]
    if args.include_embedding:
        # The arm the hybrid path is being compared against. It has no MMR stage at all, so it
        # is the reference point for "did raising lambda close the gap".
        specs.insert(0, "embedding")

    results: dict[str, Any] = {}
    for spec in specs:
        journal = args.work / f"{_slug(spec)}.journal.jsonl"
        if journal.exists() and not args.resume:
            raise EvalError(f"{journal} already exists; pass --resume to continue it deliberately")
        run_arm(spec, args, journal)

    established = established_targets(args.corpus, queries,
                                      embedding_url=args.embedding_url,
                                      embedding_model=args.embedding_model)
    for spec in specs:
        results[spec] = score(args.work / f"{_slug(spec)}.journal.jsonl", established)
    return {"arm": args.arm, "lambdas": args.lambdas, "queries": str(args.queries),
            "corpus": str(args.corpus), "established_available": len(established),
            "match_threshold": MATCH_THRESHOLD, "results": results}


def report(payload: dict[str, Any]) -> None:
    print(f"\nestablished rendering available for {payload['established_available']} queries; "
          f"'reproduced' means trigram overlap >= {payload['match_threshold']}\n")
    print(f"{'arm':22}{'measured':>10}{'reproduced':>12}{'rate':>8}"
          f"{'mean ovl':>10}{'verified':>10}{'rejected':>10}")
    for spec, r in payload["results"].items():
        print(f"{spec:22}{r['measured']:>10}{r['reproduced']:>12}{r['reproduced_rate']:>7.1%}"
              f"{r['mean_overlap']:>10.3f}{r['verified_rate']:>10.1%}{r['rejection_rate']:>10.1%}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep reference_mmr_lambda end to end.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--agents", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--arm", default="hybrid+rerank",
                        help="the hybrid arm to sweep (only hybrid arms have an MMR stage)")
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.7, 0.85, 1.0])
    parser.add_argument("--include-embedding", action="store_true", default=True)
    parser.add_argument("--no-include-embedding", dest="include_embedding",
                        action="store_false")
    parser.add_argument("--glossary", type=Path, default=None)
    parser.add_argument("--rules", type=Path, default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="qwen-local")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--embedding-model", default="local")
    parser.add_argument("--rerank-url", default="")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--source-language", default="ja")
    parser.add_argument("--target-language", default="en")
    parser.add_argument("--source-script", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # run_arm reads these off the namespace; the sweep has no separate judge stage.
    args.judge_url = args.base_url
    try:
        payload = sweep(args)
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
