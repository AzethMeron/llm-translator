"""Measure whether reference retrieval actually fetches *relevant* material, on real text.

The hermetic suite pins retrieval mechanics with hand-picked vectors; it cannot tell you whether
a retriever fetches genuinely useful examples from a real corpus in a real language. This does,
by retrieving over a real corpus and having an independent LLM judge rate each retrieved example
for relevance -- the same methodology used to calibrate the lexical retriever's default floor.

Methodology (the parts that make the number trustworthy):

* **Held-out, unique, non-duplicate queries.** A query that appears verbatim in the corpus
  retrieves itself at similarity 1.0 and inflates every score, so such queries are excluded, and
  the query set is de-duplicated against itself.
* **An independent judge.** Relevance is rated by a separate LLM (not the retriever's own score),
  0/1/2, so "relevant" is not defined circularly by the thing under test.
* **Deterministic sampling.** A fixed seed, so a rerun compares like with like.

Run it through ``tools/eval_retrieval.sh``, which checks the servers are actually up first.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import httpx  # noqa: E402

from transunit.reference import LexicalRetriever, ReferenceEntry  # noqa: E402
from translator.retrieval.embedding import EmbeddingRetriever  # noqa: E402
from translator.retrieval.hybrid import HybridRetriever  # noqa: E402

JUDGE_SYSTEM = (
    "You judge whether two lines of game text are RELATED IN MEANING or CONTENT -- that is, "
    "whether the second would be a useful reference example for a translator working on the "
    "first. Ignore shared grammatical particles, punctuation, and generic structure; those alone "
    "are not relatedness. Answer with a rating: 2 = clearly related and useful as a reference, "
    "1 = loosely related, 0 = unrelated. Respond with JSON {\"rating\": 0|1|2}."
)
JUDGE_SCHEMA = {"type": "object", "properties": {"rating": {"type": "integer"}},
                "required": ["rating"]}


class EvalError(Exception):
    """The evaluation cannot run or its quality gates failed."""


def load_entries(path: Path) -> list[ReferenceEntry]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        target = record.get("target")
        if isinstance(target, str) and target.strip():
            entries.append(ReferenceEntry(source=record.get("source", "") or "", target=target))
    return entries


def sample_queries(pool_path: Path, corpus: list[ReferenceEntry], *, count: int,
                   seed: int, script: re.Pattern[str] | None) -> list[str]:
    """Unique query sources that do NOT appear verbatim in the corpus (see the module docstring)."""
    corpus_sources = {entry.source for entry in corpus}
    seen: set[str] = set()
    pool: list[str] = []
    for line in pool_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source = json.loads(line).get("source") or ""
        if (source and source not in corpus_sources and source not in seen
                and (script is None or script.search(source))):
            seen.add(source)
            pool.append(source)
    if len(pool) < count:
        raise EvalError(
            f"only {len(pool)} unique non-corpus-duplicate queries available in {pool_path}, "
            f"need {count}; supply a larger pool or lower --queries")
    return random.Random(seed).sample(pool, count)


def build_retrievers(corpus: list[ReferenceEntry], *, embedding_url: str, rerank_url: str,
                     lexical_floor: float, embedding_floor: float, pool: int) -> dict:
    """One retriever per variant under test. Built once and shared across all queries."""
    retrievers = {
        "lexical": LexicalRetriever(corpus, index_source=True),
        "embedding": EmbeddingRetriever(corpus, base_url=embedding_url, index_source=True),
        "hybrid": HybridRetriever(
            corpus, embedding_base_url=embedding_url, index_source=True, candidate_pool=pool,
            lexical_min_score=lexical_floor, embedding_min_score=embedding_floor),
    }
    if rerank_url:
        retrievers["hybrid+rerank"] = HybridRetriever(
            corpus, embedding_base_url=embedding_url, rerank_base_url=rerank_url,
            index_source=True, candidate_pool=pool, lexical_min_score=lexical_floor,
            embedding_min_score=embedding_floor)
    return retrievers


def judge_batch(pairs: list[tuple[str, str]], *, base_url: str, model: str,
                concurrency: int) -> list[int]:
    """Rate each ``(query, retrieved)`` pair 0/1/2. A failed judgement is ``-1`` and excluded."""
    url = base_url.rstrip("/") + "/chat/completions"

    def judge(pair: tuple[str, str]) -> int:
        query, retrieved = pair
        body = {
            "model": model, "temperature": 0.0, "max_tokens": 60,
            "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                         {"role": "user", "content": f"A:\n{query}\n\nB:\n{retrieved}"}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "relevance", "strict": True, "schema": JUDGE_SCHEMA}},
        }
        try:
            response = httpx.post(url, json=body, timeout=120)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return int(json.loads(content)["rating"])
        except Exception:  # noqa: BLE001 -- one unusable judgement must not abort the sweep
            return -1

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        return list(executor.map(judge, pairs))


def evaluate(args: argparse.Namespace) -> int:
    corpus = load_entries(args.corpus)
    if not corpus:
        raise EvalError(f"no usable reference entries in {args.corpus}")
    script = re.compile(args.script_pattern) if args.script_pattern else None
    queries = sample_queries(args.queries_from, corpus, count=args.queries, seed=args.seed,
                             script=script)
    print(f"corpus {len(corpus):,} entries | {len(queries)} unique held-out queries "
          f"(seed {args.seed})", flush=True)

    retrievers = build_retrievers(
        corpus, embedding_url=args.embedding_url, rerank_url=args.rerank_url,
        lexical_floor=args.lexical_min_score, embedding_floor=args.embedding_min_score,
        pool=args.candidate_pool)
    try:
        # Retrieve first (cheap, no judge), so every variant is judged on the same query set.
        top: dict[str, list[tuple[str, str, float]]] = {}
        for name, retriever in retrievers.items():
            rows = []
            for query in queries:
                hits = retriever.by_source(query, k=1, min_score=args.min_score)
                rows.append((query, hits[0].entry.source, hits[0].score) if hits
                            else (query, "", 0.0))
            retrieved = sum(1 for _, source, _ in rows if source)
            print(f"  {name:15} retrieved something on {retrieved}/{len(queries)} queries",
                  flush=True)
            top[name] = rows

        print("\njudging relevance (independent LLM, 0/1/2)...", flush=True)
        results: dict[str, dict] = {}
        for name, rows in top.items():
            judged = [(query, source) for query, source, _ in rows if source]
            ratings = judge_batch(judged, base_url=args.judge_url, model=args.judge_model,
                                  concurrency=args.concurrency)
            usable = [r for r in ratings if r >= 0]
            if not judged:
                # Distinct from a dead judge, and diagnosed separately: nothing was ever sent to
                # it. Blaming the judge here would send the reader to the wrong server.
                raise EvalError(
                    f"the {name!r} variant retrieved nothing on any of the {len(queries)} "
                    f"queries, so there is nothing to judge; --min-score is "
                    f"{args.min_score} -- lower it, or check that the corpus is the one these "
                    f"queries were held out of")
            if not usable:
                raise EvalError(
                    f"every relevance judgement failed for {name!r}; is the judge server at "
                    f"{args.judge_url} serving {args.judge_model!r}?")
            clearly = sum(1 for r in usable if r >= 2)
            # Per-query detail, so a variant that loses overall can still be examined for WHERE
            # it lost -- an aggregate alone cannot tell "uniformly mediocre" from "excellent
            # except on one recognisable class of query".
            per_query = {query: rating
                         for (query, _), rating in zip(judged, ratings, strict=True)}
            results[name] = {
                "queries": len(queries), "retrieved": len(judged), "judged": len(usable),
                "clearly_relevant": clearly,
                "per_query": per_query,
                # Precision over what was RETRIEVED: of the examples actually shown, how many
                # were useful. A retriever that shows nothing is not rewarded here -- coverage
                # is reported separately, above and below.
                "precision": clearly / len(usable),
                # Over ALL queries, so a retriever that stays silent is not flattered by a high
                # precision on a tiny number of hits.
                "yield": clearly / len(queries),
                "ratings": dict(Counter(usable)),
            }

        print(f"\n{'variant':16}{'shown':>8}{'clearly rel.':>14}{'precision':>11}{'yield':>9}")
        for name, row in results.items():
            print(f"{name:16}{row['retrieved']:>8}{row['clearly_relevant']:>14}"
                  f"{row['precision']:>10.0%}{row['yield']:>9.0%}")

        if args.out:
            args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                encoding="utf-8")
            print(f"\nwrote {args.out}")
        return check_gates(results, floor=args.precision_floor, tolerance=args.tolerance)
    finally:
        for retriever in retrievers.values():
            if hasattr(retriever, "close"):
                retriever.close()


def check_gates(results: dict, *, floor: float, tolerance: float) -> int:
    """The assertions that make this a gate rather than a report. Non-zero exit on failure.

    What is gated is deliberately narrower than "every variant beats every other". Measured on
    Japanese game text, equal-weight rank fusion **blends** its arms rather than taking their
    maximum: where the arms disagreed, bare hybrid followed the better one only about half the
    time, so with one arm much stronger than the other it landed near the weaker arm. That is a
    property of RRF, documented in ``docs/reference-translations.md``, not a regression -- so
    bare hybrid is *reported* against its arms but not *gated* on beating them, and the gate
    holds the configuration actually recommended (hybrid+rerank, or a single arm) to account.
    """
    failures = []
    for name in ("lexical", "embedding", "hybrid"):
        row = results.get(name)
        if row is None:
            raise EvalError(f"the {name!r} variant did not run; cannot gate on it")
        if row["precision"] < floor:
            failures.append(f"{name} precision {row['precision']:.0%} is below the absolute "
                            f"floor {floor:.0%} -- retrieval is returning mostly irrelevant "
                            f"material")
    best_arm = max(results[name]["precision"] for name in ("lexical", "embedding"))
    reranked = results.get("hybrid+rerank")
    if reranked is not None:
        # The recommended configuration: it must not fall below what a single arm achieves,
        # or there is no reason to pay for three servers.
        if reranked["precision"] < best_arm - tolerance:
            failures.append(f"hybrid+rerank precision {reranked['precision']:.0%} is more than "
                            f"{tolerance:.0%} below the best single arm ({best_arm:.0%}); the "
                            f"reranking stage is not paying for itself")
        if reranked["precision"] < results["hybrid"]["precision"] - tolerance:
            failures.append(f"hybrid+rerank precision {reranked['precision']:.0%} is more than "
                            f"{tolerance:.0%} below plain hybrid "
                            f"({results['hybrid']['precision']:.0%})")
    # Reported, never gated -- see the docstring.
    delta = results["hybrid"]["precision"] - best_arm
    print(f"\nbare hybrid vs its best single arm: {delta:+.0%} "
          f"(reported, not gated -- equal-weight fusion blends its arms)")
    if failures:
        print("\nFAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("all gates passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure real-data retrieval relevance across retriever variants")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries-from", type=Path, required=True,
                        help="JSONL to sample held-out queries from (corpus duplicates excluded)")
    parser.add_argument("--queries", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--script-pattern", default="",
                        help="only sample queries matching this regex (e.g. a CJK range)")
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--rerank-url", default="")
    parser.add_argument("--judge-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--judge-model", default="qwen-local")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="final floor; 0 measures raw top-1 quality without a floor")
    parser.add_argument("--lexical-min-score", type=float, default=0.30)
    parser.add_argument("--embedding-min-score", type=float, default=0.55)
    parser.add_argument("--candidate-pool", type=int, default=40)
    parser.add_argument("--precision-floor", type=float, default=0.35,
                        help="minimum acceptable hybrid precision")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="how far a variant may fall below its comparator before failing")
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return evaluate(args)
    except EvalError as exc:
        print(f"eval_retrieval: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
