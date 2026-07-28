"""End-to-end translation A/B between two reference-retrieval configurations, on real text.

``tools/eval_retrieval.sh`` answers "does the retriever fetch relevant material?" -- cheap, no
translation. This answers the question that actually matters: **does translating with the better
retriever produce better output?** It is the "Phase B" of the staged evaluation in
``docs/feature-requests/hybrid-retrieval-reranking-rag.md``.

Three metrics, because the honest expectation is that they disagree:

* **A/B quality** -- a blinded, order-randomised judge picks between the two arms' renderings of
  the same source. Reported, never gated: the plan states up front that per-line quality may be a
  wash, and the earlier measured run bore that out. Gating on it would be gating on noise.
* **Consistency** -- how often each arm's output matches the *established* rendering already in
  the reference corpus, by trigram overlap. This is the job reference retrieval actually does, and
  it is what the gate is set on.
* **Rejection rate** -- how many units each arm failed to translate at all. A retriever that
  raises quality but poisons generation (a corrupt corpus row presented as an authoritative
  example is enough) must not pass, so this is gated too.

Every arm runs the real CLI as a subprocess against real servers; nothing here re-implements the
engine. Arms are journalled separately, so a run is resumable and a crashed arm loses nothing.

Run it through ``tools/eval_translation.sh``, which checks the servers are up first.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from translator.roles import AgentSet  # noqa: E402
from transunit.reference import read_reference  # noqa: E402

ARMS = ("none", "lexical", "embedding", "hybrid", "hybrid+rerank")

ARM_FLOORS = {
    "none": 0.0,   # unused: the "none" arm retrieves nothing, so no floor applies
    "lexical": 0.30,
    "embedding": 0.55,
    "hybrid": 0.30,
    "hybrid+rerank": 0.30,
}
"""The calibrated ``reference_min_score`` for each arm.

These are NOT interchangeable: the number means a different thing per arm (a lexical TF-IDF
score, an embedding cosine, a fused rank-agreement fraction, a reranker sigmoid), and the
measured calibration for each lives in ``docs/reference-translations.md``. Running an arm on
another arm's floor silently changes what is retrieved, so the harness writes the right one into
each arm's own agent config rather than sharing one file.
"""

CONSISTENCY_FLOOR = 0.55
"""Similarity an example must clear to count as this line's *established* rendering.

Below it there is nothing to be consistent WITH, so the unit is excluded from the consistency
measure entirely rather than counted as a miss -- counting it would punish an arm for a line the
corpus never covered.
"""

CONSISTENCY_DEADBAND = 0.02
"""Trigram-overlap difference below which the two arms are called equally close.

Without it, floating-point noise on identical renderings would be split arbitrarily into wins.
"""

JUDGE_SYSTEM = """You are comparing two translations of the same {source_language} line into \
{target_language}.

Pick the one that better preserves the meaning and force of the source: nothing added, nothing \
dropped, nothing softened or intensified, the same participants doing the same things, and the \
same speaker's register. If both are equally faithful, prefer the one that reads more naturally \
in {target_language}. If neither is better, answer "tie".

Tokens like [[0]] are placeholders; treat them as equal wherever they appear. Judge quality, \
not length. Answer with the JSON object only."""


def parse_arm(spec: str) -> tuple[str, float | None]:
    """Split an arm spec into its retriever and an optional ``reference_mmr_lambda`` override.

    ``"hybrid+rerank"`` uses the base config's lambda; ``"hybrid+rerank@1.0"`` overrides it. The
    suffix exists so the SAME retriever can be compared against itself at two settings -- the
    knob the measured consistency trade hinges on -- and it doubles as the arm's label, so the
    two journals and every reported number stay distinguishable without extra flags.
    """
    arm, _, raw = spec.partition("@")
    if arm not in ARMS:
        raise EvalError(f"unknown arm {arm!r}; choose from {', '.join(ARMS)} "
                        f"(optionally suffixed '@<mmr_lambda>')")
    if not raw:
        return arm, None
    try:
        value = float(raw)
    except ValueError:
        raise EvalError(f"{spec!r}: the '@' suffix must be a number, got {raw!r}") from None
    if not 0.0 <= value <= 1.0:
        raise EvalError(f"{spec!r}: reference_mmr_lambda must be in [0, 1], got {value}")
    if arm not in ("hybrid", "hybrid+rerank"):
        raise EvalError(f"{spec!r}: reference_mmr_lambda only affects the hybrid arms; "
                        f"setting it on {arm!r} would measure nothing")
    return arm, value


class EvalError(Exception):
    """The evaluation cannot be run or its inputs are unusable.

    Structured so the caller can tell a setup problem (a missing file, a dead server, an arm that
    failed to run) from a gate failure, which is a legitimate result rather than an error.
    """


# -- arms -----------------------------------------------------------------------


def agent_config_for(spec: str, base: Path, destination: Path) -> Path:
    """``base`` with this arm's calibrated ``reference_min_score``, written to ``destination``.

    Edited textually and then **re-loaded and verified**, rather than trusted: a silently
    ineffective edit would run the arm on the wrong floor and quietly measure the wrong thing,
    which is exactly the class of failure this evaluation exists to detect.
    """
    arm, mmr_lambda = parse_arm(spec)
    text = base.read_text(encoding="utf-8")
    if arm == "none":
        # Retrieval off, so the arm translates unaided. Both reference counts must go: leaving
        # reference_revision_examples on would quietly re-introduce retrieval during revision.
        for key in ("reference_examples", "reference_revision_examples"):
            pattern = re.compile(rf"^\s*{key}\s*=.*$", re.MULTILINE)
            text = pattern.sub(f"{key} = 0", text) if pattern.search(text) else text
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        loaded = AgentSet.load(destination)
        if loaded.context.reference_examples or loaded.context.reference_revision_examples:
            raise EvalError(
                f"the 'none' arm still has retrieval enabled in {destination} "
                f"(reference_examples={loaded.context.reference_examples}, "
                f"reference_revision_examples={loaded.context.reference_revision_examples}); "
                f"it would not be a no-reference baseline")
        return destination

    floor = ARM_FLOORS[arm]
    pattern = re.compile(r"^\s*reference_min_score\s*=.*$", re.MULTILINE)
    replacement = f"reference_min_score = {floor}"
    if pattern.search(text):
        text = pattern.sub(replacement, text, count=1)
    elif re.search(r"^\[context\]\s*$", text, re.MULTILINE):
        text = re.sub(r"^\[context\]\s*$", f"[context]\n{replacement}", text,
                      count=1, flags=re.MULTILINE)
    else:
        text += f"\n[context]\n{replacement}\n"
    if mmr_lambda is not None:
        lam = re.compile(r"^\s*reference_mmr_lambda\s*=.*$", re.MULTILINE)
        line = f"reference_mmr_lambda = {mmr_lambda}"
        text = lam.sub(line, text, count=1) if lam.search(text) else text.replace(
            replacement, f"{replacement}\n{line}", 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")

    loaded = AgentSet.load(destination)
    if mmr_lambda is not None and abs(loaded.context.reference_mmr_lambda - mmr_lambda) > 1e-9:
        raise EvalError(
            f"writing reference_mmr_lambda={mmr_lambda} for {spec!r} did not take effect "
            f"(config reports {loaded.context.reference_mmr_lambda}); {destination} is not "
            f"shaped as expected")
    if abs(loaded.context.reference_min_score - floor) > 1e-9:
        raise EvalError(
            f"writing reference_min_score={floor} for the {arm!r} arm did not take effect "
            f"(config reports {loaded.context.reference_min_score}); {destination} is not shaped "
            f"as expected")
    if loaded.context.reference_examples < 1:
        raise EvalError(
            f"{base} sets context.reference_examples = {loaded.context.reference_examples}, so "
            f"reference retrieval is OFF and both arms would translate identically. Set it above "
            f"zero in the base agent config")
    return destination


def arm_flags(spec: str, *, corpus: Path, embedding_url: str, rerank_url: str) -> list[str]:
    arm, _ = parse_arm(spec)
    if arm == "none":
        # No corpus at all, not merely a retriever that finds nothing: the point of this arm is
        # what the model writes unaided, which is the baseline every retrieval arm is really
        # being judged against.
        return []
    flags = ["--reference", str(corpus)]
    if arm == "lexical":
        return flags
    flags += ["--embedding-url", embedding_url]
    if arm in ("hybrid", "hybrid+rerank"):
        flags.append("--hybrid")
    if arm == "hybrid+rerank":
        if not rerank_url:
            raise EvalError("the 'hybrid+rerank' arm needs --rerank-url (tools/serve_reranker.sh)")
        flags += ["--rerank-url", rerank_url]
    return flags


def run_arm(arm: str, args: argparse.Namespace, journal: Path) -> None:
    """Translate the held-out set under ``arm``, into its own journal.

    A non-zero exit is fatal: continuing would compare a complete arm against a partial one and
    report the difference as a quality result.
    """
    config = agent_config_for(arm, args.agents, args.work / f"agents_{_slug(arm)}.toml")
    command = [
        str(args.python), "-m", "translator.cli",
        "-c", str(args.queries), "-j", str(journal),
        "-a", str(config),
        "--base-url", args.base_url, "--model", args.model, "--backend", args.backend,
        "--source-language", args.source_language, "--target-language", args.target_language,
        "--concurrency", str(args.concurrency),
        "--log-file", str(args.work / f"{_slug(arm)}.log"),
        *arm_flags(arm, corpus=args.corpus, embedding_url=args.embedding_url,
                   rerank_url=args.rerank_url),
    ]
    if args.glossary:
        command += ["-g", str(args.glossary)]
    if args.rules:
        command += ["-r", str(args.rules)]
    if args.source_script:
        # --languages must point at a missing path for the script detector to take over; that is
        # the documented way to disable the lexical detector, not a workaround.
        command += ["--source-script", args.source_script, "--languages", "/nonexistent"]

    environment = {"PYTHONPATH": str(REPO / "src")}
    print(f">> arm {arm}: {' '.join(command)}", file=sys.stderr, flush=True)
    completed = subprocess.run(command, env={**_environ(), **environment}, check=False)
    if completed.returncode != 0:
        raise EvalError(
            f"the {arm!r} arm exited {completed.returncode}; its journal ({journal}) is "
            f"incomplete and comparing it would misreport the result. See "
            f"{args.work / f'{_slug(arm)}.log'}")


def _environ() -> dict[str, str]:
    import os
    return dict(os.environ)


def _slug(arm: str) -> str:
    return arm.replace("+", "_").replace("@", "_at_").replace(".", "p")


# -- reading results ------------------------------------------------------------


def read_journal(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise EvalError(f"journal not found: {path}")
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue  # a torn trailing line from an interrupted run
        if isinstance(record, dict) and record.get("unit_id"):
            rows[record["unit_id"]] = record
    if not rows:
        raise EvalError(f"journal {path} has no usable rows; did the arm translate anything?")
    return rows


INJECTABLE = frozenset(("verified", "translated"))
"""Statuses whose target is actually shown to a reader.

The quality and consistency measures consider only these. A ``rejected`` unit often still
carries the text that failed its checks, and comparing that text would score renderings the
engine has already refused to ship -- measuring output nobody will ever see. Rejections are not
ignored: they are the separate, gated rejection-rate metric.
"""


def injectable(record: dict[str, Any]) -> bool:
    return str(record.get("status", "")) in INJECTABLE and bool(record.get("target"))


def status_counts(rows: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in rows.values():
        counts[str(record.get("status", "?"))] = counts.get(str(record.get("status", "?")), 0) + 1
    return counts


# -- metrics --------------------------------------------------------------------


def trigrams(text: str) -> set[str]:
    lowered = text.lower()
    return {lowered[i:i + 3] for i in range(max(0, len(lowered) - 2))} or {lowered}


def jaccard(left: str, right: str) -> float:
    first, second = trigrams(left), trigrams(right)
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def load_ground_truth(path: Path) -> dict[str, str]:
    """``unit_id`` -> the line's real established translation, from a journal.

    The only sound basis for the consistency measure. See :func:`consistency` for why the
    retriever-derived alternative is not one.
    """
    truth: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("unit_id") and record.get("target"):
            truth[record["unit_id"]] = str(record["target"])
    if not truth:
        raise EvalError(f"{path} carries no unit_id/target pairs to use as ground truth")
    return truth


def consistency(baseline: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]],
                *, corpus: Path, embedding_url: str, embedding_model: str,
                labels: tuple[str, str],
                ground_truth: dict[str, str] | None = None) -> dict[str, Any]:
    """How close each arm's output is to the line's established rendering.

    **Supply ``ground_truth`` wherever it exists.** Without it this falls back to a proxy --
    the most similar corpus entry, found with the embedding retriever -- and that proxy is
    CIRCULAR whenever an arm uses the same retriever at the same floor: measured, the embedding
    arm was shown the proxy entry as its own top example on 297 of 297 queries, against the
    hybrid arm's 93. It therefore scored a similarity of 1.000 by construction. That artifact
    produced, and then faithfully replicated across two corpora, a "the embedding arm is better
    at consistency" result that a real held-out ground truth showed to be a dead heat (223 vs
    222 of 593). A metric that reproduces cleanly can still be measuring itself.

    Ground truth means the held-out lines' OWN established translations -- available whenever
    the queries were held out of an already-translated journal, which is the shape
    ``build_eval_split.sh`` produces without ``--pool``.
    """
    if ground_truth is None and "embedding" in labels:
        raise EvalError(
            "refusing to measure consistency against a retriever-derived proxy while one arm is "
            "'embedding': the proxy IS that arm's own top pick, so the comparison is circular and "
            "the embedding arm wins by construction. Pass --ground-truth with the journal the "
            "held-out queries came from, or compare two non-embedding arms")

    retriever = None
    if ground_truth is None:
        # Only the proxy path needs a retriever -- and a server to run it on. With real ground
        # truth this is pure arithmetic over two journals and a file.
        from translator.retrieval.embedding import EmbeddingRetriever

        entries = list(read_reference(corpus))
        if not entries:
            raise EvalError(f"reference corpus {corpus} yielded no entries")
        retriever = EmbeddingRetriever(entries, base_url=embedding_url, model=embedding_model,
                                       index_source=True, index_target=False)
    try:
        shared = [uid for uid in baseline if uid in candidate]
        closer = {labels[0]: 0, labels[1]: 0, "tie": 0}
        overlaps = {labels[0]: 0.0, labels[1]: 0.0}
        measured = 0
        for uid in shared:
            first, second = baseline[uid], candidate[uid]
            if not (injectable(first) and injectable(second)):
                continue
            if ground_truth is not None:
                if uid not in ground_truth:
                    continue
                established = ground_truth[uid]
            else:
                assert retriever is not None  # ground_truth is None here, so it was built
                hits = retriever.by_source(str(first.get("source") or ""), k=1,
                                           min_score=CONSISTENCY_FLOOR)
                if not hits:
                    continue  # nothing established here: excluded, not counted as a miss
                established = hits[0].entry.target
            left = jaccard(established, str(first["target"]))
            right = jaccard(established, str(second["target"]))
            measured += 1
            overlaps[labels[0]] += left
            overlaps[labels[1]] += right
            if left > right + CONSISTENCY_DEADBAND:
                closer[labels[0]] += 1
            elif right > left + CONSISTENCY_DEADBAND:
                closer[labels[1]] += 1
            else:
                closer["tie"] += 1
    finally:
        if retriever is not None:
            retriever.close()
    if not measured:
        raise EvalError(
            f"no unit had an established rendering clearing {CONSISTENCY_FLOOR}; the consistency "
            f"measure has nothing to report. Is {corpus} the corpus these queries were held out "
            f"of?")
    return {
        "measured": measured,
        "closer": closer,
        "mean_overlap": {name: total / measured for name, total in overlaps.items()},
    }


def judge_pair(client: httpx.Client, url: str, model: str, system: str,
               source: str, left: str, right: str) -> str:
    body = {
        "model": model, "temperature": 0.0, "max_tokens": 200,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",
             "content": f"Source:\n{source}\n\nTranslation A:\n{left}\n\nTranslation B:\n{right}"},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "verdict", "strict": True,
            "schema": {"type": "object", "required": ["winner"],
                       "properties": {"winner": {"enum": ["A", "B", "tie"]}}}}},
    }
    response = client.post(url, json=body, timeout=120)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return str(json.loads(content)["winner"])


def ab_compare(baseline: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]],
               *, judge_url: str, judge_model: str, labels: tuple[str, str],
               source_language: str, target_language: str, seed: int,
               concurrency: int) -> dict[str, Any]:
    """Blinded, order-randomised head-to-head over the units where the arms actually differ.

    Identical renderings are counted but never sent to the judge: they carry no signal and cost
    real GPU time. Which arm occupies slot "A" is decided per unit by a seeded coin flip, so the
    judge cannot learn a position bias, and the verdict is de-blinded afterwards.
    """
    system = JUDGE_SYSTEM.format(source_language=source_language, target_language=target_language)
    rng = random.Random(seed)
    shared = [uid for uid in baseline if uid in candidate
              and injectable(baseline[uid]) and injectable(candidate[uid])]
    identical = [uid for uid in shared
                 if str(baseline[uid]["target"]) == str(candidate[uid]["target"])]
    differing = [uid for uid in shared if uid not in set(identical)]
    baseline_is_a = {uid: rng.random() < 0.5 for uid in differing}

    tally = {labels[0]: 0, labels[1]: 0, "tie": 0, "error": 0}
    verdicts: list[dict[str, Any]] = []
    with httpx.Client() as client:
        url = judge_url.rstrip("/") + "/chat/completions"

        def judge_one(uid: str) -> tuple[str, str]:
            first, second = str(baseline[uid]["target"]), str(candidate[uid]["target"])
            left, right = (first, second) if baseline_is_a[uid] else (second, first)
            try:
                pick = judge_pair(client, url, judge_model, system,
                                  str(baseline[uid].get("source") or ""), left, right)
            except Exception as exc:  # noqa: BLE001 -- one bad verdict must not end the sweep
                return uid, f"error:{exc}"
            if pick == "tie":
                return uid, "tie"
            chose_baseline = (pick == "A") == baseline_is_a[uid]
            return uid, labels[0] if chose_baseline else labels[1]

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for uid, verdict in pool.map(judge_one, differing):
                key = "error" if verdict.startswith("error:") else verdict
                tally[key] += 1
                verdicts.append({"unit_id": uid, "winner": verdict})

    if differing and tally["error"] == len(differing):
        raise EvalError(
            f"every A/B verdict failed; is the judge at {judge_url} serving {judge_model!r}?")
    return {"compared": len(shared), "identical": len(identical), "differing": len(differing),
            **tally, "verdicts": verdicts}


# -- gates ----------------------------------------------------------------------


def check_gates(results: dict[str, Any], *, labels: tuple[str, str],
                rejection_tolerance: float) -> int:
    """Gate on consistency and rejection rate; report A/B quality without gating on it."""
    failures: list[str] = []
    baseline_label, candidate_label = labels

    closer = results["consistency"]["closer"]
    if closer[candidate_label] < closer[baseline_label]:
        failures.append(
            f"consistency: {candidate_label} matched the established rendering "
            f"{closer[candidate_label]} times vs {baseline_label}'s {closer[baseline_label]} -- "
            f"the candidate retriever is supposed to steer toward the corpus, not away from it")

    rates = results["rejection_rate"]
    excess = rates[candidate_label] - rates[baseline_label]
    if excess > rejection_tolerance:
        failures.append(
            f"rejection rate: {candidate_label} failed {rates[candidate_label]:.1%} of units vs "
            f"{baseline_label}'s {rates[baseline_label]:.1%} (+{excess:.1%}, tolerance "
            f"{rejection_tolerance:.1%}) -- the candidate is poisoning generation")

    if failures:
        print("\nFAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("\nall gates passed")
    return 0


# -- entry point ----------------------------------------------------------------


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    labels = (args.baseline, args.candidate)
    if args.baseline == args.candidate:
        raise EvalError("--baseline and --candidate name the same arm; there is nothing to compare")
    args.work.mkdir(parents=True, exist_ok=True)

    journals = {arm: args.work / f"{_slug(arm)}.journal.jsonl" for arm in labels}
    for arm, journal in journals.items():
        if journal.exists() and not args.resume:
            raise EvalError(
                f"{journal} already exists. The CLI resumes by unit id, so running onto it would "
                f"silently translate nothing and compare a stale arm. Pass --resume to continue "
                f"it deliberately, or delete it")
        run_arm(arm, args, journal)

    rows = {arm: read_journal(journal) for arm, journal in journals.items()}
    statuses = {arm: status_counts(records) for arm, records in rows.items()}
    rejection_rate = {
        arm: counts.get("rejected", 0) / max(1, sum(counts.values()))
        for arm, counts in statuses.items()
    }

    results: dict[str, Any] = {
        "baseline": args.baseline, "candidate": args.candidate,
        "queries": str(args.queries), "corpus": str(args.corpus), "seed": args.seed,
        "statuses": statuses, "rejection_rate": rejection_rate,
        "quality": ab_compare(rows[labels[0]], rows[labels[1]], judge_url=args.judge_url,
                              judge_model=args.judge_model, labels=labels,
                              source_language=args.source_language,
                              target_language=args.target_language,
                              seed=args.seed, concurrency=args.judge_concurrency),
        "consistency": consistency(rows[labels[0]], rows[labels[1]], corpus=args.corpus,
                                   embedding_url=args.embedding_url,
                                   embedding_model=args.embedding_model, labels=labels,
                                   ground_truth=(load_ground_truth(args.ground_truth)
                                                 if args.ground_truth else None)),
    }
    return results


def report(results: dict[str, Any], labels: tuple[str, str]) -> None:
    quality, cons = results["quality"], results["consistency"]
    print(f"\n{'':22}{labels[0]:>18}{labels[1]:>18}")
    print(f"{'A/B wins':22}{quality[labels[0]]:>18}{quality[labels[1]]:>18}"
          f"   (tie {quality['tie']}, identical {quality['identical']}, "
          f"error {quality['error']})")
    print(f"{'closer to established':22}{cons['closer'][labels[0]]:>18}"
          f"{cons['closer'][labels[1]]:>18}   (of {cons['measured']} with an established example)")
    print(f"{'mean overlap':22}{cons['mean_overlap'][labels[0]]:>18.3f}"
          f"{cons['mean_overlap'][labels[1]]:>18.3f}")
    print(f"{'rejection rate':22}{results['rejection_rate'][labels[0]]:>17.1%}"
          f"{results['rejection_rate'][labels[1]]:>18.1%}")
    print("\nA/B quality is reported, not gated: the plan states per-line quality may be a wash.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end translation A/B between two "
                                                 "reference-retrieval arms.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--agents", type=Path, required=True,
                        help="base agent config; each arm gets a copy with its own calibrated "
                             "reference_min_score")
    parser.add_argument("--glossary", type=Path, default=None)
    parser.add_argument("--rules", type=Path, default=None)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--baseline", default="embedding",
                        help="retriever, optionally '@<mmr_lambda>' "
                             f"(one of: {', '.join(ARMS)})")
    parser.add_argument("--candidate", default="hybrid+rerank",
                        help="retriever, optionally '@<mmr_lambda>'")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="qwen-local")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--embedding-model", default="local")
    parser.add_argument("--rerank-url", default="")
    parser.add_argument("--judge-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--judge-model", default="qwen-local")
    parser.add_argument("--judge-concurrency", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--source-language", default="ja")
    parser.add_argument("--target-language", default="en")
    parser.add_argument("--source-script", default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rejection-tolerance", type=float, default=0.02,
                        help="how much higher the candidate's rejection rate may be before the "
                             "gate fails")
    parser.add_argument("--resume", action="store_true",
                        help="continue existing arm journals instead of refusing to overwrite")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--ground-truth", type=Path, default=None,
                        help="journal the held-out queries came from; its targets are "
                             "the only sound basis for the consistency measure")
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels = (args.baseline, args.candidate)
    try:
        results = evaluate(args)
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    report(results, labels)
    return check_gates(results, labels=labels, rejection_tolerance=args.rejection_tolerance)


if __name__ == "__main__":
    raise SystemExit(main())
