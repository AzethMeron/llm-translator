"""Tests for ``tools/lib/eval_retrieval.py``, the retrieval-relevance harness.

What makes this harness's numbers trustworthy is the sampling discipline (unique, held-out,
non-duplicate queries) and an independent judge whose failures are excluded rather than counted
as zeroes. Both are pinned here, together with the gates -- including the one deliberately left
ungated, because equal-weight fusion blends its arms instead of taking their maximum.

The judge is an in-memory ``httpx.MockTransport``; the retrievers are stubs. No server, no GPU,
no network.
"""
from __future__ import annotations

import json
import re

import httpx
import pytest

import eval_retrieval as er
from transunit.reference import ReferenceEntry, Retrieved

_REAL_CLIENT = httpx.Client


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    return path


def install_judge(monkeypatch, handler):
    """Point the harness's ``httpx.post`` at an in-memory transport, keeping real response
    semantics (status handling, JSON decoding) so the error paths are genuinely exercised."""
    transport = httpx.MockTransport(handler)

    def post(url, *, json=None, timeout=None):
        with _REAL_CLIENT(transport=transport) as client:
            return client.post(url, json=json, timeout=timeout)

    monkeypatch.setattr(er.httpx, "post", post)


def rating_handler(rating_for):
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        query, retrieved = body["messages"][-1]["content"].split("\n\nB:\n")
        answer = rating_for(query.removeprefix("A:\n"), retrieved)
        if isinstance(answer, tuple):  # (status, body) -- the failure paths
            return httpx.Response(answer[0], json=answer[1])
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({"rating": answer})}}]})
    return handle


class TestLoadEntries:
    def test_rows_without_a_usable_target_are_skipped(self, tmp_path):
        """A journal is a legitimate corpus file, and its pending/rejected rows carry no target;
        loading them as empty references would pollute every retrieval."""
        path = write_jsonl(tmp_path / "c.jsonl", [
            {"source": "a", "target": "A"},
            {"source": "b", "target": ""},
            {"source": "c", "target": "   "},
            {"source": "d"},
            {"source": "e", "target": None},
            {"source": "f", "target": 7},
            {"source": "g", "target": "G"},
        ])
        assert [(e.source, e.target) for e in er.load_entries(path)] == [("a", "A"), ("g", "G")]

    def test_a_missing_source_becomes_an_empty_one(self, tmp_path):
        """Target-only entries are a supported corpus shape, not an error."""
        path = write_jsonl(tmp_path / "c.jsonl", [{"target": "only a target"}])
        assert er.load_entries(path)[0].source == ""

    def test_blank_lines_are_ignored(self, tmp_path):
        path = tmp_path / "c.jsonl"
        path.write_text('\n{"source": "a", "target": "A"}\n\n', encoding="utf-8")
        assert len(er.load_entries(path)) == 1


class TestSampleQueries:
    def _pool(self, tmp_path, sources):
        return write_jsonl(tmp_path / "pool.jsonl", [{"source": s} for s in sources])

    def test_queries_present_in_the_corpus_are_excluded(self, tmp_path):
        """A query that appears verbatim in the corpus retrieves itself at similarity 1.0 and
        inflates every score -- the exact defect this sampling exists to avoid."""
        corpus = [ReferenceEntry("in corpus", "X")]
        pool = self._pool(tmp_path, ["in corpus", "fresh one", "fresh two"])
        assert sorted(er.sample_queries(pool, corpus, count=2, seed=1, script=None)) == [
            "fresh one", "fresh two"]

    def test_duplicate_sources_are_sampled_once(self, tmp_path):
        """Judging the same line twice would weight it double in the precision figure."""
        pool = self._pool(tmp_path, ["same", "same", "same", "other"])
        assert sorted(er.sample_queries(pool, [], count=2, seed=1, script=None)) == [
            "other", "same"]

    def test_blank_and_missing_sources_are_dropped(self, tmp_path):
        pool = write_jsonl(tmp_path / "pool.jsonl",
                           [{"source": ""}, {"source": None}, {}, {"source": "real"}])
        assert er.sample_queries(pool, [], count=1, seed=1, script=None) == ["real"]

    def test_blank_lines_in_the_pool_are_skipped(self, tmp_path):
        """A JSONL file may end with, or contain, empty lines; json.loads would raise on them."""
        pool = tmp_path / "pool.jsonl"
        pool.write_text('\n  \n{"source": "real"}\n\n', encoding="utf-8")
        assert er.sample_queries(pool, [], count=1, seed=1, script=None) == ["real"]

    def test_the_script_filter_applies(self, tmp_path):
        pool = self._pool(tmp_path, ["日本語", "ascii", "12345"])
        script = re.compile(r"[぀-ヿ㐀-鿿]")
        assert er.sample_queries(pool, [], count=1, seed=1, script=script) == ["日本語"]

    def test_the_same_seed_samples_the_same_queries(self, tmp_path):
        """A rerun must compare like with like, or two measurements cannot be put side by side."""
        pool = self._pool(tmp_path, [f"line {i}" for i in range(60)])
        first = er.sample_queries(pool, [], count=15, seed=2026, script=None)
        second = er.sample_queries(pool, [], count=15, seed=2026, script=None)
        assert first == second
        assert er.sample_queries(pool, [], count=15, seed=7, script=None) != first

    def test_too_small_a_pool_raises_instead_of_measuring_a_smaller_sample(self, tmp_path):
        """Silently returning fewer queries is how an evaluation once ran at 60% of its sample."""
        pool = self._pool(tmp_path, ["one", "two"])
        with pytest.raises(er.EvalError, match="only 2 unique"):
            er.sample_queries(pool, [], count=5, seed=1, script=None)

    def test_the_shortfall_counts_only_eligible_queries(self, tmp_path):
        corpus = [ReferenceEntry("dup", "X")]
        pool = self._pool(tmp_path, ["dup", "dup", "one"])
        with pytest.raises(er.EvalError, match="only 1 unique"):
            er.sample_queries(pool, corpus, count=2, seed=1, script=None)


class TestJudgeBatch:
    def test_ratings_are_parsed(self, monkeypatch):
        install_judge(monkeypatch, rating_handler(lambda query, retrieved: int(retrieved)))
        pairs = [("q", "0"), ("q", "1"), ("q", "2")]
        assert er.judge_batch(pairs, base_url="http://j/v1", model="m", concurrency=1) == [0, 1, 2]

    def test_order_is_preserved_under_concurrency(self, monkeypatch):
        """The ratings are zipped back onto their queries by position, so a reordering would
        attribute every judgement to the wrong line."""
        install_judge(monkeypatch, rating_handler(lambda query, retrieved: int(retrieved)))
        pairs = [("q", str(i % 3)) for i in range(30)]
        assert er.judge_batch(pairs, base_url="http://j/v1", model="m",
                              concurrency=4) == [i % 3 for i in range(30)]

    def test_a_server_error_is_minus_one_and_does_not_abort_the_sweep(self, monkeypatch):
        install_judge(monkeypatch, rating_handler(
            lambda query, retrieved: (500, {"error": "boom"}) if retrieved == "bad" else 2))
        assert er.judge_batch([("q", "bad"), ("q", "good")], base_url="http://j/v1",
                              model="m", concurrency=1) == [-1, 2]

    def test_an_unparseable_verdict_is_minus_one(self, monkeypatch):
        install_judge(monkeypatch, lambda request: httpx.Response(
            200, json={"choices": [{"message": {"content": "not json"}}]}))
        assert er.judge_batch([("q", "r")], base_url="http://j/v1", model="m",
                              concurrency=1) == [-1]

    def test_a_verdict_without_a_rating_field_is_minus_one(self, monkeypatch):
        install_judge(monkeypatch, lambda request: httpx.Response(
            200, json={"choices": [{"message": {"content": '{"verdict": 2}'}}]}))
        assert er.judge_batch([("q", "r")], base_url="http://j/v1", model="m",
                              concurrency=1) == [-1]

    def test_the_judge_url_is_built_from_the_base_url(self, monkeypatch):
        seen = []
        install_judge(monkeypatch, lambda request: (seen.append(str(request.url)), httpx.Response(
            200, json={"choices": [{"message": {"content": '{"rating": 1}'}}]}))[1])
        er.judge_batch([("q", "r")], base_url="http://j/v1/", model="m", concurrency=1)
        assert seen == ["http://j/v1/chat/completions"]


class RetrieverStub:
    """A retriever whose hits are scripted: ``answer_at_most`` of the queries it is asked get one.

    ``answer_at_most`` rather than a fixed query list, because the sampled query set is seeded and
    a test should not have to predict which lines it drew.
    """

    def __init__(self, *, score=0.8, answer_at_most=None):
        self.score = score
        self.answer_at_most = answer_at_most
        self.asked: list[str] = []
        self.closed = False

    def by_source(self, query, *, k, min_score):
        self.asked.append(query)
        answered = len(self.asked) if self.answer_at_most is None else self.answer_at_most
        if len(self.asked) > answered or self.score < min_score:
            return []
        return [Retrieved(ReferenceEntry(f"hit for {query}", "target"), self.score)][:k]

    def close(self):
        self.closed = True


def variant(precision: float, *, retrieved: int = 10) -> dict:
    clearly = round(precision * retrieved)
    return {"queries": retrieved, "retrieved": retrieved, "judged": retrieved,
            "clearly_relevant": clearly, "per_query": {}, "precision": precision,
            "yield": precision, "ratings": {}}


class TestCheckGates:
    def _results(self, **precisions):
        return {name: variant(value) for name, value in precisions.items()}

    def test_healthy_variants_pass(self, capsys):
        code = er.check_gates(self._results(lexical=0.5, embedding=0.6, hybrid=0.55),
                              floor=0.35, tolerance=0.05)
        assert code == 0
        assert "bare hybrid vs its best single arm" in capsys.readouterr().out

    def test_a_variant_below_the_absolute_floor_fails(self, capsys):
        code = er.check_gates(self._results(lexical=0.10, embedding=0.6, hybrid=0.55),
                              floor=0.35, tolerance=0.05)
        assert code == 1
        assert "mostly irrelevant" in capsys.readouterr().err

    def test_bare_hybrid_losing_to_its_best_arm_is_reported_but_not_gated(self, capsys):
        """Equal-weight RRF blends its arms rather than taking their maximum; gating on it would
        gate on a documented property of the fusion, not a regression."""
        code = er.check_gates(self._results(lexical=0.9, embedding=0.4, hybrid=0.5),
                              floor=0.35, tolerance=0.05)
        assert code == 0
        assert "-40%" in capsys.readouterr().out

    def test_a_missing_variant_raises_rather_than_passing_by_omission(self):
        """A variant that failed to run must not be silently treated as "no failures found"."""
        with pytest.raises(er.EvalError, match="'hybrid' variant did not run"):
            er.check_gates(self._results(lexical=0.5, embedding=0.6), floor=0.35, tolerance=0.05)

    def test_the_reranked_configuration_must_not_fall_below_the_best_single_arm(self, capsys):
        """It is the recommended configuration and costs three servers; if a single arm beats it
        there is no reason to pay for it."""
        results = self._results(lexical=0.8, embedding=0.4, hybrid=0.75)
        results["hybrid+rerank"] = variant(0.60)
        assert er.check_gates(results, floor=0.35, tolerance=0.05) == 1
        assert "not paying for itself" in capsys.readouterr().err

    def test_the_reranked_configuration_must_not_fall_below_plain_hybrid(self, capsys):
        results = self._results(lexical=0.5, embedding=0.5, hybrid=0.8)
        results["hybrid+rerank"] = variant(0.50)
        assert er.check_gates(results, floor=0.35, tolerance=0.05) == 1
        assert "below plain hybrid" in capsys.readouterr().err

    def test_a_shortfall_inside_the_tolerance_is_accepted(self):
        results = self._results(lexical=0.60, embedding=0.50, hybrid=0.60)
        results["hybrid+rerank"] = variant(0.55)
        assert er.check_gates(results, floor=0.35, tolerance=0.05) == 0


class TestBuildRetrievers:
    def test_the_rerank_variant_is_only_built_when_a_rerank_url_is_given(self, monkeypatch):
        """Otherwise the harness would report a "hybrid+rerank" row produced without a reranker."""
        monkeypatch.setattr(er, "LexicalRetriever", lambda *a, **k: "lexical")
        monkeypatch.setattr(er, "EmbeddingRetriever", lambda *a, **k: "embedding")
        monkeypatch.setattr(er, "HybridRetriever", lambda *a, **k: k)
        without = er.build_retrievers([], embedding_url="http://e", rerank_url="",
                                      lexical_floor=0.3, embedding_floor=0.55, pool=40)
        assert set(without) == {"lexical", "embedding", "hybrid"}
        with_rerank = er.build_retrievers([], embedding_url="http://e", rerank_url="http://r",
                                          lexical_floor=0.3, embedding_floor=0.55, pool=40)
        assert with_rerank["hybrid+rerank"]["rerank_base_url"] == "http://r"
        assert "rerank_base_url" not in with_rerank["hybrid"]


class TestEvaluate:
    """The whole sweep with the retrievers stubbed: retrieval, judging, reporting and gating."""

    def _args(self, tmp_path, **overrides):
        corpus = write_jsonl(tmp_path / "corpus.jsonl",
                             [{"source": f"corpus {i}", "target": f"T{i}"} for i in range(5)])
        pool = write_jsonl(tmp_path / "pool.jsonl",
                           [{"source": f"query {i}"} for i in range(10)])
        argv = ["--corpus", str(corpus), "--queries-from", str(pool), "--queries", "4"]
        for key, value in overrides.items():
            argv += [f"--{key.replace('_', '-')}", str(value)]
        return er.build_parser().parse_args(argv)

    def _install(self, monkeypatch, retrievers):
        monkeypatch.setattr(er, "build_retrievers", lambda *a, **k: retrievers)

    def test_precision_and_yield_are_computed_over_the_right_denominators(self, tmp_path,
                                                                         monkeypatch, capsys):
        """precision is over what was SHOWN, yield over every query: a retriever that stays
        silent must not be flattered by a high precision on a single hit."""
        silent = RetrieverStub(answer_at_most=1)   # answers only 1 of the 4 queries
        loud = RetrieverStub()
        self._install(monkeypatch, {"lexical": silent, "embedding": loud, "hybrid": loud})
        install_judge(monkeypatch, rating_handler(lambda query, retrieved: 2))
        out = tmp_path / "results.json"
        code = er.evaluate(self._args(tmp_path, out=out))
        results = json.loads(out.read_text(encoding="utf-8"))
        assert results["lexical"]["precision"] == 1.0
        assert results["lexical"]["yield"] == 0.25
        assert results["embedding"]["yield"] == 1.0
        assert code == 0
        assert "retrieved something on 1/4" in capsys.readouterr().out

    def test_failed_judgements_are_excluded_rather_than_counted_as_zero(self, tmp_path,
                                                                       monkeypatch):
        """Counting a dead request as "irrelevant" would drag every precision figure down."""
        loud = RetrieverStub()
        self._install(monkeypatch, dict.fromkeys(("lexical", "embedding", "hybrid"), loud))
        seen = {"n": 0}

        def rating(query, retrieved):
            seen["n"] += 1
            return (503, {"error": "down"}) if seen["n"] % 2 else 2

        install_judge(monkeypatch, rating_handler(rating))
        out = tmp_path / "results.json"
        er.evaluate(self._args(tmp_path, out=out))
        results = json.loads(out.read_text(encoding="utf-8"))
        assert results["lexical"]["judged"] < results["lexical"]["retrieved"]
        assert results["lexical"]["precision"] == 1.0

    def test_a_variant_that_retrieved_nothing_is_diagnosed_as_retrieval_not_a_dead_judge(
            self, tmp_path, monkeypatch):
        """Regression: a variant whose floor rejected every hit sent the judge nothing, and the
        empty rating list was reported as "every relevance judgement failed; is the judge server
        up?" -- pointing the reader at a healthy server while the real cause was --min-score."""
        mute = RetrieverStub(answer_at_most=0)
        self._install(monkeypatch, dict.fromkeys(("lexical", "embedding", "hybrid"), mute))
        install_judge(monkeypatch, rating_handler(lambda query, retrieved: 2))
        with pytest.raises(er.EvalError, match="retrieved nothing on any of the 4 queries"):
            er.evaluate(self._args(tmp_path))

    def test_every_judgement_failing_raises(self, tmp_path, monkeypatch):
        """A dead judge otherwise publishes a precision of 0% as if retrieval were broken."""
        loud = RetrieverStub()
        self._install(monkeypatch, dict.fromkeys(("lexical", "embedding", "hybrid"), loud))
        install_judge(monkeypatch, rating_handler(lambda query, retrieved: (500, {})))
        with pytest.raises(er.EvalError, match="every relevance judgement failed"):
            er.evaluate(self._args(tmp_path))

    def test_an_empty_corpus_raises(self, tmp_path):
        corpus = write_jsonl(tmp_path / "corpus.jsonl", [{"source": "a"}])
        pool = write_jsonl(tmp_path / "pool.jsonl", [{"source": "q"}])
        args = er.build_parser().parse_args(
            ["--corpus", str(corpus), "--queries-from", str(pool), "--queries", "1"])
        with pytest.raises(er.EvalError, match="no usable reference entries"):
            er.evaluate(args)

    def test_the_retrievers_are_released_even_when_the_sweep_fails(self, tmp_path, monkeypatch):
        """Each retriever may hold an HTTP connection; a failed sweep must not leak them. The
        stdlib-only lexical retriever has nothing to close, so a missing close() is not an
        error either."""
        class CloselessRetriever:
            def by_source(self, query, *, k, min_score):
                return [Retrieved(ReferenceEntry("hit", "target"), 0.8)]

        stubs = {name: RetrieverStub() for name in ("embedding", "hybrid")}
        stubs["lexical"] = CloselessRetriever()
        self._install(monkeypatch, stubs)
        install_judge(monkeypatch, rating_handler(lambda query, retrieved: (500, {})))
        with pytest.raises(er.EvalError):
            er.evaluate(self._args(tmp_path))
        assert stubs["embedding"].closed and stubs["hybrid"].closed

    def test_main_turns_an_eval_error_into_exit_two(self, tmp_path, capsys):
        corpus = write_jsonl(tmp_path / "corpus.jsonl", [{"source": "a", "target": "A"}])
        pool = write_jsonl(tmp_path / "pool.jsonl", [{"source": "q"}])
        code = er.main(["--corpus", str(corpus), "--queries-from", str(pool), "--queries", "9"])
        assert code == 2
        assert "eval_retrieval:" in capsys.readouterr().err

    def test_main_returns_the_gate_status_of_a_healthy_run(self, tmp_path, monkeypatch):
        loud = RetrieverStub()
        self._install(monkeypatch, dict.fromkeys(("lexical", "embedding", "hybrid"), loud))
        install_judge(monkeypatch, rating_handler(lambda query, retrieved: 2))
        corpus = write_jsonl(tmp_path / "corpus.jsonl",
                             [{"source": f"corpus {i}", "target": f"T{i}"} for i in range(5)])
        pool = write_jsonl(tmp_path / "pool.jsonl", [{"source": f"query {i}"} for i in range(10)])
        assert er.main(["--corpus", str(corpus), "--queries-from", str(pool),
                        "--queries", "4"]) == 0
