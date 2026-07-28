"""Tests for ``tools/lib/eval_translation.py``, the end-to-end translation A/B harness.

This is the harness whose output gets published as "arm X beat arm Y". Nothing in it crashes when
it is wrong -- it just reports the wrong arm -- so the cases here are built to make a systematic
error impossible to miss:

* :class:`TestAbCompareDeblinding` asserts the de-blinding **per unit**, against the slot the
  transport actually saw, never against a total. An inverted de-blind would flip every published
  A/B number and no total could reveal it.
* The judge is a stub over ``httpx.MockTransport``; the retriever used by the consistency measure
  is a stub class. No server, no GPU, no network.
"""
from __future__ import annotations

import argparse
import json
import re

import httpx
import pytest

import eval_translation as et
from transunit.reference import ReferenceEntry, Retrieved

VALID_AGENTS = (
    "[translate]\n"
    'instructions = "Translate from {source_language} into {target_language}."\n'
    '[[reviewer]]\nid = "style"\ninstructions = "Check the register."\n'
)
BASE_CONTEXT = "[context]\nreference_examples = 3\nreference_min_score = 0.11\n"

_REAL_CLIENT = httpx.Client
"""Captured before any patching: a second install would otherwise wrap the first stub."""


def unit(uid: str, target: str, *, status: str = "translated", source: str | None = None) -> dict:
    return {"unit_id": uid, "source": source if source is not None else f"src {uid}",
            "target": target, "status": status}


def journal(*records: dict) -> dict[str, dict]:
    return {record["unit_id"]: record for record in records}


# -- the judge stub -------------------------------------------------------------


class JudgeStub:
    """A scripted ``/chat/completions`` judge that records what each request actually showed it.

    ``slot_a`` is the *observed* blinding -- the text the transport really saw in position A --
    so a test can assert the de-blinded winner against reality rather than against a re-derivation
    of the harness's own coin flips.
    """

    _PARSE = re.compile(r"Source:\n(?P<source>.*)\n\nTranslation A:\n(?P<a>.*)"
                        r"\n\nTranslation B:\n(?P<b>.*)", re.DOTALL)

    def __init__(self, decide) -> None:
        self.decide = decide
        self.slot_a: dict[str, str] = {}
        self.slot_b: dict[str, str] = {}
        self.calls: list[str] = []
        self.systems: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.systems.append(body["messages"][0]["content"])
        match = self._PARSE.match(body["messages"][-1]["content"])
        assert match is not None, "the judge prompt no longer has the A/B shape this stub reads"
        source, left, right = match["source"], match["a"], match["b"]
        self.calls.append(source)
        self.slot_a[source], self.slot_b[source] = left, right
        winner = self.decide(left, right)
        if isinstance(winner, int):  # an HTTP status: the failure path
            return httpx.Response(winner, json={"error": "boom"})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({"winner": winner})}}]})

    def install(self, monkeypatch) -> JudgeStub:
        install_transport(monkeypatch, httpx.MockTransport(self.handle))
        return self


def install_transport(monkeypatch, transport) -> None:
    """Route ``ab_compare``'s internally-constructed client at an in-memory transport.

    The real class is captured first: the patched name would otherwise resolve to the factory
    itself and recurse.
    """
    monkeypatch.setattr(et.httpx, "Client", lambda *a, **k: _REAL_CLIENT(transport=transport))


def compare(baseline, candidate, stub, *, seed=2026, concurrency=1, labels=("base", "cand")):
    return et.ab_compare(baseline, candidate, judge_url="http://judge/v1", judge_model="m",
                         labels=labels, source_language="ja", target_language="en",
                         seed=seed, concurrency=concurrency)


# -- ab_compare -----------------------------------------------------------------


class TestAbCompareDeblinding:
    """The blinding is a coin flip per unit; the de-blinding must undo exactly that flip."""

    def _arms(self, count: int = 24):
        baseline = journal(*(unit(f"u{i}", f"BASE-{i}") for i in range(count)))
        candidate = journal(*(unit(f"u{i}", f"CAND-{i}") for i in range(count)))
        return baseline, candidate

    def _owner(self, stub: JudgeStub, uid: str) -> str:
        """Which arm the transport actually saw in slot A for ``uid`` -- ground truth."""
        return "base" if stub.slot_a[f"src {uid}"].startswith("BASE") else "cand"

    def test_a_judge_that_always_picks_a_credits_whoever_held_slot_a(self, monkeypatch):
        """If the de-blind were inverted this passes every total but fails every unit: with a
        random blinding, "always A" splits roughly 50/50 between the arms either way."""
        baseline, candidate = self._arms()
        stub = JudgeStub(lambda left, right: "A").install(monkeypatch)
        result = compare(baseline, candidate, stub)
        for verdict in result["verdicts"]:
            assert verdict["winner"] == self._owner(stub, verdict["unit_id"])
        assert result["base"] + result["cand"] == 24
        assert {self._owner(stub, v["unit_id"]) for v in result["verdicts"]} == {"base", "cand"}, \
            "the blinding never varied, so this proves nothing"

    def test_a_judge_that_always_picks_b_credits_whoever_held_slot_b(self, monkeypatch):
        baseline, candidate = self._arms()
        stub = JudgeStub(lambda left, right: "B").install(monkeypatch)
        result = compare(baseline, candidate, stub)
        for verdict in result["verdicts"]:
            loser = self._owner(stub, verdict["unit_id"])
            assert verdict["winner"] == ("cand" if loser == "base" else "base")
        assert result["base"] + result["cand"] == 24

    def test_a_marker_following_one_arm_is_never_attributed_to_the_other(self, monkeypatch):
        """The strongest form: the judge picks whichever slot holds the marker, so the winner is
        known independently of position. A systematic inversion cannot survive this."""
        baseline = journal(*(unit(f"u{i}", f"plain-{i}") for i in range(20)))
        candidate = journal(*(unit(f"u{i}", f"MARKER-{i}") for i in range(20)))
        stub = JudgeStub(lambda left, right: "A" if "MARKER" in left else "B").install(monkeypatch)
        result = compare(baseline, candidate, stub)
        assert result["cand"] == 20
        assert result["base"] == 0
        assert all(v["winner"] == "cand" for v in result["verdicts"])

    def test_the_marker_test_run_the_other_way_round(self, monkeypatch):
        """Same probe with the arms swapped: an inversion would show up as a clean 20-0 the wrong
        way, which the previous test alone could be made to pass by accident."""
        baseline = journal(*(unit(f"u{i}", f"MARKER-{i}") for i in range(20)))
        candidate = journal(*(unit(f"u{i}", f"plain-{i}") for i in range(20)))
        stub = JudgeStub(lambda left, right: "A" if "MARKER" in left else "B").install(monkeypatch)
        result = compare(baseline, candidate, stub)
        assert (result["base"], result["cand"]) == (20, 0)

    def test_ties_are_counted_as_ties_for_neither_arm(self, monkeypatch):
        baseline = journal(*(unit(f"u{i}", f"a{i}") for i in range(6)))
        candidate = journal(*(unit(f"u{i}", f"b{i}") for i in range(6)))
        stub = JudgeStub(lambda left, right: "tie").install(monkeypatch)
        result = compare(baseline, candidate, stub)
        assert result["tie"] == 6
        assert result["base"] == result["cand"] == 0

    def test_the_blinding_is_reproducible_across_runs(self, monkeypatch):
        """A rerun must ask the judge the same questions in the same order, or two runs of the
        same evaluation are not comparable."""
        baseline, candidate = self._arms()
        first = JudgeStub(lambda left, right: "A").install(monkeypatch)
        compare(baseline, candidate, first)
        second = JudgeStub(lambda left, right: "A").install(monkeypatch)
        compare(baseline, candidate, second)
        assert first.slot_a == second.slot_a
        assert first.calls == second.calls

    def test_a_different_seed_gives_a_different_blinding(self, monkeypatch):
        baseline, candidate = self._arms()
        first = JudgeStub(lambda left, right: "A").install(monkeypatch)
        compare(baseline, candidate, first, seed=1)
        second = JudgeStub(lambda left, right: "A").install(monkeypatch)
        compare(baseline, candidate, second, seed=99)
        assert first.slot_a != second.slot_a

    def test_the_judge_is_told_the_language_pair(self, monkeypatch):
        baseline, candidate = self._arms(2)
        stub = JudgeStub(lambda left, right: "tie").install(monkeypatch)
        compare(baseline, candidate, stub)
        assert "ja" in stub.systems[0] and "en" in stub.systems[0]


class TestAbCompareSelection:
    def test_identical_renderings_are_counted_and_never_judged(self, monkeypatch):
        """Sending them would cost real GPU time for a verdict that carries no signal."""
        baseline = journal(unit("same", "identical text"), unit("diff", "one"))
        candidate = journal(unit("same", "identical text"), unit("diff", "two"))
        stub = JudgeStub(lambda left, right: "A").install(monkeypatch)
        result = compare(baseline, candidate, stub)
        assert result["identical"] == 1
        assert result["differing"] == 1
        assert stub.calls == ["src diff"], "an identical unit was sent to the judge"

    def test_a_run_where_every_unit_is_identical_touches_the_transport_at_all(self, monkeypatch):
        baseline = journal(unit("a", "x"), unit("b", "y"))
        candidate = journal(unit("a", "x"), unit("b", "y"))
        stub = JudgeStub(lambda left, right: 500).install(monkeypatch)
        result = compare(baseline, candidate, stub)
        assert stub.calls == []
        assert (result["identical"], result["differing"], result["error"]) == (2, 0, 0)

    def test_only_units_both_arms_actually_translated_are_compared(self, monkeypatch):
        """A rejected or empty rendering is text the engine already refused to ship; scoring it
        would measure output no reader will ever see."""
        baseline = journal(unit("ok", "a"), unit("rej", "b", status="rejected"),
                           unit("empty", ""), unit("only-base", "c"))
        candidate = journal(unit("ok", "z"), unit("rej", "y"), unit("empty", "x"))
        stub = JudgeStub(lambda left, right: "tie").install(monkeypatch)
        result = compare(baseline, candidate, stub)
        assert result["compared"] == 1
        assert stub.calls == ["src ok"]

    def test_pending_and_verified_statuses_are_treated_by_injectability(self, monkeypatch):
        baseline = journal(unit("v", "a", status="verified"), unit("p", "a", status="pending"))
        candidate = journal(unit("v", "b", status="verified"), unit("p", "b", status="pending"))
        stub = JudgeStub(lambda left, right: "tie").install(monkeypatch)
        assert compare(baseline, candidate, stub)["compared"] == 1


class TestAbCompareFailures:
    def test_one_failed_verdict_is_counted_and_does_not_abort_the_sweep(self, monkeypatch):
        """A single dead request must not throw away the other twenty-odd verdicts."""
        baseline = journal(*(unit(f"u{i}", f"a{i}") for i in range(4)))
        candidate = journal(*(unit(f"u{i}", f"b{i}") for i in range(4)))
        stub = JudgeStub(lambda left, right: 500 if left.endswith("0") or right.endswith("0")
                         else "tie").install(monkeypatch)
        result = compare(baseline, candidate, stub)
        assert result["error"] == 1
        assert result["tie"] == 3
        assert any(v["winner"].startswith("error:") for v in result["verdicts"])

    def test_a_malformed_verdict_body_counts_as_an_error(self, monkeypatch):
        baseline = journal(unit("a", "one"))
        candidate = journal(unit("a", "two"))
        install_transport(monkeypatch, httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "{"}}]})))
        with pytest.raises(et.EvalError, match="every A/B verdict failed"):
            compare(baseline, candidate, None)

    def test_every_verdict_failing_raises_rather_than_reporting_a_zero_zero_draw(self,
                                                                                monkeypatch):
        """A dead judge otherwise publishes "0 wins each" as if the arms were equal."""
        baseline = journal(*(unit(f"u{i}", f"a{i}") for i in range(3)))
        candidate = journal(*(unit(f"u{i}", f"b{i}") for i in range(3)))
        JudgeStub(lambda left, right: 503).install(monkeypatch)
        with pytest.raises(et.EvalError, match="is the judge at"):
            compare(baseline, candidate, None)

    def test_concurrent_judging_produces_the_same_tally(self, monkeypatch):
        """The sweep is threaded; the tally must not depend on completion order."""
        baseline = journal(*(unit(f"u{i}", f"a{i}") for i in range(12)))
        candidate = journal(*(unit(f"u{i}", f"MARKER{i}") for i in range(12)))
        stub = JudgeStub(lambda left, right: "A" if "MARKER" in left else "B").install(monkeypatch)
        result = compare(baseline, candidate, stub, concurrency=4)
        assert (result["cand"], result["base"], result["error"]) == (12, 0, 0)


# -- consistency ----------------------------------------------------------------


class RetrieverStub:
    """Stands in for :class:`EmbeddingRetriever`: canned hits per source, and a close() flag."""

    def __init__(self, hits: dict[str, tuple[str, float]]) -> None:
        self.hits = hits
        self.closed = False
        self.asked: list[tuple[str, float]] = []

    def by_source(self, source: str, *, k: int, min_score: float):
        self.asked.append((source, min_score))
        found = self.hits.get(source)
        if found is None or found[1] < min_score:
            return []
        return [Retrieved(ReferenceEntry(source, found[0]), found[1])][:k]

    def close(self) -> None:
        self.closed = True


def install_retriever(monkeypatch, stub):
    import translator.retrieval.embedding as embedding

    monkeypatch.setattr(embedding, "EmbeddingRetriever", lambda *a, **k: stub)
    return stub


def corpus_file(tmp_path, entries=(("src a", "established a"),)):
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps({"source": s, "target": t}) + "\n" for s, t in entries),
                    encoding="utf-8")
    return path


def measure(baseline, candidate, corpus, labels=("base", "cand")):
    return et.consistency(baseline, candidate, corpus=corpus, embedding_url="http://x/v1",
                          embedding_model="m", labels=labels)


class TestConsistency:
    def test_the_arm_matching_the_established_rendering_is_credited(self, tmp_path, monkeypatch):
        stub = install_retriever(monkeypatch,
                                 RetrieverStub({"src a": ("the established one", 0.9)}))
        baseline = journal(unit("a", "the established one", source="src a"))
        candidate = journal(unit("a", "something entirely different", source="src a"))
        result = measure(baseline, candidate, corpus_file(tmp_path))
        assert result["measured"] == 1
        assert result["closer"] == {"base": 1, "cand": 0, "tie": 0}
        assert result["mean_overlap"]["base"] > result["mean_overlap"]["cand"]
        assert stub.closed, "the retriever's server connection was left open"

    def test_a_difference_inside_the_dead_band_is_a_tie(self, tmp_path, monkeypatch):
        """Floating-point noise on effectively-equal renderings would otherwise be split into
        wins and reported as a signal."""
        install_retriever(monkeypatch, RetrieverStub({"src a": ("established text", 0.9)}))
        baseline = journal(unit("a", "established text", source="src a"))
        candidate = journal(unit("a", "established text", source="src a"))
        result = measure(baseline, candidate, corpus_file(tmp_path))
        assert result["closer"]["tie"] == 1

    def test_a_hair_of_difference_below_the_dead_band_is_still_a_tie(self, tmp_path, monkeypatch):
        established = " ".join(f"word{i}" for i in range(60))
        install_retriever(monkeypatch, RetrieverStub({"src a": (established, 0.9)}))
        baseline = journal(unit("a", established, source="src a"))
        candidate = journal(unit("a", established + "!", source="src a"))
        overlap_gap = abs(et.jaccard(established, established)
                          - et.jaccard(established, established + "!"))
        assert 0 < overlap_gap < et.CONSISTENCY_DEADBAND, "this fixture no longer probes the band"
        assert measure(baseline, candidate, corpus_file(tmp_path))["closer"]["tie"] == 1

    def test_a_unit_with_no_established_example_is_excluded_not_counted_as_a_miss(
            self, tmp_path, monkeypatch):
        """Counting it would punish an arm for a line the corpus never covered."""
        install_retriever(monkeypatch, RetrieverStub({
            "src a": ("established a", 0.9),
            "src b": ("weak", et.CONSISTENCY_FLOOR - 0.01),  # below the floor: nothing established
        }))
        baseline = journal(unit("a", "established a", source="src a"),
                           unit("b", "whatever", source="src b"))
        candidate = journal(unit("a", "other", source="src a"),
                            unit("b", "whatever else", source="src b"))
        result = measure(baseline, candidate, corpus_file(tmp_path))
        assert result["measured"] == 1
        assert sum(result["closer"].values()) == 1

    def test_the_established_example_is_looked_up_at_the_calibrated_floor(self, tmp_path,
                                                                         monkeypatch):
        stub = install_retriever(monkeypatch, RetrieverStub({"src a": ("established a", 0.9)}))
        measure(journal(unit("a", "x", source="src a")),
                journal(unit("a", "y", source="src a")), corpus_file(tmp_path))
        assert stub.asked == [("src a", et.CONSISTENCY_FLOOR)]

    def test_no_measurable_unit_raises_instead_of_dividing_by_zero(self, tmp_path, monkeypatch):
        """mean_overlap divides by `measured`; a silent zero would be a crash or a lie."""
        stub = install_retriever(monkeypatch, RetrieverStub({}))
        baseline = journal(unit("a", "x", source="src a"))
        candidate = journal(unit("a", "y", source="src a"))
        with pytest.raises(et.EvalError, match="no unit had an established rendering"):
            measure(baseline, candidate, corpus_file(tmp_path))
        assert stub.closed, "the retriever must be released even on the failure path"

    def test_units_only_one_arm_translated_are_skipped(self, tmp_path, monkeypatch):
        install_retriever(monkeypatch, RetrieverStub({"src a": ("established a", 0.9)}))
        baseline = journal(unit("a", "established a", source="src a"),
                           unit("b", "x", source="src a", status="rejected"))
        candidate = journal(unit("a", "established a", source="src a"),
                            unit("b", "y", source="src a"))
        assert measure(baseline, candidate, corpus_file(tmp_path))["measured"] == 1

    def test_an_empty_corpus_raises(self, tmp_path, monkeypatch):
        install_retriever(monkeypatch, RetrieverStub({}))
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(et.EvalError, match="yielded no entries"):
            measure(journal(unit("a", "x")), journal(unit("a", "y")), empty)


class TestTrigramOverlap:
    def test_identical_text_scores_one(self):
        assert et.jaccard("hello world", "hello world") == 1.0

    def test_disjoint_text_scores_zero(self):
        assert et.jaccard("aaaa", "bbbb") == 0.0

    def test_text_shorter_than_a_trigram_still_compares(self):
        """A two-character rendering must not collapse to an empty set and a 0/0 division."""
        assert et.jaccard("ab", "ab") == 1.0
        assert et.jaccard("ab", "cd") == 0.0
        assert et.trigrams("ab") == {"ab"}

    def test_two_empty_strings_do_not_divide_by_zero(self):
        assert et.jaccard("", "") == 1.0

    def test_comparison_is_case_insensitive(self):
        assert et.jaccard("Hello", "hello") == 1.0


# -- arm configuration ----------------------------------------------------------


def base_config(tmp_path, text=VALID_AGENTS + BASE_CONTEXT):
    path = tmp_path / "agents.toml"
    path.write_text(text, encoding="utf-8")
    return path


class TestAgentConfigFor:
    @pytest.mark.parametrize("arm", ["lexical", "embedding", "hybrid", "hybrid+rerank"])
    def test_each_retrieval_arm_gets_its_own_calibrated_floor(self, tmp_path, arm):
        """The floors are not interchangeable -- each is on that arm's own score scale -- so
        sharing one config would silently change what is retrieved."""
        written = et.agent_config_for(arm, base_config(tmp_path), tmp_path / "out" / "a.toml")
        from translator.roles import AgentSet

        assert AgentSet.load(written).context.reference_min_score == et.ARM_FLOORS[arm]
        assert AgentSet.load(written).context.reference_examples == 3

    def test_the_none_arm_zeroes_both_reference_counts(self, tmp_path):
        """Leaving reference_revision_examples on would quietly re-introduce retrieval during
        revision, and the "unaided" baseline would not be unaided."""
        base = base_config(tmp_path, VALID_AGENTS + BASE_CONTEXT
                           + "reference_revision_examples = 2\n")
        written = et.agent_config_for("none", base, tmp_path / "none.toml")
        from translator.roles import AgentSet

        loaded = AgentSet.load(written)
        assert loaded.context.reference_examples == 0
        assert loaded.context.reference_revision_examples == 0

    def test_a_none_arm_whose_counts_could_not_be_zeroed_is_refused(self, tmp_path):
        """The textual edit only matches a `key = value` line. Written as an inline table it
        matches nothing, the arm would silently retrieve, and the "unaided" baseline would be
        anything but -- which is what re-loading the written file catches."""
        base = base_config(tmp_path, "context = { reference_examples = 2 }\n" + VALID_AGENTS)
        with pytest.raises(et.EvalError, match="still has retrieval enabled"):
            et.agent_config_for("none", base, tmp_path / "none.toml")

    def test_the_none_arm_accepts_a_config_with_no_context_section(self, tmp_path):
        """Retrieval is off by default, so there is nothing to zero -- and nothing to fail on."""
        written = et.agent_config_for("none", base_config(tmp_path, VALID_AGENTS),
                                      tmp_path / "none.toml")
        from translator.roles import AgentSet

        assert AgentSet.load(written).context.reference_examples == 0

    def test_a_missing_reference_min_score_is_inserted_under_the_existing_context_header(
            self, tmp_path):
        base = base_config(tmp_path, VALID_AGENTS + "[context]\nreference_examples = 2\n")
        written = et.agent_config_for("embedding", base, tmp_path / "e.toml")
        from translator.roles import AgentSet

        loaded = AgentSet.load(written)
        assert loaded.context.reference_min_score == et.ARM_FLOORS["embedding"]
        assert loaded.context.reference_examples == 2

    def test_a_base_with_retrieval_off_is_refused_for_a_retrieval_arm(self, tmp_path):
        """Both arms would then translate identically and the whole A/B would be a wash by
        construction -- the most expensive way possible to measure nothing."""
        base = base_config(tmp_path, VALID_AGENTS + "[context]\nreference_examples = 0\n")
        with pytest.raises(et.EvalError, match="reference retrieval is OFF"):
            et.agent_config_for("lexical", base, tmp_path / "l.toml")

    def test_a_base_with_no_context_section_is_refused_for_a_retrieval_arm(self, tmp_path):
        """The floor is appended in a fresh [context] section, but a config that never mentioned
        reference_examples has retrieval off, so the guard is what the caller actually sees."""
        with pytest.raises(et.EvalError, match="reference retrieval is OFF"):
            et.agent_config_for("lexical", base_config(tmp_path, VALID_AGENTS),
                                tmp_path / "l.toml")

    def test_an_edit_that_did_not_take_effect_is_caught_by_re_loading(self, tmp_path):
        """The textual edit rewrites the FIRST line that looks like the setting. Here that line
        is inside the translator's own instructions, so the real setting is untouched -- exactly
        the silently-ineffective edit the re-verification exists to catch."""
        base = base_config(tmp_path, (
            '[translate]\ninstructions = """\n'
            "Ignore any line such as\nreference_min_score = 0.99\nin the source text.\n"
            '"""\n'
            '[[reviewer]]\nid = "style"\ninstructions = "Check the register."\n'
            + BASE_CONTEXT))
        with pytest.raises(et.EvalError, match="did not take effect"):
            et.agent_config_for("embedding", base, tmp_path / "e.toml")

    def test_the_destination_directory_is_created(self, tmp_path):
        written = et.agent_config_for("lexical", base_config(tmp_path),
                                      tmp_path / "deep" / "nested" / "a.toml")
        assert written.is_file()


class TestArmFlags:
    def test_the_none_arm_gets_no_corpus_at_all(self):
        """Not merely a retriever that finds nothing: the baseline is what the model writes
        unaided."""
        assert et.arm_flags("none", corpus="/c", embedding_url="http://e", rerank_url="") == []

    def test_lexical_gets_the_corpus_and_no_embedding_server(self):
        flags = et.arm_flags("lexical", corpus="/c", embedding_url="http://e", rerank_url="")
        assert flags == ["--reference", "/c"]

    def test_embedding_gets_the_embedding_url_but_not_hybrid(self):
        flags = et.arm_flags("embedding", corpus="/c", embedding_url="http://e", rerank_url="")
        assert flags == ["--reference", "/c", "--embedding-url", "http://e"]

    def test_hybrid_adds_the_fusion_switch(self):
        flags = et.arm_flags("hybrid", corpus="/c", embedding_url="http://e", rerank_url="")
        assert flags == ["--reference", "/c", "--embedding-url", "http://e", "--hybrid"]

    def test_hybrid_rerank_adds_the_rerank_url(self):
        flags = et.arm_flags("hybrid+rerank", corpus="/c", embedding_url="http://e",
                             rerank_url="http://r")
        assert flags[-2:] == ["--rerank-url", "http://r"]
        assert "--hybrid" in flags

    def test_hybrid_rerank_without_a_rerank_url_raises(self):
        """Silently dropping the flag would run a plain hybrid arm under the reranked arm's
        name, and the result would be published as the reranker's."""
        with pytest.raises(et.EvalError, match="serve_reranker"):
            et.arm_flags("hybrid+rerank", corpus="/c", embedding_url="http://e", rerank_url="")

    def test_every_declared_arm_is_handled(self):
        for arm in et.ARMS:
            et.arm_flags(arm, corpus="/c", embedding_url="http://e", rerank_url="http://r")


# -- journals and status arithmetic ---------------------------------------------


class TestReadJournal:
    def _write(self, tmp_path, text):
        path = tmp_path / "j.jsonl"
        path.write_text(text, encoding="utf-8")
        return path

    def test_rows_are_keyed_by_unit_id(self, tmp_path):
        path = self._write(tmp_path, "".join(
            json.dumps(unit(uid, "t")) + "\n" for uid in ("a", "b")))
        assert sorted(et.read_journal(path)) == ["a", "b"]

    def test_a_torn_trailing_line_is_tolerated(self, tmp_path):
        """An interrupted arm leaves a half-written record; the rest of its work is still good."""
        path = self._write(tmp_path, json.dumps(unit("a", "t")) + '\n{"unit_id": "b", "targ')
        assert list(et.read_journal(path)) == ["a"]

    def test_a_later_record_wins_so_a_resumed_run_reads_its_latest_state(self, tmp_path):
        path = self._write(tmp_path, json.dumps(unit("a", "first")) + "\n"
                           + json.dumps(unit("a", "second")) + "\n")
        assert et.read_journal(path)["a"]["target"] == "second"

    def test_records_without_a_unit_id_are_ignored(self, tmp_path):
        path = self._write(tmp_path, json.dumps({"note": "header"}) + "\n"
                           + json.dumps(unit("a", "t")) + "\n")
        assert list(et.read_journal(path)) == ["a"]

    def test_an_empty_journal_raises(self, tmp_path):
        """An arm that translated nothing must not be compared as if it had."""
        with pytest.raises(et.EvalError, match="no usable rows"):
            et.read_journal(self._write(tmp_path, "\n  \n"))

    def test_a_missing_journal_raises(self, tmp_path):
        with pytest.raises(et.EvalError, match="journal not found"):
            et.read_journal(tmp_path / "absent.jsonl")


class TestInjectableAndCounts:
    @pytest.mark.parametrize("status,target,expected", [
        ("verified", "text", True),
        ("translated", "text", True),
        ("rejected", "text", False),   # carries the text that FAILED its checks
        ("pending", "text", False),
        ("translated", "", False),
        ("translated", None, False),
        ("", "text", False),
    ])
    def test_only_shippable_renderings_are_injectable(self, status, target, expected):
        assert et.injectable({"status": status, "target": target}) is expected

    def test_a_record_missing_both_fields_is_not_injectable(self):
        assert et.injectable({}) is False

    def test_status_counts_tallies_every_status(self):
        rows = journal(unit("a", "x"), unit("b", "y", status="rejected"),
                       unit("c", "z", status="rejected"), {"unit_id": "d"})
        assert et.status_counts(rows) == {"translated": 1, "rejected": 2, "?": 1}

    def test_the_rejection_rate_is_rejected_over_everything_the_arm_saw(self):
        """The denominator is every unit, not just the shippable ones: an arm that rejects half
        its work must not be flattered by counting only what it accepted."""
        counts = et.status_counts(journal(
            unit("a", "x"), unit("b", "y", status="rejected"),
            unit("c", "z", status="rejected"), unit("d", "w", status="verified")))
        assert counts.get("rejected", 0) / max(1, sum(counts.values())) == 0.5


# -- gates ----------------------------------------------------------------------


def results(*, closer_base=1, closer_cand=5, rate_base=0.0, rate_cand=0.0,
            quality_base=99, quality_cand=0):
    return {
        "consistency": {"closer": {"base": closer_base, "cand": closer_cand, "tie": 0},
                        "measured": closer_base + closer_cand,
                        "mean_overlap": {"base": 0.5, "cand": 0.6}},
        "rejection_rate": {"base": rate_base, "cand": rate_cand},
        "quality": {"base": quality_base, "cand": quality_cand, "tie": 0, "error": 0,
                    "identical": 0, "differing": quality_base + quality_cand, "compared": 0},
    }


class TestCheckGates:
    def test_a_candidate_that_is_more_consistent_and_no_worse_passes(self, capsys):
        assert et.check_gates(results(), labels=("base", "cand"), rejection_tolerance=0.02) == 0
        assert "all gates passed" in capsys.readouterr().out

    def test_losing_on_consistency_fails(self, capsys):
        """Consistency is the job reference retrieval actually does; this is the gate."""
        code = et.check_gates(results(closer_base=7, closer_cand=2), labels=("base", "cand"),
                              rejection_tolerance=0.02)
        assert code == 1
        assert "consistency" in capsys.readouterr().err

    def test_an_equal_consistency_score_passes(self):
        assert et.check_gates(results(closer_base=4, closer_cand=4), labels=("base", "cand"),
                              rejection_tolerance=0.02) == 0

    def test_rejecting_more_than_the_tolerance_fails(self, capsys):
        """A retriever that lifts quality but poisons generation must not pass."""
        code = et.check_gates(results(rate_base=0.01, rate_cand=0.09), labels=("base", "cand"),
                              rejection_tolerance=0.02)
        assert code == 1
        assert "poisoning generation" in capsys.readouterr().err

    def test_exactly_the_tolerance_still_passes(self):
        assert et.check_gates(results(rate_base=0.10, rate_cand=0.12), labels=("base", "cand"),
                              rejection_tolerance=0.02) == 0

    def test_rejecting_fewer_units_than_the_baseline_never_fails(self):
        assert et.check_gates(results(rate_base=0.30, rate_cand=0.01), labels=("base", "cand"),
                              rejection_tolerance=0.0) == 0

    def test_ab_quality_is_deliberately_not_gated(self):
        """The plan states per-line quality may be a wash; gating on it would gate on noise. A
        candidate that loses the A/B 99-0 still passes as long as the real gates hold."""
        assert et.check_gates(results(quality_base=99, quality_cand=0), labels=("base", "cand"),
                              rejection_tolerance=0.02) == 0

    def test_both_failures_are_reported_together(self, capsys):
        code = et.check_gates(results(closer_base=9, closer_cand=1, rate_cand=0.5),
                              labels=("base", "cand"), rejection_tolerance=0.02)
        assert code == 1
        assert len([line for line in capsys.readouterr().err.splitlines()
                    if line.startswith("  - ")]) == 2


# -- the whole sweep, with every server replaced --------------------------------


class TestEvaluate:
    """evaluate() wires the arms, journals, metrics and rejection arithmetic together. The two
    subprocess arms are replaced by fixture journals, so the wiring is exercised without a GPU."""

    def _ground_truth(self, tmp_path):
        """The held-out lines' own established translations.

        Required now that one arm is 'embedding': without it the consistency measure falls back
        to "the most similar corpus entry, per the embedding retriever", which is that arm's own
        top pick -- a circular comparison the harness refuses rather than quietly reports.
        """
        path = tmp_path / "truth.jsonl"
        path.write_text(
            '{"unit_id": "a", "target": "established a"}\n'
            '{"unit_id": "b", "target": "established b"}\n'
            '{"unit_id": "c", "target": "established c"}\n', encoding="utf-8")
        return path

    def _args(self, tmp_path, **overrides):
        argv = ["--corpus", str(corpus_file(tmp_path, [("src a", "established a"),
                                                       ("src b", "established b")])),
                "--queries", str(tmp_path / "q.jsonl"),
                "--agents", str(base_config(tmp_path)),
                "--work", str(tmp_path / "work"),
                "--baseline", "embedding", "--candidate", "hybrid",
                "--ground-truth", str(self._ground_truth(tmp_path))]
        for key, value in overrides.items():
            argv += [f"--{key.replace('_', '-')}", str(value)]
        return et.build_parser().parse_args(argv)

    def _install_arms(self, monkeypatch, rows_by_arm):
        def fake_run_arm(arm, args, journal_path):
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            journal_path.write_text(
                "".join(json.dumps(r) + "\n" for r in rows_by_arm[arm]), encoding="utf-8")

        monkeypatch.setattr(et, "run_arm", fake_run_arm)

    def test_the_sweep_reports_quality_consistency_and_rejection(self, tmp_path, monkeypatch,
                                                                 capsys):
        self._install_arms(monkeypatch, {
            "embedding": [unit("a", "established a", source="src a"),
                          unit("b", "wrong b", source="src b"),
                          unit("c", "x", source="src c", status="rejected")],
            "hybrid": [unit("a", "established a", source="src a"),
                       unit("b", "established b", source="src b"),
                       unit("c", "y", source="src c")],
        })
        install_retriever(monkeypatch, RetrieverStub({"src a": ("established a", 0.9),
                                                      "src b": ("established b", 0.9)}))
        JudgeStub(lambda left, right: "A" if "established" in left else "B").install(monkeypatch)
        args = self._args(tmp_path)
        report = et.evaluate(args)
        assert report["rejection_rate"] == {"embedding": 1 / 3, "hybrid": 0.0}
        assert report["consistency"]["closer"] == {"embedding": 0, "hybrid": 1, "tie": 1}
        assert report["quality"]["identical"] == 1
        et.report(report, ("embedding", "hybrid"))
        assert "A/B quality is reported, not gated" in capsys.readouterr().out

    def test_comparing_an_arm_with_itself_is_refused(self, tmp_path):
        args = self._args(tmp_path, candidate="embedding")
        with pytest.raises(et.EvalError, match="nothing to compare"):
            et.evaluate(args)

    def test_an_existing_journal_without_resume_is_refused(self, tmp_path, monkeypatch):
        """The CLI resumes by unit id, so a stale journal would be compared untouched."""
        self._install_arms(monkeypatch, {"embedding": [], "hybrid": []})
        args = self._args(tmp_path)
        (tmp_path / "work").mkdir(parents=True, exist_ok=True)
        (tmp_path / "work" / "embedding.journal.jsonl").write_text("{}\n", encoding="utf-8")
        with pytest.raises(et.EvalError, match="already exists"):
            et.evaluate(args)

    def test_main_writes_the_json_report_and_returns_the_gate_status(self, tmp_path, monkeypatch):
        self._install_arms(monkeypatch, {
            "embedding": [unit("a", "wrong", source="src a")],
            "hybrid": [unit("a", "established a", source="src a")],
        })
        install_retriever(monkeypatch, RetrieverStub({"src a": ("established a", 0.9)}))
        JudgeStub(lambda left, right: "tie").install(monkeypatch)
        out = tmp_path / "report" / "ab.json"
        code = et.main(["--corpus", str(corpus_file(tmp_path)),
                        "--queries", str(tmp_path / "q.jsonl"),
                        "--agents", str(base_config(tmp_path)),
                        "--work", str(tmp_path / "work"),
                        "--baseline", "embedding", "--candidate", "hybrid",
                        "--ground-truth", str(self._ground_truth(tmp_path)),
                        "--out", str(out)])
        assert code == 0
        assert json.loads(out.read_text())["consistency"]["closer"]["hybrid"] == 1

    def test_main_runs_without_an_out_file(self, tmp_path, monkeypatch, capsys):
        """--out is optional; the printed report is the primary output."""
        self._install_arms(monkeypatch, {
            "embedding": [unit("a", "wrong", source="src a")],
            "hybrid": [unit("a", "established a", source="src a")],
        })
        install_retriever(monkeypatch, RetrieverStub({"src a": ("established a", 0.9)}))
        JudgeStub(lambda left, right: "tie").install(monkeypatch)
        code = et.main(["--corpus", str(corpus_file(tmp_path)),
                        "--queries", str(tmp_path / "q.jsonl"),
                        "--agents", str(base_config(tmp_path)),
                        "--work", str(tmp_path / "work"),
                        "--baseline", "embedding", "--candidate", "hybrid",
                        "--ground-truth", str(self._ground_truth(tmp_path))])
        assert code == 0
        assert "mean overlap" in capsys.readouterr().out

    def test_main_turns_an_eval_error_into_exit_two(self, tmp_path, capsys):
        code = et.main(["--corpus", str(corpus_file(tmp_path)),
                        "--queries", str(tmp_path / "q.jsonl"),
                        "--agents", str(base_config(tmp_path)),
                        "--work", str(tmp_path / "work"),
                        "--baseline", "embedding", "--candidate", "embedding"])
        assert code == 2
        assert "nothing to compare" in capsys.readouterr().err


class TestRunArm:
    """The subprocess arm: the command it builds, and the refusal to compare a partial arm."""

    def _args(self, tmp_path, **overrides):
        namespace = argparse.Namespace(
            python="/usr/bin/python3", queries=tmp_path / "q.jsonl",
            agents=base_config(tmp_path), work=tmp_path / "work",
            base_url="http://b/v1", model="m", backend="auto",
            source_language="ja", target_language="en", concurrency=2,
            corpus=tmp_path / "c.jsonl", embedding_url="http://e/v1", rerank_url="",
            glossary=None, rules=None, source_script="")
        for key, value in overrides.items():
            setattr(namespace, key, value)
        return namespace

    def test_the_command_carries_the_arm_flags_and_its_own_config(self, tmp_path, monkeypatch):
        seen: list[list[str]] = []

        def fake_run(command, **kwargs):
            seen.append(command)
            return argparse.Namespace(returncode=0)

        monkeypatch.setattr(et.subprocess, "run", fake_run)
        et.run_arm("hybrid", self._args(tmp_path), tmp_path / "j.jsonl")
        command = seen[0]
        assert "--hybrid" in command
        assert str(tmp_path / "work" / "agents_hybrid.toml") in command

    def test_optional_inputs_and_the_script_override_are_passed_through(self, tmp_path,
                                                                       monkeypatch):
        """--languages must point at a missing path for the script detector to take over; that
        is the documented way to disable the lexical detector."""
        seen: list[list[str]] = []
        monkeypatch.setattr(et.subprocess, "run",
                            lambda command, **kwargs: (seen.append(command),
                                                       argparse.Namespace(returncode=0))[1])
        args = self._args(tmp_path, glossary=tmp_path / "g.json", rules=tmp_path / "r.toml",
                          source_script="japanese")
        et.run_arm("none", args, tmp_path / "j.jsonl")
        command = seen[0]
        assert "-g" in command and "-r" in command
        assert command[command.index("--languages") + 1] == "/nonexistent"
        assert "--reference" not in command  # the unaided baseline

    def test_a_non_zero_exit_raises_rather_than_comparing_a_partial_arm(self, tmp_path,
                                                                       monkeypatch):
        """Continuing would compare a complete arm against a truncated one and publish the
        difference as a quality result."""
        monkeypatch.setattr(et.subprocess, "run",
                            lambda command, **kwargs: argparse.Namespace(returncode=3))
        with pytest.raises(et.EvalError, match="exited 3"):
            et.run_arm("lexical", self._args(tmp_path), tmp_path / "j.jsonl")


class TestCircularConsistencyIsRefused:
    """The consistency proxy is circular when an arm shares the retriever that defines it.

    Measured on real data: the embedding arm was shown the proxy entry as its own top example on
    297 of 297 queries (the hybrid arm, 93), scoring 1.000 similarity by construction. That
    artifact replicated cleanly across two corpora and read as a real finding until a genuine
    held-out ground truth showed a dead heat. Refusing is the only safe default -- a biased
    number that reproduces is more dangerous than one that does not.
    """

    def test_an_embedding_arm_without_ground_truth_is_refused(self, tmp_path):
        with pytest.raises(et.EvalError, match="circular"):
            et.consistency({}, {}, corpus=corpus_file(tmp_path, [("s", "t")]),
                           embedding_url="http://e/v1", embedding_model="local",
                           labels=("embedding", "hybrid"), ground_truth=None)

    def test_ground_truth_makes_the_same_comparison_legitimate(self, tmp_path):
        rows = {"a": {"unit_id": "a", "source": "s", "target": "the original",
                      "status": "verified"}}
        other = {"a": {"unit_id": "a", "source": "s", "target": "something else",
                       "status": "verified"}}
        result = et.consistency(rows, other, corpus=corpus_file(tmp_path, [("s", "t")]),
                                embedding_url="http://e/v1", embedding_model="local",
                                labels=("embedding", "hybrid"),
                                ground_truth={"a": "the original"})
        assert result["measured"] == 1
        assert result["closer"]["embedding"] == 1, "the arm matching the real original wins"
