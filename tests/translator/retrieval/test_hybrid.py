"""Tests for HybridRetriever: fusion of the real LexicalRetriever with a dense arm and an
optional reranker, both driven over in-memory httpx transports so no server is touched.

Embedding vectors are hand-picked (as in test_embedding.py); text is chosen so the lexical
arm's real TF-IDF/char-n-gram scoring agrees with the intended narrative (shared words rank
high, unrelated wording ranks low), so both arms' real algorithms -- not stand-ins -- drive
every scenario.
"""
from __future__ import annotations

import json

import httpx
import pytest

from transunit.reference import ReferenceEntry, Retriever
from translator.retrieval.embedding import EmbeddingError
from translator.retrieval.hybrid import HybridRetriever, _sigmoid
from translator.retrieval.rerank import RerankError

# text -> embedding vector. The retriever L2-normalises, so only direction matters.
_VECTORS = {
    "a cat sat": [1.0, 0.0, 0.0],
    "alpha line one": [1.0, 0.0, 0.0],
    "alpha line two": [0.99, 0.01, 0.0],
    "gamma wildly unrelated content": [0.0, 1.0, 0.0],
    "a dog barks loudly": [0.0, 1.0, 0.0],
    "kot siedzial": [1.0, 0.0, 0.0],          # target-side vector for the target-only test
}
_DEFAULT_VEC = [0.0, 0.0, 1.0]


def _embedding_handler(requests: list | None = None):
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if requests is not None:
            requests.append(body)
        data = [{"embedding": _VECTORS.get(text, _DEFAULT_VEC)} for text in body["input"]]
        return httpx.Response(200, json={"data": data})
    return handle


def _hybrid(entries, *, embedding_handler=None, rerank_handler=None, rerank_base_url="",
           **kwargs) -> HybridRetriever:
    embedding_client = httpx.Client(
        transport=httpx.MockTransport(embedding_handler or _embedding_handler()))
    rerank_client = (httpx.Client(transport=httpx.MockTransport(rerank_handler))
                     if rerank_handler is not None else None)
    return HybridRetriever(
        entries, embedding_base_url="http://x/v1",
        rerank_base_url=(rerank_base_url or ("http://y/v1" if rerank_handler else "")),
        embedding_client=embedding_client, rerank_client=rerank_client, **kwargs)


CORPUS = [ReferenceEntry("a cat sat", "kot siedzial"),
          ReferenceEntry("a dog barks loudly", "pies glosno szczeka")]


class TestProtocolAndBasics:
    def test_it_satisfies_the_retriever_protocol(self) -> None:
        assert isinstance(_hybrid(CORPUS), Retriever)

    def test_top_hit_matches_both_arms(self) -> None:
        hits = _hybrid(CORPUS, lexical_min_score=0.0, embedding_min_score=0.0).by_source(
            "a cat sat", k=2, min_score=0.0)
        assert hits[0].entry.source == "a cat sat"

    def test_k_zero_returns_nothing(self) -> None:
        assert _hybrid(CORPUS).by_source("a cat sat", k=0, min_score=0.0) == ()

    def test_an_empty_corpus_returns_nothing(self) -> None:
        assert _hybrid([]).by_source("anything", k=3, min_score=0.0) == ()

    def test_scores_stay_within_the_unit_interval(self) -> None:
        hits = _hybrid(CORPUS, lexical_min_score=0.0, embedding_min_score=0.0).by_source(
            "a cat sat", k=2, min_score=0.0)
        assert all(0.0 <= h.score <= 1.0 for h in hits)

    def test_source_entries_and_len_delegate_to_the_lexical_arm(self) -> None:
        r = _hybrid(CORPUS)
        assert len(r) == 2 and r.source_entries == 2


class TestFusionRecall:
    def test_a_dense_only_hit_still_surfaces_via_fusion(self) -> None:
        # A high lexical floor excludes "a dog barks loudly" from the LEXICAL arm's candidates
        # for a query with no literal overlap, but the dense arm (embedding_min_score=0.0) still
        # finds it via its hand-picked vector. Fusion must not require agreement from both arms.
        r = _hybrid(CORPUS, lexical_min_score=0.99, embedding_min_score=0.0)
        hits = r.by_source("woof arf bark", k=2, min_score=0.0)
        assert any(h.entry.source == "a dog barks loudly" for h in hits)

    def test_by_target_uses_target_text_and_finds_target_only_entries(self) -> None:
        corpus = [ReferenceEntry("", "kot siedzial")]
        r = _hybrid(corpus, index_source=False, index_target=True,
                   lexical_min_score=0.0, embedding_min_score=0.0)
        hits = r.by_target("kot siedzial", k=1, min_score=0.0)
        assert hits and hits[0].entry.target == "kot siedzial"

    def test_by_target_without_a_target_index_raises(self) -> None:
        r = _hybrid(CORPUS, index_source=True, index_target=False)
        with pytest.raises(Exception, match="target index"):
            r.by_target("x", k=1, min_score=0.0)

    def test_by_source_without_a_source_index_raises(self) -> None:
        corpus = [ReferenceEntry("", "target only")]
        r = _hybrid(corpus, index_source=False, index_target=True)
        with pytest.raises(Exception, match="source index"):
            r.by_source("x", k=1, min_score=0.0)


NEAR_DUP_CORPUS = [ReferenceEntry("alpha line one", "target a"),
                   ReferenceEntry("alpha line two", "target b"),
                   ReferenceEntry("gamma wildly unrelated content", "target c")]


class TestFinalFloorAndMMR:
    def test_the_final_min_score_gates_output(self) -> None:
        r = _hybrid(CORPUS, lexical_min_score=0.0, embedding_min_score=0.0)
        loose = r.by_source("a cat sat", k=2, min_score=0.0)
        strict = r.by_source("a cat sat", k=2, min_score=0.999)
        assert len(strict) <= len(loose)

    def test_a_floor_above_the_unit_interval_admits_nothing(self) -> None:
        # No relevance value can ever reach above 1.0, so this floor rejects every candidate --
        # the "nothing survives the final gate" path, distinct from an empty candidate pool.
        r = _hybrid(CORPUS, lexical_min_score=0.0, embedding_min_score=0.0)
        assert r.by_source("a cat sat", k=2, min_score=1.01) == ()

    def test_mmr_suppresses_a_near_duplicate_example(self) -> None:
        # "alpha line one" and "alpha line two" are near-duplicates (both lexically and by their
        # hand-picked vectors); "gamma wildly unrelated content" is dissimilar to both. At a low
        # mmr_lambda, the second pick must favour the dissimilar entry over the redundant near-
        # duplicate, even though the near-duplicate has the higher raw relevance.
        r = _hybrid(NEAR_DUP_CORPUS, lexical_min_score=0.0, embedding_min_score=0.0,
                   mmr_lambda=0.3, candidate_pool=10)
        hits = r.by_source("alpha line query", k=2, min_score=0.0)
        sources = [h.entry.source for h in hits]
        assert sources == ["alpha line one", "gamma wildly unrelated content"]

    def test_lambda_one_keeps_the_near_duplicate_by_pure_relevance(self) -> None:
        # The mirror of the above: lambda_=1 disables the diversity term, so the near-duplicate
        # (genuinely more relevant) is kept over the dissimilar entry.
        r = _hybrid(NEAR_DUP_CORPUS, lexical_min_score=0.0, embedding_min_score=0.0,
                   mmr_lambda=1.0, candidate_pool=10)
        hits = r.by_source("alpha line query", k=2, min_score=0.0)
        sources = [h.entry.source for h in hits]
        assert sources == ["alpha line one", "alpha line two"]


class TestRelevanceScaleIsAbsolute:
    """Without a reranker the score is fused/best_possible_rrf, NOT min-max over the pool.

    Regression guard for a real defect: min-max normalisation made reference_min_score mean
    "relative position in this query's candidate pool" rather than the absolute relevance floor
    it is documented (and, for every other retriever, implemented) to be -- it handed the best
    of three terrible candidates a perfect 1.0 and always dropped the worst of three excellent
    ones. These tests fail if that is ever reintroduced.
    """

    def test_the_top_candidate_is_not_automatically_scored_one(self) -> None:
        # Only the lexical arm finds anything here (the dense floor is set impossibly high), so
        # even the best candidate tops out at ~0.5 -- "one arm's top hit, the other never found
        # it". Under min-max it would have been exactly 1.0.
        r = _hybrid(CORPUS, lexical_min_score=0.0, embedding_min_score=1.01)
        hits = r.by_source("a cat sat", k=2, min_score=0.0)
        assert hits, "the lexical arm should still contribute candidates"
        assert hits[0].score < 0.75
        assert hits[0].score == pytest.approx(0.5, abs=0.01)

    def test_the_worst_candidate_is_not_automatically_scored_zero(self) -> None:
        # Under min-max the last candidate is always 0.0 and so always dropped by any positive
        # floor; on an absolute scale it keeps a real, non-zero score.
        r = _hybrid(CORPUS, lexical_min_score=0.0, embedding_min_score=0.0)
        hits = r.by_source("a cat sat", k=5, min_score=0.0)
        assert len(hits) >= 2
        assert min(h.score for h in hits) > 0.0

    def test_agreement_between_both_arms_outscores_a_single_arm_hit(self) -> None:
        agreed = _hybrid(CORPUS, lexical_min_score=0.0, embedding_min_score=0.0).by_source(
            "a cat sat", k=1, min_score=0.0)
        lexical_only = _hybrid(CORPUS, lexical_min_score=0.0,
                               embedding_min_score=1.01).by_source(
            "a cat sat", k=1, min_score=0.0)
        assert agreed[0].score > lexical_only[0].score

    def test_a_floor_above_a_half_demands_both_arms_agree(self) -> None:
        # The useful dial this scale unlocks: a single-arm hit tops out at 0.5, so a floor above
        # that keeps only candidates BOTH arms found -- and a corpus only one arm can match
        # returns nothing rather than its best guess.
        r = _hybrid(CORPUS, lexical_min_score=0.0, embedding_min_score=1.01)
        assert r.by_source("a cat sat", k=5, min_score=0.55) == ()


class TestReranking:
    def test_rerank_reorders_the_fused_result(self) -> None:
        # Without a reranker the fused order favours "a cat sat" for a cat-ish query. A rerank
        # handler that scores the OTHER document higher must flip the top result.
        def rerank_handler(request):
            body = json.loads(request.content)
            docs = body["documents"]
            scores = [10.0 if "dog" in d else -10.0 for d in docs]
            return httpx.Response(200, json={
                "results": [{"index": i, "relevance_score": s}
                           for i, s in enumerate(scores)]})

        no_rerank = _hybrid(CORPUS, lexical_min_score=0.0, embedding_min_score=0.0).by_source(
            "a cat sat", k=1, min_score=0.0)
        assert no_rerank[0].entry.source == "a cat sat"

        reranked = _hybrid(CORPUS, lexical_min_score=0.0, embedding_min_score=0.0,
                           rerank_handler=rerank_handler).by_source(
            "a cat sat", k=1, min_score=0.0)
        assert reranked[0].entry.source == "a dog barks loudly"

    def test_reranked_scores_are_sigmoid_squashed_into_the_unit_interval(self) -> None:
        def rerank_handler(request):
            n = len(json.loads(request.content)["documents"])
            return httpx.Response(200, json={
                "results": [{"index": i, "relevance_score": 50.0} for i in range(n)]})
        hits = _hybrid(CORPUS, lexical_min_score=0.0, embedding_min_score=0.0,
                       rerank_handler=rerank_handler).by_source("a cat sat", k=2, min_score=0.0)
        assert all(0.0 <= h.score <= 1.0 for h in hits)

    def test_rerank_query_time_failure_propagates_with_no_silent_fallback(self) -> None:
        def dying_rerank(request):
            return httpx.Response(503, text="down")
        r = _hybrid(CORPUS, lexical_min_score=0.0, embedding_min_score=0.0,
                   rerank_handler=dying_rerank)
        with pytest.raises(RerankError, match="request failed"):
            r.by_source("a cat sat", k=1, min_score=0.0)


class TestConstructionValidation:
    def test_candidate_pool_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="candidate_pool"):
            _hybrid(CORPUS, candidate_pool=0)

    def test_mmr_lambda_out_of_range_is_refused(self) -> None:
        with pytest.raises(ValueError, match="mmr_lambda"):
            _hybrid(CORPUS, mmr_lambda=1.5)
        with pytest.raises(ValueError, match="mmr_lambda"):
            _hybrid(CORPUS, mmr_lambda=-0.1)


class TestFailureModesAndRAII:
    def test_a_dead_embedding_endpoint_raises_and_closes_an_already_built_rerank_client(
            self, monkeypatch) -> None:
        # Construction order: lexical, then the rerank client (no I/O, can't fail), then the
        # dense embed (can fail). If the dense embed fails, the already-built rerank client's
        # OWN httpx.Client -- which nothing else will ever close, since this object is discarded
        # -- must be released here rather than leaked. The client must be one the RerankClient
        # opened for itself (the test used to inject it, which under the ownership rule is
        # precisely the client that must NOT be closed, so it tested the wrong thing).
        import translator.retrieval.rerank as rr
        closed = {"value": False}

        class SpyClient(httpx.Client):
            def close(self) -> None:
                closed["value"] = True
                super().close()

        monkeypatch.setattr(rr.httpx, "Client", lambda *a, **k: SpyClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200))))
        dead_embedding = httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(503, text="down")))
        with pytest.raises(EmbeddingError):
            HybridRetriever(CORPUS, embedding_base_url="http://x/v1",
                            rerank_base_url="http://y/v1", embedding_client=dead_embedding)
        assert closed["value"] is True
        assert not dead_embedding.is_closed  # injected: still the caller's

    def test_close_releases_the_clients_the_arms_opened_themselves(self, monkeypatch) -> None:
        # close() must still release everything this retriever owns, via both arms -- the
        # ownership rule narrows what gets closed, it does not stop close() from working.
        import translator.retrieval.embedding as emb
        opened: list[httpx.Client] = []
        real_client = httpx.Client  # captured first: the patch below is global to httpx
        embed = _embedding_handler()

        # One patch, dispatching on path: both modules share the one httpx.Client symbol, so
        # patching each in turn would just leave the last handler serving both endpoints.
        def route(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/embeddings"):
                return embed(request)
            return httpx.Response(200, json={"results": []})

        def make(*_a, **_k):
            client = real_client(transport=httpx.MockTransport(route))
            opened.append(client)
            return client

        monkeypatch.setattr(emb.httpx, "Client", make)
        r = HybridRetriever(CORPUS, embedding_base_url="http://x/v1",
                            rerank_base_url="http://y/v1")
        r.close()
        assert len(opened) == 2 and all(client.is_closed for client in opened)

    def test_close_leaves_injected_clients_open(self) -> None:
        # The injector owns them: a caller sharing one client across several retrievers (or
        # reusing it after this one is discarded) must not have it closed from under them.
        embedding_client = httpx.Client(transport=httpx.MockTransport(_embedding_handler()))
        rerank_client = httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": []})))
        r = HybridRetriever(CORPUS, embedding_base_url="http://x/v1",
                            rerank_base_url="http://y/v1",
                            embedding_client=embedding_client, rerank_client=rerank_client)
        r.close()
        assert not embedding_client.is_closed and not rerank_client.is_closed
        embedding_client.close()
        rerank_client.close()

    def test_close_without_a_reranker_does_not_raise(self) -> None:
        embedding_client = httpx.Client(transport=httpx.MockTransport(_embedding_handler()))
        r = HybridRetriever(CORPUS, embedding_base_url="http://x/v1",
                            embedding_client=embedding_client)
        r.close()  # must not raise despite no rerank client existing
        assert not embedding_client.is_closed
        embedding_client.close()


class TestSigmoidStability:
    """``_sigmoid`` maps a cross-encoder's unbounded logit to a relevance in [0, 1].

    It used to be the naive ``1 / (1 + exp(-x))``, which calls ``exp`` on a large POSITIVE
    argument for very negative ``x`` and raised ``OverflowError`` past about -710: one outlier
    logit from a reranker would kill a multi-hour run with an unstructured error, hours in.
    """

    @pytest.mark.parametrize("x", [-1000.0, -709.0, -710.0, -1e308, -1.0, 0.0, 1.0,
                                   709.0, 710.0, 1000.0, 1e308])
    def test_the_whole_float_range_maps_into_the_unit_interval(self, x: float) -> None:
        assert 0.0 <= _sigmoid(x) <= 1.0

    def test_a_large_negative_logit_no_longer_overflows(self) -> None:
        assert _sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-12)

    def test_it_stays_monotonic_and_symmetric_across_the_branch_boundary(self) -> None:
        # The two branches must agree at the seam, not just each be finite on its own side.
        assert _sigmoid(-1e-9) < _sigmoid(0.0) < _sigmoid(1e-9)
        assert _sigmoid(0.0) == pytest.approx(0.5)
        for x in (0.5, 5.0, 50.0):
            assert _sigmoid(-x) == pytest.approx(1.0 - _sigmoid(x))

    def test_infinities_saturate(self) -> None:
        assert _sigmoid(float("inf")) == 1.0
        assert _sigmoid(float("-inf")) == 0.0

    def test_nan_is_refused(self) -> None:
        # No relevance honestly stands for "not a number", and passing it on would poison the
        # ranking silently (a NaN neither wins nor loses a comparison).
        with pytest.raises(ValueError, match="NaN"):
            _sigmoid(float("nan"))


class TestMisbehavingRerankerIsHandled:
    def test_a_huge_negative_rerank_logit_does_not_crash_the_search(self) -> None:
        # End-to-end form of the overflow bug: the reranker is entitled to emit any logit, and
        # the run must survive it with a valid [0, 1] relevance rather than an OverflowError.
        def rerank_handler(request: httpx.Request) -> httpx.Response:
            documents = json.loads(request.content)["documents"]
            return httpx.Response(200, json={"results": [
                {"index": i, "relevance_score": -1000.0 + i}
                for i in range(len(documents))]})

        hits = _hybrid(CORPUS, rerank_handler=rerank_handler).by_source(
            "a cat sat", k=2, min_score=0.0)
        assert hits and all(0.0 <= hit.score <= 1.0 for hit in hits)

    def test_duplicate_rerank_indices_surface_as_a_rerank_error(self) -> None:
        # Previously: the duplicate passed validation, the scored->dict comprehension dropped a
        # candidate, and MMR died on a bare KeyError naming nothing. Now it names the endpoint.
        def rerank_handler(request: httpx.Request) -> httpx.Response:
            documents = json.loads(request.content)["documents"]
            return httpx.Response(200, json={"results": [
                {"index": 0, "relevance_score": 0.9 - 0.1 * i}
                for i in range(len(documents))]})

        # Floors dropped so BOTH entries reach the reranker: a single-document rerank cannot
        # contain a duplicate, so the bug needs at least two candidates to reproduce.
        retriever = _hybrid(CORPUS, rerank_handler=rerank_handler,
                            lexical_min_score=0.0, embedding_min_score=0.0)
        with pytest.raises(RerankError, match="duplicate result indices"):
            retriever.by_source("a cat sat", k=2, min_score=0.0)
