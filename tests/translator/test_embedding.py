"""Tests for the optional embedding retriever, driven by an in-memory httpx transport so no
embedding server (and no GPU) is touched. Vectors are hand-picked per text, so cosine ranking,
the similarity floor, batching, placeholder stripping and the error taxonomy are all pinned
deterministically.
"""
from __future__ import annotations

import json

import httpx
import pytest

from transunit.reference import ReferenceEntry, Retriever
from translator.retrieval.embedding import (
    DEFAULT_EMBEDDING_MIN_SCORE,
    EmbeddingError,
    EmbeddingRetriever,
)

# cleaned-text -> vector (the retriever L2-normalises, so magnitudes need not be unit)
_VECTORS = {
    "the cat": [1.0, 0.0, 0.0],
    "a cat sat": [0.9, 0.1, 0.0],
    "the feline rested": [0.8, 0.2, 0.0],
    "a dog barks": [0.0, 1.0, 0.0],
    "cat": [1.0, 0.0, 0.0],          # for the placeholder-stripping test
    "an opposite line": [-1.0, 0.0, 0.0],  # anti-parallel to "the cat": cosine -1
}
_DEFAULT_VEC = [0.0, 0.0, 1.0]


def _handler(requests: list | None = None):
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if requests is not None:
            requests.append(body)
        data = [{"embedding": _VECTORS.get(text, _DEFAULT_VEC)} for text in body["input"]]
        return httpx.Response(200, json={"data": data})
    return handle


def _retriever(entries, *, requests=None, index_source=True, index_target=False, batch_size=64):
    client = httpx.Client(transport=httpx.MockTransport(_handler(requests)))
    return EmbeddingRetriever(entries, base_url="http://x/v1", index_source=index_source,
                              index_target=index_target, batch_size=batch_size, client=client)


CORPUS = [ReferenceEntry("a cat sat", "kot siedział"),
          ReferenceEntry("the feline rested", "kot odpoczywał"),
          ReferenceEntry("a dog barks", "pies szczeka")]


class TestProtocolAndRanking:
    def test_it_satisfies_the_retriever_protocol(self) -> None:
        assert isinstance(_retriever(CORPUS), Retriever)

    def test_by_source_ranks_by_cosine(self) -> None:
        hits = _retriever(CORPUS).by_source("the cat", k=3, min_score=0.0)
        assert [h.entry.target for h in hits] == ["kot siedział", "kot odpoczywał", "pies szczeka"]
        assert hits[0].score > hits[1].score > hits[2].score
        assert hits[2].score == pytest.approx(0.0, abs=1e-6)  # orthogonal 'dog'

    def test_scores_stay_within_the_unit_interval(self) -> None:
        # 'the cat' vs 'a cat sat' are near-parallel; the score must not round above 1.0
        hits = _retriever(CORPUS).by_source("the cat", k=1, min_score=0.0)
        assert 0.0 <= hits[0].score <= 1.0

    def test_the_floor_drops_weak_matches(self) -> None:
        hits = _retriever(CORPUS).by_source("the cat", k=3, min_score=0.99)
        assert [h.entry.target for h in hits] == ["kot siedział"]  # only the near-parallel one

    def test_k_bounds_the_number_returned(self) -> None:
        assert len(_retriever(CORPUS).by_source("the cat", k=1, min_score=0.0)) == 1


class TestDirectionsAndTargets:
    def test_by_target_matches_target_vectors(self) -> None:
        corpus = [ReferenceEntry("", "a cat sat"), ReferenceEntry("", "a dog barks")]
        hits = _retriever(corpus, index_source=False, index_target=True).by_target(
            "the cat", k=1, min_score=0.0)
        assert hits[0].entry.target == "a cat sat"

    def test_by_target_without_a_target_index_raises(self) -> None:
        with pytest.raises(EmbeddingError, match="without a target index"):
            _retriever(CORPUS).by_target("x", k=1)

    def test_by_source_without_a_source_index_raises(self) -> None:
        r = _retriever(CORPUS, index_source=False, index_target=True)
        with pytest.raises(EmbeddingError, match="without a source index"):
            r.by_source("x", k=1)

    def test_source_index_skips_entries_without_a_source(self) -> None:
        corpus = [ReferenceEntry("a cat sat", "x"), ReferenceEntry("", "target only")]
        r = _retriever(corpus, index_source=True, index_target=True)
        assert r.source_entries == 1
        assert all(h.entry.source for h in r.by_source("the cat", k=5, min_score=0.0))


class TestRequestShaping:
    def test_placeholders_are_stripped_before_embedding(self) -> None:
        requests: list = []
        r = _retriever([ReferenceEntry("[[0]] cat", "x")], requests=requests)
        r.by_source("[[0]] cat", k=1, min_score=0.0)
        # neither the corpus embed nor the query embed sends a raw [[0]]
        assert all("[[0]]" not in t for body in requests for t in body["input"])

    def test_degenerate_queries_are_bounded_not_crashing(self) -> None:
        # Live probing hammered the retriever with empty / whitespace / placeholder-only queries
        # (real game text has bare interjection and placeholder-only lines). _clean maps them to a
        # single space, so they embed and score without raising or escaping the unit interval.
        r = _retriever(CORPUS)
        for query in ("", "   ", "[[0]][[1]]", "[[0]] [[1]]", "\n\n"):
            hits = r.by_source(query, k=3, min_score=0.0)
            assert all(0.0 <= h.score <= 1.0 for h in hits), query

    def test_a_large_corpus_is_embedded_in_batches(self) -> None:
        requests: list = []
        entries = [ReferenceEntry(f"line {i}", f"t{i}") for i in range(10)]
        _retriever(entries, requests=requests, batch_size=4)
        # 10 entries / batch 4 -> 3 requests, each at most 4 inputs
        assert len(requests) == 3
        assert max(len(b["input"]) for b in requests) == 4


class TestFailureModes:
    def test_a_5xx_endpoint_is_a_clear_error(self) -> None:
        client = httpx.Client(transport=httpx.MockTransport(
            lambda req: httpx.Response(503, text="down")))
        with pytest.raises(EmbeddingError, match="request failed"):
            EmbeddingRetriever(CORPUS, base_url="http://x/v1", client=client)

    def test_a_malformed_response_is_a_clear_error(self) -> None:
        client = httpx.Client(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"nope": []})))
        with pytest.raises(EmbeddingError, match="malformed embedding response"):
            EmbeddingRetriever(CORPUS, base_url="http://x/v1", client=client)

    def test_a_count_mismatch_is_caught(self) -> None:
        client = httpx.Client(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0, 0.0]}]})))
        with pytest.raises(EmbeddingError, match="returned 1 embeddings for 3"):
            EmbeddingRetriever(CORPUS, base_url="http://x/v1", client=client)

    def test_the_error_carries_the_endpoint(self) -> None:
        exc = EmbeddingError("boom", url="http://host/v1/embeddings")
        assert exc.url == "http://host/v1/embeddings" and "http://host" in str(exc)

    def test_a_bad_batch_size_is_refused(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            _retriever(CORPUS, batch_size=0)

    def test_ragged_embedding_dimensions_are_a_clear_error(self) -> None:
        # An endpoint returning inconsistent vector lengths must fail as EmbeddingError, not a
        # raw numpy ValueError leaking out of the retriever.
        def handler(request):
            n = len(json.loads(request.content)["input"])
            return httpx.Response(200, json={"data": [{"embedding": [1.0] * (i + 2)}
                                                      for i in range(n)]})
        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(EmbeddingError, match="uniform numeric matrix"):
            EmbeddingRetriever(CORPUS, base_url="http://x/v1", client=client)

    def test_scalar_embeddings_are_rejected_as_not_2d(self) -> None:
        def handler(request):
            n = len(json.loads(request.content)["input"])
            return httpx.Response(200, json={"data": [{"embedding": 1.0} for _ in range(n)]})
        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(EmbeddingError, match="2-D embeddings"):
            EmbeddingRetriever(CORPUS, base_url="http://x/v1", client=client)

    def test_an_item_without_an_embedding_field_is_a_clear_error(self) -> None:
        # A 200 response whose data items lack "embedding" (or are not dicts) must fail as a
        # structured EmbeddingError, not a raw KeyError/TypeError escaping the retriever.
        def handler(request):
            n = len(json.loads(request.content)["input"])
            return httpx.Response(200, json={"data": [{"index": i} for i in range(n)]})
        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(EmbeddingError, match="no 'embedding'"):
            EmbeddingRetriever(CORPUS, base_url="http://x/v1", client=client)

    def test_a_failed_build_closes_the_client_it_opened(self, monkeypatch) -> None:
        # RAII: if the corpus embed fails during construction, a client the retriever opened must
        # be closed -- not leaked -- so a caught EmbeddingError leaves no dangling connection.
        import translator.retrieval.embedding as emb
        closed = {"value": False}

        class SpyClient(httpx.Client):
            def close(self) -> None:
                closed["value"] = True
                super().close()

        def make(*_a, **_k):  # the transport 503s, so the first corpus batch fails
            return SpyClient(transport=httpx.MockTransport(lambda r: httpx.Response(503, text="x")))

        monkeypatch.setattr(emb.httpx, "Client", make)
        with pytest.raises(EmbeddingError):
            EmbeddingRetriever(CORPUS, base_url="http://x/v1")  # no client -> opens (spied) one
        assert closed["value"] is True

    def test_a_failed_build_leaves_an_injected_client_open(self) -> None:
        # The mirror of the above: an injected client is the caller's, so a failed build must NOT
        # close it out from under them.
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
        with pytest.raises(EmbeddingError):
            EmbeddingRetriever(CORPUS, base_url="http://x/v1", client=client)
        assert not client.is_closed
        client.close()

    def test_missing_numpy_is_a_clear_error(self, monkeypatch) -> None:
        import sys
        # Make `import numpy` raise ImportError inside _require_numpy (None in sys.modules).
        monkeypatch.setitem(sys.modules, "numpy", None)
        with pytest.raises(EmbeddingError, match="needs numpy"):
            EmbeddingRetriever(CORPUS, base_url="http://x/v1")


class TestEdgeCases:
    def test_an_empty_corpus_returns_nothing(self) -> None:
        assert _retriever([]).by_source("anything", k=3, min_score=0.0) == ()

    def test_zero_k_returns_nothing(self) -> None:
        assert _retriever(CORPUS).by_source("the cat", k=0, min_score=0.0) == ()

    def test_len_counts_entries(self) -> None:
        assert len(_retriever(CORPUS)) == 3

    def test_a_default_embedding_floor_is_exposed(self) -> None:
        assert 0.0 < DEFAULT_EMBEDDING_MIN_SCORE <= 1.0

    def test_close_releases_a_client_it_opened(self, monkeypatch) -> None:
        # The owning half of the ownership rule: a client this retriever opened is its own to
        # release, so close() must actually close it (otherwise a real run leaks connections).
        import translator.retrieval.embedding as emb
        real_client = httpx.Client  # captured first: emb.httpx IS httpx, so the patch is global
        monkeypatch.setattr(
            emb.httpx, "Client",
            lambda *a, **k: real_client(transport=httpx.MockTransport(_handler(None))))
        r = EmbeddingRetriever(CORPUS, base_url="http://x/v1")  # no client -> opens its own
        r.close()
        assert r._client.is_closed

    def test_close_leaves_an_injected_client_open(self) -> None:
        # The other half, and the bug this pins: close() used to close the client regardless of
        # ownership, tearing down a caller's shared client while the caller still held it --
        # while the construction-failure path above already honoured ownership. One rule now.
        client = httpx.Client(transport=httpx.MockTransport(_handler(None)))
        r = EmbeddingRetriever(CORPUS, base_url="http://x/v1", client=client)
        r.close()
        assert not client.is_closed
        client.close()


class TestVectorsForAndPairwiseSimilarity:
    """The accessors HybridRetriever's MMR redundancy term is built on."""

    def test_vectors_for_returns_rows_aligned_to_the_given_entries(self) -> None:
        r = _retriever(CORPUS)
        vectors = r.vectors_for((CORPUS[1], CORPUS[0]), target=False)
        assert vectors.shape == (2, 3)
        # row 0 -> CORPUS[1] ("the feline rested" -> [0.8, 0.2, 0.0], L2-normalised)
        assert vectors[0][1] > vectors[1][1]  # feline's y-component is the larger of the two

    def test_vectors_for_an_unindexed_direction_raises(self) -> None:
        r = _retriever(CORPUS, index_source=True, index_target=False)
        with pytest.raises(EmbeddingError, match="without a target index"):
            r.vectors_for(CORPUS, target=True)

    def test_vectors_for_an_entry_this_retriever_never_indexed_raises(self) -> None:
        r = _retriever(CORPUS)
        stranger = ReferenceEntry("never seen", "nigdy nie widziany")
        with pytest.raises(EmbeddingError, match="not indexed"):
            r.vectors_for((stranger,), target=False)

    def test_pairwise_similarity_matches_the_hand_picked_vectors(self) -> None:
        r = _retriever(CORPUS)
        # 'a cat sat' [0.9, 0.1, 0] vs 'the feline rested' [0.8, 0.2, 0]: both cat-adjacent, high
        # cosine. 'a cat sat' vs 'a dog barks' [0, 1, 0]: mostly orthogonal, much lower.
        cat_cat = r.pairwise_similarity(CORPUS[0], CORPUS[1], target=False)
        cat_dog = r.pairwise_similarity(CORPUS[0], CORPUS[2], target=False)
        assert cat_cat > 0.9
        assert cat_dog < 0.2
        assert cat_cat > cat_dog

    def test_pairwise_similarity_of_an_entry_with_itself_is_one(self) -> None:
        r = _retriever(CORPUS)
        assert r.pairwise_similarity(CORPUS[0], CORPUS[0], target=False) == pytest.approx(1.0)

    def test_pairwise_similarity_returns_a_plain_float(self) -> None:
        # Not a numpy scalar: callers outside this module (fusion.mmr_order) must never need to
        # know numpy exists, per the module's numpy-confinement design.
        r = _retriever(CORPUS)
        assert type(r.pairwise_similarity(CORPUS[0], CORPUS[1], target=False)) is float


class TestBackCompatShim:
    """``translator.embedding`` is a thin re-export; existing callers must keep working."""

    def test_the_shim_re_exports_the_same_objects(self) -> None:
        import translator.embedding as shim
        import translator.retrieval.embedding as real

        assert shim.EmbeddingRetriever is real.EmbeddingRetriever
        assert shim.EmbeddingError is real.EmbeddingError
        assert shim.DEFAULT_EMBEDDING_MIN_SCORE == real.DEFAULT_EMBEDDING_MIN_SCORE


class TestScoreContract:
    """Retrieved.score is documented as being in [0, 1]; a cosine runs [-1, 1]."""

    def test_a_negative_cosine_is_clamped_to_zero(self) -> None:
        # The score used to be clamped at the top only (min(1.0, ...)), so any negative cosine
        # escaped straight into Retrieved.score whenever min_score was negative -- a contract
        # violation nothing downstream (floors, MMR relevance, reporting) expects.
        corpus = [ReferenceEntry("an opposite line", "przeciwna linia"),
                  ReferenceEntry("a cat sat", "kot siedział")]
        hits = _retriever(corpus).by_source("the cat", k=2, min_score=-1.0)
        assert len(hits) == 2
        assert all(0.0 <= hit.score <= 1.0 for hit in hits)
        assert hits[-1].entry.source == "an opposite line"  # still ranked last on raw cosine
        assert hits[-1].score == 0.0

    def test_a_negative_min_score_still_gates_on_the_raw_cosine(self) -> None:
        # The clamp must not turn the caller's floor into a no-op: gating stays on the true
        # cosine, so a floor above the anti-parallel entry's -1.0 still excludes it.
        corpus = [ReferenceEntry("an opposite line", "przeciwna linia"),
                  ReferenceEntry("a cat sat", "kot siedział")]
        hits = _retriever(corpus).by_source("the cat", k=2, min_score=-0.5)
        assert [hit.entry.source for hit in hits] == ["a cat sat"]
