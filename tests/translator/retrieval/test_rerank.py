"""Tests for the optional cross-encoder reranker, driven by an in-memory httpx transport so no
rerank server is touched. The response shape is pinned to llama.cpp's verified ``/v1/rerank``
schema: ``{"results": [{"index": i, "relevance_score": s}, ...]}``.
"""
from __future__ import annotations

import json

import httpx
import pytest

from translator.retrieval.rerank import RerankClient, RerankError

DOCUMENTS = ["a cat sat", "the feline rested", "a dog barks"]


def _results(scores: dict[int, float]) -> list[dict]:
    return [{"index": i, "relevance_score": s} for i, s in scores.items()]


def _client(handler) -> RerankClient:
    return RerankClient(base_url="http://x/v1",
                        client=httpx.Client(transport=httpx.MockTransport(handler)))


class TestHappyPath:
    def test_results_map_to_index_score_pairs_best_first(self) -> None:
        def handler(request):
            return httpx.Response(200, json={"results": _results({0: 0.2, 1: 0.9, 2: 0.05})})
        scored = _client(handler).rerank("query", DOCUMENTS)
        assert scored == [(1, 0.9), (0, 0.2), (2, 0.05)]

    def test_correct_even_when_the_server_returns_results_out_of_order(self) -> None:
        # The server is not required to pre-sort; the client must sort regardless.
        def handler(request):
            return httpx.Response(200, json={"results": _results({2: 0.05, 0: 0.2, 1: 0.9})})
        scored = _client(handler).rerank("query", DOCUMENTS)
        assert scored == [(1, 0.9), (0, 0.2), (2, 0.05)]

    def test_the_request_carries_the_model_query_and_documents(self) -> None:
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"results": _results({0: 1.0, 1: 0.5, 2: 0.0})})
        RerankClient(base_url="http://x/v1", model="my-reranker",
                    client=httpx.Client(transport=httpx.MockTransport(handler))
                    ).rerank("a query", DOCUMENTS)
        assert seen == {"model": "my-reranker", "query": "a query", "documents": DOCUMENTS}

    def test_empty_documents_short_circuits_with_no_http_call(self) -> None:
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(200, json={"results": []})
        assert _client(handler).rerank("query", []) == []
        assert calls["n"] == 0


class TestFailureModes:
    def test_a_5xx_endpoint_is_a_clear_error(self) -> None:
        def handler(request):
            return httpx.Response(503, text="down")
        with pytest.raises(RerankError, match="request failed"):
            _client(handler).rerank("query", DOCUMENTS)

    def test_a_non_json_body_is_a_clear_error(self) -> None:
        def handler(request):
            return httpx.Response(200, text="not json")
        with pytest.raises(RerankError, match="malformed rerank response"):
            _client(handler).rerank("query", DOCUMENTS)

    def test_a_response_missing_results_is_a_clear_error(self) -> None:
        def handler(request):
            return httpx.Response(200, json={"nope": []})
        with pytest.raises(RerankError, match="malformed rerank response"):
            _client(handler).rerank("query", DOCUMENTS)

    def test_a_count_mismatch_is_caught(self) -> None:
        def handler(request):
            return httpx.Response(200, json={"results": _results({0: 1.0})})
        with pytest.raises(RerankError, match="returned 1 results for 3 documents"):
            _client(handler).rerank("query", DOCUMENTS)

    def test_a_result_missing_index_is_a_clear_error(self) -> None:
        def handler(request):
            return httpx.Response(200, json={
                "results": [{"relevance_score": 0.5}] * len(DOCUMENTS)})
        with pytest.raises(RerankError, match="missing 'index' or 'relevance_score'"):
            _client(handler).rerank("query", DOCUMENTS)

    def test_a_result_missing_relevance_score_is_a_clear_error(self) -> None:
        def handler(request):
            return httpx.Response(200, json={
                "results": [{"index": i} for i in range(len(DOCUMENTS))]})
        with pytest.raises(RerankError, match="missing 'index' or 'relevance_score'"):
            _client(handler).rerank("query", DOCUMENTS)

    def test_a_non_numeric_score_is_a_clear_error(self) -> None:
        def handler(request):
            return httpx.Response(200, json={
                "results": [{"index": i, "relevance_score": "high"}
                           for i in range(len(DOCUMENTS))]})
        with pytest.raises(RerankError, match="missing 'index' or 'relevance_score'"):
            _client(handler).rerank("query", DOCUMENTS)

    def test_an_out_of_range_index_is_caught(self) -> None:
        def handler(request):
            return httpx.Response(200, json={"results": _results({0: 1.0, 1: 0.5, 99: 0.1})})
        with pytest.raises(RerankError, match="index 99 is out of range"):
            _client(handler).rerank("query", DOCUMENTS)

    def test_a_negative_index_is_caught(self) -> None:
        def handler(request):
            return httpx.Response(200, json={"results": _results({0: 1.0, 1: 0.5, -1: 0.1})})
        with pytest.raises(RerankError, match="index -1 is out of range"):
            _client(handler).rerank("query", DOCUMENTS)

    def test_duplicate_indices_are_rejected(self) -> None:
        # The count is right and every index is in range, so the old validation passed this --
        # and then a caller keying candidates by index silently lost one and blew up with a
        # bare KeyError inside MMR, pointing nowhere near the endpoint that actually misbehaved.
        def handler(request):
            return httpx.Response(200, json={"results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.8},
                {"index": 2, "relevance_score": 0.1}]})
        with pytest.raises(RerankError, match=r"duplicate result indices \[0\]"):
            _client(handler).rerank("query", DOCUMENTS)

    def test_every_duplicated_index_is_named(self) -> None:
        def handler(request):
            return httpx.Response(200, json={"results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.8},
                {"index": 0, "relevance_score": 0.5},
                {"index": 0, "relevance_score": 0.4}]})
        with pytest.raises(RerankError, match=r"duplicate result indices \[0, 1\]"):
            _client(handler).rerank("query", [*DOCUMENTS, "a fourth"])

    def test_a_nan_score_is_rejected(self) -> None:
        # NaN is unorderable: it would corrupt this client's own defensive sort and then be
        # ranked wherever a comparison happened to leave it, silently, further downstream.
        def handler(request):
            return httpx.Response(
                200, content=b'{"results": [{"index": 0, "relevance_score": NaN}, '
                             b'{"index": 1, "relevance_score": 0.5}, '
                             b'{"index": 2, "relevance_score": 0.1}]}',
                headers={"content-type": "application/json"})
        with pytest.raises(RerankError, match="document 0 has a NaN relevance_score"):
            _client(handler).rerank("query", DOCUMENTS)

    def test_an_infinite_score_is_accepted(self) -> None:
        # The other arm of that rule: +/-inf orders correctly and squashes to a sane 1.0/0.0
        # through the caller's sigmoid, so it is not treated as malformed.
        def handler(request):
            return httpx.Response(
                200, content=b'{"results": [{"index": 0, "relevance_score": Infinity}, '
                             b'{"index": 1, "relevance_score": 0.5}, '
                             b'{"index": 2, "relevance_score": 0.1}]}',
                headers={"content-type": "application/json"})
        assert _client(handler).rerank("query", DOCUMENTS)[0] == (0, float("inf"))

    def test_the_error_carries_the_endpoint(self) -> None:
        exc = RerankError("boom", url="http://host/v1/rerank")
        assert exc.url == "http://host/v1/rerank" and "http://host" in str(exc)

    def test_an_error_with_no_url_has_no_endpoint_suffix(self) -> None:
        exc = RerankError("boom")
        assert exc.url is None and str(exc) == "boom"


class TestClose:
    def test_close_releases_an_owned_client(self) -> None:
        client = RerankClient(base_url="http://x/v1")
        client.close()
        assert client._client.is_closed

    def test_close_leaves_an_injected_client_open(self) -> None:
        # One ownership rule across the retrieval layer: an object closes only what it opened.
        # An injected client belongs to its injector, who may still be using it or sharing it
        # with another retriever -- closing it here would break a connection we were only lent.
        # (This previously asserted the opposite, which contradicted both EmbeddingRetriever's
        # construction-failure path and its close().)
        raw = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
        client = RerankClient(base_url="http://x/v1", client=raw)
        client.close()
        assert not raw.is_closed
        raw.close()
