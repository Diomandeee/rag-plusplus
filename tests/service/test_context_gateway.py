"""Tests for the Smart Context Gateway.

Covers: project extraction, RLM decomposition detection, response composition,
fallback behavior, and Prometheus metrics export.
"""

import asyncio
import json
import time
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# ---------------------------------------------------------------------------
# Minimal stub to test without the full service stack
# ---------------------------------------------------------------------------

# Patch env before importing
import os
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")

from rag_plusplus.service.routes.context_gateway import (
    _extract_project,
    _should_decompose,
    _compose_response,
    _compute_novelty_batch,
    _score_with_echo_suppression,
    _mark_seen_and_filter,
    _expand_query_embedding,
    _mock_rag_results,
    _mock_gk_result,
    _mock_embedding,
    export_prometheus_metrics,
    ContextGatewayRequest,
    ContextGatewayResponse,
    ContextTurn,
    SessionBloomFilter,
    OverlapClass,
    _ENTITY_NAME_RE,
    EMBEDDING_DIM,
)


# ---------------------------------------------------------------------------
# _extract_project tests (A7: regex ordering)
# ---------------------------------------------------------------------------


class TestExtractProject:
    def test_projects_path(self):
        assert _extract_project("/Users/user/projects/creator-shield/src") == "creator-shield"

    def test_desktop_path(self):
        assert _extract_project("/Users/user/Desktop/Spore/Sources") == "Spore"

    def test_clawdbot_path(self):
        assert _extract_project("/Users/user/.clawdbot/plugins/foo") == "plugins"

    def test_monitoring_path(self):
        assert _extract_project("/Users/user/monitoring/nexus-portal/src") == "nexus-portal"

    def test_flows_path(self):
        assert _extract_project("/Users/user/flows/feed-hub/flow.py") == "feed-hub"

    def test_projects_before_desktop(self):
        """Projects path should match before Desktop for nested paths."""
        result = _extract_project("/Users/user/projects/my-project/src")
        assert result == "my-project"

    def test_empty_cwd(self):
        assert _extract_project("") is None

    def test_no_match(self):
        assert _extract_project("/usr/local/bin") is None

    def test_invalid_chars_rejected(self):
        """Entity names with special chars should be rejected (S2)."""
        # Path traversal attempt
        result = _extract_project("/Users/user/Desktop/../../etc/passwd")
        # The regex captures ".." which fails _ENTITY_NAME_RE
        assert result is None or _ENTITY_NAME_RE.match(result.lower().replace("_", "-"))


# ---------------------------------------------------------------------------
# _should_decompose tests
# ---------------------------------------------------------------------------


class TestShouldDecompose:
    def test_simple_query_not_decomposed(self):
        assert _should_decompose("How many tables?") is False

    def test_short_query_not_decomposed(self):
        """Queries under MIN_DECOMPOSE_WORDS should never decompose."""
        assert _should_decompose("how does X work") is False

    def test_multi_hop_query_decomposed(self):
        assert _should_decompose("how does the context gateway connect to the graph kernel") is True

    def test_causal_query_decomposed(self):
        assert _should_decompose("why does the RAG search fail when Supabase is down") is True

    def test_comparison_query_decomposed(self):
        assert _should_decompose("what is the difference between hybrid search and pgvector search") is True

    def test_cross_system_query_decomposed(self):
        assert _should_decompose("which services across the mesh interact with Supabase") is True

    def test_regex_pattern_between_and(self):
        assert _should_decompose("what is the relationship between Mac1 and cloud-vm infrastructure") is True

    def test_no_signal_long_query_not_decomposed(self):
        assert _should_decompose("please list all the files in the src directory now") is False


# ---------------------------------------------------------------------------
# _compose_response tests
# ---------------------------------------------------------------------------


class TestComposeResponse:
    def _make_rag_results(self, n=3, text_key="text_content"):
        return [
            {
                "id": f"doc_{i}",
                text_key: f"This is document {i} with some content about testing.",
                "similarity": 0.8 - i * 0.1,
                "metadata": {"source": "test", "model_id": "test-model", "extra": "data", "more": "stuff"},
            }
            for i in range(n)
        ]

    def _make_gk_result(self, n_paths=2):
        return {
            "paths": [
                {"edges": [{"predicate": "uses", "target": f"service_{i}"}]}
                for i in range(n_paths)
            ],
            "admissibility_token": None,
            "stats": {"raw_paths": n_paths, "filtered_paths": n_paths},
        }

    def test_basic_composition(self):
        resp = _compose_response(
            self._make_gk_result(),
            self._make_rag_results(),
            max_tokens=500,
            include_graph=True,
        )
        assert len(resp.related_turns) == 3
        assert resp.graph_context is not None
        assert "rag++" in resp.sources
        assert "gk" in resp.sources
        assert resp.fallback is False
        assert resp.token_estimate > 0

    def test_empty_rag_results(self):
        resp = _compose_response(
            self._make_gk_result(),
            [],
            max_tokens=500,
            include_graph=True,
        )
        assert len(resp.related_turns) == 0
        assert "rag++" not in resp.sources

    def test_empty_gk_results(self):
        resp = _compose_response(
            {"paths": [], "admissibility_token": None, "stats": {}},
            self._make_rag_results(),
            max_tokens=500,
            include_graph=True,
        )
        assert resp.graph_context is None
        assert "gk" not in resp.sources

    def test_graph_excluded(self):
        resp = _compose_response(
            self._make_gk_result(),
            self._make_rag_results(),
            max_tokens=500,
            include_graph=False,
        )
        assert resp.graph_context is None
        assert "gk" not in resp.sources

    def test_token_budget_respected(self):
        """With a tiny budget, should truncate results."""
        resp = _compose_response(
            self._make_gk_result(),
            self._make_rag_results(10),
            max_tokens=50,  # very small budget
            include_graph=True,
        )
        # Should have fewer turns than input
        assert len(resp.related_turns) < 10
        # Token estimate should be near the budget
        assert resp.token_estimate <= 60  # some overhead

    def test_metadata_truncated_to_3_keys(self):
        resp = _compose_response(
            {"paths": [], "admissibility_token": None, "stats": {}},
            self._make_rag_results(1),
            max_tokens=500,
            include_graph=False,
        )
        assert len(resp.related_turns[0].metadata) <= 3

    def test_content_key_fallback(self):
        """Should handle both text_content and content keys (A1)."""
        results = [{"id": "1", "content": "fallback content", "score": 0.5, "metadata": {}}]
        resp = _compose_response(
            {"paths": [], "admissibility_token": None, "stats": {}},
            results,
            max_tokens=500,
            include_graph=False,
        )
        assert resp.related_turns[0].preview == "fallback content"

    def test_similarity_key_fallback(self):
        """Should handle both similarity and score keys (A1)."""
        results = [{"id": "1", "text_content": "test", "score": 0.75, "metadata": {}}]
        resp = _compose_response(
            {"paths": [], "admissibility_token": None, "stats": {}},
            results,
            max_tokens=500,
            include_graph=False,
        )
        assert resp.related_turns[0].score == 0.75


# ---------------------------------------------------------------------------
# Prometheus metrics tests
# ---------------------------------------------------------------------------


class TestPrometheusMetrics:
    def test_export_format_valid(self):
        output = export_prometheus_metrics()
        lines = output.strip().split("\n")
        for line in lines:
            if line.startswith("#"):
                assert line.startswith("# HELP") or line.startswith("# TYPE")
            else:
                # Metric line: name value
                parts = line.split(" ")
                assert len(parts) == 2, f"Invalid metric line: {line}"
                float(parts[1])  # should be numeric

    def test_export_contains_all_counters(self):
        output = export_prometheus_metrics()
        assert "context_gateway_requests_total" in output
        assert "context_gateway_requests_success_total" in output
        assert "context_gateway_requests_fallback_total" in output
        assert "context_gateway_latency_seconds_sum" in output
        assert "context_gateway_tokens_saved_total" in output
        assert "context_gateway_rlm_decompositions_total" in output


# ---------------------------------------------------------------------------
# Request model tests
# ---------------------------------------------------------------------------


class TestRequestModel:
    def test_valid_request(self):
        req = ContextGatewayRequest(query="test query")
        assert req.max_tokens == 500
        assert req.k_rag == 5

    def test_cwd_max_length(self):
        """S6: cwd should be bounded."""
        with pytest.raises(Exception):
            ContextGatewayRequest(query="test", cwd="x" * 5000)

    def test_empty_query_rejected(self):
        with pytest.raises(Exception):
            ContextGatewayRequest(query="")

    def test_max_tokens_bounds(self):
        with pytest.raises(Exception):
            ContextGatewayRequest(query="test", max_tokens=3000)


# ---------------------------------------------------------------------------
# Response model tests
# ---------------------------------------------------------------------------


class TestResponseModel:
    def test_fallback_reason_present(self):
        """X15: fallback responses should have a reason."""
        resp = ContextGatewayResponse(fallback=True, fallback_reason="credentials_missing")
        assert resp.fallback_reason == "credentials_missing"

    def test_default_fallback_reason_none(self):
        resp = ContextGatewayResponse()
        assert resp.fallback_reason is None


# ---------------------------------------------------------------------------
# Entity name validation tests (S2)
# ---------------------------------------------------------------------------


class TestEntityValidation:
    def test_valid_names(self):
        assert _ENTITY_NAME_RE.match("spore")
        assert _ENTITY_NAME_RE.match("evolution-world")
        assert _ENTITY_NAME_RE.match("creator-shield")
        assert _ENTITY_NAME_RE.match("cc-rag-plus-plus")

    def test_invalid_names(self):
        assert not _ENTITY_NAME_RE.match("..")
        assert not _ENTITY_NAME_RE.match("../../etc")
        assert not _ENTITY_NAME_RE.match("foo bar")
        assert not _ENTITY_NAME_RE.match("foo/bar")
        assert not _ENTITY_NAME_RE.match("")


# ---------------------------------------------------------------------------
# Echo suppression tests (Phase 1)
# ---------------------------------------------------------------------------

import numpy as np


class TestNoveltyBatch:
    def _random_embedding(self, seed=42):
        rng = np.random.RandomState(seed)
        v = rng.randn(EMBEDDING_DIM).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()

    def test_no_window_returns_neutral(self):
        results = [self._random_embedding(i) for i in range(3)]
        novelties = _compute_novelty_batch(results, None)
        assert all(n == 0.5 for n in novelties)

    def test_empty_results(self):
        assert _compute_novelty_batch([], self._random_embedding()) == []

    def test_identical_vectors_zero_novelty(self):
        v = self._random_embedding()
        novelties = _compute_novelty_batch([v], v)
        assert novelties[0] < 0.05  # near zero (identical = no novelty)

    def test_orthogonal_vectors_high_novelty(self):
        v1 = [0.0] * EMBEDDING_DIM
        v1[0] = 1.0
        v2 = [0.0] * EMBEDDING_DIM
        v2[1] = 1.0
        novelties = _compute_novelty_batch([v2], v1)
        assert novelties[0] > 0.9  # orthogonal = high novelty

    def test_batch_returns_correct_length(self):
        window = self._random_embedding(0)
        results = [self._random_embedding(i) for i in range(5)]
        novelties = _compute_novelty_batch(results, window)
        assert len(novelties) == 5

    def test_novelty_bounded_0_to_1(self):
        window = self._random_embedding(0)
        results = [self._random_embedding(i) for i in range(10)]
        novelties = _compute_novelty_batch(results, window)
        assert all(0.0 <= n <= 1.0 for n in novelties)


class TestEchoSuppression:
    def test_no_window_passthrough(self):
        """Without window embedding, results pass through unchanged."""
        results = [{"text_content": "test", "similarity": 0.8}]
        out, echo_rate, novelties = _score_with_echo_suppression(results, None)
        assert out == results
        assert echo_rate == 0.0
        assert novelties == []

    def test_echo_rate_computed(self):
        """Echo rate should be computed from similarity scores."""
        results = [
            {"text_content": f"doc {i}", "similarity": 0.9 - i * 0.1}
            for i in range(5)
        ]
        window = [0.0] * EMBEDDING_DIM
        _, echo_rate, novelties = _score_with_echo_suppression(results, window)
        assert 0.0 <= echo_rate <= 1.0
        assert len(novelties) == 5

    def test_empty_results(self):
        results, rate, novelties = _score_with_echo_suppression([], [0.0] * EMBEDDING_DIM)
        assert results == []
        assert rate == 0.0


class TestEchoSuppressionRequest:
    def test_valid_embedding_accepted(self):
        req = ContextGatewayRequest(
            query="test",
            current_window_embedding=[0.1] * EMBEDDING_DIM,
            enable_echo_suppression=True,
        )
        assert req.enable_echo_suppression is True
        assert len(req.current_window_embedding) == EMBEDDING_DIM

    def test_wrong_dim_rejected(self):
        """A6: wrong dimension embeddings should be rejected."""
        with pytest.raises(Exception):
            ContextGatewayRequest(
                query="test",
                current_window_embedding=[0.1] * 512,  # wrong dim
            )

    def test_none_embedding_accepted(self):
        req = ContextGatewayRequest(query="test", current_window_embedding=None)
        assert req.current_window_embedding is None

    def test_echo_suppression_default_false(self):
        """A9: echo suppression should default to False."""
        req = ContextGatewayRequest(query="test")
        assert req.enable_echo_suppression is False


class TestEchoSuppressionResponse:
    def test_echo_telemetry_fields(self):
        resp = ContextGatewayResponse(
            echo_rate=0.45,
            echo_suppression_active=True,
            novelty_scores=[0.3, 0.5, 0.7],
        )
        assert resp.echo_rate == 0.45
        assert resp.echo_suppression_active is True
        assert len(resp.novelty_scores) == 3

    def test_default_echo_fields(self):
        resp = ContextGatewayResponse()
        assert resp.echo_rate is None
        assert resp.echo_suppression_active is False
        assert resp.novelty_scores == []


class TestEchoPrometheusMetrics:
    def test_echo_metrics_in_export(self):
        output = export_prometheus_metrics()
        assert "context_gateway_echo_suppression_requests_total" in output
        assert "context_gateway_echo_rate_sum" in output
        assert "context_gateway_echo_rate_high_total" in output


# ---------------------------------------------------------------------------
# Session Bloom Filter tests (Wave 3)
# ---------------------------------------------------------------------------


class TestSessionBloomFilter:
    def test_add_and_contains(self):
        bf = SessionBloomFilter()
        assert "doc_1" not in bf
        was_present = bf.add("doc_1")
        assert was_present is False
        assert "doc_1" in bf

    def test_duplicate_detection(self):
        bf = SessionBloomFilter()
        bf.add("doc_1")
        was_present = bf.add("doc_1")
        assert was_present is True

    def test_count_tracks_unique(self):
        bf = SessionBloomFilter()
        bf.add("a")
        bf.add("b")
        bf.add("a")  # duplicate
        assert bf.count == 2

    def test_no_false_negatives(self):
        """Bloom filters must never have false negatives."""
        bf = SessionBloomFilter()
        items = [f"item_{i}" for i in range(100)]
        for item in items:
            bf.add(item)
        for item in items:
            assert item in bf

    def test_expiry(self):
        bf = SessionBloomFilter()
        bf._last_access = time.monotonic() - 7200  # 2 hours ago
        assert bf.is_expired(ttl=3600) is True

    def test_not_expired(self):
        bf = SessionBloomFilter()
        assert bf.is_expired(ttl=3600) is False

    def test_bytearray_size(self):
        """D8: verify we use bytearray, not bitarray."""
        bf = SessionBloomFilter(m=8192)
        assert isinstance(bf._bits, bytearray)
        assert len(bf._bits) == 1024  # 8192 bits / 8

    def test_different_items_different_positions(self):
        bf = SessionBloomFilter()
        bf.add("apple")
        bf.add("banana")
        # Both should be present
        assert "apple" in bf
        assert "banana" in bf
        # Something not added should probably not be present
        # (probabilistic, but with empty filter + 2 items, very unlikely false positive)
        false_positives = sum(1 for i in range(100) if f"never_added_{i}" in bf)
        assert false_positives < 10  # ~1.5% FP rate at low load


class TestMarkSeenAndFilter:
    def test_first_seen_no_duplicates(self):
        bf = SessionBloomFilter()
        results = [{"id": f"doc_{i}", "text_content": f"text {i}"} for i in range(3)]
        out, dupes = _mark_seen_and_filter(bf, results)
        assert dupes == 0
        assert len(out) == 3

    def test_repeated_results_detected(self):
        bf = SessionBloomFilter()
        results = [{"id": "doc_1", "text_content": "text 1"}]
        _mark_seen_and_filter(bf, results)  # first pass
        _, dupes = _mark_seen_and_filter(bf, results)  # second pass
        assert dupes == 1

    def test_mixed_new_and_duplicate(self):
        bf = SessionBloomFilter()
        _mark_seen_and_filter(bf, [{"id": "doc_1"}])
        results = [{"id": "doc_1"}, {"id": "doc_2"}, {"id": "doc_3"}]
        _, dupes = _mark_seen_and_filter(bf, results)
        assert dupes == 1  # only doc_1 is duplicate

    def test_empty_id_skipped(self):
        bf = SessionBloomFilter()
        results = [{"id": "", "text_content": "no id"}]
        _, dupes = _mark_seen_and_filter(bf, results)
        assert dupes == 0


class TestBloomResponseField:
    def test_bloom_duplicates_default(self):
        resp = ContextGatewayResponse()
        assert resp.bloom_duplicates == 0

    def test_bloom_duplicates_set(self):
        resp = ContextGatewayResponse(bloom_duplicates=3)
        assert resp.bloom_duplicates == 3


class TestBloomPrometheusMetrics:
    def test_bloom_metrics_in_export(self):
        output = export_prometheus_metrics()
        assert "context_gateway_bloom_sessions_created_total" in output
        assert "context_gateway_bloom_sessions_evicted_total" in output
        assert "context_gateway_bloom_duplicates_found_total" in output
        assert "context_gateway_bloom_active_sessions" in output


# ---------------------------------------------------------------------------
# Vector Expansion tests (Wave 4, P12)
# ---------------------------------------------------------------------------


class TestVectorExpansion:
    def test_expansion_reduces_window_similarity(self):
        """Expanded embedding should be less similar to window than original."""
        # Query partially aligned with window
        q = [0.0] * EMBEDDING_DIM
        q[0] = 0.8
        q[1] = 0.6  # 37 degree angle from x-axis
        w = [0.0] * EMBEDDING_DIM
        w[0] = 1.0  # pure x-axis

        expanded = _expand_query_embedding(q, w, expansion_weight=0.5)

        # Compute cosine similarities
        q_arr = np.array(q)
        w_arr = np.array(w)
        e_arr = np.array(expanded)

        orig_cos = np.dot(q_arr, w_arr) / (np.linalg.norm(q_arr) * np.linalg.norm(w_arr))
        exp_cos = np.dot(e_arr, w_arr) / (np.linalg.norm(e_arr) * np.linalg.norm(w_arr))

        assert exp_cos < orig_cos  # expanded is less similar to window

    def test_expansion_preserves_unit_norm(self):
        """Expanded embedding should be approximately unit-normalized."""
        q = [0.0] * EMBEDDING_DIM
        q[0] = 0.7
        q[1] = 0.7
        w = [0.0] * EMBEDDING_DIM
        w[0] = 1.0

        expanded = _expand_query_embedding(q, w, expansion_weight=0.5)
        norm = np.linalg.norm(expanded)
        assert abs(norm - 1.0) < 0.01

    def test_zero_weight_returns_normalized_original(self):
        """Weight=0 should return the original direction (normalized)."""
        q = [0.0] * EMBEDDING_DIM
        q[0] = 3.0
        q[1] = 4.0  # norm = 5
        w = [0.0] * EMBEDDING_DIM
        w[0] = 1.0

        expanded = _expand_query_embedding(q, w, expansion_weight=0.0)
        # Should be in same direction as q
        q_unit = np.array(q) / np.linalg.norm(q)
        cos = np.dot(np.array(expanded), q_unit)
        assert cos > 0.99

    def test_parallel_vectors_returns_original(self):
        """If query is parallel to window, can't expand — returns original."""
        q = [0.0] * EMBEDDING_DIM
        q[0] = 1.0
        w = [0.0] * EMBEDDING_DIM
        w[0] = 2.0  # same direction

        expanded = _expand_query_embedding(q, w, expansion_weight=0.5)
        assert expanded == q  # returns original unchanged

    def test_zero_window_returns_original(self):
        """Zero window embedding should return original query."""
        q = [0.0] * EMBEDDING_DIM
        q[0] = 1.0
        w = [0.0] * EMBEDDING_DIM  # zero vector

        expanded = _expand_query_embedding(q, w, expansion_weight=0.5)
        assert expanded == q

    def test_expansion_output_length(self):
        """Output should have same dimension as input."""
        q = [0.1] * EMBEDDING_DIM
        w = [0.2] * EMBEDDING_DIM

        expanded = _expand_query_embedding(q, w, expansion_weight=0.3)
        assert len(expanded) == EMBEDDING_DIM


class TestOverlapClass:
    def test_novel(self):
        assert OverlapClass.classify(0.1) == "novel"
        assert OverlapClass.classify(0.0) == "novel"
        assert OverlapClass.classify(0.29) == "novel"

    def test_moderate(self):
        assert OverlapClass.classify(0.3) == "moderate"
        assert OverlapClass.classify(0.5) == "moderate"
        assert OverlapClass.classify(0.59) == "moderate"

    def test_high_echo(self):
        assert OverlapClass.classify(0.6) == "high_echo"
        assert OverlapClass.classify(0.9) == "high_echo"
        assert OverlapClass.classify(1.0) == "high_echo"


class TestContextTurnSource:
    def test_default_source_rag(self):
        """A5: ContextTurn should have a source field defaulting to 'rag'."""
        turn = ContextTurn()
        assert turn.source == "rag"

    def test_source_gk(self):
        turn = ContextTurn(source="gk")
        assert turn.source == "gk"


class TestWave4ResponseFields:
    def test_overlap_class_default(self):
        resp = ContextGatewayResponse()
        assert resp.overlap_class is None
        assert resp.query_expanded is False

    def test_overlap_class_set(self):
        resp = ContextGatewayResponse(overlap_class="high_echo", query_expanded=True)
        assert resp.overlap_class == "high_echo"
        assert resp.query_expanded is True


class TestWave4PrometheusMetrics:
    def test_expansion_metrics_in_export(self):
        output = export_prometheus_metrics()
        assert "context_gateway_vector_expansions_total" in output


# ---------------------------------------------------------------------------
# Mock mode tests (Wave 5, X13)
# ---------------------------------------------------------------------------


class TestMockMode:
    def test_mock_rag_results_shape(self):
        results = _mock_rag_results("test query", 5)
        assert len(results) == 3  # capped at 3 for mock
        for r in results:
            assert "id" in r
            assert "text_content" in r
            assert "similarity" in r
            assert r["metadata"]["source"] == "mock"
            assert r["id"].startswith("mock_")

    def test_mock_rag_results_deterministic(self):
        """Same query should produce same mock IDs."""
        r1 = _mock_rag_results("test", 3)
        r2 = _mock_rag_results("test", 3)
        assert r1[0]["id"] == r2[0]["id"]

    def test_mock_rag_different_queries(self):
        """Different queries should produce different mock IDs."""
        r1 = _mock_rag_results("query A", 3)
        r2 = _mock_rag_results("query B", 3)
        assert r1[0]["id"] != r2[0]["id"]

    def test_mock_gk_result_with_project(self):
        result = _mock_gk_result("spore")
        assert len(result["paths"]) == 1
        assert result["stats"]["mock"] is True

    def test_mock_gk_result_no_project(self):
        result = _mock_gk_result(None)
        assert result["paths"] == []

    def test_mock_embedding_shape(self):
        emb = _mock_embedding()
        assert len(emb) == EMBEDDING_DIM
        norm = sum(x * x for x in emb) ** 0.5
        assert abs(norm - 1.0) < 0.01  # unit normalized

    def test_mock_embedding_deterministic(self):
        e1 = _mock_embedding()
        e2 = _mock_embedding()
        assert e1 == e2

    def test_mock_results_compose_correctly(self):
        """Mock results should work with _compose_response."""
        rag = _mock_rag_results("test", 3)
        gk = _mock_gk_result("test-project")
        resp = _compose_response(gk, rag, max_tokens=500, include_graph=True)
        assert len(resp.related_turns) == 3
        assert resp.graph_context is not None
        assert resp.fallback is False
