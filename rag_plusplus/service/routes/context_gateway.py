"""Smart Context Gateway — GK + RAG++ Token Optimization.

Single endpoint that composes Graph Kernel context slicing with RAG++ hybrid
search into one compact, token-budgeted response. Replaces multiple expensive
MCP tool calls and bloated startup hooks with a single HTTP call.

Target: ~400-500 token scoped context block with HMAC provenance.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Annotated

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Config — read once at import time, fail fast on missing keys
# ---------------------------------------------------------------------------

GRAPH_KERNEL_URL = os.getenv("GRAPH_KERNEL_URL", "http://127.0.0.1:8001")
CHARS_PER_TOKEN = 4  # Matches server.py convention

# X13: Mock mode for local development without live credentials.
# Set CONTEXT_GATEWAY_MOCK=1 to enable. Returns synthetic results.
_MOCK_MODE = os.environ.get("CONTEXT_GATEWAY_MOCK", "").lower() in ("1", "true", "yes")

# Credentials: read at import. Missing keys log a warning (not crash)
# so the service can still serve /health and /embed even if RAG is unconfigured.
_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
_GOOGLE_KEY = os.environ.get("GOOGLE_API_KEY", "")

if _MOCK_MODE:
    logger.info("CONTEXT_GATEWAY_MOCK=1 — running in mock mode (synthetic results, no external calls)")
elif not _GOOGLE_KEY:
    logger.warning("GOOGLE_API_KEY not set — RAG search, RLM, and /embed will be unavailable")
if not _MOCK_MODE and (not _SUPABASE_URL or not _SUPABASE_KEY):
    logger.warning("SUPABASE_URL/SUPABASE_SERVICE_KEY not set — RAG search will be unavailable")

# Gemini embedding endpoint (key passed via x-goog-api-key header, NOT URL param)
_GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
_GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Gemini embedding dimensions (gemini-embedding-001 = 768)
EMBEDDING_DIM = 768

# High-signal edge predicates for graph context. Structural noise
# (discord_channel, completion_pct, tasks_*, status, priority, prefix) is filtered out.
# RLM signal phrases that trigger decomposition (expanded from layers/rlm.py)
# Single-word signals require query length >= 8 words (see _should_decompose)
MULTI_HOP_SIGNALS = [
    # Relational / causal (strong signals)
    "how does", "how do", "how is", "how are",
    "what connects", "relationship between", "connection between",
    "difference between", "compare",
    # Causal / temporal
    "why does", "why do", "why is",
    "what led to", "what causes", "what caused",
    "trace the", "trace how",
    # Explanatory
    "explain how", "explain why", "explain the",
    "impact of", "effect of", "consequence of",
    # Multi-entity
    "interact with", "work together", "work with",
    "relate to", "depend on", "depends on",
    "flow from", "flow between", "path from",
    # Cross-system
    "which projects", "which services", "which systems",
    "across", "between.*and",
]
# Minimum query word count for decomposition (avoids short simple questions)
MIN_DECOMPOSE_WORDS = 6

HIGH_SIGNAL_PREDICATES = frozenset({
    "works_on",
    "built_with",
    "has_feature",
    "has_component",
    "uses",
    "evolved_from",
    "potential_merge_candidate_with",
    "has_service",
    "deployed_on",
    "deployed_via",
    "is_a",
    "belongs_to",
    "tagged",
    "has_layer",
    "part_of",
    "has_product",
    "has_repositories",
    "networked_via",
    "uses_cloud",
    "building",
    "has_cron_job",
    "has_background_pipelines",
    "author",
    "created",
    "description",
    "category",
})

# Shared httpx client for GK (lazy singleton)
_GK_CLIENT: Optional[httpx.AsyncClient] = None

# Shared httpx client for Gemini + Supabase calls (lazy singleton, replaces per-request creation)
_EXTERNAL_CLIENT: Optional[httpx.AsyncClient] = None


def _get_gk_client() -> httpx.AsyncClient:
    global _GK_CLIENT
    if _GK_CLIENT is None or _GK_CLIENT.is_closed:
        _GK_CLIENT = httpx.AsyncClient(
            base_url=GRAPH_KERNEL_URL,
            timeout=httpx.Timeout(3.0, connect=2.0),
        )
    return _GK_CLIENT


def _get_external_client() -> httpx.AsyncClient:
    """Shared client for Gemini API and Supabase RPC calls (connection pooling)."""
    global _EXTERNAL_CLIENT
    if _EXTERNAL_CLIENT is None or _EXTERNAL_CLIENT.is_closed:
        _EXTERNAL_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(5.0, connect=2.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
    return _EXTERNAL_CLIENT


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ContextTurn(BaseModel):
    id: str = ""
    preview: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str = "rag"  # A5: distinguish RAG vs GK origin


class ContextGatewayRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Semantic query")
    cwd: str = Field("", max_length=4096, description="Working directory for project scoping")
    session_id: str = Field("", description="Session ID for scoping")
    max_tokens: int = Field(500, ge=50, le=2000, description="Hard output token cap")
    include_graph: bool = Field(True, description="Include GK traversal")
    k_rag: int = Field(5, ge=1, le=20, description="RAG++ result count")
    include_rlm: bool = Field(True, description="Enable recursive exploration for multi-hop queries")
    rlm_max_depth: int = Field(2, ge=0, le=5, description="Max recursion depth for exploration")
    # Echo suppression (Phase 1): default OFF until baseline measured (A9)
    enable_echo_suppression: bool = Field(
        False,
        description="Enable self-referential echo suppression. Default False until baseline established.",
    )
    current_window_embedding: Optional[List[float]] = Field(
        None,
        description=f"Current context window embedding ({EMBEDDING_DIM}-dim Gemini). Used for novelty scoring.",
    )

    @field_validator("current_window_embedding")
    @classmethod
    def validate_embedding_dim(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """A6 fix: Validate embedding dimensionality to prevent silent truncation."""
        if v is not None and len(v) != EMBEDDING_DIM:
            raise ValueError(f"current_window_embedding must be {EMBEDDING_DIM}-dimensional, got {len(v)}")
        return v


class ContextGatewayResponse(BaseModel):
    admissibility_token: Optional[str] = None
    related_turns: List[ContextTurn] = Field(default_factory=list)
    graph_context: Optional[Dict[str, Any]] = None
    token_estimate: int = 0
    sources: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    fallback: bool = False
    fallback_reason: Optional[str] = None
    # Echo suppression telemetry
    echo_rate: Optional[float] = None
    echo_suppression_active: bool = False
    novelty_scores: List[float] = Field(default_factory=list)
    # Bloom filter telemetry
    bloom_duplicates: int = 0
    # Overlap classification (A10: enum, not free string)
    overlap_class: Optional[str] = None
    query_expanded: bool = False
    rlm_decomposed: bool = False
    rlm_sub_queries: List[str] = Field(default_factory=list)
    rlm_decompose_ms: float = 0.0
    rlm_depth_reached: int = 0
    rlm_converged: bool = False
    rlm_total_nodes: int = 0


# ---------------------------------------------------------------------------
# Project extraction (A7: most-specific patterns first)
# ---------------------------------------------------------------------------

_PROJECT_PATTERNS = [
    # Most-specific first — deeper paths before shallow
    r"/projects/([^/]+)",
    r"/.clawdbot/([^/]+)",
    r"/monitoring/([^/]+)",
    r"/flows/([^/]+)",
    # Least-specific last — Desktop is a catch-all
    r"/Desktop/([^/]+)",
]

# Entity name validation: only allow safe characters for GK queries (S2)
_ENTITY_NAME_RE = re.compile(r"^[a-z0-9_-]+$")


def _extract_project(cwd: str) -> Optional[str]:
    for pattern in _PROJECT_PATTERNS:
        m = re.search(pattern, cwd)
        if m:
            name = m.group(1)
            # Validate entity name before sending to GK
            if _ENTITY_NAME_RE.match(name.lower().replace("_", "-")):
                return name
            logger.debug(f"Rejected project name '{name}' — invalid characters")
            return None
    return None


# ---------------------------------------------------------------------------
# Mock mode helpers (X13: local dev without credentials)
# ---------------------------------------------------------------------------


def _mock_rag_results(query: str, k: int) -> List[Dict[str, Any]]:
    """Generate synthetic RAG results for mock mode."""
    return [
        {
            "id": f"mock_{i}_{hashlib.md5(query.encode()).hexdigest()[:8]}",
            "text_content": f"[MOCK] Result {i+1} for '{query[:50]}': synthetic content for local development.",
            "similarity": round(0.9 - i * 0.1, 2),
            "metadata": {"source": "mock", "model_id": "mock-model"},
        }
        for i in range(min(k, 3))
    ]


def _mock_gk_result(project: Optional[str]) -> Dict[str, Any]:
    """Generate synthetic GK result for mock mode."""
    if not project:
        return {"paths": [], "admissibility_token": None, "stats": {}}
    return {
        "paths": [
            {"edges": [{"predicate": "uses", "target": "mock-service", "source": project}]},
        ],
        "admissibility_token": None,
        "stats": {"raw_paths": 1, "filtered_paths": 1, "start_entity": project, "mock": True},
    }


def _mock_embedding() -> List[float]:
    """Generate a deterministic mock embedding for local dev."""
    rng = np.random.RandomState(42)
    v = rng.randn(EMBEDDING_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ---------------------------------------------------------------------------
# GK + RAG++ queries (run in parallel)
# ---------------------------------------------------------------------------


async def _query_gk_traverse(project: Optional[str], query: str) -> Dict[str, Any]:
    """Multi-hop traversal from Graph Kernel. Returns paths + admissibility token."""
    if _MOCK_MODE:
        return _mock_gk_result(project)
    client = _get_gk_client()
    result: Dict[str, Any] = {"paths": [], "admissibility_token": None, "stats": {}}

    # NOTE: /api/slice requires anchor_turn_id (UUID) referencing a turn in GK's
    # internal store. Currently no turns are ingested into GK, so slice always
    # fails with 422. Skipping to save ~100ms latency.
    # TODO(gk-ingestion): re-enable when turn ingestion pipeline is connected.

    # Knowledge traversal for structural context (filtered to high-signal edges)
    # GK entities are bare lowercase names (e.g. "spore", "evolution"), not prefixed.
    # Try exact name first, then first word (e.g. "evolution_world" -> "evolution").
    start_candidates = []
    if project:
        name = project.lower().replace("_", "-")
        start_candidates.append(name)
        first_word = name.split("-")[0]
        if first_word != name:
            start_candidates.append(first_word)
    if not start_candidates and query:
        word = query.split()[0].lower()
        if _ENTITY_NAME_RE.match(word):
            start_candidates.append(word)

    # Try each candidate until one returns paths
    for start_entity in start_candidates:
        try:
            traverse_resp = await client.post("/api/knowledge/traverse", json={
                "start": start_entity,
                "max_hops": 2,
                "direction": "both",
                "max_results": 50,
            })
            if traverse_resp.status_code == 200:
                data = traverse_resp.json()
                raw_paths = data.get("paths", [])

                # Filter: only keep paths with high-signal edges
                filtered_paths = [
                    path for path in raw_paths
                    if any(e.get("predicate") in HIGH_SIGNAL_PREDICATES for e in path.get("edges", []))
                ][:10]

                result["paths"] = filtered_paths
                result["stats"] = data.get("stats", {})
                result["stats"]["raw_paths"] = len(raw_paths)
                result["stats"]["filtered_paths"] = len(filtered_paths)
                result["stats"]["start_entity"] = start_entity
                if filtered_paths:
                    break  # Found results, skip remaining candidates
        except Exception as e:
            logger.debug(f"GK traverse failed for '{start_entity}': {e}")

    return result


async def _generate_embedding(text: str) -> List[float]:
    """Generate a Gemini embedding. Returns [] on failure.

    S1 fix: API key sent via header, not URL parameter.
    P1 fix: uses shared pooled client.
    """
    if _MOCK_MODE:
        return _mock_embedding()
    if not _GOOGLE_KEY:
        return []
    client = _get_external_client()
    try:
        resp = await client.post(
            _GEMINI_EMBED_URL,
            json={"content": {"parts": [{"text": text[:4000]}]}},
            headers={"x-goog-api-key": _GOOGLE_KEY},
        )
        if resp.status_code != 200:
            return []
        return resp.json().get("embedding", {}).get("values", [])
    except Exception as e:
        logger.debug(f"Embedding generation failed: {e}")
        return []


async def _query_rag_search(query: str, k: int, cwd: str) -> List[Dict[str, Any]]:
    """Hybrid search via RAG++ pgvector (BM25 + dense)."""
    if _MOCK_MODE:
        return _mock_rag_results(query, k)
    if not _SUPABASE_URL or not _SUPABASE_KEY:
        return []

    embedding = await _generate_embedding(query)
    if not embedding:
        return []

    client = _get_external_client()
    try:
        search_resp = await client.post(
            f"{_SUPABASE_URL}/rest/v1/rpc/search_rag_embeddings",
            json={
                "query_embedding": json.dumps(embedding),
                "match_count": k,
            },
            headers={
                "Content-Type": "application/json",
                "apikey": _SUPABASE_KEY,
                "Authorization": f"Bearer {_SUPABASE_KEY}",
            },
        )
        if search_resp.status_code == 200:
            # P5 fix: parse JSON once, not twice
            data = search_resp.json()
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.debug(f"RAG++ search failed: {e}")

    return []


# ---------------------------------------------------------------------------
# RLM — Recursive Language Model decomposition
# ---------------------------------------------------------------------------


def _should_decompose(query: str) -> bool:
    """Detect if a query needs multi-hop decomposition.

    Uses keyword matching with a minimum word-count gate to avoid
    decomposing short simple questions like "How many tables?".
    """
    words = query.split()
    if len(words) < MIN_DECOMPOSE_WORDS:
        return False
    ql = query.lower()
    for sig in MULTI_HOP_SIGNALS:
        if ".*" in sig:
            # Regex pattern (e.g. "between.*and")
            if re.search(sig, ql):
                return True
        elif sig in ql:
            return True
    return False


async def _decompose_query_gemini(query: str) -> List[str]:
    """Decompose a complex query into 2-3 sub-queries using Gemini Flash.

    Uses the REST API directly (no SDK). Returns [query] on any failure.
    S1 fix: API key in header. P1 fix: shared client.
    """
    if not _GOOGLE_KEY:
        return [query]

    system_prompt = (
        "You decompose complex infrastructure questions into 2-3 simpler sub-questions. "
        "The system has: machines (Mac1-Mac5, cloud-vm), services (RAG++, Graph Kernel, "
        "Prefect, Nexus Portal, NUMU), projects (Cortex, Evolution World, Creator Shield, "
        "Pane Orchestrator), and data stores (Supabase, pgvector). "
        "Each sub-question should target a specific entity or relationship. "
        "Output ONLY a JSON array of strings. "
        'Example: ["What is service X and where does it run?", "How does X send data to Y?"]'
    )

    client = _get_external_client()
    try:
        resp = await client.post(
            _GEMINI_GENERATE_URL,
            json={
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"parts": [{"text": query}]}],
                "generationConfig": {
                    "maxOutputTokens": 200,
                    "temperature": 0.0,
                    "responseMimeType": "application/json",
                },
            },
            headers={"x-goog-api-key": _GOOGLE_KEY},
        )
        if resp.status_code != 200:
            logger.debug(f"RLM decompose failed: HTTP {resp.status_code}")
            return [query]

        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        # Try direct JSON parse first (responseMimeType should give clean JSON)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [s for s in parsed[:3] if isinstance(s, str)] or [query]
        except json.JSONDecodeError:
            pass
        # Fallback: extract array from text
        start = text.index("[")
        end = text.rindex("]") + 1
        subs = json.loads(text[start:end])
        return subs[:3] if isinstance(subs, list) and subs else [query]
    except Exception as e:
        logger.debug(f"RLM decompose error: {e}")
        return [query]


async def _rlm_search(
    query: str,
    k: int,
    cwd: str,
    include_rlm: bool,
    max_depth: int = 2,
) -> tuple[List[Dict[str, Any]], bool, List[str], float, int, bool, int]:
    """Recursive exploration pipeline: detect -> explore tree -> flatten.

    Returns: (results, decomposed, sub_queries, latency_ms, depth_reached, converged, total_nodes)
    """
    if not include_rlm or not _should_decompose(query):
        # Simple path: single search, no decomposition
        results = await _query_rag_search(query, k, cwd)
        return results, False, [], 0.0, 0, False, 0

    # D2 fix: Use the installed package path, remove Mac1-specific fallback
    try:
        from cognitive_twin_layers.recursive_explorer import explore, ExplorationConfig
    except ImportError:
        logger.warning(
            "recursive_explorer not available (cognitive_twin_layers not installed), "
            "falling back to flat search"
        )
        results = await _query_rag_search(query, k, cwd)
        return results, False, [], 0.0, 0, False, 0

    config = ExplorationConfig(
        max_depth=max_depth,
        max_total_queries=max(6, max_depth * 3),
        epsilon=0.15 if max_depth <= 2 else 0.1,
        k_rag=k,
        timeout_s=8.0 if max_depth <= 2 else 15.0,
        cwd=cwd,
    )

    t0 = time.monotonic()
    tree = await explore(query, config)
    elapsed_ms = (time.monotonic() - t0) * 1000

    # Track metrics
    _METRICS["rlm_decompositions_total"] += 1
    _METRICS["rlm_sub_queries_total"] += tree.state.total_queries
    _METRICS["rlm_decompose_latency_sum_ms"] += elapsed_ms

    # Flatten tree to top-k results
    flat_results = tree.flatten_results(top_k=k)

    # Collect all sub-queries from the tree
    sub_queries = tree.root.sub_queries

    depth_reached = tree.to_dict().get("max_depth_reached", 0)
    decomposed = tree.total_nodes > 1

    return flat_results, decomposed, sub_queries, elapsed_ms, depth_reached, tree.converged, tree.total_nodes


# ---------------------------------------------------------------------------
# Echo Suppression — Novelty-Aware Scoring (Phase 1)
# ---------------------------------------------------------------------------


def _compute_novelty_batch(
    result_embeddings: List[List[float]],
    window_embedding: Optional[List[float]],
) -> List[float]:
    """Vectorized cosine distance between results and window. Higher = more novel.

    P7 fix: uses numpy BLAS instead of pure Python loops.
    Returns 0.5 (neutral) when no window embedding available.
    """
    n = len(result_embeddings)
    if not window_embedding or n == 0:
        return [0.5] * n

    R = np.array(result_embeddings, dtype=np.float32)  # (k, 768)
    w = np.array(window_embedding, dtype=np.float32)    # (768,)

    R_norms = np.linalg.norm(R, axis=1)  # (k,)
    w_norm = np.linalg.norm(w)

    if w_norm < 1e-10:
        return [0.5] * n

    # Cosine similarity: dot(R, w) / (|R| * |w|)
    dots = R @ w  # (k,)
    cosines = dots / (R_norms * w_norm + 1e-10)

    # Novelty = 1 - similarity (clamped to [0, 1])
    novelties = np.clip(1.0 - cosines, 0.0, 1.0)
    return novelties.tolist()


def _score_with_echo_suppression(
    rag_results: List[Dict[str, Any]],
    window_embedding: Optional[List[float]],
) -> tuple[List[Dict[str, Any]], float, List[float]]:
    """Apply novelty-aware scoring to RAG results.

    Returns: (reranked_results, echo_rate, novelty_scores)
    Echo rate = fraction of results with novelty < 0.3 (high overlap with window).
    """
    if not rag_results or not window_embedding:
        return rag_results, 0.0, []

    # Extract embeddings from results (if available from pgvector)
    # The search_rag_embeddings RPC returns similarity scores but not raw embeddings.
    # For Phase 1, we re-embed result previews to compute novelty.
    # This is the shadow telemetry path — we measure but don't rerank yet.
    result_texts = [
        (r.get("text_content", "") or r.get("content", "") or "")[:500]
        for r in rag_results
    ]

    # Generate embeddings for result texts (batch via numpy, not API calls)
    # For Phase 1 shadow mode: use the base_score (similarity) as a proxy
    # and compute novelty relative to window embedding using cosine distance.
    # Full re-embedding will be Phase 2 when we have the embedding cache.
    base_scores = [
        float(r.get("similarity", 0.0) or r.get("score", 0.0) or 0.0)
        for r in rag_results
    ]

    # Shadow telemetry: estimate echo rate from base similarity scores
    # High similarity to query + high similarity to window = likely echo
    # For now, use base_score as novelty proxy (inverted: high score = low novelty)
    novelty_scores = [max(0.0, 1.0 - s) for s in base_scores]
    echo_rate = sum(1 for n in novelty_scores if n < 0.3) / max(len(novelty_scores), 1)

    # Phase 1: shadow mode — log metrics but don't rerank
    # Dynamic weight ramp (for future use when enable_echo_suppression=True):
    # t = max(0.0, min(1.0, (echo_rate - 0.3) / 0.4))
    # w_rel = 0.45 - 0.15 * t
    # w_nov = 0.25 + 0.15 * t

    return rag_results, echo_rate, novelty_scores


# ---------------------------------------------------------------------------
# Vector Expansion — Novelty-Biased Query Steering (Wave 4, P12)
# ---------------------------------------------------------------------------
# When echo rate is high, expand the query embedding perpendicular to the
# window embedding to bias retrieval toward novel content. Uses linear
# algebra (Gram-Schmidt orthogonalization) instead of a third Gemini API call.


class OverlapClass:
    """A10 fix: Enum-like overlap classification replacing free-string."""
    NOVEL = "novel"           # echo_rate < 0.3 — mostly new content
    MODERATE = "moderate"     # 0.3 <= echo_rate < 0.6
    HIGH_ECHO = "high_echo"   # echo_rate >= 0.6 — significant repetition

    @staticmethod
    def classify(echo_rate: float) -> str:
        if echo_rate < 0.3:
            return OverlapClass.NOVEL
        elif echo_rate < 0.6:
            return OverlapClass.MODERATE
        return OverlapClass.HIGH_ECHO


def _expand_query_embedding(
    query_embedding: List[float],
    window_embedding: List[float],
    expansion_weight: float = 0.3,
) -> List[float]:
    """Expand query embedding perpendicular to window embedding.

    P12 fix: Uses Gram-Schmidt orthogonalization to extract the component of
    the query that is NOVEL relative to the window. This steers retrieval
    toward content the user hasn't seen, without an extra Gemini API call.

    expansion_weight controls the blend: 0.0 = original query, 1.0 = pure novel component.
    Default 0.3 is conservative — enough to diversify without losing relevance.

    Returns the expanded (re-normalized) embedding vector.
    """
    q = np.array(query_embedding, dtype=np.float32)
    w = np.array(window_embedding, dtype=np.float32)

    w_norm = np.linalg.norm(w)
    if w_norm < 1e-10:
        return query_embedding  # degenerate window, skip expansion

    # Project query onto window direction
    w_unit = w / w_norm
    projection = np.dot(q, w_unit) * w_unit

    # Perpendicular component = what's novel in the query relative to window
    perp = q - projection
    perp_norm = np.linalg.norm(perp)

    if perp_norm < 1e-10:
        return query_embedding  # query is parallel to window, can't expand

    # Blend: (1 - weight) * original + weight * perpendicular
    expanded = (1.0 - expansion_weight) * q + expansion_weight * (perp / perp_norm) * np.linalg.norm(q)

    # Re-normalize to unit sphere
    exp_norm = np.linalg.norm(expanded)
    if exp_norm < 1e-10:
        return query_embedding
    expanded = expanded / exp_norm

    return expanded.tolist()


# ---------------------------------------------------------------------------
# Session Bloom Filter — Per-Session Duplicate Tracking (Wave 3)
# ---------------------------------------------------------------------------
# D8: Pure Python bytearray (no bitarray C extension)
# C2: asyncio.Lock for concurrent access to _SESSION_BLOOMS
# C5/P11: Bounded size with LRU eviction
# C11: TTL-based session expiry

_MAX_SESSIONS = 200  # Max concurrent sessions tracked
_SESSION_TTL_S = 3600  # 1 hour TTL per session
_BLOOM_SIZE = 8192  # Bits in Bloom filter (1KB per session)
_BLOOM_HASHES = 5  # Number of hash functions


class SessionBloomFilter:
    """Compact probabilistic set for tracking seen document IDs per session.

    Uses bytearray (bit-packed) with k independent hash functions derived
    from SHA-256. False positive rate ~1.5% at 1000 insertions with m=8192, k=5.
    """

    __slots__ = ("_bits", "_m", "_k", "_count", "_created_at", "_last_access")

    def __init__(self, m: int = _BLOOM_SIZE, k: int = _BLOOM_HASHES) -> None:
        self._m = m
        self._k = k
        self._bits = bytearray((m + 7) // 8)  # ceil(m/8) bytes
        self._count = 0
        self._created_at = time.monotonic()
        self._last_access = self._created_at

    def _hash_positions(self, item: str) -> List[int]:
        """Generate k hash positions from SHA-256 digest."""
        digest = hashlib.sha256(item.encode("utf-8")).digest()
        positions = []
        for i in range(self._k):
            # Use 4-byte chunks from the 32-byte digest
            offset = (i * 4) % 28  # Stay within 32-byte digest
            val = int.from_bytes(digest[offset:offset + 4], "big")
            positions.append(val % self._m)
        return positions

    def add(self, item: str) -> bool:
        """Add item. Returns True if item was already probably present."""
        positions = self._hash_positions(item)
        was_present = all(self._bits[p >> 3] & (1 << (p & 7)) for p in positions)
        for p in positions:
            self._bits[p >> 3] |= 1 << (p & 7)
        if not was_present:
            self._count += 1
        self._last_access = time.monotonic()
        return was_present

    def __contains__(self, item: str) -> bool:
        positions = self._hash_positions(item)
        self._last_access = time.monotonic()
        return all(self._bits[p >> 3] & (1 << (p & 7)) for p in positions)

    @property
    def count(self) -> int:
        return self._count

    def is_expired(self, ttl: float = _SESSION_TTL_S) -> bool:
        return (time.monotonic() - self._last_access) > ttl


# Session storage with lock + bounded size
_SESSION_BLOOMS: Dict[str, SessionBloomFilter] = {}
_SESSION_LOCK = asyncio.Lock()


async def _get_session_bloom(session_id: str) -> SessionBloomFilter:
    """Get or create a Bloom filter for a session. Evicts expired + LRU on overflow."""
    async with _SESSION_LOCK:
        # Evict expired sessions first
        expired = [sid for sid, bf in _SESSION_BLOOMS.items() if bf.is_expired()]
        for sid in expired:
            del _SESSION_BLOOMS[sid]
        if expired:
            _METRICS["bloom_sessions_evicted"] += len(expired)

        # LRU eviction if still over limit
        while len(_SESSION_BLOOMS) >= _MAX_SESSIONS:
            oldest = min(_SESSION_BLOOMS, key=lambda s: _SESSION_BLOOMS[s]._last_access)
            del _SESSION_BLOOMS[oldest]
            _METRICS["bloom_sessions_evicted"] += 1

        if session_id not in _SESSION_BLOOMS:
            _SESSION_BLOOMS[session_id] = SessionBloomFilter()
            _METRICS["bloom_sessions_created"] += 1

        return _SESSION_BLOOMS[session_id]


def _mark_seen_and_filter(
    bloom: SessionBloomFilter,
    rag_results: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], int]:
    """Mark result IDs as seen. Returns (results, duplicates_found).

    Phase 2: duplicate tracking only (telemetry). Does NOT remove duplicates
    from results yet — that comes in Phase 3 when we have baseline data.
    """
    duplicates = 0
    for r in rag_results:
        doc_id = r.get("id", "")
        if doc_id and bloom.add(doc_id):
            duplicates += 1
    return rag_results, duplicates


# ---------------------------------------------------------------------------
# Token budget composition
# ---------------------------------------------------------------------------


def _compose_response(
    gk_result: Dict[str, Any],
    rag_results: List[Dict[str, Any]],
    max_tokens: int,
    include_graph: bool,
) -> ContextGatewayResponse:
    """Compose GK + RAG++ results within token budget."""
    sources: List[str] = []
    char_budget = max_tokens * CHARS_PER_TOKEN
    chars_used = 0

    # 1. Related turns from RAG++ (60% of budget)
    turn_budget = int(char_budget * 0.6)
    related_turns: List[ContextTurn] = []
    for row in rag_results:
        text = row.get("text_content", "") or row.get("content", "") or ""
        score = row.get("similarity", 0.0) or row.get("score", 0.0) or 0.0
        doc_id = row.get("id", "")
        metadata = row.get("metadata", {}) or {}

        # Progressive truncation: fit within remaining budget
        remaining = turn_budget - chars_used
        if remaining <= 0:
            break
        preview = text[:min(len(text), remaining, 300)]
        if len(text) > len(preview):
            preview = preview.rstrip() + "..."

        related_turns.append(ContextTurn(
            id=doc_id,
            preview=preview,
            score=round(float(score), 4),
            metadata={k: v for k, v in list(metadata.items())[:3]} if metadata else {},
        ))
        chars_used += len(preview) + 50  # overhead for JSON structure

    if related_turns:
        sources.append("rag++")
        sources.append("pgvector")

    # 2. Graph context (30% of budget, if requested and available)
    graph_context = None
    gc_chars = 0
    graph_budget = int(char_budget * 0.3)
    if include_graph and gk_result.get("paths"):
        # Compact the graph paths to fit budget
        compact_paths = []
        for path in gk_result["paths"]:
            path_str = json.dumps(path, default=str)
            if gc_chars + len(path_str) > graph_budget:
                break
            compact_paths.append(path)
            gc_chars += len(path_str)

        if compact_paths:
            graph_context = {
                "paths": compact_paths,
                "stats": gk_result.get("stats", {}),
            }
            sources.append("gk")

    # 3. Admissibility token (10% overhead)
    admissibility_token = gk_result.get("admissibility_token")

    # P4 fix: estimate tokens from tracked char counts instead of full JSON serialize
    token_estimate = (chars_used + gc_chars + (len(str(admissibility_token)) if admissibility_token else 0)) // CHARS_PER_TOKEN

    return ContextGatewayResponse(
        admissibility_token=admissibility_token,
        related_turns=related_turns,
        graph_context=graph_context,
        token_estimate=token_estimate,
        sources=sources,
        fallback=False,
    )


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

_METRICS: Dict[str, Any] = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_fallback": 0,
    "latency_sum_ms": 0.0,
    "tokens_saved_total": 0,
    "rlm_decompositions_total": 0,
    "rlm_sub_queries_total": 0,
    "rlm_decompose_latency_sum_ms": 0.0,
    # Echo suppression shadow telemetry
    "echo_suppression_requests": 0,
    "echo_rate_sum": 0.0,
    "echo_rate_high_count": 0,  # requests where echo_rate > 0.6
    # Bloom filter session tracking
    "bloom_sessions_created": 0,
    "bloom_sessions_evicted": 0,
    "bloom_duplicates_found": 0,
    "bloom_lookups_total": 0,
    # Vector expansion
    "vector_expansions_triggered": 0,
}


def get_gateway_metrics() -> Dict[str, Any]:
    return dict(_METRICS)


# ---------------------------------------------------------------------------
# API Endpoint
# ---------------------------------------------------------------------------


@router.post("/gateway/context", response_model=ContextGatewayResponse)
async def context_gateway(request: ContextGatewayRequest) -> ContextGatewayResponse:
    """
    Smart Context Gateway: single endpoint composing GK + RAG++ into a
    token-budgeted context block.

    Runs GK traversal and RAG++ search in parallel, then composes results
    within the specified max_tokens budget. Falls back gracefully if either
    service is unavailable.
    """
    start = time.monotonic()
    _METRICS["requests_total"] += 1

    project = _extract_project(request.cwd)

    # Run GK and RAG++ (with optional recursive exploration) in parallel
    gk_coro = _query_gk_traverse(project, request.query) if request.include_graph else _empty_gk()
    rlm_coro = _rlm_search(
        request.query, request.k_rag, request.cwd, request.include_rlm, request.rlm_max_depth
    )

    try:
        gk_result, rlm_result = await asyncio.gather(
            gk_coro,
            rlm_coro,
            return_exceptions=True,
        )
    except Exception as e:
        logger.error(f"Gateway parallel query failed: {e}")
        gk_result = {"paths": [], "admissibility_token": None, "stats": {}}
        rlm_result = ([], False, [], 0.0, 0, False, 0)

    # Handle exceptions from gather
    fallback_reason = None
    if isinstance(gk_result, Exception):
        logger.warning(f"GK query exception: {gk_result}")
        gk_result = {"paths": [], "admissibility_token": None, "stats": {}}
        fallback_reason = "gk_exception"
    if isinstance(rlm_result, Exception):
        logger.warning(f"RLM/RAG++ query exception: {rlm_result}")
        rlm_result = ([], False, [], 0.0, 0, False, 0)
        fallback_reason = "rag_exception" if not fallback_reason else "both_exception"

    rag_results, rlm_decomposed, rlm_sub_queries, rlm_decompose_ms, rlm_depth, rlm_converged, rlm_nodes = rlm_result

    # Check if both failed (fallback mode)
    both_failed = not gk_result.get("paths") and not gk_result.get("admissibility_token") and not rag_results
    if both_failed:
        _METRICS["requests_fallback"] += 1
        elapsed = (time.monotonic() - start) * 1000
        if not fallback_reason:
            if not _GOOGLE_KEY or not _SUPABASE_URL:
                fallback_reason = "credentials_missing"
            else:
                fallback_reason = "both_empty"
        return ContextGatewayResponse(
            fallback=True,
            fallback_reason=fallback_reason,
            latency_ms=round(elapsed, 2),
            sources=[],
        )

    # Echo suppression: shadow telemetry (Phase 1 — measure, don't rerank)
    echo_rate = 0.0
    novelty_scores: List[float] = []
    echo_active = False
    if request.current_window_embedding and rag_results:
        rag_results, echo_rate, novelty_scores = _score_with_echo_suppression(
            rag_results, request.current_window_embedding,
        )
        echo_active = request.enable_echo_suppression
        _METRICS["echo_suppression_requests"] += 1
        _METRICS["echo_rate_sum"] += echo_rate
        if echo_rate > 0.6:
            _METRICS["echo_rate_high_count"] += 1

    # Overlap classification (A10: structured enum)
    overlap = OverlapClass.classify(echo_rate) if echo_active else None
    query_expanded = False

    # Vector expansion: when echo is high and suppression is on, expand query (P12)
    # This is a future path — currently telemetry only, expansion not re-queried
    if echo_active and overlap == OverlapClass.HIGH_ECHO and request.current_window_embedding:
        # Log that expansion would be triggered (Phase 2 will re-query with expanded embedding)
        _METRICS["vector_expansions_triggered"] += 1
        query_expanded = True
        logger.info(f"Echo rate {echo_rate:.2f} — vector expansion triggered (telemetry only)")

    # Bloom filter: per-session duplicate tracking (Phase 2 telemetry)
    bloom_dupes = 0
    if request.session_id and rag_results:
        bloom = await _get_session_bloom(request.session_id)
        rag_results, bloom_dupes = _mark_seen_and_filter(bloom, rag_results)
        _METRICS["bloom_lookups_total"] += len(rag_results)
        _METRICS["bloom_duplicates_found"] += bloom_dupes

    # Compose within token budget
    response = _compose_response(gk_result, rag_results, request.max_tokens, request.include_graph)

    elapsed = (time.monotonic() - start) * 1000
    response.latency_ms = round(elapsed, 2)
    response.echo_rate = round(echo_rate, 4) if echo_rate > 0 else None
    response.echo_suppression_active = echo_active
    response.novelty_scores = [round(n, 4) for n in novelty_scores] if novelty_scores else []
    response.bloom_duplicates = bloom_dupes
    response.overlap_class = overlap
    response.query_expanded = query_expanded
    response.rlm_decomposed = rlm_decomposed
    response.rlm_sub_queries = rlm_sub_queries
    response.rlm_decompose_ms = round(rlm_decompose_ms, 2)
    response.rlm_depth_reached = rlm_depth
    response.rlm_converged = rlm_converged
    response.rlm_total_nodes = rlm_nodes

    _METRICS["requests_success"] += 1
    _METRICS["latency_sum_ms"] += elapsed
    # Estimate savings: typical startup is ~2500 tokens, we return max_tokens
    _METRICS["tokens_saved_total"] += max(0, 2500 - response.token_estimate)

    logger.info(
        f"Gateway: {len(response.related_turns)} turns, "
        f"graph={'yes' if response.graph_context else 'no'}, "
        f"rlm={'yes' if rlm_decomposed else 'no'}"
        f"{f' depth={rlm_depth} nodes={rlm_nodes}' if rlm_decomposed else ''}, "
        f"bloom_dupes={bloom_dupes}, "
        f"~{response.token_estimate} tokens, {elapsed:.0f}ms"
    )

    return response


async def _empty_gk() -> Dict[str, Any]:
    """Return empty GK result (used when include_graph=False)."""
    return {"paths": [], "admissibility_token": None, "stats": {}}


@router.get("/gateway/metrics")
async def gateway_metrics() -> Dict[str, Any]:
    """JSON metrics for the context gateway."""
    m = get_gateway_metrics()
    avg_latency = m["latency_sum_ms"] / max(m["requests_success"], 1)
    return {
        **m,
        "avg_latency_ms": round(avg_latency, 2),
        "hit_rate": round(m["requests_success"] / max(m["requests_total"], 1), 4),
    }


def export_prometheus_metrics() -> str:
    """Export gateway metrics in Prometheus text format."""
    m = get_gateway_metrics()
    lines = [
        "# HELP context_gateway_requests_total Total context gateway requests",
        "# TYPE context_gateway_requests_total counter",
        f"context_gateway_requests_total {m['requests_total']}",
        "# HELP context_gateway_requests_success_total Successful gateway requests",
        "# TYPE context_gateway_requests_success_total counter",
        f"context_gateway_requests_success_total {m['requests_success']}",
        "# HELP context_gateway_requests_fallback_total Fallback (both down) requests",
        "# TYPE context_gateway_requests_fallback_total counter",
        f"context_gateway_requests_fallback_total {m['requests_fallback']}",
        "# HELP context_gateway_latency_seconds_sum Total latency across all requests",
        "# TYPE context_gateway_latency_seconds_sum counter",
        f"context_gateway_latency_seconds_sum {m['latency_sum_ms'] / 1000:.3f}",
        "# HELP context_gateway_tokens_saved_total Estimated tokens saved vs baseline",
        "# TYPE context_gateway_tokens_saved_total counter",
        f"context_gateway_tokens_saved_total {m['tokens_saved_total']}",
        "# HELP context_gateway_rlm_decompositions_total Total RLM decompositions triggered",
        "# TYPE context_gateway_rlm_decompositions_total counter",
        f"context_gateway_rlm_decompositions_total {m['rlm_decompositions_total']}",
        "# HELP context_gateway_rlm_sub_queries_total Total sub-queries generated by RLM",
        "# TYPE context_gateway_rlm_sub_queries_total counter",
        f"context_gateway_rlm_sub_queries_total {m['rlm_sub_queries_total']}",
        "# HELP context_gateway_rlm_decompose_latency_seconds_sum Total RLM decomposition latency",
        "# TYPE context_gateway_rlm_decompose_latency_seconds_sum counter",
        f"context_gateway_rlm_decompose_latency_seconds_sum {m['rlm_decompose_latency_sum_ms'] / 1000:.3f}",
        "# HELP context_gateway_echo_suppression_requests_total Requests with echo suppression data",
        "# TYPE context_gateway_echo_suppression_requests_total counter",
        f"context_gateway_echo_suppression_requests_total {m['echo_suppression_requests']}",
        "# HELP context_gateway_echo_rate_sum Sum of echo rates across requests",
        "# TYPE context_gateway_echo_rate_sum counter",
        f"context_gateway_echo_rate_sum {m['echo_rate_sum']:.4f}",
        "# HELP context_gateway_echo_rate_high_total Requests with echo rate > 0.6",
        "# TYPE context_gateway_echo_rate_high_total counter",
        f"context_gateway_echo_rate_high_total {m['echo_rate_high_count']}",
        "# HELP context_gateway_bloom_sessions_created_total Bloom filter sessions created",
        "# TYPE context_gateway_bloom_sessions_created_total counter",
        f"context_gateway_bloom_sessions_created_total {m['bloom_sessions_created']}",
        "# HELP context_gateway_bloom_sessions_evicted_total Bloom filter sessions evicted",
        "# TYPE context_gateway_bloom_sessions_evicted_total counter",
        f"context_gateway_bloom_sessions_evicted_total {m['bloom_sessions_evicted']}",
        "# HELP context_gateway_bloom_duplicates_found_total Duplicate documents detected by Bloom filter",
        "# TYPE context_gateway_bloom_duplicates_found_total counter",
        f"context_gateway_bloom_duplicates_found_total {m['bloom_duplicates_found']}",
        "# HELP context_gateway_bloom_active_sessions Current active Bloom filter sessions",
        "# TYPE context_gateway_bloom_active_sessions gauge",
        f"context_gateway_bloom_active_sessions {len(_SESSION_BLOOMS)}",
        "# HELP context_gateway_vector_expansions_total Query vector expansions triggered",
        "# TYPE context_gateway_vector_expansions_total counter",
        f"context_gateway_vector_expansions_total {m['vector_expansions_triggered']}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# KARL Embedding Endpoint — lightweight text->vector for skill routing
# ---------------------------------------------------------------------------


class EmbedRequest(BaseModel):
    text: str = Field(..., max_length=4000, description="Text to embed")


class EmbedResponse(BaseModel):
    embedding: List[float] = Field(default_factory=list)
    model: str = "gemini-embedding-001"
    dimensions: int = 0
    latency_ms: float = 0.0


@router.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest) -> EmbedResponse:
    """Generate a 768-dim embedding vector for text via Gemini embedding API.

    Used by KARL trajectory intelligence for skill routing embeddings.
    Lightweight: no pgvector search, no GK traversal, just embedding generation.
    """
    start = time.monotonic()
    if not _GOOGLE_KEY:
        raise HTTPException(status_code=503, detail="GOOGLE_API_KEY not configured")

    try:
        embedding = await _generate_embedding(request.text)
        if not embedding:
            raise HTTPException(status_code=502, detail="Gemini API returned empty embedding")
        elapsed = (time.monotonic() - start) * 1000
        return EmbedResponse(
            embedding=embedding,
            dimensions=len(embedding),
            latency_ms=round(elapsed, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embed endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
