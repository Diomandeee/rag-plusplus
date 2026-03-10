# RAG++ Context Gateway

Federated context retrieval service that unifies **pgvector search**, **Graph Kernel traversal**, and **Gemini embedding generation** behind a single endpoint.

Built for Claude Code sessions. Returns token-optimized context with echo suppression, duplicate detection, and overlap classification.

## Quick Start

```bash
# Docker (recommended)
docker build -t rag-plusplus .
docker run -d --name rag-plusplus -p 8000:8000 \
  -e SUPABASE_URL=https://your-project.supabase.co \
  -e SUPABASE_SERVICE_KEY=your-key \
  -e GOOGLE_API_KEY=your-gemini-key \
  rag-plusplus

# Or locally
pip install -e .
rag-plusplus --host 0.0.0.0 --port 8000
```

## Endpoint

```
POST /api/rag/gateway/context
```

### Request

```json
{
  "query": "how does the deployment pipeline work",
  "session_id": "abc-123",
  "project_hint": "Spore",
  "k": 10,
  "echo_suppression": true,
  "window_embedding": [0.01, -0.03, ...]
}
```

### Response

```json
{
  "related_turns": [...],
  "graph_context": { "entity": "spore", "relations": [...] },
  "token_estimate": 2400,
  "sources": ["rag++", "pgvector", "graph_kernel"],
  "latency_ms": 340.5,
  "echo_rate": 0.15,
  "echo_suppression_active": true,
  "novelty_scores": [0.92, 0.87, 0.74],
  "bloom_duplicates": 0,
  "overlap_class": "novel",
  "query_expanded": false
}
```

## Echo Suppression Pipeline

Six-stage system to prevent repetitive context within a session:

| Stage | What It Does |
|-------|-------------|
| **Bloom Filter** | Per-session duplicate tracking (SHA-256, 8192-bit, 5 hashes, ~1.5% FP rate) |
| **Embedding** | Gemini `text-embedding-004` (768d) |
| **Novelty Scoring** | Cosine distance of results against session window embedding |
| **Overlap Classification** | `novel` (<0.3), `moderate` (0.3-0.6), `high_echo` (>=0.6) |
| **Vector Expansion** | Gram-Schmidt orthogonalization steers retrieval away from seen content |
| **Metrics** | Prometheus counters for echo rate, bloom stats, expansions |

## Mock Mode

Run without external dependencies for local development:

```bash
CONTEXT_GATEWAY_MOCK=1 rag-plusplus --port 8000
```

Returns synthetic results, no Gemini/Supabase/Graph Kernel credentials needed.

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Yes | Service role key |
| `GOOGLE_API_KEY` | Yes | Gemini API key for embeddings |
| `GRAPH_KERNEL_URL` | No | Graph Kernel URL (default: `http://172.17.0.1:8001`) |
| `CONTEXT_GATEWAY_MOCK` | No | Set `1` for mock mode |
| `CORS_ORIGINS` | No | Comma-separated allowed origins |

## Monitoring

```bash
# Prometheus text format
curl http://localhost:8000/metrics

# JSON
curl http://localhost:8000/metrics/json

# Gateway-specific metrics
curl http://localhost:8000/api/rag/gateway/metrics
```

Metrics: `gateway_requests`, `echo_suppression_requests`, `echo_rate_sum`, `bloom_sessions_created`, `bloom_duplicates_found`, `vector_expansions_triggered`.

## Tests

```bash
pip install pytest
pytest tests/ -v
# 88 tests
```

## Architecture

```
Client Request
     |
     v
+----------------------------------+
|   Context Gateway (FastAPI)      |
|                                  |
|  +----------+  +--------------+  |
|  | Bloom    |  | Novelty      |  |
|  | Filter   |  | Scoring      |  |
|  +----------+  +--------------+  |
|  +----------+  +--------------+  |
|  | Overlap  |  | Vector       |  |
|  | Class    |  | Expansion    |  |
|  +----------+  +--------------+  |
+--------+-----------+-------------+
         |           |
    +----v--+   +---v---+   +--------+
    |Supabase|  |Graph  |   |Gemini  |
    |pgvector|  |Kernel |   |Embed   |
    +--------+  +-------+   +--------+
```

## License

MIT
