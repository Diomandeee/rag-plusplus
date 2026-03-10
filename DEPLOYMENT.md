# RAG++ Deployment Guide

## Production Setup (Docker)

### Prerequisites

- Docker 20+
- Supabase project with pgvector enabled
- Google Gemini API key (for embeddings)
- Graph Kernel running on the same host or reachable via network

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Yes | Service role key (not anon) |
| `GOOGLE_API_KEY` | Yes | Gemini API key for embeddings |
| `GRAPH_KERNEL_URL` | No | Graph Kernel URL (default: `http://172.17.0.1:8001`) |
| `CONTEXT_GATEWAY_MOCK` | No | Set to `1` for mock mode (no external deps) |

### Build & Run

```bash
cd core/retrieval/cc-rag-plus-plus

# Build lean image (~850MB, no torch/transformers)
docker build -t rag-plusplus .

# Run
docker run -d --name rag-plusplus \
  --restart unless-stopped \
  -p 8000:8000 \
  -e SUPABASE_URL=https://your-project.supabase.co \
  -e SUPABASE_SERVICE_KEY=your-service-key \
  -e GOOGLE_API_KEY=your-gemini-key \
  -e GRAPH_KERNEL_URL=http://172.17.0.1:8001 \
  rag-plusplus
```

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "healthy", "generator": true, "trainer": true}
```

### Verify Gateway

```bash
curl -X POST http://localhost:8000/api/rag/gateway/context \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "session_id": "test-001"}'
```

## Docker Networking

When Graph Kernel runs natively on the Docker host (not in a container), the container must use the Docker bridge gateway IP:

```
GRAPH_KERNEL_URL=http://172.17.0.1:8001
```

If the host firewall drops Docker traffic, add:

```bash
sudo iptables -I INPUT 2 -s 172.17.0.0/16 -p tcp --dport 8001 -j ACCEPT
```

## Dependencies

The production image uses `requirements.txt` with pinned, lean dependencies:

- No `torch`, `transformers`, or `faiss-cpu` (saves ~7GB)
- Single Gemini SDK: `google-genai` (not deprecated `google-generativeai`)
- Total image: ~850MB vs ~8.5GB with ML deps

## Monitoring

### Prometheus Metrics

```bash
# Text format
curl http://localhost:8000/metrics

# JSON format
curl http://localhost:8000/metrics/json
```

### Gateway-Specific Metrics

```bash
curl http://localhost:8000/api/rag/gateway/metrics
```

Key metrics:
- `gateway_requests` - Total gateway calls
- `echo_suppression_requests` - Calls with echo suppression enabled
- `echo_rate_sum` / `echo_rate_high_count` - Echo rate tracking
- `bloom_sessions_created` / `bloom_duplicates_found` - Bloom filter stats
- `vector_expansions_triggered` - Query expansion count

### Nexus Portal Dashboard

The echo suppression dashboard is at `http://<nexus-host>:3001/echo`.

## Updating the Gateway

```bash
# Copy new gateway code into running container
docker cp context_gateway.py rag-plusplus:/app/rag_plusplus/service/routes/context_gateway.py

# Verify imports
docker exec rag-plusplus python3 -c "from rag_plusplus.service.routes.context_gateway import router; print('OK')"

# Restart
docker restart rag-plusplus

# Verify health
sleep 5 && curl http://localhost:8000/health
```

## Running Tests

```bash
# Copy test file into container
docker cp test_context_gateway.py rag-plusplus:/app/tests/service/test_context_gateway.py

# Install pytest (not in production image)
docker exec rag-plusplus pip install pytest

# Run tests
docker exec rag-plusplus python3 -m pytest tests/service/test_context_gateway.py -v
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `RAG search unavailable` | Missing `SUPABASE_SERVICE_KEY` | Set env var and restart |
| Empty embedding results | Missing `GOOGLE_API_KEY` | Set env var and restart |
| Graph Kernel timeout | GK not running or wrong URL | Check `GRAPH_KERNEL_URL`, verify GK process |
| Docker can't reach GK | Firewall blocking Docker subnet | Add iptables rule for 172.17.0.0/16 |
| 850MB → 8GB image | Wrong requirements.txt | Use lean `requirements.txt` (no torch) |
