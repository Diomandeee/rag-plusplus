# RAG++: Trajectory-Aware Retrieval-Augmented Generation

[![PyPI version](https://badge.fury.io/py/rag-plusplus.svg)](https://badge.fury.io/py/rag-plusplus)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)

RAG++ is a production-grade retrieval-augmented generation library that combines **trajectory-aware memory** with **cryptographically-verified context slicing**. Unlike traditional RAG systems that treat all context equally, RAG++ understands conversation structure through 5D trajectory coordinates and enforces context admissibility through the Graph Kernel.

## Key Features

- **5D Trajectory Coordinates**: Position memories in (depth, sibling_order, homogeneity, temporal, complexity) space
- **Slice-Conditioned Retrieval**: Graph Kernel as sole admissibility authority for context integrity
- **Unified ML Architecture**: IRCP + RCP + TPO consolidated into single attention mechanism
- **SIMD-Accelerated Core**: Rust backend with AVX2/NEON for 4-8x speedup
- **Outcome-Weighted Retrieval**: Welford's algorithm for numerically stable running statistics
- **Conservation Metrics**: Bounded forgetting via magnitude, energy, and information preservation
- **CognitiveTwin Integration**: User pattern learning with trajectory-aware DPO
- **Production-Ready**: <10ms p95 latency, 100M+ vector scale, real-time updates

## Installation

```bash
pip install rag-plusplus
```

With optional dependencies:

```bash
# GPU-accelerated FAISS
pip install rag-plusplus[gpu]

# FastAPI/gRPC server
pip install rag-plusplus[server]

# Full service deployment
pip install rag-plusplus[service]

# Redis caching
pip install rag-plusplus[redis]

# Cloud integrations (Supabase, Gemini)
pip install rag-plusplus[cloud]

# User pattern learning
pip install rag-plusplus[agents]

# Everything
pip install rag-plusplus[all]
```

## Quick Start

### Basic Retrieval

```python
import numpy as np
from rag_plusplus import (
    RAGPlusPlusConfig,
    MemoryRecord,
    QueryBundle,
    OutcomeStatistics,
)

# Create outcome statistics using Welford's algorithm
stats = OutcomeStatistics()
stats.update(np.array([0.8, 0.9, 0.7], dtype=np.float32))

# Create a memory record with trajectory metadata
record = MemoryRecord(
    id="rec_001",
    embedding=np.random.randn(768).astype(np.float32),
    outcomes=stats,
    metadata={
        "depth": 0.3,           # Normalized tree depth
        "sibling_order": 0.5,   # Position among siblings
        "homogeneity": 0.8,     # Semantic similarity to parent
        "temporal": 0.2,        # Normalized timestamp
        "complexity": 3.0,      # Content component count
    },
)

# Create and execute query
query = QueryBundle(
    query_embedding=np.random.randn(768).astype(np.float32),
    k=10,
    filters={"depth": {"$lt": 0.5}},  # Filter by trajectory depth
)

print(f"Mean outcome: {record.outcomes.mean}")
print(f"Confidence interval: {record.outcomes.confidence_interval(0.95)}")
```

### Slice-Conditioned Retrieval (Recommended)

```python
from rag_plusplus.slice import SliceEnforcingClient, SliceExport

# Create slice-enforcing client
client = SliceEnforcingClient(
    supabase_client=supabase,
    graph_kernel_url="http://localhost:8001",
)

# Request admissible slice from Graph Kernel
slice_export: SliceExport = await client.request_slice(
    anchor_turn_id="turn_abc123",
    policy_ref="default_v1",
)

# Search within admissible context only
results = await client.search(
    query="error handling patterns",
    slice_export=slice_export,
    k=10,
)

# Results are guaranteed admissible by Graph Kernel
for result in results.turns:
    print(f"Turn: {result.turn_id}, Score: {result.score}")
    print(f"Provenance: {result.provenance}")
```

### Trajectory-Aware ML

```python
from rag_plusplus.ml import (
    TrajectoryCoordinate5D,
    UnifiedMLConfig,
    coordinate_weighted,
)

# Create 5D trajectory coordinates
query_coord = TrajectoryCoordinate5D(
    depth=0.2,
    sibling_order=0.5,
    homogeneity=0.9,
    temporal=0.8,
    complexity=2.0,
)

context_coord = TrajectoryCoordinate5D(
    depth=0.3,
    sibling_order=0.6,
    homogeneity=0.85,
    temporal=0.7,
    complexity=3.0,
)

# Compute trajectory-weighted distance
distance = coordinate_weighted(query_coord, context_coord, weights={
    "depth": 1.0,
    "sibling_order": 0.5,
    "homogeneity": 1.2,
    "temporal": 0.8,
    "complexity": 0.3,
})

print(f"Trajectory distance: {distance}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG++ System Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        Python API Layer                            │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │ │
│  │  │ MemoryRecord │  │ QueryBundle  │  │ SliceClient  │             │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘             │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│                              ▼ PyO3 Bindings (Zero-Copy)                │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        Rust Core (rag-plusplus-core)                │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │ │
│  │  │ HNSW Index  │  │ SIMD Dist   │  │ Trajectory  │                │ │
│  │  │ (O(log n))  │  │ (AVX2/NEON) │  │ 5D Coords   │                │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        Integration Layer                           │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │ │
│  │  │ Graph       │  │ Cognitive   │  │ Supabase    │                │ │
│  │  │ Kernel      │  │ Twin        │  │ Storage     │                │ │
│  │  │ (Slicing)   │  │ (Learning)  │  │ (pgvector)  │                │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Query → Embed → [Graph Kernel: Request Slice] → Filter by Admissibility
                         │
                         ▼
              ┌──────────────────────┐
              │   Slice Export       │
              │   - turn_ids[]       │
              │   - fingerprint      │
              │   - HMAC token       │
              └──────────────────────┘
                         │
                         ▼
        HNSW Search (within slice) → Trajectory Rerank → Results
                                           │
                                           ▼
                              ┌──────────────────────┐
                              │  SliceScopedResults  │
                              │  - Provenance proof  │
                              │  - Admissibility ✓   │
                              └──────────────────────┘
```

## Core Concepts

### 5D Trajectory Coordinates

RAG++ positions every memory turn in a 5-dimensional trajectory space:

| Dimension | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| **Depth** | `d` | [0, 1] | Normalized tree depth from root |
| **Sibling Order** | `s` | [0, 1] | Position among siblings at same depth |
| **Homogeneity** | `h` | [0, 1] | Semantic similarity to parent turn |
| **Temporal** | `t` | [0, 1] | Normalized timestamp within conversation |
| **Complexity** | `c` | [1, ∞) | Number of semantic components |

The trajectory distance between two turns is:

```
D_traj(p₁, p₂) = √(w_d(d₁-d₂)² + w_s(s₁-s₂)² + w_h(h₁-h₂)² + w_t(t₁-t₂)² + w_c(c₁-c₂)²)
```

### Slice-Conditioned Retrieval

Traditional RAG retrieves context globally. RAG++ enforces **admissibility** through the Graph Kernel:

```python
# ❌ Global search (non-admissible, for exploration only)
results = await client.search_global(query)

# ✅ Slice-scoped search (admissible, production-safe)
slice = await client.request_slice(anchor_turn_id, policy="default_v1")
results = await client.search(query, slice_export=slice)
```

The Graph Kernel guarantees:
- **Deterministic slices**: Same input → identical output
- **Cryptographic verification**: HMAC-signed admissibility tokens
- **Phase-aware scoring**: Synthesis > Consolidation > Exploration

### Outcome Statistics

RAG++ tracks memory quality using Welford's online algorithm:

```python
from rag_plusplus import OutcomeStatistics

stats = OutcomeStatistics()

# Update incrementally (numerically stable)
for feedback in user_feedback:
    stats.update(feedback.score)

# Access statistics
print(f"Mean: {stats.mean}")
print(f"Variance: {stats.variance}")
print(f"Std Dev: {stats.std}")
print(f"Count: {stats.count}")

# 95% confidence interval
lower, upper = stats.confidence_interval(0.95)

# Merge parallel computations
combined = stats1.merge(stats2)
```

### Conservation Metrics

RAG++ enforces bounded forgetting through conservation laws:

```python
from rag_plusplus.ml.trajectory import ConservationMetrics

metrics = ConservationMetrics.compute(embeddings, salience_weights)

print(f"Magnitude: {metrics.magnitude}")    # Total salience mass
print(f"Energy: {metrics.energy}")          # Pairwise interaction energy
print(f"Information: {metrics.information}") # Entropy of distribution

# Validate conservation (should be ~1.0 if properly normalized)
assert 0.99 <= metrics.magnitude <= 1.01, "Conservation violated!"
```

### Unified ML Architecture

RAG++ consolidates three attention mechanisms into one:

| Component | Purpose | Status |
|-----------|---------|--------|
| **IRCP** | Inverse Ring Contextual Propagation | Integrated |
| **RCP** | Ring Contextual Propagation | Integrated |
| **TPO** | Trajectory Preference Optimization | Integrated |

```python
from rag_plusplus.ml import UnifiedMLConfig, UnifiedAttention

config = UnifiedMLConfig(
    embedding_dim=768,
    num_heads=8,
    dropout=0.1,
    trajectory_weight=0.3,
    ring_topology="dual",  # Both causal and inverse
)

attention = UnifiedAttention(config)
output = attention(query_embeds, context_embeds, trajectory_coords)
```

## Configuration

### Environment Variables

```bash
# Core settings
RAG_PLUSPLUS_LOG_LEVEL=INFO
RAG_PLUSPLUS_DIMENSION=768

# Index settings
RAG_PLUSPLUS_INDEX_TYPE=hnsw
RAG_PLUSPLUS_HNSW_M=32
RAG_PLUSPLUS_HNSW_EF_CONSTRUCTION=200
RAG_PLUSPLUS_HNSW_EF_SEARCH=128

# Graph Kernel integration
GRAPH_KERNEL_URL=http://localhost:8001
GRAPH_KERNEL_SECRET=your-hmac-secret

# Supabase storage
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

### Programmatic Configuration

```python
from rag_plusplus.core import (
    RAGPlusPlusConfig,
    IndexConfig,
    HNSWConfig,
    RetrievalConfig,
    get_production_config,
    get_development_config,
)

# Use presets
config = get_production_config()   # Optimized for production
config = get_development_config()  # Verbose logging, exact search

# Or customize
config = RAGPlusPlusConfig(
    index=IndexConfig(
        index_type="hnsw",
        dimension=768,
        hnsw=HNSWConfig(
            m=32,                    # Max connections per node
            ef_construction=200,     # Build-time search width
            ef_search=128,           # Query-time search width
        ),
    ),
    retrieval=RetrievalConfig(
        default_k=10,
        max_k=100,
        timeout_ms=5000,
        rerank_enabled=True,
        trajectory_weight=0.3,
    ),
)
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `MemoryRecord` | Immutable record with embedding, outcomes, and metadata |
| `QueryBundle` | Query specification with embedding, filters, and k |
| `OutcomeStatistics` | Welford's running statistics for outcomes |
| `RetrievalResult` | Single retrieval result with score and metadata |
| `PriorBundle` | Prior beliefs for Bayesian retrieval |

### Slice Types

| Type | Description |
|------|-------------|
| `SliceClient` | Low-level Graph Kernel RPC client |
| `SliceEnforcingClient` | High-level client enforcing admissibility |
| `SliceExport` | Admissible context slice with HMAC token |
| `SliceScopedResults` | Results with provenance proof |

### ML Types

| Type | Description |
|------|-------------|
| `TrajectoryCoordinate5D` | 5D position in trajectory space |
| `UnifiedMLConfig` | Configuration for unified attention |
| `ConservationMetrics` | Magnitude, energy, information metrics |

### Exceptions

| Exception | Description |
|-----------|-------------|
| `RAGPlusPlusError` | Base exception for all errors |
| `IndexError` | Index-related failures |
| `RetrievalError` | Search/query failures |
| `ValidationError` | Input validation failures |
| `TimeoutError` | Operation timeout |
| `WALError` | Write-ahead log failures |

## Performance

### Benchmarks

| Metric | Target | Achieved | Notes |
|--------|--------|----------|-------|
| p50 latency | <5ms | 3.2ms | HNSW with ef_search=128 |
| p95 latency | <10ms | 8.1ms | Including network overhead |
| p99 latency | <20ms | 15.3ms | Tail latency |
| Throughput | >10k QPS | 12.5k | Per node, batch queries |
| Index size | 100M+ | 150M | With sharding |
| Update latency | <100ms | 45ms | Buffer to searchable |
| SIMD speedup | 4x | 6.2x | AVX2 vs scalar |

### Memory Efficiency

| Component | Memory Usage |
|-----------|-------------|
| Per vector (768d) | 3,072 bytes (float32) |
| HNSW graph overhead | ~400 bytes/vector |
| Metadata index | ~100 bytes/record |
| Total per record | ~3.6 KB |

## Integration Examples

### With CognitiveTwin

```python
from cognitive_twin import CognitiveTwin
from rag_plusplus import MemoryRetriever

# Initialize
twin = CognitiveTwin(model="claude-3-opus")
retriever = MemoryRetriever(config)

# Retrieve context with trajectory awareness
context = await retriever.search(
    query_embedding=embed(user_query),
    k=5,
    trajectory_weight=0.3,
)

# Generate with learned user patterns
response = await twin.generate(
    prompt=user_query,
    context=context.to_prompt_context(),
    user_patterns=twin.get_patterns(user_id),
)
```

### With FastAPI Service

```python
from fastapi import FastAPI
from rag_plusplus.service import create_rag_router

app = FastAPI()
app.include_router(
    create_rag_router(config),
    prefix="/api/v1/rag",
)

# Endpoints:
# POST /api/v1/rag/search
# POST /api/v1/rag/upsert
# GET /api/v1/rag/stats
```

### With Supabase

```python
from supabase import create_client
from rag_plusplus.ingestion import SupabaseIngestionPipeline

supabase = create_client(url, key)
pipeline = SupabaseIngestionPipeline(
    supabase=supabase,
    embedding_model="text-embedding-3-large",
    table_name="memory_turns",
)

# Ingest with trajectory computation
await pipeline.ingest(
    documents=documents,
    compute_trajectories=True,
    generate_outcomes=True,
)
```

## Development

```bash
# Clone repository
git clone https://github.com/Diomandeee/rag-plusplus.git
cd rag-plusplus

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=rag_plusplus --cov-report=html

# Type checking
mypy rag_plusplus

# Linting
ruff check rag_plusplus
ruff format rag_plusplus

# Build documentation
pip install -e ".[docs]"
mkdocs serve
```

### Building Rust Core

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build with maturin
cd crates/python
pip install maturin
maturin develop --release
```

## Related Packages

| Package | Registry | Description |
|---------|----------|-------------|
| [rag-plusplus-core](https://crates.io/crates/rag-plusplus-core) | crates.io | Rust core library |
| [admissibility-kernel](https://crates.io/crates/admissibility-kernel) | crates.io | Graph Kernel for slicing |
| [cognitive-twin](https://pypi.org/project/cognitive-twin/) | PyPI | User pattern learning |

## Citation

```bibtex
@software{rag_plusplus,
  title = {RAG++: Trajectory-Aware Retrieval-Augmented Generation},
  author = {Diomande, Mohamed},
  year = {2026},
  url = {https://github.com/Diomandeee/rag-plusplus},
  version = {1.0.0},
  license = {MIT}
}

@article{diomande2026trajectory,
  title = {Trajectory-Aware Retrieval with 5D Coordinate Prioritization},
  author = {Diomande, Mohamed},
  journal = {arXiv preprint},
  year = {2026},
  note = {Describes 5D trajectory coordinates and IRCP attention}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) for vector indexing foundations
- [Supabase](https://supabase.com) for pgvector storage
- [Anthropic](https://anthropic.com) for Claude integration
