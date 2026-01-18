# rag-plusplus

High-performance retrieval for memory-conditioned selection with Python bindings to a Rust core.

## Installation

```bash
pip install rag-plusplus
```

## Features

- **SIMD-accelerated vector search** - AVX2-optimized distance calculations
- **HNSW indexing** - Approximate nearest neighbor with sub-linear scaling
- **Trajectory coordinates** - 5D positioning for conversation context
- **IRCP propagator** - Information-Relevance Causal Propagation
- **Salience scoring** - Turn importance with recency decay
- **Chain management** - Cross-conversation knowledge links

## Quick Start

```python
import numpy as np
from rag_plusplus import HNSWIndex, FlatIndex, InMemoryStore, MemoryRecord

# Create an HNSW index for fast approximate search
index = HNSWIndex(dimension=128, m=16, ef_construction=200)

# Add vectors
for i in range(1000):
    vec = np.random.randn(128).astype(np.float32)
    index.add(f"vec_{i}", vec)

# Search
query = np.random.randn(128).astype(np.float32)
results = index.search(query, k=10)

for result in results:
    print(f"{result.id}: score={result.score:.4f}")
```

## Trajectory Coordinates

```python
from rag_plusplus import TrajectoryCoordinate5D, trajectory_weighted_cosine_5d

# Create 5D trajectory coordinates
# (depth, sibling_order, homogeneity, temporal, complexity)
coord1 = TrajectoryCoordinate5D(3, 0, 0.9, 0.7, 1)
coord2 = TrajectoryCoordinate5D(4, 1, 0.8, 0.6, 2)

# Trajectory-weighted similarity
score = trajectory_weighted_cosine_5d(vec1, vec2, coord1, coord2, coord_weight=0.3)
```

## IRCP Propagation

```python
from rag_plusplus import IRCPPropagator, IRCPConfig

config = IRCPConfig()  # temperature=1.0, spatial_weight=0.3
propagator = IRCPPropagator(config)

# Propagate attention through conversation turns
# attention_weights = propagator.propagate(...)
```

## Available Classes

### Core
- `HNSWIndex` - Hierarchical Navigable Small World graph for ANN
- `FlatIndex` - Exact brute-force search
- `InMemoryStore` - Fast in-memory record storage
- `MemoryRecord` - Record with embedding and outcome
- `QueryCache` - LRU cache with TTL

### Trajectory
- `TrajectoryCoordinate` - 4D coordinate (depth, sibling, homogeneity, temporal)
- `TrajectoryCoordinate5D` - 5D coordinate (+complexity)
- `TrajectoryPhase` - Conversation phase enum
- `PhaseInferencer` - Infer phase from turn features

### IRCP & Salience
- `IRCPPropagator` - Attention propagation
- `SalienceScorer` - Turn importance scoring
- `TurnSalience` - Salience result with factors

### Chain Systems
- `ChainManager` - Cross-conversation chain management
- `ChainLinkEstimator` - Link strength estimation
- `TrajectoryGraph` - Episode graph structure

## Performance

The Rust core provides:
- ~10x faster vector distance than pure Python
- Sub-linear search with HNSW indexing
- Zero-copy numpy array handling
- Parallel search across multiple indexes

## License

MIT
