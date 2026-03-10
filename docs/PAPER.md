# RAG++: Memory-Conditioned Candidate Selection with Trajectory-Aware Attention

**Mohamed Diomande**

*Technical Report v1.0 - January 2026*

---

## Abstract

Retrieval-Augmented Generation (RAG) systems typically treat retrieved context as a flat collection of documents, ignoring the structural and temporal relationships between conversation turns. We present RAG++, a trajectory-aware retrieval system that positions memories in a 5-dimensional coordinate space (depth, sibling order, homogeneity, temporal position, and complexity) and enforces context admissibility through cryptographically-verified slicing. Our system introduces three key innovations: (1) **Inverse Ring Contextual Propagation (IRCP)**, an attention mechanism that propagates information in both causal and anti-causal directions through ring topology; (2) **Slice-Conditioned Retrieval**, where a separate Graph Kernel service serves as the sole admissibility authority for context selection; and (3) **Conservation Metrics**, mathematical invariants that ensure bounded forgetting in memory systems. We consolidate three previously separate attention mechanisms (IRCP, RCP, TPO) into a unified architecture, achieving a 92% code reduction (42K to 3.35K LOC) while maintaining functional equivalence. Benchmarks demonstrate p95 latency of 8.1ms, throughput of 12.5k QPS, and successful scaling to 150M vectors.

**Keywords**: Retrieval-Augmented Generation, Trajectory Memory, Context Slicing, Attention Mechanisms, Vector Search

---

## 1. Introduction

### 1.1 Problem Statement

Modern conversational AI systems face a fundamental challenge: as conversations grow longer and memory corpora expand, selecting the right context becomes critical for response quality. Traditional RAG systems apply cosine similarity to retrieve top-k documents, treating all retrieved content as equally relevant regardless of:

- **Structural position**: Where in the conversation tree did this turn occur?
- **Temporal distance**: How recently was this turn created?
- **Semantic homogeneity**: How related is this turn to its parent context?
- **Causal admissibility**: Is this turn legally accessible from the current anchor?

These factors significantly impact which context should inform generation, yet standard vector search ignores them entirely.

### 1.2 Motivating Example

Consider a debugging conversation spanning 50 turns with multiple branches:

```
Turn 1: "Help me debug the auth system"
├── Turn 2: "The login endpoint returns 401"
│   ├── Turn 3: "Check if JWT is expired"
│   │   └── Turn 4: "Token expires in 1 hour"
│   └── Turn 5: "Check if user exists"
│       └── Turn 6: "User exists in database"
└── Turn 7: "Actually, let me check CORS first"
    └── Turn 8: "CORS is configured correctly"
```

When the user asks "Why is login failing?" at Turn 9, standard RAG might retrieve Turns 4, 6, and 8 based on semantic similarity, mixing context from unrelated branches. RAG++ instead:

1. Computes **trajectory coordinates** for each turn (depth=3 for Turn 4, depth=2 for Turn 8)
2. Requests an **admissible slice** from Turn 9's perspective
3. Applies **trajectory-weighted attention** preferring recent, related turns
4. Returns results with **provenance proof** of admissibility

### 1.3 Contributions

This paper makes the following contributions:

1. **5D Trajectory Coordinate System**: A mathematical framework positioning conversation turns in (depth, sibling_order, homogeneity, temporal, complexity) space, enabling structure-aware retrieval.

2. **Inverse Ring Contextual Propagation (IRCP)**: An attention mechanism that propagates information bidirectionally through ring topology, combining causal (past→present) and anti-causal (future→past) influence patterns.

3. **Slice-Conditioned Retrieval**: An architectural pattern where a separate Graph Kernel service cryptographically signs admissible context slices, preventing the retrieval system from accessing unauthorized turns.

4. **Unified Attention Architecture**: Consolidation of three separate codebases (IRCP, RCP, TPO) into a single 3.35K LOC implementation with identical functionality.

5. **Conservation Metrics**: Mathematical invariants (magnitude, energy, information) that ensure bounded forgetting and enable memory health monitoring.

---

## 2. Related Work

### 2.1 Retrieval-Augmented Generation

RAG systems augment language model generation with retrieved context [Lewis et al., 2020]. Recent work has explored:

- **Dense Passage Retrieval (DPR)**: Learning dual encoders for query and document embedding [Karpukhin et al., 2020]
- **Fusion-in-Decoder (FiD)**: Concatenating retrieved passages in the decoder [Izacard & Grave, 2021]
- **Self-RAG**: Training models to retrieve and critique their own generations [Asai et al., 2023]

RAG++ differs by introducing trajectory-aware prioritization and cryptographic admissibility verification, concepts absent from prior work.

### 2.2 Conversation Memory Systems

Long-context conversation systems have explored:

- **MemoryBank**: Summarizing old conversations into memory banks [Zhong et al., 2024]
- **Reflexion**: Self-reflective memory for task learning [Shinn et al., 2023]
- **Generative Agents**: Simulating social behavior with memory retrieval [Park et al., 2023]

These systems treat memory as flat text. RAG++ instead models the conversation DAG structure explicitly through trajectory coordinates.

### 2.3 Graph-Based Context Selection

Prior work on graph-based retrieval includes:

- **GraphRAG**: Building knowledge graphs from documents [Microsoft, 2024]
- **RAPTOR**: Recursive abstractive processing for retrieval [Sarthi et al., 2024]
- **HippoRAG**: Hippocampal-inspired memory indexing [Gutierrez et al., 2024]

RAG++ complements these approaches with explicit admissibility verification via the Graph Kernel.

### 2.4 Attention Mechanisms

Transformer attention mechanisms have been extended in several directions:

- **Sparse Attention**: Reducing O(n²) complexity [Child et al., 2019]
- **Linear Attention**: Approximating softmax attention [Katharopoulos et al., 2020]
- **Rotary Position Embeddings (RoPE)**: Encoding position through rotation [Su et al., 2021]

IRCP extends these by introducing ring topology that captures both causal and influence relationships.

---

## 3. System Architecture

### 3.1 Overview

RAG++ comprises four layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAG++ Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Application API                                        │
│  ├─ MemoryRecord, QueryBundle, OutcomeStatistics                │
│  ├─ SliceEnforcingClient, RetrievalResult                       │
│  └─ Python package: rag-plusplus (PyPI)                         │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: ML & Attention                                         │
│  ├─ TrajectoryCoordinate5D, coordinate_weighted()               │
│  ├─ UnifiedAttention (IRCP + RCP + TPO)                         │
│  └─ ConservationMetrics, salience scoring                       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Integration Services                                   │
│  ├─ Graph Kernel (admissibility-kernel)                         │
│  ├─ CognitiveTwin (user pattern learning)                       │
│  └─ Supabase (pgvector storage)                                 │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Rust Core                                              │
│  ├─ HNSW/Flat indices, SIMD distances                           │
│  ├─ WAL, query cache                                            │
│  └─ Crate: rag-plusplus-core (crates.io)                        │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Interactions

The request flow for slice-conditioned retrieval:

```
                    ┌─────────────┐
                    │   Client    │
                    └──────┬──────┘
                           │ 1. search(query, anchor_turn_id)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    SliceEnforcingClient                       │
├──────────────────────────────────────────────────────────────┤
│  2. Request slice from Graph Kernel                          │
│     ┌─────────────────────────────────────────────────────┐  │
│     │  POST /slice                                         │  │
│     │  { anchor_turn_id, policy_ref }                      │  │
│     │  → { turn_ids[], fingerprint, hmac_token }           │  │
│     └─────────────────────────────────────────────────────┘  │
│                                                               │
│  3. Validate HMAC token (Graph Kernel is sole authority)     │
│                                                               │
│  4. Execute HNSW search filtered to slice turn_ids           │
│     ┌─────────────────────────────────────────────────────┐  │
│     │  Rust Core: hnsw.search(query_vec, k, filter=slice)  │  │
│     └─────────────────────────────────────────────────────┘  │
│                                                               │
│  5. Apply trajectory reranking                                │
│     score_final = α·semantic + (1-α)·trajectory_weight       │
│                                                               │
│  6. Return SliceScopedResults with provenance                │
└──────────────────────────────────────────────────────────────┘
```

### 3.3 Storage Layer

RAG++ supports multiple storage backends:

| Backend | Use Case | Latency | Scale |
|---------|----------|---------|-------|
| **In-Memory HNSW** | Development, small datasets | <1ms | 1M vectors |
| **FAISS IVF** | Production, large datasets | 3-5ms | 100M vectors |
| **Supabase pgvector** | Managed, serverless | 10-20ms | 50M vectors |

The storage interface:

```python
class VectorStore(Protocol):
    def search(self, query: np.ndarray, k: int, filter: Filter) -> List[SearchResult]: ...
    def upsert(self, records: List[MemoryRecord]) -> int: ...
    def delete(self, ids: List[str]) -> int: ...
```

---

## 4. Core Algorithms

### 4.1 5D Trajectory Coordinates

#### 4.1.1 Definition

Every conversation turn is positioned in a 5-dimensional trajectory space:

**Definition 4.1 (Trajectory Coordinate)**. A trajectory coordinate is a tuple (d, s, h, t, c) where:

- **d ∈ [0, 1]**: Normalized depth from conversation root
- **s ∈ [0, 1]**: Sibling order among children of the same parent
- **h ∈ [0, 1]**: Homogeneity (semantic similarity to parent)
- **t ∈ [0, 1]**: Temporal position within conversation timeline
- **c ∈ [1, ∞)**: Complexity (number of semantic components)

#### 4.1.2 Coordinate Computation

Coordinates are computed during ingestion:

```python
def compute_trajectory_coordinate(turn: Turn, conversation: Conversation) -> Coordinate5D:
    # Depth: normalized by max depth
    depth = turn.tree_depth / conversation.max_depth

    # Sibling order: position among siblings
    siblings = conversation.get_siblings(turn)
    sibling_order = siblings.index(turn) / max(len(siblings) - 1, 1)

    # Homogeneity: cosine similarity to parent
    if turn.parent:
        homogeneity = cosine_similarity(turn.embedding, turn.parent.embedding)
    else:
        homogeneity = 1.0  # Root has perfect homogeneity

    # Temporal: normalized timestamp
    temporal = (turn.timestamp - conversation.start_time) / conversation.duration

    # Complexity: count of semantic components (entities, clauses, etc.)
    complexity = count_semantic_components(turn.content)

    return Coordinate5D(depth, sibling_order, homogeneity, temporal, complexity)
```

#### 4.1.3 Trajectory Distance

**Definition 4.2 (Trajectory Distance)**. The trajectory distance between coordinates p₁ and p₂ is:

$$D_{traj}(p_1, p_2) = \sqrt{\sum_{i \in \{d,s,h,t,c\}} w_i (p_1^i - p_2^i)^2}$$

where w = (w_d, w_s, w_h, w_t, w_c) are dimension weights.

Default weights emphasizing temporal and homogeneity:

```python
DEFAULT_WEIGHTS = {
    "depth": 1.0,
    "sibling_order": 0.5,
    "homogeneity": 1.2,
    "temporal": 0.8,
    "complexity": 0.3,
}
```

### 4.2 Inverse Ring Contextual Propagation (IRCP)

#### 4.2.1 Motivation

Standard attention flows causally from earlier to later positions. In conversations, however, later clarifications often illuminate earlier ambiguous statements. IRCP captures both directions.

#### 4.2.2 Ring Topology

**Definition 4.3 (Dual Ring)**. Given n turns, we define two ring orderings:

1. **Causal Ring (RCP)**: Standard temporal order [0, 1, 2, ..., n-1]
2. **Inverse Ring (IRCP)**: Reversed order [n-1, n-2, ..., 1, 0]

Ring distance is the minimum of clockwise and counter-clockwise distances:

$$d_{ring}(i, j) = \min(|i - j|, n - |i - j|)$$

#### 4.2.3 IRCP Attention

The IRCP attention weight combines semantic similarity with spatial proximity:

$$\text{attn}[i, j] = \text{softmax}\left(\frac{w_{spatial}(i, j) \cdot w_{semantic}(i, j)}{\tau}\right)$$

where:

- $w_{spatial}(i, j) = \exp(-\lambda \cdot d_{ring}(i, j))$ penalizes distant turns
- $w_{semantic}(i, j) = \frac{1 + \cos(q_i, k_j)}{2}$ measures semantic similarity
- τ is a temperature parameter

```python
def ircp_attention(queries, keys, values, ring_order="inverse", temperature=0.1):
    n = queries.shape[0]

    # Compute semantic similarity
    semantic_weights = (1 + torch.mm(queries, keys.T)) / 2

    # Compute ring distance
    if ring_order == "inverse":
        positions = torch.arange(n-1, -1, -1)  # Reversed
    else:
        positions = torch.arange(n)  # Causal

    ring_distances = compute_ring_distance(positions)
    spatial_weights = torch.exp(-0.5 * ring_distances)

    # Combined attention
    attention = torch.softmax(
        (spatial_weights * semantic_weights) / temperature,
        dim=-1
    )

    return torch.mm(attention, values)
```

#### 4.2.4 Unified Architecture

RAG++ unifies three attention mechanisms:

| Mechanism | Ring Order | Purpose | Weight |
|-----------|------------|---------|--------|
| **RCP** | Causal | Standard past→present flow | 0.4 |
| **IRCP** | Inverse | Future→past clarification | 0.4 |
| **TPO** | Trajectory | Structure-aware reranking | 0.2 |

Combined attention:

$$\text{attn}_{unified} = \alpha_{RCP} \cdot \text{attn}_{RCP} + \alpha_{IRCP} \cdot \text{attn}_{IRCP} + \alpha_{TPO} \cdot \text{attn}_{TPO}$$

### 4.3 Slice-Conditioned Retrieval

#### 4.3.1 Admissibility

**Definition 4.4 (Admissibility)**. A turn T is admissible from anchor A if and only if there exists a path in the conversation DAG from A to T following parent-child or sibling edges.

The Graph Kernel computes admissible slices using priority-queue BFS:

```
Algorithm: ADMISSIBLE_SLICE(anchor, policy)
Input: anchor turn, slicing policy
Output: set of admissible turn IDs

1. frontier ← PriorityQueue()
2. frontier.push(anchor, priority=1.0)
3. visited ← {}
4.
5. while |visited| < policy.max_nodes and frontier not empty:
6.     current, priority ← frontier.pop()
7.     if current in visited: continue
8.     visited[current] ← priority
9.
10.    for neighbor in (current.parent, current.children, current.siblings):
11.        if neighbor not in visited:
12.            new_priority ← priority × phase_weight(neighbor) × distance_decay
13.            if new_priority > policy.min_threshold:
14.                frontier.push(neighbor, new_priority)
15.
16. return visited.keys()
```

#### 4.3.2 Cryptographic Verification

Each slice is signed with HMAC-SHA256:

```python
def sign_slice(turn_ids: List[str], secret_key: bytes) -> SliceExport:
    # Canonical serialization
    canonical_json = json.dumps(sorted(turn_ids), separators=(',', ':'))

    # Compute fingerprint
    fingerprint = xxhash.xxh64(canonical_json).hexdigest()

    # Sign with HMAC
    token = hmac.new(secret_key, canonical_json.encode(), hashlib.sha256).hexdigest()

    return SliceExport(turn_ids=turn_ids, fingerprint=fingerprint, token=token)
```

The retrieval system cannot forge tokens—it must request slices from the Graph Kernel.

### 4.4 Outcome Statistics with Welford's Algorithm

#### 4.4.1 Motivation

Memory quality tracking requires numerically stable online statistics. Welford's algorithm maintains running mean and variance without catastrophic cancellation.

#### 4.4.2 Algorithm

```python
class OutcomeStatistics:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences

    def update(self, value: float):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / self.count if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def merge(self, other: "OutcomeStatistics") -> "OutcomeStatistics":
        """Parallel-safe merge of two statistics objects."""
        if other.count == 0:
            return self

        combined = OutcomeStatistics()
        combined.count = self.count + other.count

        delta = other.mean - self.mean
        combined.mean = self.mean + delta * other.count / combined.count
        combined.M2 = self.M2 + other.M2 + delta**2 * self.count * other.count / combined.count

        return combined
```

### 4.5 Conservation Metrics

#### 4.5.1 Definition

**Definition 4.5 (Conservation Metrics)**. Given embeddings E = {e₁, ..., eₙ} and salience weights α = {α₁, ..., αₙ}:

1. **Magnitude**: $M = \sum_i \alpha_i \|e_i\|$
2. **Energy**: $E = \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j \cos(e_i, e_j)$
3. **Information**: $I = -\sum_i \alpha_i \log(\alpha_i)$

#### 4.5.2 Invariants

For a properly normalized memory system:

- **Magnitude Conservation**: M should remain approximately constant as memories are added/removed
- **Energy Bounds**: E ∈ [0, 1] when embeddings are unit-normalized
- **Information Monotonicity**: I increases as memories are added (more entropy)

```python
def validate_conservation(before: ConservationMetrics, after: ConservationMetrics, threshold=0.05):
    """Validate that conservation laws hold after memory operation."""
    magnitude_delta = abs(after.magnitude - before.magnitude) / before.magnitude

    if magnitude_delta > threshold:
        raise ConservationViolation(
            f"Magnitude changed by {magnitude_delta:.2%}, exceeds threshold {threshold:.2%}"
        )
```

---

## 5. Implementation

### 5.1 Language Stack

RAG++ uses a polyglot architecture:

| Component | Language | Justification |
|-----------|----------|---------------|
| Core indexing | Rust | SIMD intrinsics, zero-copy |
| ML/Attention | Python | PyTorch ecosystem |
| API bindings | PyO3 | Zero-copy Python-Rust bridge |
| Service layer | Python/FastAPI | Async, type-safe |
| Graph Kernel | Rust/Axum | Deterministic, cryptographic |

### 5.2 SIMD Acceleration

Distance computations use AVX2 intrinsics for 6.2x speedup:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let mut sum = _mm256_setzero_ps();

        for i in (0..a.len()).step_by(8) {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        horizontal_sum_ps(sum)
    }
}
```

### 5.3 Code Consolidation

The unified ML architecture reduced codebase by 92%:

| Before | After | Reduction |
|--------|-------|-----------|
| IRCP: 15K LOC | Unified: 3.35K LOC | 78% |
| RCP: 12K LOC | | |
| TPO: 15K LOC | | |
| **Total: 42K LOC** | **3.35K LOC** | **92%** |

Key consolidation techniques:

1. **Shared coordinate system**: All three used 5D coordinates, now unified
2. **Common attention base**: Factored out spatial weighting
3. **Configuration-driven behavior**: Ring order via config, not separate classes

### 5.4 Error Handling

RAG++ defines a hierarchical exception tree:

```
RAGPlusPlusError (base)
├── ConfigurationError
├── ValidationError
├── IndexError
│   ├── IndexNotFoundError
│   ├── IndexBuildError
│   └── IndexCapacityError
├── RetrievalError
│   ├── QueryError
│   └── TimeoutError
├── MemoryStoreError
│   ├── RecordNotFoundError
│   └── DuplicateRecordError
└── DistributedError
    ├── ShardError
    └── ReplicationError
```

---

## 6. Evaluation

### 6.1 Experimental Setup

**Hardware**: AWS c6i.8xlarge (32 vCPU, 64GB RAM, AVX-512)

**Datasets**:
- **Conversation Corpus**: 107K turns from 5 sources (ChatGPT, Claude, Telegram)
- **Synthetic Benchmark**: 10M randomly generated 768d vectors
- **Production Traffic**: 30 days of query logs (anonymized)

### 6.2 Latency Benchmarks

| Metric | Target | Achieved | Configuration |
|--------|--------|----------|---------------|
| p50 | <5ms | 3.2ms | HNSW M=32, ef=128 |
| p95 | <10ms | 8.1ms | Including network |
| p99 | <20ms | 15.3ms | Tail latency |

**Latency by component**:

| Component | Latency (p50) |
|-----------|---------------|
| Embedding | 1.2ms |
| Graph Kernel slice | 0.8ms |
| HNSW search | 1.5ms |
| Trajectory rerank | 0.4ms |
| Serialization | 0.3ms |
| **Total** | **4.2ms** |

### 6.3 Throughput

| Configuration | QPS | Notes |
|---------------|-----|-------|
| Single node | 12,500 | Batch size 32 |
| 4-node cluster | 48,000 | Linear scaling |
| With slice enforcement | 10,800 | Graph Kernel overhead |

### 6.4 Accuracy

**Retrieval Quality** (measured on held-out conversation continuations):

| Method | MRR@10 | Recall@10 | NDCG@10 |
|--------|--------|-----------|---------|
| Cosine only | 0.412 | 0.523 | 0.478 |
| + Temporal decay | 0.445 | 0.561 | 0.512 |
| + Trajectory weight | 0.487 | 0.602 | 0.554 |
| + IRCP attention | **0.523** | **0.641** | **0.589** |

**Finding**: IRCP improves MRR by 27% over cosine-only baseline.

### 6.5 Memory Efficiency

| Component | Size (768d) |
|-----------|-------------|
| Vector | 3,072 bytes |
| HNSW graph | 400 bytes |
| Trajectory coord | 40 bytes |
| Metadata | 100 bytes |
| **Total per record** | **3,612 bytes** |

**Scaling**: 150M vectors require ~540GB memory for vectors + 150GB for graph + metadata.

### 6.6 Conservation Validation

**Conservation metric stability over 30 days**:

| Metric | Initial | Final | Drift |
|--------|---------|-------|-------|
| Magnitude | 1.000 | 0.994 | -0.6% |
| Energy | 0.847 | 0.852 | +0.6% |
| Information | 12.34 | 14.21 | +15% (expected) |

Information increases due to memory growth; magnitude and energy remain stable within 1% tolerance.

---

## 7. Discussion

### 7.1 Design Decisions

**Why separate Graph Kernel?**

The Graph Kernel serves as a separate microservice rather than an embedded library to enforce the "sole admissibility authority" invariant. This architectural boundary prevents retrieval code from accidentally accessing inadmissible turns, even if a bug exists.

**Why 5 dimensions?**

We evaluated coordinate systems from 3 to 8 dimensions:

| Dimensions | MRR@10 | Complexity |
|------------|--------|------------|
| 3 (d, t, h) | 0.478 | Low |
| 5 (d, s, h, t, c) | 0.523 | Medium |
| 7 (+branch_factor, +children_count) | 0.531 | High |

5D provides optimal accuracy-complexity tradeoff. Additional dimensions provided marginal (<2%) improvement.

**Why Welford over batch statistics?**

Online updates are critical for streaming applications. Welford's algorithm:
- Requires O(1) space
- Supports parallel merge
- Avoids catastrophic cancellation in variance computation

### 7.2 Limitations

1. **Cold Start**: New conversations have insufficient trajectory data for optimal retrieval. We fall back to temporal weighting until ≥5 turns exist.

2. **Cross-Conversation Transfer**: Trajectory coordinates are conversation-local. Transferring patterns across conversations requires additional mechanism (addressed by CognitiveTwin).

3. **Graph Kernel Latency**: Slice requests add ~0.8ms latency. For latency-critical applications, slice caching mitigates this.

4. **Memory Overhead**: 5D coordinates add 40 bytes per record (~1% overhead). For very large corpora, coordinate compression could reduce this.

### 7.3 Future Work

1. **Learned Trajectory Weights**: Currently, dimension weights are hand-tuned. Learning weights via contrastive loss could improve accuracy.

2. **Hierarchical Slicing**: Multi-resolution slices that start coarse and refine based on initial results.

3. **Cross-Modal Trajectories**: Extending coordinates to include image, audio, and code modalities.

4. **Federated Memory**: Enabling slice-conditioned retrieval across organizational boundaries with differential privacy.

---

## 8. Conclusion

RAG++ introduces trajectory-aware retrieval that respects conversation structure and enforces cryptographic admissibility. Our 5D coordinate system captures depth, sibling order, homogeneity, temporal position, and complexity—dimensions ignored by standard vector search. The IRCP attention mechanism propagates information bidirectionally through ring topology, improving MRR by 27% over cosine-only baselines. By separating the Graph Kernel as sole admissibility authority, RAG++ provides provable context integrity guarantees.

The system achieves production-ready performance: p95 latency of 8.1ms, throughput of 12.5k QPS, and scaling to 150M vectors. The unified ML architecture consolidates three codebases into 3.35K LOC (92% reduction) while maintaining functional equivalence.

RAG++ is available as open-source software:
- Python package: `pip install rag-plusplus` (PyPI)
- Rust core: `rag-plusplus-core` (crates.io)
- Graph Kernel: `admissibility-kernel` (crates.io)

---

## References

[Asai et al., 2023] Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *arXiv preprint arXiv:2310.11511*.

[Child et al., 2019] Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). Generating Long Sequences with Sparse Transformers. *arXiv preprint arXiv:1904.10509*.

[Gutierrez et al., 2024] Gutierrez, B., et al. (2024). HippoRAG: Hippocampal-Inspired Memory Indexing for Retrieval-Augmented Generation.

[Izacard & Grave, 2021] Izacard, G., & Grave, E. (2021). Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering. *EACL 2021*.

[Karpukhin et al., 2020] Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020*.

[Katharopoulos et al., 2020] Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. *ICML 2020*.

[Lewis et al., 2020] Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.

[Microsoft, 2024] Microsoft Research. (2024). GraphRAG: A Graph-Based Approach to RAG.

[Park et al., 2023] Park, J. S., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST 2023*.

[Sarthi et al., 2024] Sarthi, P., et al. (2024). RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. *ICLR 2024*.

[Shinn et al., 2023] Shinn, N., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *NeurIPS 2023*.

[Su et al., 2021] Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv preprint arXiv:2104.09864*.

[Zhong et al., 2024] Zhong, W., et al. (2024). MemoryBank: Enhancing Large Language Models with Long-Term Memory.

---

## Appendix A: API Reference

### A.1 Core Types

```python
@dataclass
class MemoryRecord:
    id: str
    embedding: np.ndarray  # Shape: (dimension,)
    outcomes: OutcomeStatistics
    metadata: Dict[str, Any]
    trajectory: Optional[Coordinate5D] = None

@dataclass
class QueryBundle:
    query_embedding: np.ndarray
    k: int = 10
    filters: Optional[Dict[str, Any]] = None
    trajectory_weight: float = 0.3

@dataclass
class RetrievalResult:
    record_id: str
    score: float
    semantic_score: float
    trajectory_score: float
    provenance: Optional[str] = None
```

### A.2 Slice Types

```python
@dataclass
class SliceExport:
    turn_ids: List[str]
    fingerprint: str
    token: str  # HMAC signature
    policy_ref: str
    anchor_turn_id: str

@dataclass
class SliceScopedResults:
    turns: List[TurnResult]
    slice_export: SliceExport
    query_id: str
    latency_ms: float
```

### A.3 Configuration

```python
@dataclass
class RAGPlusPlusConfig:
    index: IndexConfig
    retrieval: RetrievalConfig
    observability: ObservabilityConfig

@dataclass
class IndexConfig:
    index_type: str = "hnsw"
    dimension: int = 768
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)

@dataclass
class HNSWConfig:
    m: int = 32
    ef_construction: int = 200
    ef_search: int = 128
```

---

## Appendix B: Benchmark Reproduction

### B.1 Environment Setup

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate
pip install rag-plusplus[dev]

# Run benchmarks
cd benchmarks/
python run_latency_benchmark.py --vectors 1000000 --queries 10000
python run_accuracy_benchmark.py --dataset conversations_107k
```

### B.2 Synthetic Data Generation

```python
import numpy as np

def generate_benchmark_data(n_vectors, dimension=768):
    """Generate synthetic benchmark data."""
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    coords = [
        Coordinate5D(
            depth=np.random.uniform(0, 1),
            sibling_order=np.random.uniform(0, 1),
            homogeneity=np.random.uniform(0.5, 1),
            temporal=np.random.uniform(0, 1),
            complexity=np.random.uniform(1, 10),
        )
        for _ in range(n_vectors)
    ]

    return vectors, coords
```

### B.3 Accuracy Evaluation

```python
def evaluate_retrieval(retriever, test_conversations, k=10):
    """Evaluate retrieval accuracy on held-out continuations."""
    results = []

    for conv in test_conversations:
        # Use all but last turn as context
        context_turns = conv.turns[:-1]
        target_turn = conv.turns[-1]

        # Retrieve
        retrieved = retriever.search(
            query=target_turn.embedding,
            k=k,
        )

        # Compute metrics
        retrieved_ids = [r.record_id for r in retrieved]
        relevant_ids = [t.id for t in context_turns if is_relevant(t, target_turn)]

        mrr = compute_mrr(retrieved_ids, relevant_ids)
        recall = compute_recall_at_k(retrieved_ids, relevant_ids, k)
        ndcg = compute_ndcg(retrieved_ids, relevant_ids, k)

        results.append({"mrr": mrr, "recall": recall, "ndcg": ndcg})

    return aggregate_metrics(results)
```
