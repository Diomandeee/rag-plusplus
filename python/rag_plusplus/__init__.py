"""RAG++: High-performance retrieval for memory-conditioned selection.

This package provides Python bindings to the Rust core library.

Example:
    >>> import numpy as np
    >>> from rag_plusplus import RAGPlusPlus, DistanceType
    >>>
    >>> # Create a RAG++ instance
    >>> rag = RAGPlusPlus(dimension=256)
    >>>
    >>> # Add records
    >>> rag.add(
    ...     id="rec_001",
    ...     embedding=np.random.randn(256).astype(np.float32),
    ...     context="User asked about Python",
    ...     outcome=0.85,
    ... )
    >>>
    >>> # Query for similar records
    >>> results = rag.query(
    ...     embedding=np.random.randn(256).astype(np.float32),
    ...     k=10,
    ... )
    >>> for result in results:
    ...     print(f"{result.id}: {result.score:.4f}")

Index Types:
    - FlatIndex: Exact brute-force search for small datasets
    - HNSWIndex: Approximate search using HNSW for large datasets
    - IndexRegistry: Multi-index management with score fusion

Storage:
    - InMemoryStore: Fast in-memory record storage
    - QueryCache: LRU cache with TTL for query results
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._rag_plusplus_rs import (
    # Core types
    MemoryRecord,
    OutcomeStats,
    SearchResult,
    # Distance types
    DistanceType,
    # Indexes
    FlatIndex,
    HNSWIndex,
    IndexRegistry,
    # Storage
    InMemoryStore,
    # Cache
    CacheStats,
    QueryCache,
    # Utility functions
    cosine_similarity,
    euclidean_distance,
    normalize,
    # Version
    __version__ as __rust_version__,
)

__version__ = __rust_version__


@dataclass
class QueryResult:
    """Result from a RAG++ query.

    Attributes:
        id: Record identifier
        embedding: Record embedding vector
        score: Similarity score (higher is better)
        distance: Distance from query
        context: Context string for this record
        outcome: Outcome value for this record
    """
    id: str
    embedding: np.ndarray
    score: float
    distance: float
    context: str
    outcome: float


class RAGPlusPlus:
    """High-level RAG++ interface for memory-conditioned retrieval.

    Provides a unified API for vector indexing, record storage, and retrieval.

    Example:
        >>> rag = RAGPlusPlus(dimension=256, use_hnsw=True)
        >>>
        >>> # Add records
        >>> for i in range(1000):
        ...     embedding = np.random.randn(256).astype(np.float32)
        ...     rag.add(f"rec_{i}", embedding, f"context_{i}", outcome=0.5)
        >>>
        >>> # Query
        >>> query_vec = np.random.randn(256).astype(np.float32)
        >>> results = rag.query(query_vec, k=10)
        >>> print(f"Found {len(results)} results")

    Args:
        dimension: Vector dimension
        use_hnsw: Use HNSW for approximate search (default: False for exact)
        distance_type: Distance metric (default: L2)
        m: HNSW M parameter (connections per node)
        ef_construction: HNSW ef_construction parameter
        cache_max_entries: Query cache size (0 to disable)
        cache_ttl_seconds: Cache TTL in seconds
    """

    def __init__(
        self,
        dimension: int,
        use_hnsw: bool = False,
        distance_type: DistanceType = DistanceType.L2,
        m: int = 16,
        ef_construction: int = 200,
        cache_max_entries: int = 10000,
        cache_ttl_seconds: int = 300,
    ):
        self.dimension = dimension
        self._use_hnsw = use_hnsw

        # Create index
        if use_hnsw:
            self._index = HNSWIndex(
                dimension=dimension,
                m=m,
                ef_construction=ef_construction,
                distance_type=distance_type,
            )
        else:
            self._index = FlatIndex(
                dimension=dimension,
                distance_type=distance_type,
            )

        # Create store
        self._store = InMemoryStore()

        # Create cache (optional)
        if cache_max_entries > 0:
            self._cache = QueryCache(
                max_entries=cache_max_entries,
                ttl_seconds=cache_ttl_seconds,
            )
        else:
            self._cache = None

    @property
    def index_type(self) -> str:
        """Return the type of index being used."""
        return "hnsw" if self._use_hnsw else "flat"

    def add(
        self,
        id: str,
        embedding: np.ndarray,
        context: str = "",
        outcome: float = 0.0,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a record to the store and index.

        Args:
            id: Unique record identifier
            embedding: Vector embedding (must match dimension)
            context: Context string describing this record
            outcome: Outcome value (e.g., relevance score)
            metadata: Optional metadata dictionary

        Raises:
            ValueError: If embedding dimension doesn't match
            RuntimeError: If record already exists
        """
        if embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} doesn't match "
                f"index dimension {self.dimension}"
            )

        # Ensure correct dtype
        embedding = np.asarray(embedding, dtype=np.float32)

        # Create record (matches Rust API: id, embedding, context, outcome, metadata)
        record = MemoryRecord(id, embedding, context, outcome, metadata)

        # Add to store
        self._store.insert(record)

        # Add to index
        self._index.add(id, embedding)

    def add_batch(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        contexts: Sequence[str],
        outcomes: Sequence[float],
    ) -> None:
        """Add multiple records in batch.

        Args:
            ids: List of unique record identifiers
            embeddings: 2D array of embeddings (n_records x dimension)
            contexts: List of context strings
            outcomes: List of outcome values
        """
        n = len(ids)
        if len(embeddings) != n or len(contexts) != n or len(outcomes) != n:
            raise ValueError("ids, embeddings, contexts, and outcomes must have same length")

        for id_, emb, ctx, out in zip(ids, embeddings, contexts, outcomes):
            self.add(id_, emb, ctx, out)

    def get(self, id: str) -> Optional[MemoryRecord]:
        """Get a record by ID.

        Args:
            id: Record identifier

        Returns:
            MemoryRecord if found, None otherwise
        """
        return self._store.get(id)

    def remove(self, id: str) -> Optional[MemoryRecord]:
        """Remove a record from store and index.

        Args:
            id: Record identifier

        Returns:
            The removed record if found, None otherwise
        """
        self._index.remove(id)
        return self._store.remove(id)

    def query(
        self,
        embedding: np.ndarray,
        k: int = 10,
    ) -> List[QueryResult]:
        """Query for similar records.

        Args:
            embedding: Query vector
            k: Number of results to return

        Returns:
            List of QueryResult ordered by similarity (descending)
        """
        if embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension {embedding.shape[0]} doesn't match "
                f"index dimension {self.dimension}"
            )

        embedding = np.asarray(embedding, dtype=np.float32)

        # Search index
        search_results = self._index.search(embedding, k)

        # Fetch full records
        results = []
        for sr in search_results:
            record = self._store.get(sr.id)
            if record is not None:
                results.append(QueryResult(
                    id=sr.id,
                    embedding=record.embedding,
                    score=sr.score,
                    distance=sr.distance,
                    context=record.context,
                    outcome=record.outcome,
                ))

        return results

    def contains(self, id: str) -> bool:
        """Check if a record exists.

        Args:
            id: Record identifier

        Returns:
            True if record exists
        """
        return self._store.contains(id)

    def __len__(self) -> int:
        """Number of records in the store."""
        return self._store.len

    def __contains__(self, id: str) -> bool:
        """Check if record exists."""
        return self.contains(id)

    @property
    def record_count(self) -> int:
        """Number of records."""
        return self._store.len

    @property
    def index_size(self) -> int:
        """Number of vectors in index."""
        return self._index.len

    @property
    def memory_bytes(self) -> int:
        """Estimated memory usage in bytes."""
        return self._store.memory_bytes

    def cache_stats(self) -> Optional[CacheStats]:
        """Get cache statistics if caching is enabled."""
        if self._cache is not None:
            return self._cache.stats()
        return None

    def clear_cache(self) -> None:
        """Clear the query cache."""
        if self._cache is not None:
            self._cache.clear()

    def ids(self) -> List[str]:
        """Get all record IDs."""
        return self._store.ids()

    def __repr__(self) -> str:
        return (
            f"RAGPlusPlus(dimension={self.dimension}, "
            f"index_type={self.index_type!r}, "
            f"records={len(self)})"
        )


# Export all public symbols
__all__ = [
    # Version
    "__version__",
    # High-level API
    "RAGPlusPlus",
    "QueryResult",
    # Core types
    "OutcomeStats",
    "MemoryRecord",
    "SearchResult",
    # Distance types
    "DistanceType",
    # Indexes
    "FlatIndex",
    "HNSWIndex",
    "IndexRegistry",
    # Storage
    "InMemoryStore",
    # Cache
    "QueryCache",
    "CacheStats",
    # Utility functions
    "cosine_similarity",
    "euclidean_distance",
    "normalize",
]
