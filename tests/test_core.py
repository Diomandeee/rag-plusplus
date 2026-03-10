"""Tests for RAG++ Python bindings."""

import numpy as np
import pytest


class TestOutcomeStats:
    """Tests for OutcomeStats."""

    def test_create_empty(self):
        """Test creating empty statistics."""
        from rag_plusplus import OutcomeStats

        stats = OutcomeStats(dim=3)
        assert stats.count == 0
        assert stats.dim == 3
        assert stats.mean is None

    def test_single_update(self):
        """Test single update."""
        from rag_plusplus import OutcomeStats

        stats = OutcomeStats(dim=3)
        outcome = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        stats.update(outcome)

        assert stats.count == 1
        np.testing.assert_array_almost_equal(stats.mean, outcome)

    def test_multiple_updates(self):
        """Test multiple updates compute correct mean."""
        from rag_plusplus import OutcomeStats

        stats = OutcomeStats(dim=2)
        stats.update(np.array([1.0, 2.0], dtype=np.float32))
        stats.update(np.array([3.0, 4.0], dtype=np.float32))
        stats.update(np.array([5.0, 6.0], dtype=np.float32))

        assert stats.count == 3
        np.testing.assert_array_almost_equal(stats.mean, [3.0, 4.0])

    def test_merge(self):
        """Test merging statistics."""
        from rag_plusplus import OutcomeStats

        stats1 = OutcomeStats(dim=2)
        stats2 = OutcomeStats(dim=2)

        stats1.update(np.array([1.0, 2.0], dtype=np.float32))
        stats2.update(np.array([3.0, 4.0], dtype=np.float32))

        merged = stats1.merge(stats2)
        assert merged.count == 2
        np.testing.assert_array_almost_equal(merged.mean, [2.0, 3.0])

    def test_variance(self):
        """Test variance computation."""
        from rag_plusplus import OutcomeStats

        stats = OutcomeStats(dim=1)
        for val in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            stats.update(np.array([val], dtype=np.float32))

        # Variance should be 4.0 for this dataset
        assert stats.variance is not None
        assert abs(stats.variance[0] - 4.0) < 0.1


class TestMemoryRecord:
    """Tests for MemoryRecord."""

    def test_create_record(self):
        """Test creating a memory record."""
        from rag_plusplus import OutcomeStats, MemoryRecord

        stats = OutcomeStats(dim=2)
        stats.update(np.array([0.8, 0.9], dtype=np.float32))

        embedding = np.random.randn(256).astype(np.float32)
        record = MemoryRecord(
            id="test_001",
            embedding=embedding,
            outcomes=stats,
        )

        assert record.id == "test_001"
        assert record.dim == 256
        assert record.version == 1
        np.testing.assert_array_equal(record.embedding, embedding)

    def test_update_outcomes(self):
        """Test updating outcomes creates new record."""
        from rag_plusplus import OutcomeStats, MemoryRecord

        stats = OutcomeStats(dim=2)
        embedding = np.random.randn(128).astype(np.float32)
        record = MemoryRecord(id="test", embedding=embedding, outcomes=stats)

        new_outcome = np.array([0.5, 0.6], dtype=np.float32)
        updated = record.with_updated_outcomes(new_outcome)

        assert record.version == 1
        assert updated.version == 2
        assert updated.outcomes.count == 1


class TestQueryBundle:
    """Tests for QueryBundle."""

    def test_create_query(self):
        """Test creating a query."""
        from rag_plusplus import QueryBundle

        embedding = np.random.randn(256).astype(np.float32)
        query = QueryBundle(query_embedding=embedding, k=10)

        assert query.k == 10
        assert query.dim == 256
        assert query.ef_search == 128
        assert query.timeout_ms == 10000

    def test_default_k(self):
        """Test default k value."""
        from rag_plusplus import QueryBundle

        embedding = np.random.randn(128).astype(np.float32)
        query = QueryBundle(query_embedding=embedding)

        assert query.k == 10  # Default


class TestFlatIndex:
    """Tests for FlatIndex (exact search)."""

    def test_create_index(self):
        """Test creating a flat index."""
        from rag_plusplus import FlatIndex, DistanceType

        index = FlatIndex(dimension=128, distance_type=DistanceType.Euclidean)
        assert index.dimension == 128
        assert index.len == 0
        assert index.is_empty()

    def test_add_and_search(self):
        """Test adding vectors and searching."""
        from rag_plusplus import FlatIndex

        index = FlatIndex(dimension=4)

        # Add some vectors
        index.add("vec1", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        index.add("vec2", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        index.add("vec3", np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))

        assert index.len == 3
        assert index.contains("vec1")
        assert not index.contains("vec4")

        # Search for nearest
        query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=2)

        assert len(results) == 2
        assert results[0].id == "vec1"  # Closest to query

    def test_remove(self):
        """Test removing a vector."""
        from rag_plusplus import FlatIndex

        index = FlatIndex(dimension=4)
        index.add("vec1", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

        assert index.contains("vec1")
        removed = index.remove("vec1")
        assert removed
        assert not index.contains("vec1")


class TestHNSWIndex:
    """Tests for HNSWIndex (approximate search)."""

    def test_create_index(self):
        """Test creating an HNSW index."""
        from rag_plusplus import HNSWIndex

        index = HNSWIndex(dimension=128, m=16, ef_construction=200)
        assert index.dimension == 128
        assert index.len == 0

    def test_add_and_search(self):
        """Test adding vectors and searching."""
        from rag_plusplus import HNSWIndex

        index = HNSWIndex(dimension=4)

        # Add vectors
        index.add("vec1", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        index.add("vec2", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        index.add("vec3", np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))

        assert index.len == 3

        # Search
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=2)

        assert len(results) == 2
        assert results[0].id == "vec1"  # Exact match


class TestIndexRegistry:
    """Tests for IndexRegistry (multi-index)."""

    def test_create_registry(self):
        """Test creating a registry."""
        from rag_plusplus import IndexRegistry

        registry = IndexRegistry()
        assert registry.len == 0
        assert registry.is_empty()

    def test_register_and_search(self):
        """Test registering indexes and searching."""
        from rag_plusplus import IndexRegistry, FlatIndex

        registry = IndexRegistry()

        # Create and register indexes
        index1 = FlatIndex(dimension=4)
        index1.add("rec1", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

        index2 = FlatIndex(dimension=4)
        index2.add("rec2", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))

        registry.register_flat("index1", index1)
        registry.register_flat("index2", index2)

        assert registry.len == 2
        assert "index1" in registry.names()
        assert "index2" in registry.names()

        # Search all indexes
        query = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        results = registry.search_all(query, k=2)

        assert "index1" in results
        assert "index2" in results


class TestInMemoryStore:
    """Tests for InMemoryStore."""

    def test_create_store(self):
        """Test creating a store."""
        from rag_plusplus import InMemoryStore

        store = InMemoryStore()
        assert store.len == 0
        assert store.is_empty()

    def test_insert_and_get(self):
        """Test inserting and retrieving records."""
        from rag_plusplus import InMemoryStore, MemoryRecord, OutcomeStats

        store = InMemoryStore()

        stats = OutcomeStats(dim=2)
        stats.update(np.array([0.5, 0.5], dtype=np.float32))

        record = MemoryRecord(
            id="rec1",
            embedding=np.random.randn(128).astype(np.float32),
            outcomes=stats,
        )

        store.insert(record)
        assert store.len == 1

        retrieved = store.get("rec1")
        assert retrieved is not None
        assert retrieved.id == "rec1"

    def test_update_stats(self):
        """Test updating record statistics."""
        from rag_plusplus import InMemoryStore, MemoryRecord, OutcomeStats

        store = InMemoryStore()
        stats = OutcomeStats(dim=2)
        record = MemoryRecord(
            id="rec1",
            embedding=np.random.randn(64).astype(np.float32),
            outcomes=stats,
        )
        store.insert(record)

        # Update stats
        updated = store.update_stats("rec1", np.array([0.8, 0.9], dtype=np.float32))
        assert updated

        # Verify update
        retrieved = store.get("rec1")
        assert retrieved.outcomes.count == 1

    def test_remove(self):
        """Test removing a record."""
        from rag_plusplus import InMemoryStore, MemoryRecord, OutcomeStats

        store = InMemoryStore()
        stats = OutcomeStats(dim=2)
        record = MemoryRecord(
            id="rec1",
            embedding=np.random.randn(64).astype(np.float32),
            outcomes=stats,
        )
        store.insert(record)

        removed = store.remove("rec1")
        assert removed is not None
        assert store.len == 0


class TestQueryCache:
    """Tests for QueryCache."""

    def test_create_cache(self):
        """Test creating a cache."""
        from rag_plusplus import QueryCache

        cache = QueryCache(max_entries=100, ttl_seconds=60)
        assert cache.len == 0
        assert cache.is_empty()

    def test_cache_stats(self):
        """Test cache statistics."""
        from rag_plusplus import QueryCache

        cache = QueryCache()
        stats = cache.stats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_ratio() == 0.0


class TestRAGPlusPlus:
    """Tests for the high-level RAGPlusPlus class."""

    def test_create_instance(self):
        """Test creating a RAG++ instance."""
        from rag_plusplus import RAGPlusPlus

        rag = RAGPlusPlus(dimension=128)
        assert rag.dimension == 128
        assert rag.index_type == "flat"
        assert len(rag) == 0

    def test_add_and_query(self):
        """Test adding records and querying."""
        from rag_plusplus import RAGPlusPlus, OutcomeStats

        rag = RAGPlusPlus(dimension=4)

        # Add records
        for i in range(5):
            stats = OutcomeStats(dim=2)
            stats.update(np.array([i * 0.1, i * 0.2], dtype=np.float32))

            embedding = np.zeros(4, dtype=np.float32)
            embedding[i % 4] = 1.0

            rag.add(f"rec_{i}", embedding, stats)

        assert len(rag) == 5
        assert rag.contains("rec_0")

        # Query
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = rag.query(query, k=3)

        assert len(results) == 3
        assert results[0].id == "rec_0"  # Closest

    def test_update_stats(self):
        """Test updating outcome statistics."""
        from rag_plusplus import RAGPlusPlus, OutcomeStats

        rag = RAGPlusPlus(dimension=64)

        stats = OutcomeStats(dim=2)
        embedding = np.random.randn(64).astype(np.float32)
        rag.add("rec1", embedding, stats)

        # Update stats
        result = rag.update_stats("rec1", np.array([0.7, 0.8], dtype=np.float32))
        assert result

        # Verify
        record = rag.get("rec1")
        assert record.outcomes.count == 1

    def test_remove(self):
        """Test removing records."""
        from rag_plusplus import RAGPlusPlus, OutcomeStats

        rag = RAGPlusPlus(dimension=32)

        stats = OutcomeStats(dim=1)
        rag.add("rec1", np.random.randn(32).astype(np.float32), stats)

        assert len(rag) == 1
        removed = rag.remove("rec1")
        assert removed is not None
        assert len(rag) == 0

    def test_hnsw_mode(self):
        """Test using HNSW index."""
        from rag_plusplus import RAGPlusPlus, OutcomeStats

        rag = RAGPlusPlus(dimension=64, use_hnsw=True, m=16, ef_construction=100)
        assert rag.index_type == "hnsw"

        # Add and query should work
        stats = OutcomeStats(dim=2)
        embedding = np.random.randn(64).astype(np.float32)
        rag.add("rec1", embedding, stats)

        results = rag.query(embedding, k=1)
        assert len(results) == 1
        assert results[0].id == "rec1"


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        from rag_plusplus import cosine_similarity

        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        sim = cosine_similarity(a, b)
        assert abs(sim - 1.0) < 1e-6

        c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sim = cosine_similarity(a, c)
        assert abs(sim - 0.0) < 1e-6

    def test_euclidean_distance(self):
        """Test Euclidean distance computation."""
        from rag_plusplus import euclidean_distance

        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([3.0, 4.0, 0.0], dtype=np.float32)

        dist = euclidean_distance(a, b)
        assert abs(dist - 5.0) < 1e-6

    def test_normalize(self):
        """Test vector normalization."""
        from rag_plusplus import normalize

        v = np.array([3.0, 4.0], dtype=np.float32)
        normalized = normalize(v)

        np.testing.assert_array_almost_equal(normalized, [0.6, 0.8])

        # Check unit length
        norm = np.sqrt(np.sum(normalized ** 2))
        assert abs(norm - 1.0) < 1e-6


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_welford_large_values(self):
        """Test Welford handles large values correctly."""
        from rag_plusplus import OutcomeStats

        stats = OutcomeStats(dim=1)
        base = 1e9

        for i in range(1000):
            stats.update(np.array([base + i * 0.001], dtype=np.float32))

        assert stats.count == 1000
        # Mean should be close to base
        assert abs(stats.mean[0] - base) < 1.0
        # Variance must be non-negative (INV-002)
        assert stats.variance[0] >= 0

    def test_welford_merge_stability(self):
        """Test merge maintains numerical stability."""
        from rag_plusplus import OutcomeStats

        # Create many small batches and merge
        stats_list = []
        for _ in range(100):
            s = OutcomeStats(dim=1)
            for _ in range(10):
                s.update(np.array([np.random.randn()], dtype=np.float32))
            stats_list.append(s)

        # Merge all
        merged = stats_list[0]
        for s in stats_list[1:]:
            merged = merged.merge(s)

        assert merged.count == 1000
        assert merged.variance[0] >= 0  # INV-002


class TestDistanceTypes:
    """Tests for different distance types."""

    def test_euclidean(self):
        """Test Euclidean distance."""
        from rag_plusplus import FlatIndex, DistanceType

        index = FlatIndex(dimension=3, distance_type=DistanceType.Euclidean)
        index.add("a", np.array([0.0, 0.0, 0.0], dtype=np.float32))
        index.add("b", np.array([1.0, 0.0, 0.0], dtype=np.float32))
        index.add("c", np.array([2.0, 0.0, 0.0], dtype=np.float32))

        # Query at origin
        query = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=3)

        # Should be ordered by distance from origin
        assert results[0].id == "a"

    def test_cosine(self):
        """Test Cosine distance."""
        from rag_plusplus import FlatIndex, DistanceType

        index = FlatIndex(dimension=3, distance_type=DistanceType.Cosine)
        index.add("north", np.array([0.0, 1.0, 0.0], dtype=np.float32))
        index.add("east", np.array([1.0, 0.0, 0.0], dtype=np.float32))
        index.add("northeast", np.array([1.0, 1.0, 0.0], dtype=np.float32))

        # Query north direction
        query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        results = index.search(query, k=3)

        # North should be most similar
        assert results[0].id == "north"
