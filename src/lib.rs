//! RAG++ Python Bindings
//!
//! PyO3 bindings exposing the Rust core to Python.

use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::time::Duration;

use rag_plusplus_core::{
    // Types
    MemoryRecord as RustMemoryRecord,
    OutcomeStats as RustOutcomeStats,
    RecordId,
    types::RecordStatus as RustRecordStatus,
    types::MetadataValue as RustMetadataValue,
    // Index
    DistanceType as RustDistanceType,
    FlatIndex as RustFlatIndex,
    HNSWConfig as RustHNSWConfig,
    HNSWIndex as RustHNSWIndex,
    IndexConfig as RustIndexConfig,
    IndexRegistry as RustIndexRegistry,
    SearchResult as RustSearchResult,
    VectorIndex,
    // Store
    InMemoryStore as RustInMemoryStore,
    RecordStore,
    // Cache
    CacheConfig as RustCacheConfig,
    CacheStats as RustCacheStats,
    QueryCache as RustQueryCache,
    // Error
    Error as RustError,
    // Trajectory (IRCP/RCP/TPO)
    TrajectoryCoordinate as RustTrajectoryCoordinate,
    TrajectoryCoordinate5D as RustTrajectoryCoordinate5D,
    TrajectoryPhase as RustTrajectoryPhase,
    PhaseInferencer as RustPhaseInferencer,
    TurnFeatures as RustTurnFeatures,
    SalienceScorer as RustSalienceScorer,
    SalienceFactors as RustSalienceFactors,
    TurnSalience as RustTurnSalience,
    Feedback as RustFeedback,
    PathQuality as RustPathQuality,
    PathQualityWeights as RustPathQualityWeights,
    PathQualityFactors as RustPathQualityFactors,
    // DLM types
    trajectory::DLMWeights as RustDLMWeights,
    // I-RCP types
    trajectory::IRCPConfig as RustIRCPConfig,
    trajectory::IRCPPropagator as RustIRCPPropagator,
    trajectory::AttentionWeights as RustAttentionWeights,
    // ChainLink types
    trajectory::ChainLink as RustChainLink,
    trajectory::ChainLinkEstimator as RustChainLinkEstimator,
    trajectory::ChainLinkEstimatorConfig as RustChainLinkEstimatorConfig,
    trajectory::ChainLinkEstimate as RustChainLinkEstimate,
    trajectory::LinkType as RustLinkType,
    // CoordinateTree types
    trajectory::CoordinateTree as RustCoordinateTree,
    // Graph types
    trajectory::TrajectoryGraph as RustTrajectoryGraph,
    trajectory::Episode as RustEpisode,
    trajectory::Edge as RustEdge,
    trajectory::EdgeType as RustEdgeType,
    // Conservation types
    trajectory::ConservationMetrics as RustConservationMetrics,
    // Branch types
    trajectory::BranchStateMachine as RustBranchStateMachine,
    trajectory::Branch as RustBranch,
    trajectory::BranchStatus as RustBranchStatus,
    trajectory::BranchResolver as RustBranchResolver,
    trajectory::RecoverableBranch as RustRecoverableBranch,
    trajectory::RecoveryStrategy as RustRecoveryStrategy,
    trajectory::LostReason as RustLostReason,
    // Chain manager types
    trajectory::ChainManager as RustChainManager,
    trajectory::ChainMetadata as RustChainMetadata,
    trajectory::CrossChainLink as RustCrossChainLink,
    trajectory::CrossChainLinkType as RustCrossChainLinkType,
    trajectory::LinkStrength as RustLinkStrength,
    trajectory::detect_knowledge_transfer as rust_detect_knowledge_transfer,
    // Trajectory-weighted distance
    trajectory_weighted_cosine as rust_trajectory_weighted_cosine,
    trajectory_weighted_cosine_5d as rust_trajectory_weighted_cosine_5d,
};

// ============================================================================
// Error Conversion
// ============================================================================

fn to_py_err(e: RustError) -> PyErr {
    PyRuntimeError::new_err(format!("{e}"))
}

// ============================================================================
// OutcomeStats
// ============================================================================

/// Python wrapper for OutcomeStats (Welford's algorithm).
#[pyclass(name = "OutcomeStats")]
#[derive(Clone)]
pub struct PyOutcomeStats {
    inner: RustOutcomeStats,
}

#[pymethods]
impl PyOutcomeStats {
    /// Create new empty statistics for given dimension.
    #[new]
    #[pyo3(signature = (dim=1))]
    fn new(dim: usize) -> Self {
        Self {
            inner: RustOutcomeStats::new(dim),
        }
    }

    /// Update statistics with a new observation.
    fn update(&mut self, outcome: PyReadonlyArray1<f32>) -> PyResult<()> {
        let slice = outcome.as_slice()?;
        self.inner.update(slice);
        Ok(())
    }

    /// Update with a scalar value.
    fn update_scalar(&mut self, value: f64) {
        self.inner.update_scalar(value);
    }

    /// Merge with another OutcomeStats.
    fn merge(&self, other: &PyOutcomeStats) -> Self {
        Self {
            inner: self.inner.merge(&other.inner),
        }
    }

    /// Number of observations.
    #[getter]
    fn count(&self) -> u64 {
        self.inner.count()
    }

    /// Dimension of outcome vectors.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Current mean estimate.
    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        self.inner.mean().map(|m| m.to_pyarray(py))
    }

    /// Scalar mean (for 1D stats).
    #[getter]
    fn mean_scalar(&self) -> Option<f64> {
        self.inner.mean_scalar()
    }

    /// Population variance.
    #[getter]
    fn variance<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        self.inner.variance().map(|v| v.to_pyarray(py))
    }

    /// Population standard deviation.
    #[getter]
    fn std<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        self.inner.std().map(|s| s.to_pyarray(py))
    }

    /// Minimum observed values.
    #[getter]
    fn min<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        self.inner.min().map(|m| m.to_pyarray(py))
    }

    /// Maximum observed values.
    #[getter]
    fn max<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        self.inner.max().map(|m| m.to_pyarray(py))
    }

    /// Confidence interval for the mean.
    fn confidence_interval<'py>(
        &self,
        py: Python<'py>,
        confidence: f32,
    ) -> Option<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
        self.inner
            .confidence_interval(confidence)
            .map(|(l, u)| (l.to_pyarray(py), u.to_pyarray(py)))
    }

    fn __repr__(&self) -> String {
        format!(
            "OutcomeStats(count={}, dim={})",
            self.inner.count(),
            self.inner.dim()
        )
    }
}

// ============================================================================
// MemoryRecord
// ============================================================================

/// Python wrapper for MemoryRecord.
#[pyclass(name = "MemoryRecord")]
#[derive(Clone)]
pub struct PyMemoryRecord {
    inner: RustMemoryRecord,
}

#[pymethods]
impl PyMemoryRecord {
    /// Create a new memory record.
    #[new]
    #[pyo3(signature = (id, embedding, context, outcome, metadata=None))]
    fn new(
        id: String,
        embedding: PyReadonlyArray1<f32>,
        context: String,
        outcome: f64,
        metadata: Option<HashMap<String, PyObject>>,
    ) -> PyResult<Self> {
        let embedding_vec = embedding.as_slice()?.to_vec();

        let mut rust_metadata = HashMap::new();
        if let Some(meta) = metadata {
            for (k, v) in meta {
                // Convert Python values to RustMetadataValue
                Python::with_gil(|py| {
                    if let Ok(s) = v.extract::<String>(py) {
                        rust_metadata.insert(k, RustMetadataValue::String(s));
                    } else if let Ok(i) = v.extract::<i64>(py) {
                        rust_metadata.insert(k, RustMetadataValue::Int(i));
                    } else if let Ok(f) = v.extract::<f64>(py) {
                        rust_metadata.insert(k, RustMetadataValue::Float(f));
                    } else if let Ok(b) = v.extract::<bool>(py) {
                        rust_metadata.insert(k, RustMetadataValue::Bool(b));
                    }
                });
            }
        }

        Ok(Self {
            inner: RustMemoryRecord {
                id,
                embedding: embedding_vec,
                context,
                outcome,
                metadata: rust_metadata,
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                status: RustRecordStatus::Active,
                stats: RustOutcomeStats::new(1),
            },
        })
    }

    /// Record ID.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// Primary embedding.
    #[getter]
    fn embedding<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.embedding.to_pyarray(py)
    }

    /// Embedding dimension.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.embedding.len()
    }

    /// Context string.
    #[getter]
    fn context(&self) -> &str {
        &self.inner.context
    }

    /// Outcome value.
    #[getter]
    fn outcome(&self) -> f64 {
        self.inner.outcome
    }

    /// Creation timestamp (Unix seconds).
    #[getter]
    fn created_at(&self) -> u64 {
        self.inner.created_at
    }

    /// Outcome statistics.
    #[getter]
    fn stats(&self) -> PyOutcomeStats {
        PyOutcomeStats {
            inner: self.inner.stats.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MemoryRecord(id='{}', dim={}, outcome={:.3})",
            self.inner.id,
            self.inner.embedding.len(),
            self.inner.outcome
        )
    }
}

// ============================================================================
// SearchResult
// ============================================================================

/// Python wrapper for SearchResult.
#[pyclass(name = "SearchResult")]
#[derive(Clone)]
pub struct PySearchResult {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub distance: f32,
    #[pyo3(get)]
    pub score: f32,
}

#[pymethods]
impl PySearchResult {
    fn __repr__(&self) -> String {
        format!(
            "SearchResult(id='{}', distance={:.4}, score={:.4})",
            self.id, self.distance, self.score
        )
    }
}

impl From<RustSearchResult> for PySearchResult {
    fn from(r: RustSearchResult) -> Self {
        Self {
            id: r.id,
            distance: r.distance,
            score: r.score,
        }
    }
}

// ============================================================================
// DistanceType
// ============================================================================

/// Distance metric type.
#[pyclass(name = "DistanceType", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyDistanceType {
    L2,
    Cosine,
    InnerProduct,
}

impl From<PyDistanceType> for RustDistanceType {
    fn from(dt: PyDistanceType) -> Self {
        match dt {
            PyDistanceType::L2 => RustDistanceType::L2,
            PyDistanceType::Cosine => RustDistanceType::Cosine,
            PyDistanceType::InnerProduct => RustDistanceType::InnerProduct,
        }
    }
}

// ============================================================================
// FlatIndex
// ============================================================================

/// Python wrapper for FlatIndex (exact brute-force search).
#[pyclass(name = "FlatIndex")]
pub struct PyFlatIndex {
    inner: RustFlatIndex,
}

#[pymethods]
impl PyFlatIndex {
    /// Create a new flat index.
    #[new]
    #[pyo3(signature = (dimension, distance_type=None))]
    fn new(dimension: usize, distance_type: Option<PyDistanceType>) -> Self {
        let dt = distance_type.map(Into::into).unwrap_or(RustDistanceType::L2);
        let config = RustIndexConfig::new(dimension).with_distance(dt);
        Self {
            inner: RustFlatIndex::new(config),
        }
    }

    /// Add a vector to the index.
    fn add(&mut self, id: String, vector: PyReadonlyArray1<f32>) -> PyResult<()> {
        let slice = vector.as_slice()?;
        self.inner.add(id, slice).map_err(to_py_err)
    }

    /// Search for nearest neighbors.
    fn search(&self, query: PyReadonlyArray1<f32>, k: usize) -> PyResult<Vec<PySearchResult>> {
        let slice = query.as_slice()?;
        let results = self.inner.search(slice, k).map_err(to_py_err)?;
        Ok(results.into_iter().map(Into::into).collect())
    }

    /// Remove a vector from the index.
    fn remove(&mut self, id: &str) -> PyResult<bool> {
        self.inner.remove(id).map_err(to_py_err)
    }

    /// Check if ID exists in index.
    fn contains(&self, id: &str) -> bool {
        self.inner.contains(id)
    }

    /// Number of vectors in index.
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether index is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Vector dimension.
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn __repr__(&self) -> String {
        format!("FlatIndex(len={}, dim={})", self.inner.len(), self.inner.dimension())
    }
}

// ============================================================================
// HNSWIndex
// ============================================================================

/// Python wrapper for HNSWIndex (approximate nearest neighbor).
#[pyclass(name = "HNSWIndex")]
pub struct PyHNSWIndex {
    inner: RustHNSWIndex,
}

#[pymethods]
impl PyHNSWIndex {
    /// Create a new HNSW index.
    #[new]
    #[pyo3(signature = (dimension, m=16, ef_construction=200, distance_type=None))]
    fn new(
        dimension: usize,
        m: usize,
        ef_construction: usize,
        distance_type: Option<PyDistanceType>,
    ) -> Self {
        let dt = distance_type.map(Into::into).unwrap_or(RustDistanceType::L2);
        let config = RustHNSWConfig::new(dimension)
            .with_m(m)
            .with_ef_construction(ef_construction)
            .with_distance(dt);
        Self {
            inner: RustHNSWIndex::new(config),
        }
    }

    /// Add a vector to the index.
    fn add(&mut self, id: String, vector: PyReadonlyArray1<f32>) -> PyResult<()> {
        let slice = vector.as_slice()?;
        self.inner.add(id, slice).map_err(to_py_err)
    }

    /// Search for nearest neighbors.
    fn search(&self, query: PyReadonlyArray1<f32>, k: usize) -> PyResult<Vec<PySearchResult>> {
        let slice = query.as_slice()?;
        let results = self.inner.search(slice, k).map_err(to_py_err)?;
        Ok(results.into_iter().map(Into::into).collect())
    }

    /// Remove a vector from the index.
    fn remove(&mut self, id: &str) -> PyResult<bool> {
        self.inner.remove(id).map_err(to_py_err)
    }

    /// Check if ID exists in index.
    fn contains(&self, id: &str) -> bool {
        self.inner.contains(id)
    }

    /// Number of vectors in index.
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether index is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Vector dimension.
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn __repr__(&self) -> String {
        format!("HNSWIndex(len={}, dim={})", self.inner.len(), self.inner.dimension())
    }
}

// ============================================================================
// IndexRegistry
// ============================================================================

/// Python wrapper for IndexRegistry.
#[pyclass(name = "IndexRegistry")]
pub struct PyIndexRegistry {
    inner: RustIndexRegistry,
}

#[pymethods]
impl PyIndexRegistry {
    /// Create a new index registry.
    #[new]
    fn new() -> Self {
        Self {
            inner: RustIndexRegistry::new(),
        }
    }

    /// Register a FlatIndex by creating a new one with same config.
    fn register_flat(&mut self, name: String, dimension: usize) -> PyResult<()> {
        let config = RustIndexConfig::new(dimension);
        let index = RustFlatIndex::new(config);
        self.inner.register(&name, index).map_err(to_py_err)
    }

    /// Register an HNSWIndex by creating a new one.
    fn register_hnsw(&mut self, name: String, dimension: usize, m: usize, ef_construction: usize) -> PyResult<()> {
        let config = RustHNSWConfig::new(dimension)
            .with_m(m)
            .with_ef_construction(ef_construction);
        let index = RustHNSWIndex::new(config);
        self.inner.register(&name, index).map_err(to_py_err)
    }

    /// Add a vector to a specific index.
    fn add(&mut self, index_name: &str, id: String, vector: PyReadonlyArray1<f32>) -> PyResult<()> {
        let slice = vector.as_slice()?;
        self.inner.add(index_name, id, slice).map_err(to_py_err)
    }

    /// Remove an index by name.
    fn remove(&mut self, name: &str) -> bool {
        self.inner.remove(name).is_some()
    }

    /// Check if registry has an index.
    fn contains(&self, name: &str) -> bool {
        self.inner.contains(name)
    }

    /// Number of registered indexes.
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether registry is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Search all indexes.
    fn search_all(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<HashMap<String, Vec<PySearchResult>>> {
        let slice = query.as_slice()?;
        let multi_results = self.inner.search_all(slice, k).map_err(to_py_err)?;

        let mut result = HashMap::new();
        for mir in multi_results.by_index {
            result.insert(mir.index_name, mir.results.into_iter().map(Into::into).collect());
        }
        Ok(result)
    }

    fn __repr__(&self) -> String {
        format!("IndexRegistry(indexes={})", self.inner.len())
    }
}

// ============================================================================
// InMemoryStore
// ============================================================================

/// Python wrapper for InMemoryStore.
#[pyclass(name = "InMemoryStore")]
pub struct PyInMemoryStore {
    inner: RustInMemoryStore,
}

#[pymethods]
impl PyInMemoryStore {
    /// Create a new in-memory store.
    #[new]
    fn new() -> Self {
        Self {
            inner: RustInMemoryStore::new(),
        }
    }

    /// Insert a record.
    fn insert(&mut self, record: &PyMemoryRecord) -> PyResult<String> {
        let id = self.inner.insert(record.inner.clone()).map_err(to_py_err)?;
        Ok(id)
    }

    /// Get a record by ID.
    fn get(&self, id: &str) -> Option<PyMemoryRecord> {
        let record_id: RecordId = id.to_string();
        self.inner.get(&record_id).map(|r| PyMemoryRecord { inner: r })
    }

    /// Update outcome statistics for a record.
    fn update_stats(&mut self, id: &str, outcome: f64) -> PyResult<()> {
        let record_id: RecordId = id.to_string();
        self.inner.update_stats(&record_id, outcome).map_err(to_py_err)
    }

    /// Remove a record.
    fn remove(&mut self, id: &str) -> PyResult<bool> {
        let record_id: RecordId = id.to_string();
        self.inner.remove(&record_id).map_err(to_py_err)
    }

    /// Check if record exists.
    fn contains(&self, id: &str) -> bool {
        let record_id: RecordId = id.to_string();
        self.inner.contains(&record_id)
    }

    /// Get all record IDs.
    fn ids(&self) -> Vec<String> {
        self.inner.ids()
    }

    /// Number of records.
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether store is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Estimated memory usage in bytes.
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.memory_usage()
    }

    fn __repr__(&self) -> String {
        format!("InMemoryStore(records={})", self.inner.len())
    }
}

// ============================================================================
// QueryCache
// ============================================================================

/// Python wrapper for CacheStats.
#[pyclass(name = "CacheStats")]
#[derive(Clone)]
pub struct PyCacheStats {
    #[pyo3(get)]
    pub hits: u64,
    #[pyo3(get)]
    pub misses: u64,
    #[pyo3(get)]
    pub entries: usize,
    #[pyo3(get)]
    pub evictions: u64,
    #[pyo3(get)]
    pub expirations: u64,
}

#[pymethods]
impl PyCacheStats {
    fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CacheStats(hits={}, misses={}, hit_ratio={:.2}%)",
            self.hits,
            self.misses,
            self.hit_ratio() * 100.0
        )
    }
}

impl From<RustCacheStats> for PyCacheStats {
    fn from(s: RustCacheStats) -> Self {
        Self {
            hits: s.hits,
            misses: s.misses,
            entries: s.entries,
            evictions: s.evictions,
            expirations: s.expirations,
        }
    }
}

/// Python wrapper for QueryCache.
#[pyclass(name = "QueryCache")]
pub struct PyQueryCache {
    inner: RustQueryCache,
}

#[pymethods]
impl PyQueryCache {
    /// Create a new query cache.
    #[new]
    #[pyo3(signature = (max_entries=10000, ttl_seconds=300))]
    fn new(max_entries: usize, ttl_seconds: u64) -> Self {
        let config = RustCacheConfig::new()
            .with_max_entries(max_entries)
            .with_ttl(Duration::from_secs(ttl_seconds));
        Self {
            inner: RustQueryCache::new(config),
        }
    }

    /// Clear all cached entries.
    fn clear(&self) {
        self.inner.clear();
    }

    /// Remove expired entries.
    fn cleanup_expired(&self) {
        self.inner.cleanup_expired();
    }

    /// Get cache statistics.
    fn stats(&self) -> PyCacheStats {
        self.inner.stats().into()
    }

    /// Current number of cached entries.
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether cache is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.stats();
        format!(
            "QueryCache(entries={}, hit_ratio={:.2}%)",
            stats.entries,
            stats.hit_ratio() * 100.0
        )
    }
}

// ============================================================================
// Trajectory Module (IRCP/RCP/TPO)
// ============================================================================

/// 4D Trajectory Coordinate for episode positioning.
///
/// Dimensions:
/// - depth: Distance from root in conversation tree (0 = root)
/// - sibling_order: Position among siblings at branching points
/// - homogeneity: Semantic similarity to parent [0, 1]
/// - temporal: Normalized timestamp within trajectory [0, 1]
#[pyclass(name = "TrajectoryCoordinate")]
#[derive(Clone)]
pub struct PyTrajectoryCoordinate {
    inner: RustTrajectoryCoordinate,
}

#[pymethods]
impl PyTrajectoryCoordinate {
    #[new]
    fn new(depth: u32, sibling_order: u32, homogeneity: f32, temporal: f32) -> Self {
        Self {
            inner: RustTrajectoryCoordinate::new(depth, sibling_order, homogeneity, temporal),
        }
    }

    /// Create a root coordinate.
    #[staticmethod]
    fn root() -> Self {
        Self {
            inner: RustTrajectoryCoordinate::root(),
        }
    }

    #[getter]
    fn depth(&self) -> u32 {
        self.inner.depth
    }

    #[getter]
    fn sibling_order(&self) -> u32 {
        self.inner.sibling_order
    }

    #[getter]
    fn homogeneity(&self) -> f32 {
        self.inner.homogeneity
    }

    #[getter]
    fn temporal(&self) -> f32 {
        self.inner.temporal
    }

    /// Euclidean distance to another coordinate.
    fn distance(&self, other: &PyTrajectoryCoordinate) -> f32 {
        self.inner.distance(&other.inner)
    }

    /// Manhattan distance to another coordinate.
    fn manhattan_distance(&self, other: &PyTrajectoryCoordinate) -> f32 {
        self.inner.manhattan_distance(&other.inner)
    }

    /// Weighted distance with custom dimension weights.
    fn weighted_distance(&self, other: &PyTrajectoryCoordinate, weights: (f32, f32, f32, f32)) -> f32 {
        self.inner.weighted_distance(&other.inner, weights)
    }

    /// Convert to tuple.
    fn to_tuple(&self) -> (u32, u32, f32, f32) {
        (self.inner.depth, self.inner.sibling_order, self.inner.homogeneity, self.inner.temporal)
    }

    /// Convert to dict.
    fn to_dict(&self) -> HashMap<String, f32> {
        let mut d = HashMap::new();
        d.insert("depth".to_string(), self.inner.depth as f32);
        d.insert("sibling_order".to_string(), self.inner.sibling_order as f32);
        d.insert("homogeneity".to_string(), self.inner.homogeneity);
        d.insert("temporal".to_string(), self.inner.temporal);
        d
    }

    fn __repr__(&self) -> String {
        format!(
            "TrajectoryCoordinate(depth={}, sibling={}, homo={:.2}, time={:.2})",
            self.inner.depth, self.inner.sibling_order, self.inner.homogeneity, self.inner.temporal
        )
    }
}

/// 5D Trajectory Coordinate with complexity dimension (TPO extension).
///
/// Adds the complexity dimension from TPO:
/// - complexity: Message complexity (n_parts: text + images + code blocks)
#[pyclass(name = "TrajectoryCoordinate5D")]
#[derive(Clone)]
pub struct PyTrajectoryCoordinate5D {
    inner: RustTrajectoryCoordinate5D,
}

#[pymethods]
impl PyTrajectoryCoordinate5D {
    #[new]
    fn new(depth: u32, sibling_order: u32, homogeneity: f32, temporal: f32, complexity: u32) -> Self {
        Self {
            inner: RustTrajectoryCoordinate5D::new(depth, sibling_order, homogeneity, temporal, complexity),
        }
    }

    /// Create from a 4D coordinate with default complexity (1).
    #[staticmethod]
    fn from_4d(coord: &PyTrajectoryCoordinate) -> Self {
        Self {
            inner: RustTrajectoryCoordinate5D::from_4d(&coord.inner),
        }
    }

    /// Create from a 4D coordinate with specified complexity.
    #[staticmethod]
    fn from_4d_with_complexity(coord: &PyTrajectoryCoordinate, complexity: u32) -> Self {
        Self {
            inner: RustTrajectoryCoordinate5D::from_4d_with_complexity(&coord.inner, complexity),
        }
    }

    /// Create a root coordinate.
    #[staticmethod]
    fn root() -> Self {
        Self {
            inner: RustTrajectoryCoordinate5D::root(),
        }
    }

    #[getter]
    fn depth(&self) -> u32 {
        self.inner.depth
    }

    #[getter]
    fn sibling_order(&self) -> u32 {
        self.inner.sibling_order
    }

    #[getter]
    fn homogeneity(&self) -> f32 {
        self.inner.homogeneity
    }

    #[getter]
    fn temporal(&self) -> f32 {
        self.inner.temporal
    }

    #[getter]
    fn complexity(&self) -> u32 {
        self.inner.complexity
    }

    /// Euclidean distance (log-scaled complexity).
    fn distance(&self, other: &PyTrajectoryCoordinate5D) -> f32 {
        self.inner.distance(&other.inner)
    }

    /// Normalized complexity factor [0, 1].
    fn complexity_factor(&self) -> f32 {
        self.inner.complexity_factor()
    }

    /// Check if this is a multimodal message (complexity > 1).
    fn is_multimodal(&self) -> bool {
        self.inner.is_multimodal()
    }

    /// Convert to 4D coordinate (discards complexity).
    fn to_4d(&self) -> PyTrajectoryCoordinate {
        PyTrajectoryCoordinate {
            inner: self.inner.to_4d(),
        }
    }

    /// Convert to dict.
    fn to_dict(&self) -> HashMap<String, f32> {
        let mut d = HashMap::new();
        d.insert("depth".to_string(), self.inner.depth as f32);
        d.insert("sibling_order".to_string(), self.inner.sibling_order as f32);
        d.insert("homogeneity".to_string(), self.inner.homogeneity);
        d.insert("temporal".to_string(), self.inner.temporal);
        d.insert("complexity".to_string(), self.inner.complexity as f32);
        d
    }

    fn __repr__(&self) -> String {
        format!(
            "TrajectoryCoordinate5D(depth={}, sibling={}, homo={:.2}, time={:.2}, complex={})",
            self.inner.depth, self.inner.sibling_order, self.inner.homogeneity,
            self.inner.temporal, self.inner.complexity
        )
    }
}

/// Trajectory phase representing qualitative state within a conversation.
#[pyclass(name = "TrajectoryPhase", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyTrajectoryPhase {
    Exploration,
    Consolidation,
    Synthesis,
    Debugging,
    Planning,
}

impl From<RustTrajectoryPhase> for PyTrajectoryPhase {
    fn from(p: RustTrajectoryPhase) -> Self {
        match p {
            RustTrajectoryPhase::Exploration => PyTrajectoryPhase::Exploration,
            RustTrajectoryPhase::Consolidation => PyTrajectoryPhase::Consolidation,
            RustTrajectoryPhase::Synthesis => PyTrajectoryPhase::Synthesis,
            RustTrajectoryPhase::Debugging => PyTrajectoryPhase::Debugging,
            RustTrajectoryPhase::Planning => PyTrajectoryPhase::Planning,
        }
    }
}

impl From<PyTrajectoryPhase> for RustTrajectoryPhase {
    fn from(p: PyTrajectoryPhase) -> Self {
        match p {
            PyTrajectoryPhase::Exploration => RustTrajectoryPhase::Exploration,
            PyTrajectoryPhase::Consolidation => RustTrajectoryPhase::Consolidation,
            PyTrajectoryPhase::Synthesis => RustTrajectoryPhase::Synthesis,
            PyTrajectoryPhase::Debugging => RustTrajectoryPhase::Debugging,
            PyTrajectoryPhase::Planning => RustTrajectoryPhase::Planning,
        }
    }
}

/// User feedback type for salience scoring.
#[pyclass(name = "Feedback", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyFeedback {
    ThumbsUp,
    ThumbsDown,
}

impl From<PyFeedback> for RustFeedback {
    fn from(f: PyFeedback) -> Self {
        match f {
            PyFeedback::ThumbsUp => RustFeedback::ThumbsUp,
            PyFeedback::ThumbsDown => RustFeedback::ThumbsDown,
        }
    }
}

/// Turn features for phase inference.
#[pyclass(name = "TurnFeatures")]
#[derive(Clone)]
pub struct PyTurnFeatures {
    inner: RustTurnFeatures,
}

#[pymethods]
impl PyTurnFeatures {
    /// Create turn features from message content.
    #[staticmethod]
    fn from_content(turn_id: u64, role: &str, content: &str) -> Self {
        Self {
            inner: RustTurnFeatures::from_content(turn_id, role, content),
        }
    }

    #[getter]
    fn turn_id(&self) -> u64 {
        self.inner.turn_id
    }

    #[getter]
    fn role(&self) -> &str {
        &self.inner.role
    }

    #[getter]
    fn is_user(&self) -> bool {
        self.inner.role == "user"
    }

    #[getter]
    fn content_length(&self) -> usize {
        self.inner.content_length
    }

    #[getter]
    fn question_count(&self) -> usize {
        self.inner.question_count
    }

    #[getter]
    fn code_block_count(&self) -> usize {
        self.inner.code_block_count
    }

    #[getter]
    fn list_item_count(&self) -> usize {
        self.inner.list_item_count
    }

    #[getter]
    fn has_error_keywords(&self) -> bool {
        self.inner.has_error_keywords
    }

    #[getter]
    fn has_decision_keywords(&self) -> bool {
        self.inner.has_decision_keywords
    }

    #[getter]
    fn has_planning_keywords(&self) -> bool {
        self.inner.has_planning_keywords
    }

    fn __repr__(&self) -> String {
        format!(
            "TurnFeatures(id={}, role='{}', len={}, questions={})",
            self.inner.turn_id, self.inner.role, self.inner.content_length, self.inner.question_count
        )
    }
}

/// Phase inferencer for detecting conversation phases.
#[pyclass(name = "PhaseInferencer")]
pub struct PyPhaseInferencer {
    inner: RustPhaseInferencer,
}

#[pymethods]
impl PyPhaseInferencer {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustPhaseInferencer::new(),
        }
    }

    /// Infer phase from turn features.
    ///
    /// Returns (phase, confidence) tuple.
    fn infer_single(&self, features: &PyTurnFeatures) -> PyResult<(PyTrajectoryPhase, f32)> {
        self.inner
            .infer_single(&features.inner)
            .map(|(phase, conf)| (phase.into(), conf))
            .ok_or_else(|| PyValueError::new_err("Could not infer phase"))
    }

    fn __repr__(&self) -> String {
        "PhaseInferencer()".to_string()
    }
}

/// Salience factors for computing turn importance.
#[pyclass(name = "SalienceFactors")]
#[derive(Clone)]
pub struct PySalienceFactors {
    inner: RustSalienceFactors,
}

#[pymethods]
impl PySalienceFactors {
    #[new]
    #[pyo3(signature = (turn_id, role, content_length, feedback=None, reference_count=0, is_phase_transition=false, embedding=None))]
    fn new(
        turn_id: u64,
        role: String,
        content_length: usize,
        feedback: Option<PyFeedback>,
        reference_count: usize,
        is_phase_transition: bool,
        embedding: Option<Vec<f32>>,
    ) -> Self {
        Self {
            inner: RustSalienceFactors {
                turn_id,
                feedback: feedback.map(Into::into),
                is_phase_transition,
                reference_count,
                embedding,
                role,
                content_length,
            },
        }
    }

    #[getter]
    fn turn_id(&self) -> u64 {
        self.inner.turn_id
    }

    #[getter]
    fn role(&self) -> &str {
        &self.inner.role
    }

    #[getter]
    fn content_length(&self) -> usize {
        self.inner.content_length
    }

    #[getter]
    fn is_phase_transition(&self) -> bool {
        self.inner.is_phase_transition
    }

    #[getter]
    fn reference_count(&self) -> usize {
        self.inner.reference_count
    }

    fn __repr__(&self) -> String {
        format!("SalienceFactors(turn_id={}, role='{}')", self.inner.turn_id, self.inner.role)
    }
}

/// Turn salience result with score breakdown.
#[pyclass(name = "TurnSalience")]
#[derive(Clone)]
pub struct PyTurnSalience {
    inner: RustTurnSalience,
}

#[pymethods]
impl PyTurnSalience {
    #[getter]
    fn turn_id(&self) -> u64 {
        self.inner.turn_id
    }

    #[getter]
    fn score(&self) -> f32 {
        self.inner.score
    }

    #[getter]
    fn feedback_contribution(&self) -> f32 {
        self.inner.feedback_contribution
    }

    #[getter]
    fn phase_contribution(&self) -> f32 {
        self.inner.phase_contribution
    }

    #[getter]
    fn reference_contribution(&self) -> f32 {
        self.inner.reference_contribution
    }

    #[getter]
    fn novelty_contribution(&self) -> f32 {
        self.inner.novelty_contribution
    }

    fn __repr__(&self) -> String {
        format!(
            "TurnSalience(turn_id={}, score={:.3})",
            self.inner.turn_id, self.inner.score
        )
    }
}

/// Salience scorer for computing turn importance weights.
#[pyclass(name = "SalienceScorer")]
pub struct PySalienceScorer {
    inner: RustSalienceScorer,
}

#[pymethods]
impl PySalienceScorer {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustSalienceScorer::new(),
        }
    }

    /// Score a single turn.
    ///
    /// Returns TurnSalience with score and contribution breakdown.
    #[pyo3(signature = (factors, recent_embeddings=None))]
    fn score_single(
        &self,
        factors: &PySalienceFactors,
        recent_embeddings: Option<Vec<Vec<f32>>>,
    ) -> PyTurnSalience {
        let embeddings_ref = recent_embeddings.as_ref().map(|v| v.as_slice());
        PyTurnSalience {
            inner: self.inner.score_single(&factors.inner, embeddings_ref),
        }
    }

    fn __repr__(&self) -> String {
        "SalienceScorer()".to_string()
    }
}

/// Path quality weights for TPO scoring.
#[pyclass(name = "PathQualityWeights")]
#[derive(Clone)]
pub struct PyPathQualityWeights {
    inner: RustPathQualityWeights,
}

#[pymethods]
impl PyPathQualityWeights {
    #[new]
    #[pyo3(signature = (alpha=0.25, beta=0.30, gamma=0.25, delta=0.20))]
    fn new(alpha: f32, beta: f32, gamma: f32, delta: f32) -> Self {
        Self {
            inner: RustPathQualityWeights { alpha, beta, gamma, delta },
        }
    }

    /// Default TPO weights.
    #[staticmethod]
    fn default_weights() -> Self {
        Self {
            inner: RustPathQualityWeights::default(),
        }
    }

    #[getter]
    fn alpha(&self) -> f32 {
        self.inner.alpha
    }

    #[getter]
    fn beta(&self) -> f32 {
        self.inner.beta
    }

    #[getter]
    fn gamma(&self) -> f32 {
        self.inner.gamma
    }

    #[getter]
    fn delta(&self) -> f32 {
        self.inner.delta
    }

    fn __repr__(&self) -> String {
        format!(
            "PathQualityWeights(α={:.2}, β={:.2}, γ={:.2}, δ={:.2})",
            self.inner.alpha, self.inner.beta, self.inner.gamma, self.inner.delta
        )
    }
}

/// Path quality factors for TPO scoring.
#[pyclass(name = "PathQualityFactors")]
#[derive(Clone)]
pub struct PyPathQualityFactors {
    inner: RustPathQualityFactors,
}

#[pymethods]
impl PyPathQualityFactors {
    #[new]
    fn new(linearity: f32, terminal_score: f32, coherence: f32, completion: f32) -> Self {
        Self {
            inner: RustPathQualityFactors {
                linearity,
                terminal_score,
                coherence,
                completion,
            },
        }
    }

    #[getter]
    fn linearity(&self) -> f32 {
        self.inner.linearity
    }

    #[getter]
    fn terminal_score(&self) -> f32 {
        self.inner.terminal_score
    }

    #[getter]
    fn coherence(&self) -> f32 {
        self.inner.coherence
    }

    #[getter]
    fn completion(&self) -> f32 {
        self.inner.completion
    }

    /// Compute quality score with given weights.
    fn compute_quality(&self, weights: &PyPathQualityWeights) -> f32 {
        self.inner.compute_quality(&weights.inner)
    }

    /// Compute quality score with default weights.
    fn quality(&self) -> f32 {
        self.inner.compute_quality(&RustPathQualityWeights::default())
    }

    fn __repr__(&self) -> String {
        format!(
            "PathQualityFactors(L={:.2}, T={:.2}, S={:.2}, C={:.2})",
            self.inner.linearity, self.inner.terminal_score,
            self.inner.coherence, self.inner.completion
        )
    }
}

/// Path quality utilities for TPO-based salience enhancement.
#[pyclass(name = "PathQuality")]
pub struct PyPathQuality;

#[pymethods]
impl PyPathQuality {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Enhance base salience with path quality.
    #[staticmethod]
    fn enhance_salience(base_salience: f32, quality: f32, blend: f32) -> f32 {
        RustPathQuality::enhance_salience(base_salience, quality, blend)
    }

    /// Boost terminal nodes based on path quality.
    #[staticmethod]
    fn terminal_boost(base_salience: f32, quality: f32, is_terminal: bool) -> f32 {
        RustPathQuality::terminal_boost(base_salience, quality, is_terminal)
    }

    /// Get terminal quality score from phase name.
    #[staticmethod]
    #[pyo3(signature = (phase=None))]
    fn terminal_quality_from_phase(phase: Option<&str>) -> f32 {
        RustPathQuality::terminal_quality_from_phase(phase)
    }

    /// Get terminal quality score from feedback.
    #[staticmethod]
    fn terminal_quality_from_feedback(has_thumbs_up: bool, has_thumbs_down: bool) -> f32 {
        RustPathQuality::terminal_quality_from_feedback(has_thumbs_up, has_thumbs_down)
    }
}

// ============================================================================
// DLM Weights
// ============================================================================

/// DLM coordinate weight configuration for distance calculations.
///
/// Default weights (from DLM config):
/// - depth: 0.25 (structural)
/// - sibling: 0.15 (structural)
/// - homogeneity: 0.30 (semantic)
/// - temporal: 0.20 (temporal)
/// - complexity: 0.10 (from TPO n_parts)
#[pyclass(name = "DLMWeights")]
#[derive(Clone)]
pub struct PyDLMWeights {
    inner: RustDLMWeights,
}

#[pymethods]
impl PyDLMWeights {
    #[new]
    #[pyo3(signature = (depth=0.25, sibling=0.15, homogeneity=0.30, temporal=0.20, complexity=0.10))]
    fn new(depth: f32, sibling: f32, homogeneity: f32, temporal: f32, complexity: f32) -> Self {
        Self {
            inner: RustDLMWeights {
                depth,
                sibling,
                homogeneity,
                temporal,
                complexity,
            },
        }
    }

    /// Create default weights.
    #[staticmethod]
    fn default_weights() -> Self {
        Self {
            inner: RustDLMWeights::default(),
        }
    }

    /// Create weights optimized for semantic similarity.
    #[staticmethod]
    fn semantic_focused() -> Self {
        Self {
            inner: RustDLMWeights::semantic_focused(),
        }
    }

    /// Create weights optimized for structural similarity.
    #[staticmethod]
    fn structural_focused() -> Self {
        Self {
            inner: RustDLMWeights::structural_focused(),
        }
    }

    /// Create weights optimized for temporal proximity.
    #[staticmethod]
    fn temporal_focused() -> Self {
        Self {
            inner: RustDLMWeights::temporal_focused(),
        }
    }

    /// Create equal weights for all dimensions.
    #[staticmethod]
    fn equal() -> Self {
        Self {
            inner: RustDLMWeights::equal(),
        }
    }

    #[getter]
    fn depth(&self) -> f32 {
        self.inner.depth
    }

    #[getter]
    fn sibling(&self) -> f32 {
        self.inner.sibling
    }

    #[getter]
    fn homogeneity(&self) -> f32 {
        self.inner.homogeneity
    }

    #[getter]
    fn temporal(&self) -> f32 {
        self.inner.temporal
    }

    #[getter]
    fn complexity(&self) -> f32 {
        self.inner.complexity
    }

    /// Convert to array [depth, sibling, homogeneity, temporal, complexity].
    fn to_array(&self) -> [f32; 5] {
        self.inner.to_array()
    }

    /// Convert to dict.
    fn to_dict(&self) -> HashMap<String, f32> {
        let mut d = HashMap::new();
        d.insert("depth".to_string(), self.inner.depth);
        d.insert("sibling".to_string(), self.inner.sibling);
        d.insert("homogeneity".to_string(), self.inner.homogeneity);
        d.insert("temporal".to_string(), self.inner.temporal);
        d.insert("complexity".to_string(), self.inner.complexity);
        d
    }

    fn __repr__(&self) -> String {
        format!(
            "DLMWeights(d={:.2}, s={:.2}, h={:.2}, t={:.2}, c={:.2})",
            self.inner.depth, self.inner.sibling, self.inner.homogeneity,
            self.inner.temporal, self.inner.complexity
        )
    }
}

// ============================================================================
// I-RCP (Inverse Ring Contextual Propagation)
// ============================================================================

/// I-RCP configuration for attention computation.
#[pyclass(name = "IRCPConfig")]
#[derive(Clone)]
pub struct PyIRCPConfig {
    inner: RustIRCPConfig,
}

#[pymethods]
impl PyIRCPConfig {
    #[new]
    #[pyo3(signature = (
        temperature=1.0,
        spatial_weight=0.3,
        causal_mask=false
    ))]
    fn new(temperature: f32, spatial_weight: f32, causal_mask: bool) -> Self {
        Self {
            inner: RustIRCPConfig {
                temperature,
                coord_weights: RustDLMWeights::default(),
                spatial_weight,
                use_coordinate_cosine: false,
                min_attention: 1e-10,
                causal_mask,
            },
        }
    }

    /// Default I-RCP config.
    #[staticmethod]
    fn default_config() -> Self {
        Self {
            inner: RustIRCPConfig::default(),
        }
    }

    /// Semantic-focused attention (favor embedding similarity).
    #[staticmethod]
    fn semantic_focused() -> Self {
        Self {
            inner: RustIRCPConfig::semantic_focused(),
        }
    }

    /// Spatial-focused attention (favor coordinate distance).
    #[staticmethod]
    fn spatial_focused() -> Self {
        Self {
            inner: RustIRCPConfig::spatial_focused(),
        }
    }

    /// Causal attention (no looking at future).
    #[staticmethod]
    fn causal() -> Self {
        Self {
            inner: RustIRCPConfig::causal(),
        }
    }

    /// Sharp attention (low temperature).
    #[staticmethod]
    fn sharp() -> Self {
        Self {
            inner: RustIRCPConfig::sharp(),
        }
    }

    /// Diffuse attention (high temperature).
    #[staticmethod]
    fn diffuse() -> Self {
        Self {
            inner: RustIRCPConfig::diffuse(),
        }
    }

    #[getter]
    fn temperature(&self) -> f32 {
        self.inner.temperature
    }

    #[getter]
    fn spatial_weight(&self) -> f32 {
        self.inner.spatial_weight
    }

    #[getter]
    fn causal_mask(&self) -> bool {
        self.inner.causal_mask
    }

    fn __repr__(&self) -> String {
        format!(
            "IRCPConfig(temp={:.2}, spatial={:.2}, causal={})",
            self.inner.temperature, self.inner.spatial_weight, self.inner.causal_mask
        )
    }
}

/// Attention weights from I-RCP propagation.
#[pyclass(name = "AttentionWeights")]
#[derive(Clone)]
pub struct PyAttentionWeights {
    inner: RustAttentionWeights,
}

#[pymethods]
impl PyAttentionWeights {
    #[getter]
    fn forward<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.forward.to_pyarray(py)
    }

    #[getter]
    fn inverse<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.inverse.to_pyarray(py)
    }

    #[getter]
    fn cross<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.cross.to_pyarray(py)
    }

    #[getter]
    fn raw_scores<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.raw_scores.to_pyarray(py)
    }

    #[getter]
    fn total_mass(&self) -> f32 {
        self.inner.total_mass
    }

    fn __len__(&self) -> usize {
        self.inner.forward.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "AttentionWeights(len={}, total_mass={:.3})",
            self.inner.forward.len(),
            self.inner.total_mass
        )
    }
}

/// I-RCP propagator for computing attention weights.
#[pyclass(name = "IRCPPropagator")]
pub struct PyIRCPPropagator {
    inner: RustIRCPPropagator,
}

#[pymethods]
impl PyIRCPPropagator {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<&PyIRCPConfig>) -> Self {
        let inner = match config {
            Some(c) => RustIRCPPropagator::new(c.inner.clone()),
            None => RustIRCPPropagator::new(RustIRCPConfig::default()),
        };
        Self { inner }
    }

    /// Compute full I-RCP attention (forward + inverse + cross).
    fn compute_attention(
        &self,
        query_coord: &PyTrajectoryCoordinate5D,
        context_coords: Vec<PyTrajectoryCoordinate5D>,
        query_emb: PyReadonlyArray1<f32>,
        context_embs: Vec<PyReadonlyArray1<f32>>,
    ) -> PyResult<PyAttentionWeights> {
        let query_slice = query_emb.as_slice()?;
        let context_rust: Vec<RustTrajectoryCoordinate5D> =
            context_coords.iter().map(|c| c.inner).collect();

        // Convert context embeddings
        let mut context_vecs: Vec<Vec<f32>> = Vec::with_capacity(context_embs.len());
        for emb in &context_embs {
            context_vecs.push(emb.as_slice()?.to_vec());
        }
        let context_refs: Vec<&[f32]> = context_vecs.iter().map(|v| v.as_slice()).collect();

        let weights = self.inner.compute_attention(
            &query_coord.inner,
            &context_rust,
            query_slice,
            &context_refs,
        );

        Ok(PyAttentionWeights { inner: weights })
    }

    /// Compute trajectory coordinates for a new turn based on context.
    ///
    /// Args:
    ///     embedding: The embedding vector for the new turn.
    ///     neighbors: List of (turn_id, similarity) tuples for neighbor turns.
    ///     neighbor_coords: Trajectory coordinates of neighbor turns.
    ///     session_depth: Current depth within the session.
    ///     turn_index: Index of this turn within its conversation.
    ///     sibling_count: Number of sibling branches at this depth.
    ///     sibling_order: Order among siblings (0-indexed).
    ///     project_phase: Project phase string (e.g., "planning", "implementation").
    ///
    /// Returns:
    ///     PyTrajectoryCoordinate5D with computed coordinates.
    #[pyo3(signature = (embedding, neighbors, neighbor_coords, session_depth=0, turn_index=0, sibling_count=1, sibling_order=0, project_phase="implementation"))]
    #[allow(unused_variables)]
    fn compute_trajectory_coordinates(
        &self,
        embedding: PyReadonlyArray1<f32>,
        neighbors: Vec<(String, f32)>,
        neighbor_coords: Vec<PyTrajectoryCoordinate5D>,
        session_depth: i32,
        turn_index: i32,
        sibling_count: i32,
        sibling_order: i32,
        project_phase: &str,
    ) -> PyResult<PyTrajectoryCoordinate5D> {
        let emb_slice = embedding.as_slice()?;

        // Compute depth (raw value, not normalized)
        let depth = session_depth.max(0) as u32;

        // Compute sibling order (raw value)
        let sibling_order_u32 = sibling_order.max(0) as u32;

        // Compute homogeneity based on neighbor similarities
        let homogeneity = if !neighbors.is_empty() {
            let sum: f32 = neighbors.iter().map(|(_, sim)| *sim).sum();
            (sum / neighbors.len() as f32).clamp(0.0, 1.0)
        } else {
            0.5 // Default homogeneity when no neighbors
        };

        // Compute temporal: normalized turn index
        let max_turns = 100.0_f32;
        let temporal = (turn_index as f32 / max_turns).min(1.0);

        // Compute complexity based on embedding variance
        let mean: f32 = emb_slice.iter().sum::<f32>() / emb_slice.len() as f32;
        let variance: f32 = emb_slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / emb_slice.len() as f32;
        let complexity = (variance.sqrt() * 10.0).round().max(1.0) as u32;

        Ok(PyTrajectoryCoordinate5D {
            inner: RustTrajectoryCoordinate5D::new(depth, sibling_order_u32, homogeneity, temporal, complexity),
        })
    }

    fn __repr__(&self) -> String {
        "IRCPPropagator()".to_string()
    }
}

// ============================================================================
// ChainLink Types
// ============================================================================

/// Link type categorizing the relationship between turns.
#[pyclass(name = "LinkType", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyLinkType {
    Continuation,
    Elaboration,
    Summary,
    Contradiction,
    Tangent,
    Question,
    Answer,
    Code,
    Error,
    Solution,
    Meta,
    Unknown,
}

impl From<RustLinkType> for PyLinkType {
    fn from(lt: RustLinkType) -> Self {
        match lt {
            RustLinkType::Continuation => PyLinkType::Continuation,
            RustLinkType::Elaboration => PyLinkType::Elaboration,
            RustLinkType::Summary => PyLinkType::Summary,
            RustLinkType::Contradiction => PyLinkType::Contradiction,
            RustLinkType::Tangent => PyLinkType::Tangent,
            RustLinkType::Question => PyLinkType::Question,
            RustLinkType::Answer => PyLinkType::Answer,
            RustLinkType::Code => PyLinkType::Code,
            RustLinkType::Error => PyLinkType::Error,
            RustLinkType::Solution => PyLinkType::Solution,
            RustLinkType::Meta => PyLinkType::Meta,
            RustLinkType::Unknown => PyLinkType::Unknown,
        }
    }
}

impl From<PyLinkType> for RustLinkType {
    fn from(lt: PyLinkType) -> Self {
        match lt {
            PyLinkType::Continuation => RustLinkType::Continuation,
            PyLinkType::Elaboration => RustLinkType::Elaboration,
            PyLinkType::Summary => RustLinkType::Summary,
            PyLinkType::Contradiction => RustLinkType::Contradiction,
            PyLinkType::Tangent => RustLinkType::Tangent,
            PyLinkType::Question => RustLinkType::Question,
            PyLinkType::Answer => RustLinkType::Answer,
            PyLinkType::Code => RustLinkType::Code,
            PyLinkType::Error => RustLinkType::Error,
            PyLinkType::Solution => RustLinkType::Solution,
            PyLinkType::Meta => RustLinkType::Meta,
            PyLinkType::Unknown => RustLinkType::Unknown,
        }
    }
}

/// A link in the response chain with coordinate and embedding.
#[pyclass(name = "ChainLink")]
#[derive(Clone)]
pub struct PyChainLink {
    inner: RustChainLink,
}

#[pymethods]
impl PyChainLink {
    #[new]
    #[pyo3(signature = (coordinate, embedding, link_type=None, influence=1.0, is_user=true, id=None))]
    fn new(
        coordinate: &PyTrajectoryCoordinate5D,
        embedding: Vec<f32>,
        link_type: Option<PyLinkType>,
        influence: f32,
        is_user: bool,
        id: Option<String>,
    ) -> Self {
        Self {
            inner: RustChainLink {
                coordinate: coordinate.inner,
                embedding,
                link_type: link_type.map(Into::into).unwrap_or(RustLinkType::Unknown),
                influence,
                is_user,
                attention: None,
                id,
            },
        }
    }

    #[getter]
    fn coordinate(&self) -> PyTrajectoryCoordinate5D {
        PyTrajectoryCoordinate5D {
            inner: self.inner.coordinate,
        }
    }

    #[getter]
    fn embedding<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.embedding.to_pyarray(py)
    }

    #[getter]
    fn link_type(&self) -> PyLinkType {
        self.inner.link_type.into()
    }

    #[getter]
    fn influence(&self) -> f32 {
        self.inner.influence
    }

    #[getter]
    fn is_user(&self) -> bool {
        self.inner.is_user
    }

    #[getter]
    fn id(&self) -> Option<&str> {
        self.inner.id.as_deref()
    }

    /// Cosine similarity to another chain link.
    fn similarity(&self, other: &PyChainLink) -> f32 {
        self.inner.semantic_similarity(&other.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "ChainLink(type={:?}, influence={:.2}, user={})",
            self.inner.link_type, self.inner.influence, self.inner.is_user
        )
    }
}

/// ChainLink estimator configuration.
#[pyclass(name = "ChainLinkEstimatorConfig")]
#[derive(Clone)]
pub struct PyChainLinkEstimatorConfig {
    inner: RustChainLinkEstimatorConfig,
}

#[pymethods]
impl PyChainLinkEstimatorConfig {
    #[new]
    #[pyo3(signature = (baseline_weight=0.2, relationship_weight=0.3, type_weight=0.2, context_weight=0.3))]
    fn new(
        baseline_weight: f32,
        relationship_weight: f32,
        type_weight: f32,
        context_weight: f32,
    ) -> Self {
        Self {
            inner: RustChainLinkEstimatorConfig::with_weights(
                baseline_weight,
                relationship_weight,
                type_weight,
                context_weight,
            ),
        }
    }

    /// Default estimator config.
    #[staticmethod]
    fn default_config() -> Self {
        Self {
            inner: RustChainLinkEstimatorConfig::default(),
        }
    }

    #[getter]
    fn baseline_weight(&self) -> f32 {
        self.inner.baseline_weight
    }

    #[getter]
    fn relationship_weight(&self) -> f32 {
        self.inner.relationship_weight
    }

    #[getter]
    fn type_weight(&self) -> f32 {
        self.inner.type_weight
    }

    #[getter]
    fn context_weight(&self) -> f32 {
        self.inner.context_weight
    }

    fn __repr__(&self) -> String {
        format!(
            "ChainLinkEstimatorConfig(b={:.2}, r={:.2}, t={:.2}, c={:.2})",
            self.inner.baseline_weight, self.inner.relationship_weight,
            self.inner.type_weight, self.inner.context_weight
        )
    }
}

/// Detailed estimate result from ChainLink estimator.
#[pyclass(name = "ChainLinkEstimate")]
#[derive(Clone)]
pub struct PyChainLinkEstimate {
    inner: RustChainLinkEstimate,
}

#[pymethods]
impl PyChainLinkEstimate {
    #[getter]
    fn total(&self) -> f32 {
        self.inner.total
    }

    #[getter]
    fn baseline(&self) -> f32 {
        self.inner.baseline
    }

    #[getter]
    fn relationship(&self) -> f32 {
        self.inner.relationship
    }

    #[getter]
    fn type_based(&self) -> f32 {
        self.inner.type_based
    }

    #[getter]
    fn context(&self) -> f32 {
        self.inner.context
    }

    fn __repr__(&self) -> String {
        format!(
            "ChainLinkEstimate(total={:.3}, base={:.3}, rel={:.3}, type={:.3}, ctx={:.3})",
            self.inner.total, self.inner.baseline, self.inner.relationship,
            self.inner.type_based, self.inner.context
        )
    }
}

/// ChainLink estimator for computing relationship strength between links.
#[pyclass(name = "ChainLinkEstimator")]
pub struct PyChainLinkEstimator {
    inner: RustChainLinkEstimator,
}

#[pymethods]
impl PyChainLinkEstimator {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<&PyChainLinkEstimatorConfig>) -> Self {
        let inner = match config {
            Some(c) => RustChainLinkEstimator::new(c.inner.clone()),
            None => RustChainLinkEstimator::new(RustChainLinkEstimatorConfig::default()),
        };
        Self { inner }
    }

    /// Estimate relationship strength between two links.
    fn estimate(&self, link_a: &PyChainLink, link_b: &PyChainLink) -> f32 {
        self.inner.estimate(&link_a.inner, &link_b.inner)
    }

    /// Get detailed estimate with component breakdown.
    fn estimate_detailed(&self, link_a: &PyChainLink, link_b: &PyChainLink) -> PyChainLinkEstimate {
        PyChainLinkEstimate {
            inner: self.inner.estimate_detailed(&link_a.inner, &link_b.inner),
        }
    }

    fn __repr__(&self) -> String {
        "ChainLinkEstimator()".to_string()
    }
}

// ============================================================================
// Ring Structures
// ============================================================================

// Note: Ring and DualRing are generic types in Rust.
// For Python bindings, we use specialized wrappers storing coordinate tuples.

/// Simple ring structure for Python (stores NodeId + coordinate pairs).
///
/// Ring topology provides:
/// - Circular access (wrapping indices)
/// - Ring distance (shortest path in either direction)
/// - Neighbor iteration within radius
#[pyclass(name = "Ring")]
pub struct PyRing {
    nodes: Vec<(u64, RustTrajectoryCoordinate5D)>,
}

#[pymethods]
impl PyRing {
    #[new]
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
        }
    }

    /// Create ring with initial capacity.
    #[staticmethod]
    fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
        }
    }

    /// Add an element to the ring.
    fn push(&mut self, id: u64, coord: &PyTrajectoryCoordinate5D) {
        self.nodes.push((id, coord.inner));
    }

    /// Get element at index (wrapping).
    fn get(&self, idx: usize) -> Option<(u64, PyTrajectoryCoordinate5D)> {
        if self.nodes.is_empty() {
            return None;
        }
        let idx = idx % self.nodes.len();
        self.nodes.get(idx).map(|(id, c)| {
            (*id, PyTrajectoryCoordinate5D { inner: *c })
        })
    }

    /// Number of elements in ring.
    #[getter]
    fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if ring is empty.
    fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get ring distance between two indices.
    fn ring_distance(&self, a: usize, b: usize) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        let n = self.nodes.len();
        let a = a % n;
        let b = b % n;
        let forward = (b + n - a) % n;
        let backward = (a + n - b) % n;
        forward.min(backward)
    }

    /// Get next index (wrapping).
    fn next_idx(&self, idx: usize) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        (idx + 1) % self.nodes.len()
    }

    /// Get previous index (wrapping).
    fn prev_idx(&self, idx: usize) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        if idx == 0 {
            self.nodes.len() - 1
        } else {
            idx - 1
        }
    }

    /// Get neighbor indices within radius.
    fn neighbor_indices(&self, idx: usize, radius: usize) -> Vec<usize> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let n = self.nodes.len();
        let idx = idx % n;
        let mut indices = Vec::new();
        for d in 1..=radius {
            let prev_idx = (idx + n - d) % n;
            let next_idx = (idx + d) % n;
            if prev_idx != next_idx {
                indices.push(prev_idx);
                indices.push(next_idx);
            } else {
                indices.push(prev_idx);
            }
        }
        indices
    }

    fn __repr__(&self) -> String {
        format!("Ring(len={})", self.nodes.len())
    }
}

/// Dual-ring structure for I-RCP (forward + inverse rings).
///
/// Forward ring: assistant responses
/// Inverse ring: user messages
/// Used for computing I-RCP attention.
#[pyclass(name = "DualRing")]
pub struct PyDualRing {
    forward: Vec<(u64, RustTrajectoryCoordinate5D)>,
    inverse: Vec<(u64, RustTrajectoryCoordinate5D)>,
}

#[pymethods]
impl PyDualRing {
    #[new]
    fn new() -> Self {
        Self {
            forward: Vec::new(),
            inverse: Vec::new(),
        }
    }

    /// Add a node to the appropriate ring based on is_user.
    fn add(&mut self, id: u64, coord: &PyTrajectoryCoordinate5D, is_user: bool) {
        if is_user {
            self.inverse.push((id, coord.inner));
        } else {
            self.forward.push((id, coord.inner));
        }
    }

    /// Get total count of nodes.
    #[getter]
    fn len(&self) -> usize {
        self.forward.len() + self.inverse.len()
    }

    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.forward.is_empty() && self.inverse.is_empty()
    }

    /// Get forward ring length (assistant responses).
    fn forward_len(&self) -> usize {
        self.forward.len()
    }

    /// Get inverse ring length (user messages).
    fn inverse_len(&self) -> usize {
        self.inverse.len()
    }

    /// Get node from forward ring by index.
    fn get_forward(&self, idx: usize) -> Option<(u64, PyTrajectoryCoordinate5D)> {
        if self.forward.is_empty() {
            return None;
        }
        let idx = idx % self.forward.len();
        self.forward.get(idx).map(|(id, c)| (*id, PyTrajectoryCoordinate5D { inner: *c }))
    }

    /// Get node from inverse ring by index.
    fn get_inverse(&self, idx: usize) -> Option<(u64, PyTrajectoryCoordinate5D)> {
        if self.inverse.is_empty() {
            return None;
        }
        let idx = idx % self.inverse.len();
        self.inverse.get(idx).map(|(id, c)| (*id, PyTrajectoryCoordinate5D { inner: *c }))
    }

    fn __repr__(&self) -> String {
        format!(
            "DualRing(forward={}, inverse={})",
            self.forward.len(),
            self.inverse.len()
        )
    }
}

// ============================================================================
// Trajectory Graph
// ============================================================================

/// Edge type in trajectory graph.
///
/// - Continuation: Normal sequential flow
/// - Regeneration: Alternative response (regeneration)
/// - Branch: User-initiated branch
#[pyclass(name = "EdgeType", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyEdgeType {
    Continuation,
    Regeneration,
    Branch,
}

impl From<RustEdgeType> for PyEdgeType {
    fn from(et: RustEdgeType) -> Self {
        match et {
            RustEdgeType::Continuation => PyEdgeType::Continuation,
            RustEdgeType::Regeneration => PyEdgeType::Regeneration,
            RustEdgeType::Branch => PyEdgeType::Branch,
        }
    }
}

impl From<PyEdgeType> for RustEdgeType {
    fn from(et: PyEdgeType) -> Self {
        match et {
            PyEdgeType::Continuation => RustEdgeType::Continuation,
            PyEdgeType::Regeneration => RustEdgeType::Regeneration,
            PyEdgeType::Branch => RustEdgeType::Branch,
        }
    }
}

/// Episode in trajectory graph (node).
///
/// Episodes are the nodes in the conversation DAG. Each episode has:
/// - A unique NodeId (u64)
/// - Optional parent reference
/// - Children references for branching
/// - Metadata like weight, feedback, timestamps
#[pyclass(name = "Episode")]
#[derive(Clone)]
pub struct PyEpisode {
    inner: RustEpisode,
}

#[pymethods]
impl PyEpisode {
    #[new]
    #[pyo3(signature = (id, parent=None, weight=1.0, has_thumbs_up=false, has_thumbs_down=false, content_length=0, has_error=false, created_at=0))]
    fn new(
        id: u64,
        parent: Option<u64>,
        weight: f32,
        has_thumbs_up: bool,
        has_thumbs_down: bool,
        content_length: usize,
        has_error: bool,
        created_at: i64,
    ) -> Self {
        Self {
            inner: RustEpisode {
                id,
                parent,
                children: Vec::new(),
                weight,
                has_thumbs_up,
                has_thumbs_down,
                content_length,
                has_error,
                created_at,
            },
        }
    }

    #[getter]
    fn id(&self) -> u64 {
        self.inner.id
    }

    #[getter]
    fn parent(&self) -> Option<u64> {
        self.inner.parent
    }

    #[getter]
    fn children(&self) -> Vec<u64> {
        self.inner.children.clone()
    }

    #[getter]
    fn weight(&self) -> f32 {
        self.inner.weight
    }

    #[getter]
    fn has_thumbs_up(&self) -> bool {
        self.inner.has_thumbs_up
    }

    #[getter]
    fn has_thumbs_down(&self) -> bool {
        self.inner.has_thumbs_down
    }

    #[getter]
    fn content_length(&self) -> usize {
        self.inner.content_length
    }

    #[getter]
    fn has_error(&self) -> bool {
        self.inner.has_error
    }

    #[getter]
    fn created_at(&self) -> i64 {
        self.inner.created_at
    }

    /// Check if this is a leaf node (no children).
    fn is_leaf(&self) -> bool {
        self.inner.is_leaf()
    }

    /// Check if this is a root node (no parent).
    fn is_root(&self) -> bool {
        self.inner.is_root()
    }

    /// Check if this is a branch point (multiple children).
    fn is_branch_point(&self) -> bool {
        self.inner.is_branch_point()
    }

    fn __repr__(&self) -> String {
        format!(
            "Episode(id={}, parent={:?}, children={}, weight={:.2})",
            self.inner.id, self.inner.parent, self.inner.children.len(), self.inner.weight
        )
    }
}

/// Trajectory graph for managing conversation DAG.
///
/// The graph stores episodes (nodes) connected by edges representing
/// parent-child relationships. Supports branching and regeneration.
#[pyclass(name = "TrajectoryGraph")]
pub struct PyTrajectoryGraph {
    inner: RustTrajectoryGraph,
}

#[pymethods]
impl PyTrajectoryGraph {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustTrajectoryGraph::new(),
        }
    }

    /// Create graph from a list of edges.
    ///
    /// Each edge is (parent_id, child_id, edge_type).
    #[staticmethod]
    fn from_edges(edges: Vec<(u64, u64, Option<PyEdgeType>)>) -> Self {
        let rust_edges: Vec<RustEdge> = edges.iter().map(|(parent, child, et)| {
            RustEdge {
                parent: *parent,
                child: *child,
                edge_type: et.map(Into::into).unwrap_or(RustEdgeType::Continuation),
            }
        }).collect();
        Self {
            inner: RustTrajectoryGraph::from_edges(rust_edges.into_iter()),
        }
    }

    /// Add an episode (node) to the graph.
    fn add_node(&mut self, episode: &PyEpisode) {
        self.inner.add_node(episode.inner.clone());
    }

    /// Get an episode by ID.
    fn get_node(&self, id: u64) -> Option<PyEpisode> {
        self.inner.get_node(id).map(|e| PyEpisode { inner: e.clone() })
    }

    /// Number of episodes (nodes).
    #[getter]
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Get root episodes (no parent).
    fn roots(&self) -> Vec<u64> {
        self.inner.roots().to_vec()
    }

    /// Get leaf episodes (no children).
    fn leaves(&self) -> Vec<u64> {
        self.inner.leaves().to_vec()
    }

    /// Check if an episode is a branch point.
    fn is_branch_point(&self, id: u64) -> bool {
        self.inner.is_branch_point(id)
    }

    /// Find all branch points in the graph.
    fn find_branch_points(&self) -> Vec<(u64, Vec<u64>)> {
        self.inner.find_branch_points()
            .into_iter()
            .map(|info| (info.branch_point, info.children))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "TrajectoryGraph(nodes={}, roots={}, leaves={})",
            self.inner.node_count(),
            self.inner.roots().len(),
            self.inner.leaves().len()
        )
    }
}

// ============================================================================
// Conservation Metrics
// ============================================================================

/// Conservation metrics for trajectory validation.
#[pyclass(name = "ConservationMetrics")]
#[derive(Clone)]
pub struct PyConservationMetrics {
    inner: RustConservationMetrics,
}

#[pymethods]
impl PyConservationMetrics {
    /// Compute conservation metrics from embeddings and attention weights.
    #[staticmethod]
    fn compute(
        embeddings: Vec<PyReadonlyArray1<f32>>,
        attention: Vec<f32>,
    ) -> PyResult<Self> {
        let mut emb_vecs: Vec<Vec<f32>> = Vec::with_capacity(embeddings.len());
        for emb in &embeddings {
            emb_vecs.push(emb.as_slice()?.to_vec());
        }
        let emb_refs: Vec<&[f32]> = emb_vecs.iter().map(|v| v.as_slice()).collect();

        let inner = RustConservationMetrics::compute(&emb_refs, &attention);
        Ok(Self { inner })
    }

    #[getter]
    fn magnitude(&self) -> f32 {
        self.inner.magnitude
    }

    #[getter]
    fn energy(&self) -> f32 {
        self.inner.energy
    }

    #[getter]
    fn information(&self) -> f32 {
        self.inner.information
    }

    /// Check if metrics are conserved within tolerance.
    fn is_conserved(&self, other: &PyConservationMetrics, tolerance: f32) -> bool {
        self.inner.is_conserved(&other.inner, tolerance)
    }

    fn __repr__(&self) -> String {
        format!(
            "ConservationMetrics(mag={:.3}, energy={:.3}, info={:.3})",
            self.inner.magnitude, self.inner.energy, self.inner.information
        )
    }
}

// ============================================================================
// Branch State Machine
// ============================================================================

/// Branch status.
#[pyclass(name = "BranchStatus", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyBranchStatus {
    Active,
    Archived,
    Merged,
    Recovered,
    Orphaned,
}

impl From<RustBranchStatus> for PyBranchStatus {
    fn from(s: RustBranchStatus) -> Self {
        match s {
            RustBranchStatus::Active => PyBranchStatus::Active,
            RustBranchStatus::Archived => PyBranchStatus::Archived,
            RustBranchStatus::Merged => PyBranchStatus::Merged,
            RustBranchStatus::Recovered => PyBranchStatus::Recovered,
            RustBranchStatus::Orphaned => PyBranchStatus::Orphaned,
        }
    }
}

impl From<PyBranchStatus> for RustBranchStatus {
    fn from(s: PyBranchStatus) -> Self {
        match s {
            PyBranchStatus::Active => RustBranchStatus::Active,
            PyBranchStatus::Archived => RustBranchStatus::Archived,
            PyBranchStatus::Merged => RustBranchStatus::Merged,
            PyBranchStatus::Recovered => RustBranchStatus::Recovered,
            PyBranchStatus::Orphaned => RustBranchStatus::Orphaned,
        }
    }
}

/// Recovery strategy for lost branches.
#[pyclass(name = "RecoveryStrategy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyRecoveryStrategy {
    Reactivate,
    Copy,
    MergeInto,
    SplitIndependent,
}

impl From<&RustRecoveryStrategy> for PyRecoveryStrategy {
    fn from(s: &RustRecoveryStrategy) -> Self {
        match s {
            RustRecoveryStrategy::Reactivate => PyRecoveryStrategy::Reactivate,
            RustRecoveryStrategy::Copy => PyRecoveryStrategy::Copy,
            RustRecoveryStrategy::MergeInto(_) => PyRecoveryStrategy::MergeInto,
            RustRecoveryStrategy::SplitIndependent => PyRecoveryStrategy::SplitIndependent,
        }
    }
}

/// Reason a branch was lost.
#[pyclass(name = "LostReason", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyLostReason {
    Archived,
    Untracked,
    OrphanedByDeletion,
    UnselectedRegeneration,
    Abandoned,
    ExplorationDivergence,
}

impl From<&RustLostReason> for PyLostReason {
    fn from(r: &RustLostReason) -> Self {
        match r {
            RustLostReason::Archived => PyLostReason::Archived,
            RustLostReason::Untracked => PyLostReason::Untracked,
            RustLostReason::OrphanedByDeletion => PyLostReason::OrphanedByDeletion,
            RustLostReason::UnselectedRegeneration => PyLostReason::UnselectedRegeneration,
            RustLostReason::Abandoned => PyLostReason::Abandoned,
            RustLostReason::ExplorationDivergence => PyLostReason::ExplorationDivergence,
        }
    }
}

/// Branch in conversation trajectory.
#[pyclass(name = "Branch")]
#[derive(Clone)]
pub struct PyBranch {
    inner: RustBranch,
}

#[pymethods]
impl PyBranch {
    #[getter]
    fn id(&self) -> u64 {
        self.inner.id
    }

    #[getter]
    fn fork_point(&self) -> u64 {
        self.inner.fork_point
    }

    #[getter]
    fn head(&self) -> u64 {
        self.inner.head
    }

    #[getter]
    fn status(&self) -> PyBranchStatus {
        self.inner.status.into()
    }

    #[getter]
    fn parent_branch(&self) -> Option<u64> {
        self.inner.parent_branch
    }

    #[getter]
    fn child_branches(&self) -> Vec<u64> {
        self.inner.child_branches.clone()
    }

    #[getter]
    fn nodes(&self) -> Vec<u64> {
        self.inner.nodes.clone()
    }

    #[getter]
    fn label(&self) -> Option<String> {
        self.inner.label.clone()
    }

    #[getter]
    fn created_at(&self) -> i64 {
        self.inner.created_at
    }

    #[getter]
    fn updated_at(&self) -> i64 {
        self.inner.updated_at
    }

    /// Check if this branch is active.
    fn is_active(&self) -> bool {
        self.inner.is_active()
    }

    /// Check if this is the root branch.
    fn is_root(&self) -> bool {
        self.inner.is_root()
    }

    /// Check if this branch has been merged.
    fn is_merged(&self) -> bool {
        self.inner.is_merged()
    }

    /// Get the number of nodes in this branch.
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Branch(id={}, head={}, status={:?}, nodes={})",
            self.inner.id, self.inner.head, self.inner.status, self.inner.len()
        )
    }
}

/// Recoverable branch information.
#[pyclass(name = "RecoverableBranch")]
#[derive(Clone)]
pub struct PyRecoverableBranch {
    inner: RustRecoverableBranch,
}

#[pymethods]
impl PyRecoverableBranch {
    #[getter]
    fn branch_id(&self) -> Option<u64> {
        self.inner.branch_id
    }

    #[getter]
    fn fork_point(&self) -> u64 {
        self.inner.fork_point
    }

    #[getter]
    fn entry_node(&self) -> u64 {
        self.inner.entry_node
    }

    #[getter]
    fn nodes(&self) -> Vec<u64> {
        self.inner.nodes.clone()
    }

    #[getter]
    fn head(&self) -> u64 {
        self.inner.head
    }

    #[getter]
    fn depth(&self) -> u32 {
        self.inner.depth
    }

    #[getter]
    fn lost_reason(&self) -> PyLostReason {
        (&self.inner.lost_reason).into()
    }

    #[getter]
    fn recovery_score(&self) -> f32 {
        self.inner.recovery_score
    }

    #[getter]
    fn suggested_strategy(&self) -> PyRecoveryStrategy {
        (&self.inner.suggested_strategy).into()
    }

    fn __repr__(&self) -> String {
        format!(
            "RecoverableBranch(id={:?}, fork={}, score={:.2}, strategy={:?})",
            self.inner.branch_id, self.inner.fork_point,
            self.inner.recovery_score, self.inner.suggested_strategy
        )
    }
}

/// Branch state machine for managing conversation branches.
#[pyclass(name = "BranchStateMachine")]
pub struct PyBranchStateMachine {
    inner: RustBranchStateMachine,
}

#[pymethods]
impl PyBranchStateMachine {
    /// Create a new state machine from a trajectory graph.
    #[staticmethod]
    fn from_graph(graph: &PyTrajectoryGraph) -> Self {
        Self {
            inner: RustBranchStateMachine::from_graph(graph.inner.clone()),
        }
    }

    /// Split a branch at a given node, creating a new independent branch.
    ///
    /// Returns a tuple of (original_branch_id, new_branch_id, split_point, moved_nodes).
    fn split(&mut self, node_id: u64) -> PyResult<(u64, u64, u64, Vec<u64>)> {
        let result = self.inner.split(node_id).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok((
            result.original_branch,
            result.new_branch,
            result.split_point,
            result.moved_nodes,
        ))
    }

    /// Merge a branch into another.
    ///
    /// Returns a tuple of (target_branch, merged_branch, merge_point).
    fn merge(&mut self, from_branch: u64, into_branch: u64) -> PyResult<(u64, u64, u64)> {
        let result = self.inner.merge(from_branch, into_branch)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok((result.target_branch, result.merged_branch, result.merge_point))
    }

    /// Traverse from current branch to another branch.
    fn traverse(&mut self, target_branch: u64) -> PyResult<()> {
        self.inner.traverse(target_branch)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Archive a branch (preserve but mark inactive).
    #[pyo3(signature = (branch_id, reason=None))]
    fn archive(&mut self, branch_id: u64, reason: Option<String>) -> PyResult<()> {
        self.inner.archive(branch_id, reason)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Get the current branch.
    fn current(&self) -> Option<PyBranch> {
        self.inner.current().map(|b| PyBranch { inner: b.clone() })
    }

    /// Get branch by ID.
    fn get_branch(&self, branch_id: u64) -> Option<PyBranch> {
        self.inner.get_branch(branch_id).map(|b| PyBranch { inner: b.clone() })
    }

    /// Get all active branches.
    fn active_branches(&self) -> Vec<PyBranch> {
        self.inner.active_branches()
            .map(|b| PyBranch { inner: b.clone() })
            .collect()
    }

    /// Get all branches (including inactive).
    fn all_branches(&self) -> Vec<PyBranch> {
        self.inner.all_branches()
            .map(|b| PyBranch { inner: b.clone() })
            .collect()
    }

    /// Get number of branches.
    #[getter]
    fn branch_count(&self) -> usize {
        self.inner.branch_count()
    }

    /// Find which branch contains a node.
    fn find_branch_for_node(&self, node_id: u64) -> Option<u64> {
        self.inner.find_branch_for_node(node_id)
    }

    /// Get child branches of a given branch.
    fn child_branches(&self, branch_id: u64) -> Vec<u64> {
        self.inner.child_branches(branch_id)
    }

    fn __repr__(&self) -> String {
        format!(
            "BranchStateMachine(branches={}, current={:?})",
            self.inner.branch_count(),
            self.inner.current().map(|b| b.id)
        )
    }
}

/// Branch resolver for finding and recovering lost branches.
///
/// This is a static utility class that operates on BranchStateMachine instances.
/// The Rust BranchResolver uses borrowed references, so we expose static methods.
#[pyclass(name = "BranchResolver")]
pub struct PyBranchResolver;

#[pymethods]
impl PyBranchResolver {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Find all branches that could be recovered from the state machine.
    ///
    /// This analyzes the DAG to find paths that aren't actively tracked.
    #[staticmethod]
    fn find_recoverable_branches(machine: &PyBranchStateMachine) -> Vec<PyRecoverableBranch> {
        let resolver = RustBranchResolver::new(&machine.inner);
        resolver.find_recoverable_branches()
            .into_iter()
            .map(|r| PyRecoverableBranch { inner: r })
            .collect()
    }

    /// Find branches created by regeneration that weren't selected.
    #[staticmethod]
    fn find_unselected_regenerations(machine: &PyBranchStateMachine) -> Vec<PyRecoverableBranch> {
        let resolver = RustBranchResolver::new(&machine.inner);
        resolver.find_unselected_regenerations()
            .into_iter()
            .map(|r| PyRecoverableBranch { inner: r })
            .collect()
    }

    /// Recover a lost branch using its suggested strategy.
    ///
    /// Returns the recovered branch ID.
    #[staticmethod]
    fn recover(machine: &mut PyBranchStateMachine, recoverable: &PyRecoverableBranch) -> PyResult<u64> {
        // Clone machine to create resolver (avoids borrow conflict)
        let machine_clone = machine.inner.clone();
        let resolver = RustBranchResolver::new(&machine_clone);
        resolver.recover(&mut machine.inner, &recoverable.inner)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        "BranchResolver()".to_string()
    }
}

// ============================================================================
// Chain Manager
// ============================================================================

/// Cross-chain link type.
#[pyclass(name = "CrossChainLinkType", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyCrossChainLinkType {
    Semantic,
    Reference,
    Continuation,
    Related,
    Alternative,
    KnowledgeTransfer,
}

impl From<RustCrossChainLinkType> for PyCrossChainLinkType {
    fn from(lt: RustCrossChainLinkType) -> Self {
        match lt {
            RustCrossChainLinkType::Semantic => PyCrossChainLinkType::Semantic,
            RustCrossChainLinkType::Reference => PyCrossChainLinkType::Reference,
            RustCrossChainLinkType::Continuation => PyCrossChainLinkType::Continuation,
            RustCrossChainLinkType::Related => PyCrossChainLinkType::Related,
            RustCrossChainLinkType::Alternative => PyCrossChainLinkType::Alternative,
            RustCrossChainLinkType::KnowledgeTransfer => PyCrossChainLinkType::KnowledgeTransfer,
        }
    }
}

impl From<PyCrossChainLinkType> for RustCrossChainLinkType {
    fn from(lt: PyCrossChainLinkType) -> Self {
        match lt {
            PyCrossChainLinkType::Semantic => RustCrossChainLinkType::Semantic,
            PyCrossChainLinkType::Reference => RustCrossChainLinkType::Reference,
            PyCrossChainLinkType::Continuation => RustCrossChainLinkType::Continuation,
            PyCrossChainLinkType::Related => RustCrossChainLinkType::Related,
            PyCrossChainLinkType::Alternative => RustCrossChainLinkType::Alternative,
            PyCrossChainLinkType::KnowledgeTransfer => RustCrossChainLinkType::KnowledgeTransfer,
        }
    }
}

/// Link strength with semantic, temporal, and thematic components.
#[pyclass(name = "LinkStrength")]
#[derive(Clone)]
pub struct PyLinkStrength {
    inner: RustLinkStrength,
}

#[pymethods]
impl PyLinkStrength {
    #[new]
    fn new(semantic: f32, temporal: f32, thematic: f32) -> Self {
        Self {
            inner: RustLinkStrength::new(semantic, temporal, thematic),
        }
    }

    /// Create link strength from semantic similarity only.
    #[staticmethod]
    fn from_semantic(similarity: f32) -> Self {
        Self {
            inner: RustLinkStrength::from_semantic(similarity),
        }
    }

    #[getter]
    fn semantic(&self) -> f32 {
        self.inner.semantic
    }

    #[getter]
    fn temporal(&self) -> f32 {
        self.inner.temporal
    }

    #[getter]
    fn thematic(&self) -> f32 {
        self.inner.thematic
    }

    #[getter]
    fn combined(&self) -> f32 {
        self.inner.combined
    }

    /// Check if this is a strong link (above threshold).
    fn is_strong(&self, threshold: f32) -> bool {
        self.inner.is_strong(threshold)
    }

    fn __repr__(&self) -> String {
        format!(
            "LinkStrength(semantic={:.2}, temporal={:.2}, thematic={:.2}, combined={:.2})",
            self.inner.semantic, self.inner.temporal, self.inner.thematic, self.inner.combined
        )
    }
}

/// Chain metadata.
#[pyclass(name = "ChainMetadata")]
#[derive(Clone)]
pub struct PyChainMetadata {
    inner: RustChainMetadata,
}

#[pymethods]
impl PyChainMetadata {
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    #[getter]
    fn title(&self) -> Option<&str> {
        self.inner.title.as_deref()
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.inner.node_count
    }

    #[getter]
    fn branch_count(&self) -> usize {
        self.inner.branch_count
    }

    #[getter]
    fn is_active(&self) -> bool {
        self.inner.is_active
    }

    #[getter]
    fn tags(&self) -> Vec<String> {
        self.inner.tags.clone()
    }

    #[getter]
    fn created_at(&self) -> i64 {
        self.inner.created_at
    }

    #[getter]
    fn last_accessed_at(&self) -> i64 {
        self.inner.last_accessed_at
    }

    fn __repr__(&self) -> String {
        format!(
            "ChainMetadata(id='{}', nodes={}, branches={})",
            self.inner.id, self.inner.node_count, self.inner.branch_count
        )
    }
}

/// Cross-chain link between nodes in different chains.
#[pyclass(name = "CrossChainLink")]
#[derive(Clone)]
pub struct PyCrossChainLink {
    inner: RustCrossChainLink,
}

#[pymethods]
impl PyCrossChainLink {
    /// Create a new cross-chain link.
    #[new]
    fn new(
        source_chain: String,
        source_node: u64,
        target_chain: String,
        target_node: u64,
        strength: &PyLinkStrength,
        link_type: PyCrossChainLinkType,
    ) -> Self {
        Self {
            inner: RustCrossChainLink::new(
                source_chain,
                source_node,
                target_chain,
                target_node,
                strength.inner.clone(),
                link_type.into(),
            ),
        }
    }

    /// Create a semantic link (convenience method).
    #[staticmethod]
    fn semantic(
        source_chain: String,
        source_node: u64,
        target_chain: String,
        target_node: u64,
        similarity: f32,
    ) -> Self {
        Self {
            inner: RustCrossChainLink::semantic(
                source_chain,
                source_node,
                target_chain,
                target_node,
                similarity,
            ),
        }
    }

    #[getter]
    fn source_chain(&self) -> &str {
        &self.inner.source_chain
    }

    #[getter]
    fn source_node(&self) -> u64 {
        self.inner.source_node
    }

    #[getter]
    fn target_chain(&self) -> &str {
        &self.inner.target_chain
    }

    #[getter]
    fn target_node(&self) -> u64 {
        self.inner.target_node
    }

    #[getter]
    fn link_type(&self) -> PyCrossChainLinkType {
        self.inner.link_type.into()
    }

    #[getter]
    fn strength(&self) -> PyLinkStrength {
        PyLinkStrength { inner: self.inner.strength.clone() }
    }

    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }

    /// Check if this link is bidirectional.
    fn is_bidirectional(&self) -> bool {
        self.inner.is_bidirectional()
    }

    fn __repr__(&self) -> String {
        format!(
            "CrossChainLink('{}':{}  -> '{}':{}, type={:?})",
            self.inner.source_chain, self.inner.source_node,
            self.inner.target_chain, self.inner.target_node,
            self.inner.link_type
        )
    }
}

/// Chain manager for multi-conversation tracking.
#[pyclass(name = "ChainManager")]
pub struct PyChainManager {
    inner: RustChainManager,
}

#[pymethods]
impl PyChainManager {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustChainManager::new(),
        }
    }

    /// Create a new chain with optional ID.
    ///
    /// If chain_id is None, a unique ID is generated.
    /// Returns the chain ID.
    #[pyo3(signature = (chain_id=None))]
    fn create_chain(&mut self, chain_id: Option<String>) -> PyResult<String> {
        self.inner.create_chain(chain_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Create a chain from an existing trajectory graph.
    #[pyo3(signature = (graph, chain_id=None))]
    fn create_chain_from_graph(&mut self, graph: &PyTrajectoryGraph, chain_id: Option<String>) -> PyResult<String> {
        self.inner.create_chain_from_graph(chain_id, graph.inner.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Get a chain's state machine by ID.
    fn get_chain(&self, chain_id: &str) -> Option<PyBranchStateMachine> {
        self.inner.get_chain(&chain_id.to_string())
            .map(|m| PyBranchStateMachine { inner: m.clone() })
    }

    /// Get chain metadata by ID.
    fn get_metadata(&self, chain_id: &str) -> Option<PyChainMetadata> {
        self.inner.get_metadata(&chain_id.to_string())
            .map(|m| PyChainMetadata { inner: m.clone() })
    }

    /// Check if a chain exists.
    fn contains(&self, chain_id: &str) -> bool {
        self.inner.contains(&chain_id.to_string())
    }

    /// Get all chain IDs.
    fn chain_ids(&self) -> Vec<String> {
        self.inner.chain_ids().map(|s| s.clone()).collect()
    }

    /// Get number of chains.
    #[getter]
    fn chain_count(&self) -> usize {
        self.inner.chain_count()
    }

    /// Get the currently active chain ID.
    #[getter]
    fn active_chain(&self) -> Option<String> {
        self.inner.active_chain().cloned()
    }

    /// Set the active chain.
    fn set_active_chain(&mut self, chain_id: &str) -> PyResult<()> {
        self.inner.set_active_chain(chain_id.to_string())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Get the active chain's state machine.
    fn active_chain_machine(&self) -> Option<PyBranchStateMachine> {
        self.inner.get_active_chain()
            .map(|m| PyBranchStateMachine { inner: m.clone() })
    }

    /// Delete a chain.
    fn delete_chain(&mut self, chain_id: &str) -> bool {
        self.inner.delete_chain(&chain_id.to_string())
    }

    /// Split a chain at a node, creating a new chain.
    fn split_chain(&mut self, chain_id: &str, node_id: u64) -> PyResult<String> {
        self.inner.split_chain(&chain_id.to_string(), node_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Merge two chains (from is merged into into, then deleted).
    fn merge_chains(&mut self, from_chain: &str, into_chain: &str) -> PyResult<()> {
        self.inner.merge_chains(&from_chain.to_string(), &into_chain.to_string())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Deactivate a chain.
    fn deactivate_chain(&mut self, chain_id: &str) -> PyResult<()> {
        self.inner.deactivate_chain(&chain_id.to_string())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Reactivate a chain.
    fn reactivate_chain(&mut self, chain_id: &str) -> PyResult<()> {
        self.inner.reactivate_chain(&chain_id.to_string())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Get statistics about all managed chains.
    fn stats(&self) -> HashMap<String, usize> {
        let s = self.inner.stats();
        let mut map = HashMap::new();
        map.insert("total_chains".to_string(), s.total_chains);
        map.insert("active_chains".to_string(), s.active_chains);
        map.insert("total_nodes".to_string(), s.total_nodes);
        map.insert("total_branches".to_string(), s.total_branches);
        map
    }

    /// Find chains by tag.
    fn find_by_tag(&self, tag: &str) -> Vec<String> {
        self.inner.find_by_tag(tag).into_iter().cloned().collect()
    }

    /// Find chains by title (partial match).
    fn find_by_title(&self, query: &str) -> Vec<String> {
        self.inner.find_by_title(query).into_iter().cloned().collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChainManager(chains={}, active={:?})",
            self.inner.chain_count(),
            self.inner.active_chain()
        )
    }
}

/// Detect knowledge transfer patterns in text.
/// Returns the matched pattern if found, None otherwise.
#[pyfunction]
fn detect_knowledge_transfer(text: &str) -> Option<String> {
    rust_detect_knowledge_transfer(text).map(|s| s.to_string())
}

// ============================================================================
// Coordinate Tree with LCA
// ============================================================================

/// Coordinate tree with O(log n) LCA queries via binary lifting.
#[pyclass(name = "CoordinateTree")]
pub struct PyCoordinateTree {
    inner: RustCoordinateTree,
}

#[pymethods]
impl PyCoordinateTree {
    #[new]
    #[pyo3(signature = (capacity=16))]
    fn new(capacity: usize) -> Self {
        Self {
            inner: RustCoordinateTree::new(capacity),
        }
    }

    /// Add a node to the tree.
    #[pyo3(signature = (id, parent, coordinate))]
    fn add_node(
        &mut self,
        id: usize,
        parent: Option<usize>,
        coordinate: &PyTrajectoryCoordinate5D,
    ) -> bool {
        self.inner.add_node(id, parent, coordinate.inner)
    }

    /// Preprocess the tree for LCA queries (must call after adding nodes).
    fn preprocess(&mut self) {
        self.inner.preprocess();
    }

    /// Get the root node ID.
    #[getter]
    fn root(&self) -> Option<usize> {
        self.inner.root()
    }

    /// Number of nodes in tree.
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if tree is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get depth of a node.
    fn depth(&self, id: usize) -> Option<usize> {
        self.inner.depth(id)
    }

    /// Maximum depth in tree.
    fn max_depth(&self) -> usize {
        self.inner.max_depth()
    }

    /// Compute Lowest Common Ancestor of two nodes.
    fn lca(&self, a: usize, b: usize) -> Option<usize> {
        self.inner.lca(a, b)
    }

    /// Compute path distance between two nodes via LCA.
    fn path_distance(&self, a: usize, b: usize) -> Option<usize> {
        self.inner.path_distance(a, b)
    }

    /// Get k-th ancestor of a node.
    fn kth_ancestor(&self, node: usize, k: usize) -> Option<usize> {
        self.inner.kth_ancestor(node, k)
    }

    /// Check if node a is ancestor of node b.
    fn is_ancestor(&self, a: usize, b: usize) -> bool {
        self.inner.is_ancestor(a, b)
    }

    /// Check if node a is descendant of node b.
    fn is_descendant(&self, a: usize, b: usize) -> bool {
        self.inner.is_descendant(a, b)
    }

    /// Get path from node a to node b.
    fn path(&self, a: usize, b: usize) -> Option<Vec<usize>> {
        self.inner.path(a, b)
    }

    /// Get all nodes at specified depth.
    fn nodes_at_depth(&self, depth: usize) -> Vec<usize> {
        self.inner.nodes_at_depth(depth)
    }

    /// Get all leaf nodes.
    fn leaves(&self) -> Vec<usize> {
        self.inner.leaves()
    }

    /// Get all node IDs in preorder.
    fn preorder(&self) -> Vec<usize> {
        self.inner.preorder()
    }

    fn __repr__(&self) -> String {
        format!(
            "CoordinateTree(nodes={}, max_depth={})",
            self.inner.len(),
            self.inner.max_depth()
        )
    }
}

// ============================================================================
// Trajectory-Weighted Distance Functions
// ============================================================================

/// Compute trajectory-weighted cosine similarity (4D coordinates).
///
/// Blends semantic similarity (cosine) with spatial similarity (coordinate distance).
///
/// Args:
///     a: First embedding vector
///     b: Second embedding vector
///     coord_a: Trajectory coordinate of first episode
///     coord_b: Trajectory coordinate of second episode
///     coord_weight: Weight for coordinate component [0, 1]
///         - 0.0 = pure semantic similarity
///         - 0.5 = balanced
///         - 1.0 = pure spatial similarity
#[pyfunction]
fn trajectory_weighted_cosine(
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
    coord_a: &PyTrajectoryCoordinate,
    coord_b: &PyTrajectoryCoordinate,
    coord_weight: f32,
) -> PyResult<f32> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;

    if a_slice.len() != b_slice.len() {
        return Err(PyValueError::new_err("Vectors must have same dimension"));
    }

    Ok(rust_trajectory_weighted_cosine(
        a_slice,
        b_slice,
        &coord_a.inner,
        &coord_b.inner,
        coord_weight,
    ))
}

/// Compute trajectory-weighted cosine similarity (5D coordinates with complexity).
#[pyfunction]
fn trajectory_weighted_cosine_5d(
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
    coord_a: &PyTrajectoryCoordinate5D,
    coord_b: &PyTrajectoryCoordinate5D,
    coord_weight: f32,
) -> PyResult<f32> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;

    if a_slice.len() != b_slice.len() {
        return Err(PyValueError::new_err("Vectors must have same dimension"));
    }

    Ok(rust_trajectory_weighted_cosine_5d(
        a_slice,
        b_slice,
        &coord_a.inner,
        &coord_b.inner,
        coord_weight,
    ))
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Compute cosine similarity between two vectors.
#[pyfunction]
fn cosine_similarity(a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> PyResult<f32> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;

    if a_slice.len() != b_slice.len() {
        return Err(PyValueError::new_err("Vectors must have same dimension"));
    }

    let dot: f32 = a_slice.iter().zip(b_slice.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a_slice.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b_slice.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        Ok(0.0)
    } else {
        Ok(dot / (norm_a * norm_b))
    }
}

/// Compute Euclidean distance between two vectors.
#[pyfunction]
fn euclidean_distance(a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> PyResult<f32> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;

    if a_slice.len() != b_slice.len() {
        return Err(PyValueError::new_err("Vectors must have same dimension"));
    }

    let sum: f32 = a_slice
        .iter()
        .zip(b_slice.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();

    Ok(sum.sqrt())
}

/// Compute trajectory coordinates for a new turn.
///
/// Standalone function for computing 5D trajectory coordinates from context.
/// This is a convenience wrapper around IRCPPropagator.compute_trajectory_coordinates.
///
/// Args:
///     embedding: The embedding vector for the new turn.
///     neighbor_similarities: List of similarity scores to neighbor turns.
///     session_depth: Current depth within the session.
///     turn_index: Index of this turn within its conversation.
///     sibling_count: Number of sibling branches at this depth.
///     sibling_order: Order among siblings (0-indexed).
///     project_phase: Project phase string ("exploration", "planning", "implementation", etc.).
///
/// Returns:
///     PyTrajectoryCoordinate5D with computed coordinates.
#[pyfunction]
#[pyo3(signature = (embedding, neighbor_similarities, session_depth=0, turn_index=0, sibling_count=1, sibling_order=0, project_phase="implementation"))]
#[allow(unused_variables)]
fn compute_trajectory_coordinates(
    embedding: PyReadonlyArray1<f32>,
    neighbor_similarities: Vec<f32>,
    session_depth: i32,
    turn_index: i32,
    sibling_count: i32,
    sibling_order: i32,
    project_phase: &str,
) -> PyResult<PyTrajectoryCoordinate5D> {
    let emb_slice = embedding.as_slice()?;

    // Compute depth (raw value as u32)
    let depth = session_depth.max(0) as u32;

    // Compute sibling order (raw value as u32)
    let sibling_order_u32 = sibling_order.max(0) as u32;

    // Compute homogeneity based on neighbor similarities
    let homogeneity = if !neighbor_similarities.is_empty() {
        let sum: f32 = neighbor_similarities.iter().sum();
        (sum / neighbor_similarities.len() as f32).clamp(0.0, 1.0)
    } else {
        0.5 // Default homogeneity when no neighbors
    };

    // Compute temporal: normalized turn index
    let max_turns = 100.0_f32;
    let temporal = (turn_index as f32 / max_turns).min(1.0);

    // Compute complexity based on embedding variance
    let mean: f32 = emb_slice.iter().sum::<f32>() / emb_slice.len() as f32;
    let variance: f32 = emb_slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / emb_slice.len() as f32;
    let complexity = (variance.sqrt() * 10.0).round().max(1.0) as u32;

    Ok(PyTrajectoryCoordinate5D {
        inner: RustTrajectoryCoordinate5D::new(depth, sibling_order_u32, homogeneity, temporal, complexity),
    })
}

/// Normalize a vector to unit length.
#[pyfunction]
fn normalize<'py>(py: Python<'py>, v: PyReadonlyArray1<f32>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let slice = v.as_slice()?;
    let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm == 0.0 {
        return Err(PyValueError::new_err("Cannot normalize zero vector"));
    }

    let normalized: Vec<f32> = slice.iter().map(|x| x / norm).collect();
    Ok(normalized.to_pyarray(py))
}

// ============================================================================
// Module Definition
// ============================================================================

/// RAG++ Python module
#[pymodule]
fn _rag_plusplus_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core types
    m.add_class::<PyOutcomeStats>()?;
    m.add_class::<PyMemoryRecord>()?;
    m.add_class::<PySearchResult>()?;

    // Distance types
    m.add_class::<PyDistanceType>()?;

    // Indexes
    m.add_class::<PyFlatIndex>()?;
    m.add_class::<PyHNSWIndex>()?;
    m.add_class::<PyIndexRegistry>()?;

    // Storage
    m.add_class::<PyInMemoryStore>()?;

    // Cache
    m.add_class::<PyQueryCache>()?;
    m.add_class::<PyCacheStats>()?;

    // Trajectory module (IRCP/RCP/TPO)
    m.add_class::<PyTrajectoryCoordinate>()?;
    m.add_class::<PyTrajectoryCoordinate5D>()?;
    m.add_class::<PyTrajectoryPhase>()?;
    m.add_class::<PyFeedback>()?;
    m.add_class::<PyTurnFeatures>()?;
    m.add_class::<PyPhaseInferencer>()?;
    m.add_class::<PySalienceFactors>()?;
    m.add_class::<PyTurnSalience>()?;
    m.add_class::<PySalienceScorer>()?;
    m.add_class::<PyPathQualityWeights>()?;
    m.add_class::<PyPathQualityFactors>()?;
    m.add_class::<PyPathQuality>()?;

    // DLM types
    m.add_class::<PyDLMWeights>()?;

    // I-RCP types
    m.add_class::<PyIRCPConfig>()?;
    m.add_class::<PyAttentionWeights>()?;
    m.add_class::<PyIRCPPropagator>()?;

    // ChainLink types
    m.add_class::<PyLinkType>()?;
    m.add_class::<PyChainLink>()?;
    m.add_class::<PyChainLinkEstimatorConfig>()?;
    m.add_class::<PyChainLinkEstimate>()?;
    m.add_class::<PyChainLinkEstimator>()?;

    // Coordinate tree
    m.add_class::<PyCoordinateTree>()?;

    // Ring structures
    m.add_class::<PyRing>()?;
    m.add_class::<PyDualRing>()?;

    // Trajectory graph
    m.add_class::<PyEdgeType>()?;
    m.add_class::<PyEpisode>()?;
    m.add_class::<PyTrajectoryGraph>()?;

    // Conservation metrics
    m.add_class::<PyConservationMetrics>()?;

    // Branch state machine
    m.add_class::<PyBranchStatus>()?;
    m.add_class::<PyRecoveryStrategy>()?;
    m.add_class::<PyLostReason>()?;
    m.add_class::<PyBranch>()?;
    m.add_class::<PyRecoverableBranch>()?;
    m.add_class::<PyBranchStateMachine>()?;
    m.add_class::<PyBranchResolver>()?;

    // Chain manager
    m.add_class::<PyCrossChainLinkType>()?;
    m.add_class::<PyLinkStrength>()?;
    m.add_class::<PyChainMetadata>()?;
    m.add_class::<PyCrossChainLink>()?;
    m.add_class::<PyChainManager>()?;

    // Utility functions
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(euclidean_distance, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;

    // Trajectory-weighted distance functions
    m.add_function(wrap_pyfunction!(trajectory_weighted_cosine, m)?)?;
    m.add_function(wrap_pyfunction!(trajectory_weighted_cosine_5d, m)?)?;

    // Knowledge transfer detection
    m.add_function(wrap_pyfunction!(detect_knowledge_transfer, m)?)?;

    // Trajectory coordinate computation
    m.add_function(wrap_pyfunction!(compute_trajectory_coordinates, m)?)?;

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
