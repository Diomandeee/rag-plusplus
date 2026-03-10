"""RAG++: Context Gateway for Trajectory-Aware Retrieval.

A production-grade context federation service that unifies pgvector search,
Graph Kernel traversal, and Gemini embedding generation behind a single endpoint.

Features:
- Echo suppression with novelty scoring
- Session-level Bloom filter for duplicate detection
- Vector expansion via Gram-Schmidt orthogonalization
- Overlap classification (novel / moderate / high_echo)
- Mock mode for local development
- Prometheus metrics export
"""

from .version import __version__, __version_info__

__all__ = ["__version__", "__version_info__"]
