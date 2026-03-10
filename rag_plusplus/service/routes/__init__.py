"""Service routes package.

Exposes the Context Gateway router for federated context retrieval.
"""

from .context_gateway import router as gateway_router

__all__ = ["gateway_router"]
