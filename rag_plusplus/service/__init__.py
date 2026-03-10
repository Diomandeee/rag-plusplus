"""RAG++ Service Layer.

Provides the FastAPI application for the Context Gateway.

Usage:
    python -m rag_plusplus.service.app --host 0.0.0.0 --port 8000

    # Or with uvicorn
    uvicorn rag_plusplus.service.app:app --host 0.0.0.0 --port 8000
"""

from .app import create_app

__all__ = ["create_app"]
