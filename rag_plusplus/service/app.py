"""FastAPI application for RAG++ Context Gateway.

Standalone service that federates context retrieval across:
- Supabase pgvector (semantic search)
- Graph Kernel (entity traversal)
- Gemini (embedding generation)

Usage:
    python -m rag_plusplus.service.app --host 0.0.0.0 --port 8000

    # Or with uvicorn
    uvicorn rag_plusplus.service.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(
    title: str = "RAG++ Context Gateway",
    version: str = "2.0.0",
    debug: bool = False,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=title,
        version=version,
        description=(
            "Context federation service that unifies pgvector search, "
            "Graph Kernel traversal, and Gemini embeddings behind a single endpoint. "
            "Includes echo suppression, session Bloom filters, and vector expansion."
        ),
        debug=debug,
    )

    # CORS
    origins = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ]
    extra_origins = os.getenv("CORS_ORIGINS", "").split(",")
    origins.extend([o.strip() for o in extra_origins if o.strip()])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    register_routes(app)

    return app


def register_routes(app: FastAPI):
    """Register API routes."""
    from .routes.context_gateway import router as gateway_router
    from .routes.context_gateway import export_prometheus_metrics as gw_metrics

    # Context Gateway
    app.include_router(
        gateway_router,
        prefix="/api/rag",
        tags=["context-gateway"],
    )

    @app.get("/health", tags=["system"])
    async def health_check():
        return {"status": "healthy"}

    @app.get("/metrics", tags=["system"])
    async def prometheus_metrics():
        return PlainTextResponse(
            content=gw_metrics(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/metrics/json", tags=["system"])
    async def metrics_json():
        from .routes.context_gateway import _METRICS
        return dict(_METRICS)

    @app.get("/", tags=["system"])
    async def root():
        return {
            "name": "RAG++ Context Gateway",
            "version": "2.0.0",
            "docs": "/docs",
        }


# Default app instance for uvicorn
app = create_app()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="RAG++ Context Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        "rag_plusplus.service.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
