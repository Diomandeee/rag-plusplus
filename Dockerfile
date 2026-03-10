# =============================================================================
# RAG++ FastAPI Service Dockerfile
# =============================================================================
FROM python:3.11-slim

# Install build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install uvicorn for serving
RUN pip install --no-cache-dir uvicorn[standard]

# Copy code
COPY rag_plusplus ./rag_plusplus

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Service entry point
CMD ["python", "-m", "uvicorn", "rag_plusplus.service.app:app", "--host", "0.0.0.0", "--port", "8000"]

