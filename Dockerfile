# Dockerfile for RAG System
# Multi-stage build for smaller image size

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    libxml2-dev \
    libxslt1-dev \
    antiword \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    curl \
    libgl1 \
    libglx0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY rag-admin.py rag-user.py ./

# Create directories for volumes and uploads
RUN mkdir -p /tmp/uploads

# Expose Streamlit port
EXPOSE 8501 8502

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command (can be overridden in docker-compose)
CMD ["streamlit", "run", "rag-admin.py", "--server.address=0.0.0.0"]
