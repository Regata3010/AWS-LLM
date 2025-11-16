# deployment/backend/Dockerfile
# Build from repo root: docker build -f deployment/backend/Dockerfile -t biasguard-backend ./backend

# ============================================
# Stage 1: Builder (Install dependencies)
# ============================================
FROM python:3.10-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    postgresql-dev \
    libffi-dev \
    openssl-dev \
    cargo \
    rust \
    make \
    cmake \
    openblas-dev \
    lapack-dev \
    gfortran

WORKDIR /build

# Copy requirements (build context is ./backend)
COPY requirements.txt .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# ============================================
# Stage 2: Runtime (Minimal production image)
# ============================================
FROM python:3.10-alpine

# Install only runtime dependencies
RUN apk add --no-cache \
    postgresql-libs \
    libstdc++ \
    libgomp \
    openblas \
    lapack

# Create non-root user
RUN addgroup -g 1000 biasguard && \
    adduser -D -u 1000 -G biasguard biasguard

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code (respects ./backend/.dockerignore)
COPY --chown=biasguard:biasguard . .

# Create MLflow tracking directory
RUN mkdir -p /app/mlruns && \
    mkdir -p /app/logsnewer && \
    mkdir -p /app/static/reportspdfs && \
    mkdir -p /app/charts && \
    chown -R biasguard:biasguard /app
# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8001

# Switch to non-root user
USER biasguard

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')" || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]

# Image size: ~400-500MB
# Memory usage: ~200-300MB at runtime