# ==============================================================================
# SalesSight AI - Production Multi-Stage Dockerfile
# Stage 1: Build & Dependency Resolution | Stage 2: Minimal Distroless/Slim Runtime
# ==============================================================================

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build

# Install compilation prerequisites and OpenMP support for LightGBM/XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production Slim Runner
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime shared libraries and curl for container health probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create non-root application security user
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

# Copy application source code, models, data, and configs
COPY --chown=appuser:appgroup . .

# Set up logs directory permissions
RUN mkdir -p logs && chown -R appuser:appgroup logs

# Switch to unprivileged security user
USER appuser

# Expose default ports (8000 for FastAPI REST backend, 8501 for Streamlit frontend)
EXPOSE 8000 8501

# Healthcheck probe querying FastAPI /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default container entrypoint: Launch FastAPI REST backend
# Override in docker-compose.yml or Render for Streamlit: streamlit run app.py
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
