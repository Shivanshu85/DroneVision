# syntax=docker/dockerfile:1
# Stage 1: Builder stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies needed for compiling package wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the requirements directory first to leverage Docker layer caching
COPY requirements/ requirements/
COPY requirements.txt .

# Upgrade pip and install packages, using BuildKit cache mounts to accelerate builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt

# Stage 2: Production runtime stage
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime dependencies for OpenCV and system execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Configure security with a non-privileged user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
RUN chown -R appuser:appgroup /app

USER appuser

# Copy application files (with correct user/group ownership)
COPY --chown=appuser:appgroup dronevision/ dronevision/
COPY --chown=appuser:appgroup demo/ demo/
COPY --chown=appuser:appgroup configs/ configs/
COPY --chown=appuser:appgroup runs/phase1/best.pth runs/phase1/best.pth
COPY --chown=appuser:appgroup app.py .
COPY --chown=appuser:appgroup pyproject.toml .
COPY --chown=appuser:appgroup VERSION .

# Set environment variables for Gradio inside the container
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

EXPOSE 7860

CMD ["python", "app.py"]
