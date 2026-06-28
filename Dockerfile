# Stage 1: Build virtual environment
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY requirements/ requirements/
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Final runner stage
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime dependencies (OpenCV and system requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Create a non-root user and group
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Set up ownership for /app
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Copy application files
COPY --chown=appuser:appgroup dronevision/ dronevision/
COPY --chown=appuser:appgroup demo/ demo/
COPY --chown=appuser:appgroup configs/ configs/
COPY --chown=appuser:appgroup runs/phase1/best.pth runs/phase1/best.pth
COPY --chown=appuser:appgroup app.py .
COPY --chown=appuser:appgroup pyproject.toml .
COPY --chown=appuser:appgroup VERSION .

# Set environment variables for Gradio to run correctly inside container
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

EXPOSE 7860

CMD ["python", "app.py"]
