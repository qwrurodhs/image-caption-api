# --- Stage 1: Build the dependencies ---
FROM python:3.12-slim-bookworm AS builder

# Install system packages required for compiling Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install core dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch (CPU-only to reduce image size)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# --- Stage 2: Final runtime image ---
FROM python:3.12-slim-bookworm

# Install curl, which is required for the HEALTHCHECK command
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Configure environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    LOG_LEVEL=INFO \
    MODEL_NAME=nlpconnect/vit-gpt2-image-captioning \
    CACHE_MAXSIZE=512 \
    HTTP_TIMEOUT=10 \
    MAX_IMAGE_SIZE=10485760 \
    MAX_URLS_PER_REQUEST=20 \
    DEVICE=cpu \
    HTTP_USER_AGENT=ImageCaptionBot/1.0 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface

# Create a non-root user and group
RUN groupadd --system --gid 1001 appuser && \
    useradd --system --uid 1001 --gid 1001 appuser

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy application code and set ownership
COPY --chown=appuser:appuser app.py /app/
# Uncomment the line below if you have a .env.example file to include
# COPY --chown=appuser:appuser .env.example /app/.env.example

# Create the Hugging Face cache directory and set ownership
RUN mkdir -p $HF_HOME && \
    chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

EXPOSE 8000

# Configure the health check to monitor the application's status
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]