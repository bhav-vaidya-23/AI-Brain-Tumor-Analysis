# 1. Use Python 3.11 slim base image for consistent deployment
FROM python:3.11-slim

# 2. Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# 3. Set working directory to /app
WORKDIR /app

# 4. Install essential build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# 6. Copy application packages and source code
COPY backend /app/backend
COPY src /app/src

# 7. Model checkpoints setup:
# Railway clones Git LFS pointer stubs by default, causing PyTorch "invalid load key, 'v'".
# Download authentic PyTorch binary checkpoints directly from GitHub Release assets into /app/outputs during build.
ARG CLASSIFIER_URL="https://github.com/bhav-vaidya-23/AI-Brain-Tumor-Analysis/releases/download/v1.0.0/best_classifier.pth"
ARG UNET_URL="https://github.com/bhav-vaidya-23/AI-Brain-Tumor-Analysis/releases/download/v1.0.0/best_unet.pth"

RUN python /app/backend/download_checkpoints.py \
    --classifier-url "${CLASSIFIER_URL}" \
    --unet-url "${UNET_URL}" \
    --dest-dir /app/outputs

# 8. Expose default port
EXPOSE 8000

# 9. Start FastAPI with uvicorn expanding $PORT supplied by Railway runtime
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
