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

# 7. Copy model checkpoints into /app/outputs
COPY outputs/best_classifier.pth /app/outputs/best_classifier.pth
COPY outputs/best_unet.pth /app/outputs/best_unet.pth

# 8. Expose default port
EXPOSE 8000

# 9. Start FastAPI with uvicorn expanding $PORT supplied by Railway runtime
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
