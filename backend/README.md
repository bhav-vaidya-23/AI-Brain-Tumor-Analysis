# Brain Tumor MRI Analysis Backend (FastAPI)

FastAPI inference service powering the web application for brain tumor classification, Grad-CAM explainability, and U-Net segmentation.

## Features
- **In-Memory Inference**: Zero disk retention of uploaded patient MRIs.
- **Pre-Loaded Models**: ResNet-18 and U-Net are loaded once during application startup.
- **REST Endpoints**:
  - `GET /api/health`: Health status and model readiness.
  - `POST /api/analyze`: Multipart upload returning classification probabilities, Grad-CAM overlays, and U-Net segmentation masks.
- **Strict Security & Privacy**: File type validation, 10MB payload limit, CORS protection.

## Running Locally

```bash
# From the project root
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Test the health check:
```bash
curl http://localhost:8000/api/health
```
