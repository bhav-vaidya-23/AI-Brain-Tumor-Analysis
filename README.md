# NeuroVision: AI-Based Brain Tumor Analysis

An AI-assisted medical imaging research platform for multi-task brain MRI analysis, combining multi-class classification, gradient-based explainability, and dense boundary segmentation in a unified pipeline.

---

## Architecture Overview

```
                        +----------------------------+
                        |   T1 Axial Brain MRI Scan  |
                        +----------------------------+
                                      |
                                      v
                        +----------------------------+
                        |  In-Memory Preprocessing   |
                        |     (256 x 256 Tensor)     |
                        +----------------------------+
                                      |
                +---------------------+---------------------+
                |                                           |
                v                                           v
  +---------------------------+               +---------------------------+
  |    ResNet-18 Classifier   |               |     4-Level U-Net         |
  |  (ImageNet Pretrained)    |               |  (31.0M Params, Ep. 23)   |
  +---------------------------+               +---------------------------+
                |                                           |
        +-------+-------+                                   |
        |               |                                   |
        v               v                                   v
  +-----------+   +-----------+                       +-----------+
  | Category  |   | Grad-CAM  |                       | Tumor     |
  | & Softmax |   | Saliency  |                       | Boundary  |
  | Probs     |   | (layer4)  |                       | Mask      |
  +-----------+   +-----------+                       +-----------+
```

### Core Components
- **Classification:** ResNet-18 fine-tuned on the BRISC2025 benchmark, predicting 4 classes: *Glioma*, *Meningioma*, *Pituitary*, and *No Tumor*.
- **Explainability:** Layer4 Gradient-weighted Class Activation Mapping (Grad-CAM) generating spatial visual attribution heatmaps and overlays.
- **Segmentation:** 4-level encoder-decoder U-Net model performing pixel-level tumor boundary delineation (decision threshold $\ge 0.50$).
- **Non-Lesional Safety Protocol:** Non-tumor scans (`is_non_lesional=true`) safely yield a 0.0% predicted area without generating false boundaries.

---

## Quick Start Guide

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### 1. Backend Setup (FastAPI)

```bash
# Install backend dependencies
pip install -r backend/requirements.txt

# Start the FastAPI server (runs on http://localhost:8000)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Verify the backend is running:
```bash
curl http://localhost:8000/api/health
```

### 2. Frontend Setup (React + Vite)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server (runs on http://localhost:5173)
npm run dev
```

Open `http://localhost:5173` in your browser to access the application.

---

## API Endpoints

### 1. Health Check
`GET /api/health`

Verifies backend status and model readiness.

```json
{
  "status": "healthy",
  "classifier_loaded": true,
  "segmentation_loaded": true,
  "explainability_loaded": true,
  "device": "cpu",
  "version": "1.0.0"
}
```

### 2. Multi-Modal Analysis
`POST /api/analyze`

Accepts an uploaded MRI slice via `multipart/form-data` (`file`: JPG, JPEG, PNG, max 10 MB).

Returns:
- `prediction`: Top category and confidence percentage.
- `probabilities`: Softmax distribution across all 4 categories.
- `segmentation`: Boundary indicator, area percentage, threshold used.
- `visualizations`: Base64 data URIs for Original MRI, Grad-CAM Heatmap, Grad-CAM Overlay, and Segmentation Overlay.
- `metadata`: Model architectures and input resolution.
- `disclaimer`: Verbatim research prototype disclaimer.

---

## Privacy & Security

- **Zero-Retention:** Scans are processed entirely in memory and immediately discarded. No patient data or medical images are stored on disk.
- **In-House Inference:** All deep learning inference executes locally or within dedicated container environments without routing medical images to third-party APIs.

---

## Medical & Research Disclaimer

> **Notice:** This application is an **AI-assisted research prototype** and is **not a clinically validated diagnostic system**. Its predictions and visualizations should not be used for diagnosis, treatment decisions, or surgical planning. Please consult a qualified medical professional for medical interpretation.
