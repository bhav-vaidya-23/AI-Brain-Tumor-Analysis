# Production Deployment & Hosting Guide

This guide documents the full setup and deployment procedure for the **AI-Based Brain Tumor Classification, Segmentation, and Explainable MRI Analysis** web application.

---

## 1. Architecture Overview

```
                          +-------------------------+
                          |   User Web Browser      |
                          +-------------------------+
                                       |
                                       | HTTPS (React UI)
                                       v
                          +-------------------------+
                          |   Vercel (Frontend)     |
                          |   React + Vite + Tailwind|
                          +-------------------------+
                                       |
                                       | REST API (multipart/form-data)
                                       v
                          +-------------------------+
                          |   Render (Backend)      |
                          |   FastAPI + Uvicorn     |
                          +-------------------------+
                                       |
                  +--------------------+--------------------+
                  |                                         |
                  v                                         v
       +----------------------+                  +----------------------+
       | ResNet-18 Classifier |                  | 4-Level U-Net        |
       | outputs/best_classifier.pth             | outputs/best_unet.pth|
       | (106.8 MB)           |                  | (355.3 MB, Epoch 23) |
       +----------------------+                  +----------------------+
                  |
                  v
       +----------------------+
       | Grad-CAM (layer4)    |
       +----------------------+
```

---

## 2. Local Development Setup

### A. Backend (FastAPI)
```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Run local FastAPI dev server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Base URL: `http://localhost:8000`
- Health Check: `http://localhost:8000/api/health`
- Interactive API Docs: `http://localhost:8000/docs`

### B. Frontend (React + Vite)
```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Start local development server
npm run dev
```
- Frontend UI: `http://localhost:5173`

---

## 3. Environment Variables

### Backend (`backend/.env`)
| Variable | Description | Local Default | Production Example (Render) |
| :--- | :--- | :--- | :--- |
| `FRONTEND_URL` | Allowed origin(s) for CORS | `http://localhost:5173` | `https://brain-mri-ai.vercel.app` |
| `PORT` | Server listening port | `8000` | Render assigns `$PORT` automatically |

### Frontend (`frontend/.env`)
| Variable | Description | Local Default | Production Example (Vercel) |
| :--- | :--- | :--- | :--- |
| `VITE_API_BASE_URL` | Base URL of the FastAPI backend | `http://localhost:8000` | `https://brain-tumor-mri-api.onrender.com` |

---

## 4. Model Checkpoint Storage & Deployment Strategy

The application requires two trained checkpoints in `outputs/`:
- `outputs/best_classifier.pth`: **106.8 MB** (ResNet-18)
- `outputs/best_unet.pth`: **355.3 MB** (4-Level U-Net, Epoch 23)

### A. Local & Repository Tracking (Git LFS)
In the Git repository, these files are tracked using **Git Large File Storage (Git LFS)** via `.gitattributes`:
```bash
outputs/*.pth filter=lfs diff=lfs merge=lfs -text
```

### B. Deployment Asset Strategy (GitHub Releases)
Cloud platforms such as Railway clone Git repositories without LFS object storage enabled, resulting in small ~130-byte Git LFS pointer text files (`version https://git-lfs.github.com/spec/v1...`). When PyTorch attempts to load these text stubs, it fails with:
```
[FastAPI ERROR] Failed to load models on startup: invalid load key, 'v'.
```

To resolve this reliably in production, model weights are hosted as **GitHub Release Assets** and automatically downloaded/verified inside Docker at build time:

#### Exact Release Asset URLs:
| Model | Release Asset URL |
| :--- | :--- |
| **ResNet-18 Classifier** | `https://github.com/bhav-vaidya-23/AI-Brain-Tumor-Analysis/releases/download/v1.0.0/best_classifier.pth` |
| **U-Net Segmentation** | `https://github.com/bhav-vaidya-23/AI-Brain-Tumor-Analysis/releases/download/v1.0.0/best_unet.pth` |

#### How to Set Up the GitHub Release:
1. Navigate to the GitHub repository: `https://github.com/bhav-vaidya-23/AI-Brain-Tumor-Analysis`
2. Click **Releases > Draft a new release** (or go to `releases/new`).
3. Set **Choose a tag**: `v1.0.0` (create tag on publish).
4. Set **Release title**: `v1.0.0 - Production Model Checkpoints`.
5. Attach the two checkpoint binaries from your local `outputs/` folder:
   - `outputs/best_classifier.pth`
   - `outputs/best_unet.pth`
6. Click **Publish release**.

#### Automated Docker Build Verification:
During the Docker build on Railway, `backend/download_checkpoints.py`:
1. Fetches both model binaries from the exact GitHub Release URLs.
2. Checks that neither file is a Git LFS pointer text stub.
3. Validates archive magic bytes and tests PyTorch structure with `torch.load`.
4. Saves them to `/app/outputs/best_classifier.pth` and `/app/outputs/best_unet.pth`.
5. Fails the build immediately with actionable instructions if the download or verification fails.

---

## 5. Hugging Face Spaces Deployment (Backend - Recommended Free Tier: 16 GB RAM)

Hugging Face Spaces provides **2 vCPU and 16 GB RAM** on its free community tier, easily accommodating the exact ResNet-18 + Epoch-23 U-Net models (~800 MB footprint):

1. Sign in to [Hugging Face](https://huggingface.co) and click **New Space**.
2. Set Space properties:
   - **Space name**: `brain-tumor-mri-api` (or preferred name).
   - **License**: `mit`.
   - **Space SDK**: **Docker** (Blank).
   - **Space hardware**: `CPU Basic • 2 vCPU • 16GB RAM` (**Free**).
   - **Visibility**: `Public`.
3. Connect your GitHub repository (`AI-Brain-Tumor-Analysis`, branch `deployment-free`) or push via git to the Hugging Face Space remote.
4. Hugging Face automatically reads `README.md` (`app_port: 7860`) and builds `Dockerfile`.
5. Under Space **Settings > Variables and secrets**, add:
   - `FRONTEND_URL`: `https://your-frontend.vercel.app` (or `http://localhost:5173` for local frontend testing).
6. Your live public API URL will be:
   `https://<username>-brain-tumor-mri-api.hf.space`
   - Healthcheck: `https://<username>-brain-tumor-mri-api.hf.space/api/health`
   - Interactive Docs: `https://<username>-brain-tumor-mri-api.hf.space/docs`

---

## 6. Render Deployment (Backend - Alternative)

1. Sign in to [Render](https://render.com) and click **New + > Web Service**.
2. Connect your GitHub repository.
3. Configure the service settings:
   - **Name**: `brain-tumor-mri-api`
   - **Runtime**: `Python 3`
   - **Region**: `Oregon (US West)` or preferred region.
   - **Build Command**:
     ```bash
     pip install -r backend/requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn backend.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type**: `Standard` (Recommended: At least 2GB RAM to hold the PyTorch models in memory).
4. In **Environment Variables**, add:
   - `PYTHON_VERSION`: `3.11.9`
   - `FRONTEND_URL`: `https://your-frontend.vercel.app` (replace with your Vercel URL once deployed).
5. In **Health Check Path**, set:
   - `/api/health`
6. Click **Deploy Web Service**. Render will build and expose your backend at `https://brain-tumor-mri-api.onrender.com`.

---

## 7. Vercel Deployment (Frontend)

1. Sign in to [Vercel](https://vercel.com) and click **Add New > Project**.
2. Select your repository.
3. In **Project Settings**:
   - **Root Directory**: `frontend`
   - **Framework Preset**: `Vite`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. In **Environment Variables**, add:
   - `VITE_API_BASE_URL`: `https://brain-tumor-mri-api.onrender.com` (your Render backend URL).
5. Click **Deploy**. Vercel will build and host the React frontend on HTTPS.

---

## 7. Connecting Frontend & Backend (CORS Alignment)

Once both services are deployed:
1. Copy your live Vercel frontend URL (e.g. `https://brain-mri-app.vercel.app`).
2. Go to **Render Dashboard > Web Service > Environment**.
3. Update `FRONTEND_URL` to `https://brain-mri-app.vercel.app`.
4. Render will automatically reload the application with the updated CORS policy.

---

## 8. Privacy & Security Implementation

1. **Zero-Retention Processing**: Uploaded MRI files are read directly into memory (`BytesIO`), processed, and discarded upon response completion. No files are written to disk.
2. **Safe Image Decoding**: Input bytes are validated and sanitized via Pillow with strict format validation (`.jpg`, `.jpeg`, `.png`).
3. **Payload Protection**: File sizes exceeding 10MB are rejected with `HTTP 413 Payload Too Large`.
4. **No External APIs**: All inference executes locally on the server using PyTorch. No patient data is transmitted to third-party AI APIs.

---

## 9. Medical & Research Disclaimer

This application is developed as an **AI-assisted research prototype** based on the BRISC2025 brain tumor MRI dataset. It is **not** a clinically certified diagnostic device. 

All user-facing interfaces and backend responses strictly maintain non-diagnostic medical terminology (`AI-predicted category`, `Model confidence`, `Predicted tumor region`) and explicitly instruct users to seek qualified clinical consultation for any diagnostic or treatment planning.
