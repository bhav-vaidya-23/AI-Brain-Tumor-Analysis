import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.model_loader import ModelManager
from backend.inference import run_full_inference

# Maximum allowed upload size (10 MB)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager:
    Loads ResNet-18 Classifier, Grad-CAM hooks, and U-Net into memory once at server startup.
    """
    print("[FastAPI] Starting application and pre-loading models...", flush=True)
    manager = ModelManager.get_instance()
    try:
        manager.load_models()
        print("[FastAPI] Models pre-loaded successfully. Ready for inference.", flush=True)
    except Exception as e:
        print(f"[FastAPI ERROR] Failed to load models on startup: {str(e)}", file=sys.stderr, flush=True)
    yield
    print("[FastAPI] Shutting down application...", flush=True)


app = FastAPI(
    title="Brain Tumor MRI Analysis API",
    description="Production-grade AI inference API for brain tumor classification, Grad-CAM explainability, and U-Net segmentation.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
frontend_url_env = os.getenv("FRONTEND_URL", "http://localhost:5173")
# Parse multiple allowed origins if comma-separated
allowed_origins = [origin.strip() for origin in frontend_url_env.split(",") if origin.strip()]

# Include local defaults if in development
if "http://localhost:5173" not in allowed_origins:
    allowed_origins.append("http://localhost:5173")
if "http://127.0.0.1:5173" not in allowed_origins:
    allowed_origins.append("http://127.0.0.1:5173")
if "http://localhost:3000" not in allowed_origins:
    allowed_origins.append("http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Verifies that the server is online and required models are loaded in memory.
    """
    manager = ModelManager.get_instance()
    is_ready = bool(manager.is_loaded and manager.classifier is not None and manager.unet is not None)
    
    return {
        "status": "healthy" if is_ready else "degraded",
        "classifier_loaded": bool(manager.classifier is not None),
        "segmentation_loaded": bool(manager.unet is not None),
        "explainability_loaded": bool(manager.gradcam is not None),
        "device": str(manager.device),
        "version": "1.0.0"
    }


@app.post("/api/analyze", tags=["Inference"])
async def analyze_mri(file: UploadFile = File(...)):
    """
    Accepts an uploaded MRI image (JPG, JPEG, PNG), runs multi-task AI inference
    (Classification + Grad-CAM Explainability + U-Net Segmentation), and returns
    structured results with base64 visual overlays.
    
    No patient data or uploaded images are permanently stored on disk.
    """
    # 1. Validate filename extension
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file submitted in the upload request."
        )

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Supported formats: JPG, JPEG, PNG."
        )

    # 2. Validate MIME type if provided
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported content type '{file.content_type}'. Must be an image (JPEG, PNG)."
        )

    # 3. Read image bytes into memory safely
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    # 4. Validate file size
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the maximum upload size limit of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB."
        )

    # 5. Execute unified in-memory inference pipeline
    try:
        results = run_full_inference(image_bytes)
        return JSONResponse(status_code=status.HTTP_200_OK, content=results)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as exc:
        # Log error securely without image content
        print(f"[Inference Error] Unexpected error during analysis: {type(exc).__name__}: {str(exc)}", file=sys.stderr, flush=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during MRI analysis. Please ensure the image is a valid brain MRI slice and try again."
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
