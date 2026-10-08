# AI-Based Brain Tumor Classification, Segmentation and Explainable MRI Analysis

Production-grade FastAPI inference backend for multi-task brain MRI analysis, powered by PyTorch:
- **Classification:** ResNet-18 (ImageNet fine-tuned) predicting Glioma, Meningioma, Pituitary, or No Tumor.
- **Explainability:** Grad-CAM saliency heatmaps generated from ResNet-18 `layer4`.
- **Segmentation:** 4-level U-Net (Epoch 23 checkpoint, 31.0M parameters) performing pixel-level tumor delineation.

---

## API Endpoints

### 1. Health Check
`GET /api/health`

Verifies that the backend server is running and all models are loaded in memory.

**Response:**
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

Accepts an uploaded brain MRI image (`multipart/form-data`, key: `file`, formats: `.jpg`, `.jpeg`, `.png`, max size: 10 MB).

**Response:**
```json
{
  "prediction": {
    "class": "Glioma",
    "raw_class": "glioma",
    "confidence": 0.999999,
    "confidence_percentage": 100.0
  },
  "probabilities": {
    "Glioma": 0.999999,
    "Meningioma": 0.000001,
    "Pituitary": 0.0,
    "No Tumor": 0.0
  },
  "segmentation": {
    "has_predicted_region": true,
    "is_non_lesional": false,
    "tumor_area_percentage": 2.46,
    "raw_mask_area_percentage": 2.46,
    "threshold_used": 0.50,
    "model_epoch": 23
  },
  "visualizations": {
    "original_mri": "data:image/png;base64,...",
    "gradcam_heatmap": "data:image/png;base64,...",
    "gradcam_overlay": "data:image/png;base64,...",
    "segmentation_mask": "data:image/png;base64,...",
    "segmentation_overlay": "data:image/png;base64,..."
  },
  "metadata": {
    "classifier_architecture": "ResNet-18 (layer4 Fine-tuned)",
    "explainability_method": "Grad-CAM (Target Layer: model.layer4)",
    "segmentation_architecture": "4-Level U-Net (31.0M parameters)",
    "input_resolution": "256x256",
    "device": "CPU"
  },
  "disclaimer": "This application is an AI-assisted research prototype and is not a clinically validated diagnostic system. Its predictions and visualizations should not be used for diagnosis, treatment decisions, or surgical planning. Please consult a qualified medical professional for medical interpretation."
}
```

---

## Medical & Research Disclaimer

This application is an **AI-assisted research prototype** and is **not a clinically validated diagnostic system**. Its predictions and visualizations should not be used for diagnosis, treatment decisions, or surgical planning. Please consult a qualified medical professional for medical interpretation.
