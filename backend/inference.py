import io
import base64
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from src.config import (
    CLASS_TO_IDX, IDX_TO_CLASS, CLASSES,
    IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE
)
from src.transforms import get_test_transforms
from backend.model_loader import ModelManager

# Human-readable class names display mapping
DISPLAY_CLASS_MAP = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary",
    "no_tumor": "No Tumor"
}

def image_to_base64_png(img_array: np.ndarray) -> str:
    """Converts uint8 [H, W] or [H, W, 3] numpy array into a base64 encoded PNG string."""
    if img_array.ndim == 2:
        pil_img = Image.fromarray(img_array, mode="L")
    elif img_array.ndim == 3 and img_array.shape[2] == 3:
        pil_img = Image.fromarray(img_array, mode="RGB")
    elif img_array.ndim == 3 and img_array.shape[2] == 4:
        pil_img = Image.fromarray(img_array, mode="RGBA")
    else:
        raise ValueError(f"Unsupported image shape for base64 encoding: {img_array.shape}")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def create_segmentation_overlay(orig_rgb: np.ndarray, binary_mask: np.ndarray, color=(255, 50, 50), alpha=0.45) -> np.ndarray:
    """
    Creates an intuitive semi-transparent colored overlay and contour of the predicted tumor region on the MRI.
    """
    overlay = orig_rgb.copy().astype(np.float32)
    mask_bool = binary_mask > 0

    if np.any(mask_bool):
        # Color the tumor region
        color_arr = np.array(color, dtype=np.float32)
        overlay[mask_bool] = (1.0 - alpha) * overlay[mask_bool] + alpha * color_arr
        
        # Add simple border highlighting around mask perimeter
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(mask_bool, iterations=2)
        border = np.logical_and(dilated, ~mask_bool)
        overlay[border] = [0, 255, 255] # Cyan contour boundary

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def run_full_inference(image_bytes: bytes) -> Dict[str, Any]:
    """
    Executes the unified inference pipeline in-memory:
    1. ResNet-18 Classification + Softmax Probabilities
    2. Grad-CAM visual explainability from model.layer4
    3. 4-Level U-Net Tumor Segmentation (Epoch 23 checkpoint, Threshold >= 0.50)
    4. Base64 visualization generation for frontend rendering.
    """
    manager = ModelManager.get_instance()
    if not manager.is_loaded:
        manager.load_models()

    # 1. Decode Image Safely in Memory
    try:
        raw_pil = Image.open(io.BytesIO(image_bytes))
        orig_rgb_pil = raw_pil.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        orig_gray_pil = raw_pil.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    except Exception as e:
        raise ValueError(f"Could not decode image: {str(e)}")

    orig_rgb_np = np.array(orig_rgb_pil)

    # 2. Classification & Grad-CAM Inference
    cls_transforms = get_test_transforms()
    cls_tensor = cls_transforms(orig_rgb_pil).unsqueeze(0).to(manager.device)

    # Generate Grad-CAM from layer4
    heatmap_np, pred_idx, confidence, logits_np = manager.gradcam.generate_cam(cls_tensor, target_class=None)
    pred_class_raw = IDX_TO_CLASS[pred_idx]
    pred_class_display = DISPLAY_CLASS_MAP[pred_class_raw]

    # Calculate probabilities dictionary
    probs_np = F.softmax(torch.from_numpy(logits_np), dim=0).numpy()
    probabilities_dict = {
        DISPLAY_CLASS_MAP[IDX_TO_CLASS[i]]: float(probs_np[i])
        for i in range(len(CLASSES))
    }

    # 3. Grad-CAM Visualizations
    # Colored heatmap (jet colormap)
    cmap = plt.get_cmap('jet')
    colored_heatmap = (cmap(heatmap_np)[:, :, :3] * 255).astype(np.uint8)
    # Overlay on original MRI
    gradcam_overlay_np = manager.gradcam.overlay_heatmap(orig_rgb_np, heatmap_np, alpha=0.45, colormap='jet')

    # 4. U-Net Segmentation Inference
    unet_tensor = TF.normalize(TF.to_tensor(orig_gray_pil), mean=[0.5], std=[0.5]).unsqueeze(0).to(manager.device)
    with torch.no_grad():
        unet_logits = manager.unet(unet_tensor)
        unet_probs = torch.sigmoid(unet_logits).squeeze().cpu().numpy()
        unet_pred_bin = (unet_probs >= 0.50).astype(np.uint8)

    # Mask visualization & Overlay
    mask_visual = (unet_pred_bin * 255).astype(np.uint8)
    
    # Try scipy for contour boundary if available, else standard fallback
    try:
        seg_overlay_np = create_segmentation_overlay(orig_rgb_np, unet_pred_bin)
    except Exception:
        # Fallback simple overlay
        seg_overlay_np = orig_rgb_np.copy()
        seg_overlay_np[unet_pred_bin > 0] = (0.55 * seg_overlay_np[unet_pred_bin > 0] + 0.45 * np.array([255, 50, 50])).astype(np.uint8)

    # Compute tumor pixel percentage
    tumor_pixels = int(np.sum(unet_pred_bin))
    total_pixels = unet_pred_bin.size
    tumor_area_pct = float((tumor_pixels / total_pixels) * 100.0)
    is_tumor_class = bool(pred_class_raw != "no_tumor")
    has_predicted_region = bool(tumor_pixels > 0 and is_tumor_class)

    # 5. Base64 Encode Visualizations for Zero-Disk Client Transfer
    orig_b64 = image_to_base64_png(orig_rgb_np)
    heatmap_b64 = image_to_base64_png(colored_heatmap)
    cam_overlay_b64 = image_to_base64_png(gradcam_overlay_np)
    mask_b64 = image_to_base64_png(mask_visual)
    seg_overlay_b64 = image_to_base64_png(seg_overlay_np)

    return {
        "prediction": {
            "class": pred_class_display,
            "raw_class": pred_class_raw,
            "confidence": float(confidence),
            "confidence_percentage": round(float(confidence) * 100.0, 2)
        },
        "probabilities": probabilities_dict,
        "segmentation": {
            "has_predicted_region": has_predicted_region,
            "is_non_lesional": not is_tumor_class,
            "tumor_area_percentage": round(tumor_area_pct, 2) if is_tumor_class else 0.0,
            "raw_mask_area_percentage": round(tumor_area_pct, 2),
            "threshold_used": 0.50,
            "model_epoch": 23
        },
        "visualizations": {
            "original_mri": orig_b64,
            "gradcam_heatmap": heatmap_b64,
            "gradcam_overlay": cam_overlay_b64,
            "segmentation_mask": mask_b64,
            "segmentation_overlay": seg_overlay_b64
        },
        "metadata": {
            "classifier_architecture": "ResNet-18 (layer4 Fine-tuned)",
            "explainability_method": "Grad-CAM (Target Layer: model.layer4)",
            "segmentation_architecture": "4-Level U-Net (31.0M parameters)",
            "input_resolution": "256x256",
            "device": "CPU"
        },
        "disclaimer": (
            "This application is an AI-assisted research prototype and is not a clinically validated diagnostic system. "
            "Its predictions and visualizations should not be used for diagnosis, treatment decisions, or surgical planning. "
            "Please consult a qualified medical professional for medical interpretation."
        )
    }
