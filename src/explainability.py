import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Tuple, Dict, Union
from pathlib import Path


class GradCAM:
    """
    Reusable Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
    
    Extracts feature maps and gradients from a designated convolutional layer
    (defaults to ResNet-18 layer4) and computes saliency heatmaps for specified
    or predicted target classes.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            # Default to layer4 for ResNet architectures
            if hasattr(model, 'layer4'):
                self.target_layer = model.layer4
            else:
                raise ValueError("Target layer not specified and model has no 'layer4' attribute.")
        else:
            self.target_layer = target_layer

        self.activations = None
        self.gradients = None
        self._handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._handlers.append(self.target_layer.register_forward_hook(forward_hook))
        if hasattr(self.target_layer, 'register_full_backward_hook'):
            self._handlers.append(self.target_layer.register_full_backward_hook(backward_hook))
        else:
            self._handlers.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        """Removes registered PyTorch hooks to release memory."""
        for handle in self._handlers:
            handle.remove()
        self._handlers = []

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float, np.ndarray]:
        """
        Computes the Grad-CAM heatmap for a single input tensor.
        
        Args:
            input_tensor: Tensor of shape [1, C, H, W]
            target_class: Integer target class index (None to use predicted class)
            
        Returns:
            heatmap: 2D numpy array [H, W] normalized to [0.0, 1.0]
            pred_class: Integer index of predicted class
            confidence: Softmax confidence of predicted class
            raw_logits: 1D numpy array of output logits
        """
        self.model.zero_grad()
        
        # 1. Forward inference
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
        
        pred_class = int(torch.argmax(probs).item())
        confidence = float(probs[pred_class].item())

        if target_class is None:
            chosen_class = pred_class
        else:
            chosen_class = target_class

        # 2. Backward pass for the target score
        target_score = logits[0, chosen_class]
        target_score.backward(retain_graph=True)

        # 3. Compute neuron importance weights alpha_k via Global Average Pooling
        # Gradients shape: [1, Channels, H_feat, W_feat]
        # Activations shape: [1, Channels, H_feat, W_feat]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 4. Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # 5. Apply ReLU to isolate features with a positive influence
        cam = F.relu(cam)

        # 6. Upsample CAM to input spatial dimensions
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear',
            align_corners=False
        )

        cam_np = cam.squeeze().cpu().numpy()

        # 7. Normalize heatmap to [0, 1]
        max_val = np.max(cam_np)
        min_val = np.min(cam_np)
        if max_val - min_val > 1e-8:
            heatmap = (cam_np - min_val) / (max_val - min_val)
        else:
            heatmap = np.zeros_like(cam_np)

        return heatmap, pred_class, confidence, logits.squeeze(0).detach().cpu().numpy()

    @staticmethod
    def overlay_heatmap(
        image_np: np.ndarray,
        heatmap_np: np.ndarray,
        alpha: float = 0.45,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Overlays a normalized Grad-CAM heatmap [0, 1] onto an RGB image [0, 255] uint8.
        """
        # Ensure image is in uint8 [0, 255]
        if image_np.dtype != np.uint8:
            if image_np.max() <= 1.0:
                img_uint8 = (image_np * 255).astype(np.uint8)
            else:
                img_uint8 = image_np.astype(np.uint8)
        else:
            img_uint8 = image_np

        # Apply colormap to heatmap
        try:
            cmap = plt.get_cmap(colormap)
        except AttributeError:
            import matplotlib as mpl
            cmap = mpl.colormaps[colormap]
        colored_heatmap = cmap(heatmap_np)[:, :, :3]  # [H, W, 3] in [0, 1]
        colored_heatmap_uint8 = (colored_heatmap * 255).astype(np.uint8)

        # Alpha blend: (1 - alpha) * Image + alpha * Heatmap
        overlay = (1.0 - alpha) * img_uint8.astype(np.float32) + alpha * colored_heatmap_uint8.astype(np.float32)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return overlay

    @staticmethod
    def compute_heatmap_stats(heatmap_np: np.ndarray) -> Dict[str, float]:
        """
        Calculates quantitative activation metrics for a Grad-CAM heatmap.
        """
        total_pixels = heatmap_np.size
        return {
            "mean_activation": float(np.mean(heatmap_np)),
            "max_activation": float(np.max(heatmap_np)),
            "pct_above_25": float(np.sum(heatmap_np >= 0.25) / total_pixels * 100.0),
            "pct_above_50": float(np.sum(heatmap_np >= 0.50) / total_pixels * 100.0),
            "pct_above_75": float(np.sum(heatmap_np >= 0.75) / total_pixels * 100.0)
        }

    @staticmethod
    def save_3panel_figure(
        original_img: Union[Image.Image, np.ndarray],
        heatmap_np: np.ndarray,
        overlay_np: np.ndarray,
        info: Dict[str, str],
        save_path: Union[str, Path]
    ) -> None:
        """
        Saves a publication-quality 3-panel figure: [Original MRI] [Grad-CAM] [Overlay].
        """
        if isinstance(original_img, Image.Image):
            orig_np = np.array(original_img.convert('RGB'))
        else:
            orig_np = original_img

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Original MRI
        axes[0].imshow(orig_np)
        axes[0].set_title("Original T1 MRI", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Panel 2: Grad-CAM Heatmap
        im_cam = axes[1].imshow(heatmap_np, cmap='jet', vmin=0.0, vmax=1.0)
        axes[1].set_title(f"Grad-CAM Heatmap ({info.get('target_class', 'Target')})", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im_cam, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Overlay
        axes[2].imshow(overlay_np)
        axes[2].set_title("Grad-CAM Overlay", fontsize=12, fontweight='bold')
        axes[2].axis('off')

        # Main Title Header with metadata
        is_correct = info.get('is_correct', True)
        status_color = '#1f77b4' if is_correct else '#d62728'
        status_str = "CORRECT" if is_correct else "INCORRECT"

        fig.suptitle(
            f"File: {info.get('filename', 'Unknown')} | True: {info.get('true_class', 'N/A').capitalize()} | "
            f"Pred: {info.get('pred_class', 'N/A').capitalize()} ({info.get('confidence', 0.0):.2f}%) | "
            f"Status: {status_str}",
            fontsize=13,
            fontweight='bold',
            color=status_color,
            y=0.98
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
