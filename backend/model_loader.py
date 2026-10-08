import os
from pathlib import Path
import torch
from src.models import build_resnet18_classifier
from src.segmentation_model import UNet
from src.explainability import GradCAM

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER_PATH = PROJECT_ROOT / "outputs" / "best_classifier.pth"
UNET_PATH = PROJECT_ROOT / "outputs" / "best_unet.pth"

class ModelManager:
    """
    Singleton manager that loads and holds the ResNet-18 Classifier,
    Grad-CAM explainability hooks, and U-Net Segmentation model in memory.
    """
    _instance = None

    def __init__(self):
        self.device = torch.device("cpu")
        self.classifier = None
        self.unet = None
        self.gradcam = None
        self.is_loaded = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def load_models(self):
        if self.is_loaded:
            return

        print("[Backend] Loading ResNet-18 Classifier checkpoint...", flush=True)
        if not CLASSIFIER_PATH.exists():
            raise FileNotFoundError(f"Classifier checkpoint not found at: {CLASSIFIER_PATH}")
        
        self.classifier = build_resnet18_classifier(num_classes=4, pretrained=False, freeze_backbone=False)
        cls_ckpt = torch.load(CLASSIFIER_PATH, map_location=self.device, weights_only=False)
        cls_state_dict = cls_ckpt["model_state_dict"]
        del cls_ckpt  # Free optimizer_state_dict and training metadata immediately
        import gc
        gc.collect()

        self.classifier.load_state_dict(cls_state_dict)
        del cls_state_dict
        self.classifier.to(self.device)
        self.classifier.eval()
        gc.collect()

        # Initialize Grad-CAM on layer4
        self.gradcam = GradCAM(model=self.classifier, target_layer=self.classifier.layer4)

        print("[Backend] Loading U-Net Segmentation checkpoint (Epoch 23)...", flush=True)
        if not UNET_PATH.exists():
            raise FileNotFoundError(f"U-Net checkpoint not found at: {UNET_PATH}")

        unet_ckpt = torch.load(UNET_PATH, map_location=self.device, weights_only=False)
        unet_state_dict = unet_ckpt["model_state_dict"]
        del unet_ckpt  # Free 248MB optimizer_state_dict immediately
        gc.collect()

        self.unet = UNet(n_channels=1, n_classes=1)
        self.unet.load_state_dict(unet_state_dict)
        del unet_state_dict
        self.unet.to(self.device)
        self.unet.eval()
        gc.collect()

        self.is_loaded = True
        print("[Backend] All models successfully loaded into memory.", flush=True)
