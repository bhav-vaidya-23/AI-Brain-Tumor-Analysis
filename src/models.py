import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple

try:
    from src.config import NUM_CLASSES
except ImportError:
    from config import NUM_CLASSES


def build_resnet18_classifier(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Constructs a ResNet-18 model for brain tumor classification.
    
    Architecture:
        Input: [3, 256, 256]
          ↓
        Pretrained ResNet-18
          ↓
        Feature extraction (AdaptiveAvgPool2d -> 512-dim)
          ↓
        Fully Connected Layer (Linear(512, 4))
          ↓
        4 output logits
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze all layers initially if freeze_backbone is True
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final classification head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # Ensure the newly instantiated fc layer is trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_layer4_and_fc(model: nn.Module) -> None:
    """
    Unfreezes layer4 and the fc classification head for Stage 2 fine-tuning.
    Layers 1, 2, and 3 remain frozen.
    """
    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Ensure fc is trainable
    for param in model.fc.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Returns (total_parameters, trainable_parameters).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
