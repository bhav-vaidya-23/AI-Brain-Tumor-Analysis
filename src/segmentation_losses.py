import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Computes soft Dice Loss for binary segmentation:
        Dice = (2 * |P ∩ Y| + smooth) / (|P| + |Y| + smooth)
        DiceLoss = 1.0 - Dice
    where P = sigmoid(logits) and Y = binary ground truth.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Flatten batch and spatial dimensions
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        total_cardinality = probs_flat.sum() + targets_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (total_cardinality + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combined Binary Cross Entropy with Logits and Soft Dice Loss:
        TotalLoss = bce_weight * BCEWithLogitsLoss + dice_weight * DiceLoss
    Default: 0.5 * BCE + 0.5 * Dice
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
