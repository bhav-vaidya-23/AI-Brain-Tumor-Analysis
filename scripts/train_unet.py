import os
import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from PIL import Image

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Optimize CPU threads if running on CPU
if not torch.cuda.is_available():
    num_threads = min(os.cpu_count() or 4, 8)
    torch.set_num_threads(num_threads)
    print(f"Configured PyTorch CPU threads: {num_threads}", flush=True)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SEED, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, OUTPUTS_DIR, REPORTS_DIR
from src.segmentation_model import UNet, count_parameters
from src.segmentation_losses import CombinedLoss
from src.segmentation_metrics import compute_sample_metrics, aggregate_metrics
from src.segmentation_dataset import get_segmentation_dataloaders


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Runs one training epoch for U-Net."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, targets, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return running_loss / total_samples


def evaluate_segmentation(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[float, Dict[str, Dict[str, float]], List[Dict]]:
    """Evaluates U-Net on validation/test set."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_sample_metrics = []

    with torch.no_grad():
        for images, targets, metas in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            probs_np = probs.squeeze(1).cpu().numpy()
            preds_np = preds.squeeze(1).cpu().numpy()
            targets_np = targets.squeeze(1).cpu().numpy()

            for i in range(images.size(0)):
                s_metrics = compute_sample_metrics(preds_np[i], targets_np[i])
                s_metrics["filename"] = metas["filename"][i]
                s_metrics["tumor_code"] = metas["tumor_code"][i]
                s_metrics["plane_code"] = metas["plane_code"][i]
                s_metrics["img_path"] = metas["img_path"][i]
                s_metrics["msk_path"] = metas["msk_path"][i]
                all_sample_metrics.append(s_metrics)

    eval_loss = running_loss / total_samples
    summary_metrics = aggregate_metrics(all_sample_metrics)

    return eval_loss, summary_metrics, all_sample_metrics


def plot_learning_curves(history: Dict[str, List[float]], save_path: Path):
    """Plots training/validation loss and Dice curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    axes[0].plot(epochs, history["train_loss"], 'o-', label='Train Loss', color='#1f77b4', lw=2)
    axes[0].plot(epochs, history["val_loss"], 's-', label='Val Loss', color='#ff7f0e', lw=2)
    axes[0].set_title('Combined Loss (BCE + Dice)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Dice Curve
    axes[1].plot(epochs, history["val_dice"], 's-', label='Val Dice (%)', color='#2ca02c', lw=2)
    axes[1].plot(epochs, history["val_iou"], '^-', label='Val IoU (%)', color='#d62728', lw=2)
    axes[1].set_title('Validation Dice & IoU (%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score (%)')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"U-Net learning curves saved to: {save_path}")


def save_4panel_visualization(
    img_path: str,
    msk_path: str,
    pred_mask: np.ndarray,
    meta_info: Dict,
    save_path: Path
):
    """Saves 4-panel figure: [Original MRI] [Ground Truth] [Predicted Mask] [Overlay]."""
    with Image.open(img_path) as im, Image.open(msk_path) as mk:
        im_gray = im.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
        mk_bin = (np.array(mk.convert("L").resize((256, 256), Image.Resampling.NEAREST)) > 0).astype(np.uint8)

    im_np = np.array(im_gray)
    
    # Create RGB image
    im_rgb = np.stack([im_np] * 3, axis=-1)

    # Overlay: Ground Truth in Green, Prediction in Red, Overlap in Yellow
    overlay = im_rgb.copy()
    
    gt_only = np.logical_and(mk_bin == 1, pred_mask == 0)
    pred_only = np.logical_and(mk_bin == 0, pred_mask == 1)
    overlap = np.logical_and(mk_bin == 1, pred_mask == 1)

    # Green for GT only [0, 255, 0]
    overlay[gt_only] = [0, 255, 0]
    # Red for Pred only [255, 0, 0]
    overlay[pred_only] = [255, 0, 0]
    # Yellow for Overlap (True Positive) [255, 255, 0]
    overlay[overlap] = [255, 255, 0]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    axes[0].imshow(im_np, cmap='gray')
    axes[0].set_title(f"Original MRI ({meta_info.get('plane_code', '').upper()})", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(mk_bin, cmap='gray')
    axes[1].set_title(f"Ground Truth ({meta_info.get('tumor_area_pct', 0):.2f}%)", fontsize=11, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title(f"U-Net Prediction", fontsize=11, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay (Yellow=TP, Red=FP, Green=FN)", fontsize=10, fontweight='bold')
    axes[3].axis('off')

    fig.suptitle(
        f"File: {meta_info.get('filename', '')} | Tumor: {meta_info.get('tumor_code', '').upper()} | "
        f"Dice: {meta_info.get('dice', 0)*100:.2f}% | IoU: {meta_info.get('iou', 0)*100:.2f}% | "
        f"Prec: {meta_info.get('precision', 0)*100:.2f}% | Rec: {meta_info.get('recall', 0)*100:.2f}%",
        fontsize=12,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    print("=" * 70)
    print("BRAIN TUMOR SEGMENTATION: U-NET TRAINING PIPELINE")
    print("=" * 70)

    # 1. Device and Reproducibility
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Using CPU")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # 2. Output Paths
    model_save_path = OUTPUTS_DIR / "best_unet.pth"
    history_save_path = OUTPUTS_DIR / "unet_training_history.json"
    curves_save_path = OUTPUTS_DIR / "unet_learning_curves.png"
    config_save_path = OUTPUTS_DIR / "unet_config.json"
    val_vis_dir = OUTPUTS_DIR / "segmentation_validation"
    val_vis_dir.mkdir(parents=True, exist_ok=True)

    # 3. DataLoaders
    print("\n[Step 1] Initializing Stratified Segmentation DataLoaders...")
    batch_size = 16
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_segmentation_dataloaders(
        val_split=0.2,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        seed=SEED
    )

    print(f"  - U-Net Train Set (80%): {len(train_dataset)} pairs")
    print(f"  - U-Net Val Set   (20%): {len(val_dataset)} pairs")
    print(f"  - U-Net Test Holdout   : {len(test_dataset)} pairs (Untouched)")

    # 4. Model Construction
    print("\n[Step 2] Building U-Net Model from Scratch...")
    model = UNet(n_channels=1, n_classes=1).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"U-Net Parameter Summary:")
    print(f"  - Total Parameters:     {total_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,}")

    # 5. Loss, Optimizer, and Scheduler
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    learning_rate = 1e-4
    weight_decay = 1e-4
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Scheduler: Reduce LR when Validation Dice plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    # 6. Save Config
    config_metadata = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "batch_size": batch_size,
        "image_resolution": [256, 256],
        "optimizer": "AdamW",
        "initial_learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss_function": "0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss",
        "augmentation": {
            "horizontal_flip_prob": 0.5,
            "vertical_flip_prob": 0.5,
            "rotation_degrees": [-10, 10]
        },
        "max_epochs": 25,
        "early_stopping_patience": 10,
        "seed": SEED,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset)
    }

    with open(config_save_path, "w") as f:
        json.dump(config_metadata, f, indent=4)
    print(f"Configuration metadata saved to: {config_save_path}")

    # 7. Training Loop with Early Stopping
    print("\n" + "=" * 70)
    print("STARTING U-NET TRAINING (Max 25 Epochs | Early Stopping Patience = 10)")
    print("=" * 70)

    epochs = 25
    patience = 10
    best_val_dice = -1.0
    best_epoch = 0
    patience_counter = 0
    start_epoch = 1

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_precision": [],
        "val_recall": [],
        "learning_rates": []
    }

    if model_save_path.exists():
        try:
            print(f"\n[Checkpoint Found] Resuming from existing checkpoint: {model_save_path.name}...")
            ckpt = torch.load(model_save_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            best_val_dice = ckpt.get("val_dice", -1.0)
            best_epoch = ckpt.get("epoch", 0)
            
            if history_save_path.exists():
                with open(history_save_path, "r") as f:
                    history = json.load(f)
                # If history has entries recorded starting from epoch 4:
                completed_epochs = len(history["train_loss"]) + 3
                start_epoch = completed_epochs + 1
                patience_counter = max(0, completed_epochs - best_epoch)
            else:
                start_epoch = best_epoch + 1
                patience_counter = 0

            print(f"Resumed at epoch {start_epoch} (Previous Best Val Dice: {best_val_dice:.2f}% at epoch {best_epoch}, Current Patience: {patience_counter}/{patience})\n", flush=True)
        except Exception as e:
            print(f"Could not resume checkpoint ({e}); starting from epoch 1.", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_metrics, val_samples_data = evaluate_segmentation(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        val_dice_mean = val_metrics["dice"]["mean"] * 100.0
        val_iou_mean = val_metrics["iou"]["mean"] * 100.0
        val_prec_mean = val_metrics["precision"]["mean"] * 100.0
        val_rec_mean = val_metrics["recall"]["mean"] * 100.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice_mean)
        history["val_iou"].append(val_iou_mean)
        history["val_precision"].append(val_prec_mean)
        history["val_recall"].append(val_rec_mean)
        history["learning_rates"].append(current_lr)

        # Persist history
        with open(history_save_path, "w") as f:
            json.dump(history, f, indent=4)

        # Step scheduler
        scheduler.step(val_dice_mean)

        print(f"Epoch {epoch:02d}/{epochs} [{elapsed:4.1f}s] "
              f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| Val Dice: {val_dice_mean:5.2f}% | Val IoU: {val_iou_mean:5.2f}% "
              f"| Val Prec: {val_prec_mean:5.2f}% | Val Rec: {val_rec_mean:5.2f}% | LR: {current_lr:.1e}", flush=True)

        # Checkpoint based on Validation Dice
        if val_dice_mean > best_val_dice:
            best_val_dice = val_dice_mean
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice_mean,
                "val_iou": val_iou_mean,
                "val_loss": val_loss,
                "best_val_metrics": val_metrics,
                "config": config_metadata
            }, model_save_path)
            print(f"  --> Saved new best U-Net checkpoint (Val Dice: {val_dice_mean:.2f}%) to {model_save_path.name}", flush=True)
        else:
            patience_counter += 1
            print(f"  --> No improvement in Val Dice. Patience: {patience_counter}/{patience}", flush=True)
            if patience_counter >= patience:
                print(f"\n[Early Stopping Triggered] Stopped training at epoch {epoch}.", flush=True)
                break

    # 8. Save History & Learning Curves
    with open(history_save_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"\nTraining history saved to: {history_save_path}")

    plot_learning_curves(history, curves_save_path)

    # 9. Qualitative Validation Visualizations using Best Checkpoint
    print("\n[Step 3] Generating Qualitative Validation Visualizations from Best Model...")
    checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Re-evaluate validation set with best model to get per-sample predictions
    _, _, best_val_samples = evaluate_segmentation(model, val_loader, criterion, device)

    # Sort validation samples by tumor area and by Dice
    sorted_by_area = sorted(best_val_samples, key=lambda x: x["tumor_area_pct"])
    sorted_by_dice = sorted(best_val_samples, key=lambda x: x["dice"])

    small_val_sample = sorted_by_area[int(len(sorted_by_area) * 0.05)]
    med_val_sample = sorted_by_area[int(len(sorted_by_area) * 0.50)]
    large_val_sample = sorted_by_area[int(len(sorted_by_area) * 0.95)]
    high_dice_sample = sorted_by_dice[-1]
    low_dice_sample = sorted_by_dice[int(len(sorted_by_dice) * 0.05)]
    diff_sample = [s for s in sorted_by_dice if 0.4 < s["dice"] < 0.6][0] if any(0.4 < s["dice"] < 0.6 for s in sorted_by_dice) else sorted_by_dice[0]

    val_vis_cases = [
        ("small_tumor", small_val_sample),
        ("medium_tumor", med_val_sample),
        ("large_tumor", large_val_sample),
        ("high_dice", high_dice_sample),
        ("low_dice", low_dice_sample),
        ("difficult_case", diff_sample)
    ]

    transform_val = val_dataset
    for label, s_meta in val_vis_cases:
        img_p = s_meta["img_path"]
        msk_p = s_meta["msk_path"]

        with Image.open(img_p) as im:
            im_t = im.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
            t_img = TF.normalize(TF.to_tensor(im_t), mean=[0.5], std=[0.5]).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = model(t_img)
            pred_mask = (torch.sigmoid(logit) >= 0.5).squeeze().cpu().numpy().astype(np.uint8)

        out_fname = f"val_{label}_{s_meta['filename'].replace('.jpg', '')}.png"
        save_4panel_visualization(img_p, msk_p, pred_mask, s_meta, val_vis_dir / out_fname)

    print(f"Validation visualizations saved to: {val_vis_dir}")
    print("\n" + "=" * 70)
    print(f"U-NET TRAINING COMPLETED SUCCESSFULLY (Best Epoch: {best_epoch}, Best Val Dice: {best_val_dice:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
