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
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_DIR,
    TEST_DIR,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CLASSES,
    NUM_CLASSES,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    SEED,
    VAL_SPLIT,
    STAGE1_EPOCHS,
    STAGE1_LR,
    STAGE2_EPOCHS,
    STAGE2_LR,
    EARLY_STOPPING_PATIENCE,
    MODEL_SAVE_PATH,
    METRICS_SAVE_PATH,
    CURVES_SAVE_PATH,
    REPORTS_DIR,
    OUTPUTS_DIR
)
from src.dataset import get_classification_dataloaders
from src.models import (
    build_resnet18_classifier,
    unfreeze_layer4_and_fc,
    count_parameters
)


def set_seed(seed: int = SEED) -> None:
    """Ensures deterministic execution across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_class_weights(train_dataset, device: torch.device) -> torch.Tensor:
    """
    Computes balanced class weights for CrossEntropyLoss:
    weight[c] = Total_samples / (n_classes * count[c])
    """
    counts = train_dataset.get_class_distribution()
    total_samples = len(train_dataset)
    num_classes = len(counts)
    
    weights = []
    print("\nTraining Set Class Distribution & Calculated Weights:")
    for cname in CLASSES:
        count = counts[cname]
        pct = (count / total_samples) * 100
        # Inverse frequency weighting
        w = total_samples / (num_classes * count) if count > 0 else 1.0
        weights.append(w)
        print(f"  - {cname:<12} (class {CLASS_TO_IDX[cname]}): {count:>4} samples ({pct:>5.2f}%) | Weight: {w:.4f}")
        
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    return weight_tensor


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Runs one training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluates model on validation/test set."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    total = len(all_targets)
    epoch_loss = running_loss / total
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    epoch_acc = (np.sum(all_preds == all_targets) / total) * 100.0
    epoch_macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100.0

    return epoch_loss, epoch_acc, epoch_macro_f1, all_targets, all_preds


def plot_learning_curves(history: Dict[str, List[float]], save_path: Path) -> None:
    """Plots and saves train/val loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Curve
    axes[0].plot(epochs, history["train_loss"], 'o-', label='Train Loss', color='#1f77b4', lw=2)
    axes[0].plot(epochs, history["val_loss"], 's-', label='Val Loss', color='#ff7f0e', lw=2)
    if history.get("stage_transition_epoch"):
        axes[0].axvline(x=history["stage_transition_epoch"], color='red', linestyle='--', label='Stage 2 Start')
    axes[0].set_title('Cross-Entropy Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Accuracy Curve
    axes[1].plot(epochs, history["train_acc"], 'o-', label='Train Acc (%)', color='#1f77b4', lw=2)
    axes[1].plot(epochs, history["val_acc"], 's-', label='Val Acc (%)', color='#ff7f0e', lw=2)
    if history.get("stage_transition_epoch"):
        axes[1].axvline(x=history["stage_transition_epoch"], color='red', linestyle='--', label='Stage 2 Start')
    axes[1].set_title('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Learning curves saved to: {save_path}")


def main():
    print("=" * 70)
    print("BRAIN TUMOR CLASSIFICATION: RESNET-18 TWO-STAGE TRAINING PIPELINE")
    print("=" * 70)

    # 1. Reproducibility & Device Check
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Using CPU")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # 2. Data Preparation
    print("\n[Step 1] Initializing Stratified Datasets and DataLoaders...")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_classification_dataloaders(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        val_split=VAL_SPLIT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        seed=SEED
    )

    print(f"  - Train Set (80%): {len(train_dataset)} samples")
    print(f"  - Val Set   (20%): {len(val_dataset)} samples")
    print(f"  - Test Set       : {len(test_dataset)} samples")

    # 3. Class Weighting Computation
    class_weights = calculate_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. Model Construction
    print("\n[Step 2] Building Pretrained ResNet-18 Classifier...")
    model = build_resnet18_classifier(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Model Architecture:\n{model}")
    print(f"\nParameter Summary (Stage 1 Initial):")
    print(f"  - Total Parameters:     {total_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,} (Only 'fc' layer)")
    print(f"  - Frozen Parameters:    {total_params - trainable_params:,}")

    # History Tracking
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "stage_transition_epoch": None
    }

    best_val_f1 = -1.0
    best_epoch = 0

    # =========================================================================
    # STAGE 1: Linear Probing (Train FC Head Only, Backbone Frozen)
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"STAGE 1: Feature Extraction (Backbone Frozen) | {STAGE1_EPOCHS} Epochs | LR = {STAGE1_LR}")
    print("=" * 70)

    optimizer_stage1 = Adam(model.fc.parameters(), lr=STAGE1_LR)

    for epoch in range(1, STAGE1_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage1, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        print(f"Epoch {epoch}/{STAGE1_EPOCHS} (Stage 1) [{elapsed:.1f}s] "
              f"| Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                "epoch": int(epoch),
                "stage": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_stage1.state_dict(),
                "val_f1": float(val_f1),
                "val_acc": float(val_acc),
                "val_loss": float(val_loss),
                "class_to_idx": CLASS_TO_IDX
            }, MODEL_SAVE_PATH)
            print(f"  --> Saved new best model checkpoint (Val F1: {val_f1:.2f}%) to {MODEL_SAVE_PATH.name}")

    # =========================================================================
    # STAGE 2: Fine-Tuning (Unfreeze layer4 + FC)
    # =========================================================================
    history["stage_transition_epoch"] = STAGE1_EPOCHS
    print("\n" + "=" * 70)
    print(f"STAGE 2: Fine-Tuning (Unfreeze layer4 + FC) | Max {STAGE2_EPOCHS} Epochs | LR = {STAGE2_LR}")
    print("=" * 70)

    unfreeze_layer4_and_fc(model)
    total_params, trainable_params = count_parameters(model)
    print(f"Updated Parameter Summary (Stage 2):")
    print(f"  - Total Parameters:     {total_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,} (layer4 + fc)")
    print(f"  - Frozen Parameters:    {total_params - trainable_params:,}")

    # Differential or uniform fine-tuning optimizer
    optimizer_stage2 = Adam([
        {'params': model.layer4.parameters(), 'lr': STAGE2_LR},
        {'params': model.fc.parameters(), 'lr': STAGE2_LR}
    ])

    patience_counter = 0

    for epoch_idx in range(1, STAGE2_EPOCHS + 1):
        global_epoch = STAGE1_EPOCHS + epoch_idx
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage2, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        print(f"Epoch {epoch_idx}/{STAGE2_EPOCHS} (Total Epoch {global_epoch}) [{elapsed:.1f}s] "
              f"| Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%")

        # Early stopping & model checkpointing based on Validation Macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = global_epoch
            patience_counter = 0
            torch.save({
                "epoch": int(global_epoch),
                "stage": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_stage2.state_dict(),
                "val_f1": float(val_f1),
                "val_acc": float(val_acc),
                "val_loss": float(val_loss),
                "class_to_idx": CLASS_TO_IDX
            }, MODEL_SAVE_PATH)
            print(f"  --> Saved new best model checkpoint (Val F1: {val_f1:.2f}%) to {MODEL_SAVE_PATH.name}")
        else:
            patience_counter += 1
            print(f"  --> No improvement in Val F1. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n[Early Stopping Triggered] Stopping fine-tuning at total epoch {global_epoch}.")
                break

    # Plot learning curves
    plot_learning_curves(history, CURVES_SAVE_PATH)

    # =========================================================================
    # FINAL EVALUATION ON INDEPENDENT TEST SET (1,000 SAMPLES)
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON INDEPENDENT TEST SET (1,000 Samples)")
    print("=" * 70)

    # Load best checkpoint
    print(f"Loading best checkpoint from {MODEL_SAVE_PATH.name} (from Epoch {best_epoch})...")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_macro_f1, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    # Generate metrics report
    report_dict = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, digits=4)
    report_text = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"\nOverall Test Accuracy: {test_acc:.2f}%")
    print(f"Overall Test Loss:     {test_loss:.4f}")
    print(f"Overall Test Macro F1: {test_macro_f1:.2f}%\n")
    print("Classification Report:")
    print(report_text)
    print("\nConfusion Matrix:")
    print(f"{'':<12} " + " ".join([f"{c[:4]:>6}" for c in CLASSES]))
    for i, row in enumerate(conf_matrix):
        print(f"{CLASSES[i]:<12} " + " ".join([f"{val:>6}" for val in row]))

    # Save metrics JSON
    metrics_summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_val_f1),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "classification_report": report_dict,
        "confusion_matrix": conf_matrix.tolist(),
        "history": history
    }

    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"\nMetrics summary saved to: {METRICS_SAVE_PATH}")

    # Save Report to text file
    report_file = REPORTS_DIR / "classification_model_report.txt"
    with open(report_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("BRISC2025 BRAIN TUMOR CLASSIFICATION REPORT\n")
        f.write("Model: ResNet-18 (Two-Stage Transfer Learning)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Best Validation Epoch: {best_epoch}\n")
        f.write(f"Best Validation Macro F1: {best_val_f1:.2f}%\n\n")
        f.write(f"Independent Test Set Evaluation (1,000 Samples):\n")
        f.write(f"  - Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"  - Test Loss:     {test_loss:.4f}\n")
        f.write(f"  - Test Macro F1: {test_macro_f1:.2f}%\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report_text + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{'':<12} " + " ".join([f"{c[:4]:>6}" for c in CLASSES]) + "\n")
        for i, row in enumerate(conf_matrix):
            f.write(f"{CLASSES[i]:<12} " + " ".join([f"{val:>6}" for val in row]) + "\n")

    print(f"Classification report saved to: {report_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
