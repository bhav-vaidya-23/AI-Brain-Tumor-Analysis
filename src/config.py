import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "brisc2025"
CLASSIFICATION_DIR = DATASET_DIR / "classification_task"
TRAIN_DIR = CLASSIFICATION_DIR / "train"
TEST_DIR = CLASSIFICATION_DIR / "test"
MANIFEST_CSV_PATH = DATASET_DIR / "manifest.csv"
MANIFEST_JSON_PATH = DATASET_DIR / "manifest.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Classification Class Mappings
# Strict mapping requirement:
# glioma -> 0, meningioma -> 1, pituitary -> 2, no_tumor -> 3
CLASS_TO_IDX = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
    "no_tumor": 3
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
CLASSES = list(CLASS_TO_IDX.keys())
NUM_CLASSES = len(CLASSES)

# Image Preprocessing & Input Config
IMAGE_SIZE = (256, 256)
CHANNELS = 3

# ImageNet Normalization Parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# DataLoader Configuration (Optimized for 16 GB RAM / CPU execution)
BATCH_SIZE = 16
NUM_WORKERS = 0
PIN_MEMORY = False
SHUFFLE_TRAIN = True
SHUFFLE_TEST = False

# Training Hyperparameters & Reproducibility
SEED = 42
VAL_SPLIT = 0.2  # 80% Train (4000), 20% Val (1000)

# Stage 1: Frozen Backbone (Linear Probing)
STAGE1_EPOCHS = 5
STAGE1_LR = 1e-3

# Stage 2: Fine-Tuning layer4 + FC
STAGE2_EPOCHS = 10
STAGE2_LR = 1e-4
EARLY_STOPPING_PATIENCE = 3

# Output Paths
MODEL_SAVE_PATH = OUTPUTS_DIR / "best_classifier.pth"
METRICS_SAVE_PATH = OUTPUTS_DIR / "classifier_metrics.json"
CURVES_SAVE_PATH = OUTPUTS_DIR / "classifier_learning_curves.png"
