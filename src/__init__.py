"""
Brain Tumor Detection, Segmentation, and Explainable AI Pipeline.
"""
from src.config import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CLASSES,
    NUM_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEED,
    VAL_SPLIT,
    STAGE1_EPOCHS,
    STAGE1_LR,
    STAGE2_EPOCHS,
    STAGE2_LR,
    EARLY_STOPPING_PATIENCE,
    MODEL_SAVE_PATH
)
from src.transforms import get_train_transforms, get_test_transforms
from src.dataset import (
    BrainTumorClassificationDataset,
    create_stratified_train_val_split,
    get_classification_dataloaders
)
from src.models import (
    build_resnet18_classifier,
    unfreeze_layer4_and_fc,
    count_parameters
)
from src.explainability import GradCAM
from src.segmentation_model import UNet
from src.segmentation_losses import DiceLoss, CombinedLoss
from src.segmentation_metrics import compute_sample_metrics, aggregate_metrics, compute_metrics_by_tumor_size
from src.segmentation_dataset import BrainTumorSegmentationDataset, get_segmentation_dataloaders
