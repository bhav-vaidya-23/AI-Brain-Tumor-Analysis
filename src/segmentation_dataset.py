import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import train_test_split

try:
    from src.config import DATASET_DIR, SEED, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
except ImportError:
    from config import DATASET_DIR, SEED, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY


class BrainTumorSegmentationDataset(Dataset):
    """
    PyTorch Dataset for BRISC2025 Brain Tumor MRI Segmentation.
    
    Dynamically loads paired MRI images and masks.
    Applies synchronized spatial data augmentation during training.
    Strictly uses Nearest-Neighbor interpolation for mask resizing.
    """

    def __init__(
        self,
        samples: List[Tuple[Union[str, Path], Union[str, Path], str, str]],
        image_size: Tuple[int, int] = (256, 256),
        is_train: bool = False
    ):
        """
        Args:
            samples: List of (img_path, mask_path, tumor_code, plane_code) tuples.
            image_size: Target (H, W) spatial resolution.
            is_train: If True, applies random synchronized spatial augmentations.
        """
        self.samples = samples
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_synchronized_transforms(
        self,
        image: Image.Image,
        mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Resize: Image -> Bilinear, Mask -> Nearest Neighbor
        image = TF.resize(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=InterpolationMode.NEAREST)

        # 2. Synchronized Augmentations (Training Only)
        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random Vertical Flip (medical MRI slices)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Small Random Rotation (-10 to +10 degrees)
            angle = random.uniform(-10.0, 10.0)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        # 3. Convert Image to Tensor & Normalize to [-1.0, 1.0]
        img_tensor = TF.to_tensor(image)  # [1, 256, 256] in [0.0, 1.0]
        img_tensor = TF.normalize(img_tensor, mean=[0.5], std=[0.5])

        # 4. Convert Mask to Binary Target Tensor [1, 256, 256] in {0.0, 1.0}
        mask_np = np.array(mask)
        mask_binary = (mask_np > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # [1, 256, 256]

        return img_tensor, mask_tensor

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Union[str, int]]]:
        img_path, msk_path, tcode, pcode = self.samples[index]

        try:
            with Image.open(img_path) as raw_img:
                image = raw_img.convert("L")  # Grayscale MRI slice
            with Image.open(msk_path) as raw_msk:
                mask = raw_msk.convert("L")   # Single-channel ground truth mask
        except Exception as e:
            raise RuntimeError(f"Failed to load sample at {img_path}: {e}")

        img_tensor, mask_tensor = self._apply_synchronized_transforms(image, mask)

        meta = {
            "img_path": str(img_path),
            "msk_path": str(msk_path),
            "filename": Path(img_path).name,
            "tumor_code": tcode,
            "plane_code": pcode
        }

        return img_tensor, mask_tensor, meta


def load_segmentation_pairs_from_dir(base_dir: Path) -> List[Tuple[Path, Path, str, str]]:
    """Loads and pairs image and mask paths from a split directory (e.g. segmentation_task/train)."""
    img_dir = base_dir / "images"
    msk_dir = base_dir / "masks"

    pairs = []
    if not img_dir.exists() or not msk_dir.exists():
        return pairs

    for img_name in sorted(os.listdir(img_dir)):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            base_name = os.path.splitext(img_name)[0]
            mask_name = base_name + ".png"
            msk_path = msk_dir / mask_name

            if msk_path.exists():
                img_path = img_dir / img_name
                parts = img_name.split('_')
                tcode = parts[3] if len(parts) > 3 else "unknown"
                pcode = parts[4] if len(parts) > 4 else "unknown"
                pairs.append((img_path, msk_path, tcode, pcode))

    return pairs


def create_segmentation_splits(
    dataset_root: Optional[Union[str, Path]] = None,
    val_split: float = 0.2,
    seed: int = SEED
) -> Tuple[List, List, List]:
    """
    Creates stratified 80% Train / 20% Val split from the 3,933 official training pairs.
    Preserves 100% of the official 860 test pairs as isolated holdout.
    """
    if dataset_root is None:
        dataset_root = DATASET_DIR
    dataset_root = Path(dataset_root)

    train_base = dataset_root / "segmentation_task" / "train"
    test_base = dataset_root / "segmentation_task" / "test"

    official_train_pairs = load_segmentation_pairs_from_dir(train_base)
    official_test_pairs = load_segmentation_pairs_from_dir(test_base)

    tumor_codes = [p[2] for p in official_train_pairs]

    train_pairs, val_pairs = train_test_split(
        official_train_pairs,
        test_size=val_split,
        random_state=seed,
        stratify=tumor_codes
    )

    return train_pairs, val_pairs, official_test_pairs


def get_segmentation_dataloaders(
    dataset_root: Optional[Union[str, Path]] = None,
    val_split: float = 0.2,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    seed: int = SEED
) -> Tuple[DataLoader, DataLoader, DataLoader, BrainTumorSegmentationDataset, BrainTumorSegmentationDataset, BrainTumorSegmentationDataset]:
    """
    Creates DataLoaders for U-Net Train, Validation, and Test sets.
    """
    train_pairs, val_pairs, test_pairs = create_segmentation_splits(
        dataset_root=dataset_root,
        val_split=val_split,
        seed=seed
    )

    train_dataset = BrainTumorSegmentationDataset(train_pairs, is_train=True)
    val_dataset = BrainTumorSegmentationDataset(val_pairs, is_train=False)
    test_dataset = BrainTumorSegmentationDataset(test_pairs, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
