import os
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

try:
    from src.config import (
        TRAIN_DIR,
        TEST_DIR,
        CLASS_TO_IDX,
        IDX_TO_CLASS,
        CLASSES,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
        SEED,
        VAL_SPLIT
    )
    from src.transforms import get_train_transforms, get_test_transforms
except ImportError:
    from config import (
        TRAIN_DIR,
        TEST_DIR,
        CLASS_TO_IDX,
        IDX_TO_CLASS,
        CLASSES,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
        SEED,
        VAL_SPLIT
    )
    from transforms import get_train_transforms, get_test_transforms


class BrainTumorClassificationDataset(Dataset):
    """
    PyTorch Dataset for BRISC2025 Brain Tumor MRI Classification.
    
    Dynamically loads images on demand to ensure minimal RAM usage.
    Automatically converts all images (RGB or Grayscale) to 3-channel RGB.
    
    Class mapping:
        glioma    -> 0
        meningioma -> 1
        pituitary -> 2
        no_tumor  -> 3
    """

    def __init__(
        self,
        root_dir: Optional[str | Path] = None,
        samples: Optional[List[Tuple[Path, int]]] = None,
        transform: Optional[Callable] = None,
        class_to_idx: Dict[str, int] = CLASS_TO_IDX
    ):
        """
        Args:
            root_dir: Directory containing class subfolders (e.g. classification_task/train).
            samples: Optional pre-compiled list of (image_path, label_idx) tuples.
            transform: Optional torchvision transform to be applied on a sample.
            class_to_idx: Dictionary mapping class folder names to class index.
        """
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        if samples is not None:
            self.samples = samples
        elif root_dir is not None:
            self.root_dir = Path(root_dir)
            self.samples = self._load_samples_from_dir(self.root_dir)
        else:
            raise ValueError("Either 'root_dir' or 'samples' must be provided.")

    def _load_samples_from_dir(self, root_dir: Path) -> List[Tuple[Path, int]]:
        samples = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        
        for class_name, class_idx in self.class_to_idx.items():
            class_folder = root_dir / class_name
            if not class_folder.exists() or not class_folder.is_dir():
                continue
            
            for file_path in sorted(class_folder.iterdir()):
                if file_path.suffix.lower() in valid_extensions:
                    samples.append((file_path, class_idx))
                    
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[index]

        # 1. Open safely with PIL
        # 2. Convert to RGB (standardizes both 1-channel grayscale and 3-channel RGB slices)
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image at {img_path}: {e}")

        # 3. Apply transformation pipeline
        if self.transform is not None:
            image = self.transform(image)

        # Return image tensor and label tensor (int64 / long)
        return image, torch.tensor(label, dtype=torch.long)

    def get_class_distribution(self) -> Dict[str, int]:
        counts = {c: 0 for c in self.class_to_idx.keys()}
        for _, label in self.samples:
            cname = self.idx_to_class[label]
            counts[cname] += 1
        return counts


def create_stratified_train_val_split(
    train_dir: str | Path = TRAIN_DIR,
    val_split: float = VAL_SPLIT,
    seed: int = SEED
) -> Tuple[BrainTumorClassificationDataset, BrainTumorClassificationDataset]:
    """
    Creates reproducible, stratified 80% train / 20% validation datasets from the training folder.
    """
    base_dataset = BrainTumorClassificationDataset(root_dir=train_dir)
    all_samples = base_dataset.samples
    labels = [label for _, label in all_samples]

    train_samples, val_samples = train_test_split(
        all_samples,
        test_size=val_split,
        random_state=seed,
        stratify=labels
    )

    train_dataset = BrainTumorClassificationDataset(
        samples=train_samples,
        transform=get_train_transforms()
    )
    
    val_dataset = BrainTumorClassificationDataset(
        samples=val_samples,
        transform=get_test_transforms()
    )

    return train_dataset, val_dataset


def get_classification_dataloaders(
    train_dir: str | Path = TRAIN_DIR,
    test_dir: str | Path = TEST_DIR,
    val_split: float = VAL_SPLIT,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    seed: int = SEED
) -> Tuple[DataLoader, DataLoader, DataLoader, BrainTumorClassificationDataset, BrainTumorClassificationDataset, BrainTumorClassificationDataset]:
    """
    Creates DataLoaders for Train (80%), Validation (20%), and Test (1000).
    """
    train_dataset, val_dataset = create_stratified_train_val_split(
        train_dir=train_dir,
        val_split=val_split,
        seed=seed
    )
    
    test_dataset = BrainTumorClassificationDataset(
        root_dir=test_dir,
        transform=get_test_transforms()
    )

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
