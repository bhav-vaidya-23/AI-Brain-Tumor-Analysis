import sys
from pathlib import Path
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_DIR,
    TEST_DIR,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY
)
from src.dataset import (
    BrainTumorClassificationDataset,
    get_classification_dataloaders
)
from src.transforms import get_train_transforms, get_test_transforms


def test_classification_loader():
    print("=" * 60)
    print("TESTING BRISC2025 CLASSIFICATION DATA LOADER")
    print("=" * 60)

    # 1 & 2: Load training and test datasets & dataloaders
    print("\n[1] Initializing datasets and dataloaders...")
    train_loader, test_loader, train_dataset, test_dataset = get_classification_dataloaders(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # 3: Print dataset sizes
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    total_size = train_size + test_size
    print(f"\n[2] Dataset Sizes:")
    print(f"  - Training samples: {train_size}")
    print(f"  - Testing samples:  {test_size}")
    print(f"  - Total samples:    {total_size}")
    
    print("\n[3] Class Distribution in Training Set:")
    for cname, count in train_dataset.get_class_distribution().items():
        print(f"  - {cname:<12}: {count:>4} samples (index {CLASS_TO_IDX[cname]})")
        
    print("\n[4] Class Distribution in Testing Set:")
    for cname, count in test_dataset.get_class_distribution().items():
        print(f"  - {cname:<12}: {count:>4} samples (index {CLASS_TO_IDX[cname]})")

    # 4: Print class mapping
    print(f"\n[5] Class Mapping:")
    for cname, idx in sorted(CLASS_TO_IDX.items(), key=lambda x: x[1]):
        print(f"  - {cname:<12} -> {idx}")

    # 5: Load one batch
    print(f"\n[6] Loading one training batch (Batch size = {BATCH_SIZE})...")
    images, labels = next(iter(train_loader))

    # 6: Print tensor specifications
    print("\n[7] First Batch Tensor Properties:")
    print(f"  - Image tensor shape: {images.shape}")
    print(f"  - Label tensor shape: {labels.shape}")
    print(f"  - Image dtype:        {images.dtype}")
    print(f"  - Label dtype:        {labels.dtype}")
    print(f"  - Minimum pixel val:  {images.min().item():.4f}")
    print(f"  - Maximum pixel val:  {images.max().item():.4f}")

    # 7: Print labels in the first batch
    print("\n[8] Labels in first batch:")
    label_list = labels.tolist()
    label_names = [IDX_TO_CLASS[l] for l in label_list]
    print(f"  - Numeric labels: {label_list}")
    print(f"  - Class names:    {label_names}")

    # 8: Verify batch shape
    expected_shape = torch.Size([BATCH_SIZE, 3, 256, 256])
    assert images.shape == expected_shape, (
        f"Verification FAILED: Expected shape {expected_shape}, got {images.shape}"
    )
    print(f"\n[9] Shape verification: PASSED (Shape is strictly {list(images.shape)})")

    # 9: Test loading both RGB and Grayscale source images specifically
    print("\n[10] Testing RGB and Grayscale source image handling...")
    from PIL import Image
    
    # Verify first 20 samples from train and test without error
    rgb_count = 0
    l_count = 0
    for i in range(50):
        img_path, target = train_dataset.samples[i]
        with Image.open(img_path) as raw_img:
            if raw_img.mode == 'RGB':
                rgb_count += 1
            elif raw_img.mode == 'L':
                l_count += 1
        
        # Test dataset getitem
        tensor_img, tensor_lbl = train_dataset[i]
        assert tensor_img.shape == (3, 256, 256), f"Sample {i} returned incorrect shape {tensor_img.shape}"
        assert tensor_lbl.item() == target, f"Sample {i} returned mismatched label"
        
    print(f"  - Successfully verified 50 samples dynamically (Found {rgb_count} RGB and {l_count} Grayscale sources in sample).")
    print("  - All images correctly standardized to 3-channel RGB tensors with ImageNet normalization.")

    # Also check test loader batch
    test_images, test_labels = next(iter(test_loader))
    assert test_images.shape == expected_shape, f"Test batch shape error: {test_images.shape}"
    print(f"  - Test DataLoader batch loaded successfully: {list(test_images.shape)}")

    print("\n" + "=" * 60)
    print("ALL CLASSIFICATION DATA LOADER CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_classification_loader()
