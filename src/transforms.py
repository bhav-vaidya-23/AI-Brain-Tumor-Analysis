import torchvision.transforms as transforms
from typing import Tuple, List, Optional
try:
    from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

def get_train_transforms(
    image_size: Tuple[int, int] = IMAGE_SIZE,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> transforms.Compose:
    """
    Returns moderate, MRI-appropriate training data transformations.
    
    Augmentations:
    - Resize to target dimension (256x256)
    - Random Horizontal Flip (p=0.5)
    - Small Random Rotation (degrees=10)
    - Convert to Tensor [0.0, 1.0]
    - ImageNet Normalization
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def get_test_transforms(
    image_size: Tuple[int, int] = IMAGE_SIZE,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> transforms.Compose:
    """
    Returns deterministic validation/test preprocessing transformations without augmentation.
    
    Pipeline:
    - Resize to target dimension (256x256)
    - Convert to Tensor [0.0, 1.0]
    - ImageNet Normalization
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
