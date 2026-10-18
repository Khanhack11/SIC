"""Shared training/inference image contract, maintained by the data engineer."""

from functools import lru_cache
from PIL import ImageOps
from torchvision import transforms

from backend.config import MODEL_INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD


def orient_rgb(image):
    """Respect camera EXIF orientation before discarding metadata and alpha."""
    return ImageOps.exif_transpose(image).convert("RGB")


@lru_cache(maxsize=2)
def image_transform(training=False):
    operations = [transforms.Resize(MODEL_INPUT_SIZE, antialias=True)]
    if training:
        operations.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2),
        ])
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(operations)
