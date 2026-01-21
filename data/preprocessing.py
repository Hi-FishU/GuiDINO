import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2


def _resolve_hw(image_size):
    """Convert an int or tuple into a (H, W) pair."""
    if isinstance(image_size, (tuple, list)):
        if len(image_size) != 2:
            raise ValueError("image_size tuple must be (height, width)")
        return int(image_size[0]), int(image_size[1])
    return int(image_size), int(image_size)


def _zscore_image(image: np.ndarray, **kwargs) -> np.ndarray:
    image = image.astype(np.float32)
    if image.ndim == 2:
        mean = float(image.mean())
        std = float(image.std())
        return (image - mean) / (std + 1e-8)
    mean = image.mean(axis=(0, 1), keepdims=True)
    std = image.std(axis=(0, 1), keepdims=True)
    return (image - mean) / (std + 1e-8)

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=10, p=0.5),
        A.Resize(1024, 1024),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def get_val_transform():
    return A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


def get_segmentation_train_transform(image_size=512):
    height, width = _resolve_hw(image_size)
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.2),
        A.ElasticTransform(alpha=120, sigma=6, p=0.1),
        A.Affine(scale=(0.8, 1.2), rotate=(-30, 30), translate_percent=0.0, p=0.2),
        A.RandomGamma(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.15),
        A.GaussianBlur(blur_limit=(3, 5), p=0.05),
        A.Resize(height, width),
        A.Lambda(image=_zscore_image),
        ToTensorV2(),
    ])


def get_segmentation_val_transform(image_size=512):
    height, width = _resolve_hw(image_size)
    return A.Compose([
        A.Resize(height, width),
        A.Lambda(image=_zscore_image),
        ToTensorV2(),
    ])
