from albumentations.pytorch import ToTensorV2
import albumentations as A


def _resolve_hw(image_size):
    """Convert an int or tuple into a (H, W) pair."""
    if isinstance(image_size, (tuple, list)):
        if len(image_size) != 2:
            raise ValueError("image_size tuple must be (height, width)")
        return int(image_size[0]), int(image_size[1])
    return int(image_size), int(image_size)

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
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, mask_value=0, p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.05),
        A.Resize(height, width),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ])


def get_segmentation_val_transform(image_size=512):
    height, width = _resolve_hw(image_size)
    return A.Compose([
        A.Resize(height, width),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ])
