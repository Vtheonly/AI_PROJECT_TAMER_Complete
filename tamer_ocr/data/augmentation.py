import albumentations as A
import cv2

def get_train_augmentation(height: int, width: int) -> A.Compose:
    """Training augmentation — applied AFTER the 256x1024 padding."""
    return A.Compose([
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.03, 0.03), rotate=(-3, 3), p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 2),
            hole_height_range=(1, max(2, int(height * 0.05))),
            hole_width_range=(1, max(2, int(width * 0.05))),
            p=0.1,
        ),
    ])

def get_val_augmentation() -> A.Compose:
    """No augmentation for validation."""
    return A.Compose([])
