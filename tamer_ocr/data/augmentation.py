import albumentations as A
import cv2

def get_train_augmentation(height: int, width: int) -> A.Compose:
    """
    Training augmentation — applied to the 256x1024 padded image.
    
    Fixes:
    - CoarseDropout now uses fill_value=255 (white) to match math backgrounds.
    - Gentle affine transforms to prevent breaking LaTeX structural integrity.
    """
    return A.Compose([
        # Subtle geometric distortions
        A.ShiftScaleRotate(
            scale_limit=0.05,
            shift_limit=0.02,
            rotate_limit=3,
            border_mode=cv2.BORDER_CONSTANT,
            value=255, # Pad with white (cv2 convention for this transform)
            p=0.5
        ) if int(A.__version__.split('.')[0]) < 1 else A.Affine(
            scale=(0.95, 1.05),
            translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
            rotate=(-3, 3),
            cval=255,
            p=0.5
        ),
        
        # Subtle blurring to simulate low-res scans or handwriting bleed
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 3)),
            A.MedianBlur(blur_limit=3),
        ], p=0.2),
        
        # Simulating lighting/scanning variations
        A.RandomBrightnessContrast(
            brightness_limit=0.1, 
            contrast_limit=0.1, 
            p=0.3
        ),
        
        # Simulating "missing" parts of strokes
        A.CoarseDropout(
            max_holes=4,
            max_height=int(height * 0.1),
            max_width=int(width * 0.05),
            min_holes=1,
            fill_value=255, # Match white background
            p=0.2,
        ) if int(A.__version__.split('.')[0]) < 1 else A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(1, int(height * 0.1)),
            hole_width_range=(1, int(width * 0.05)),
            fill_value=255,
            p=0.2,
        ),
        
        # Simulating digital sensor noise
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.2) if int(A.__version__.split('.')[0]) < 1 else A.GaussianNoise(var_limit=(10.0, 30.0), p=0.2),
    ])

def get_val_augmentation() -> A.Compose:
    """No augmentation for validation to ensure consistent metrics."""
    return A.Compose([])