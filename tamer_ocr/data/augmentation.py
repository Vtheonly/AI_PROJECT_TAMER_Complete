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
        A.Affine(
            scale=(0.95, 1.05), 
            translate_percent=(-0.02, 0.02), 
            rotate=(-3, 3), 
            shear=(-2, 2),
            mode=cv2.BORDER_CONSTANT,
            cval=255, # Pad with white
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
        
        # FIX: Drop white holes instead of black holes
        # This simulates "missing" parts of strokes rather than "ink blots"
        A.CoarseDropout(
            max_holes=4,
            max_height=int(height * 0.1),
            max_width=int(width * 0.05),
            min_holes=1,
            fill_value=255, # Match white background
            p=0.2,
        ),
        
        # Simulating digital sensor noise
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
    ])

def get_val_augmentation() -> A.Compose:
    """No augmentation for validation to ensure consistent metrics."""
    return A.Compose([])