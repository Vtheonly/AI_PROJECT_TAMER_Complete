import albumentations as A
import cv2

def get_train_augmentation(height: int, width: int) -> A.Compose:
    return A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, 
                           border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.ElasticTransform(alpha=30, sigma=5, border_mode=cv2.BORDER_CONSTANT, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.CoarseDropout(max_holes=3,
                        hole_height_range=(1, max(2, int(height * 0.1))),
                        hole_width_range=(1, max(2, int(width * 0.1))),
                        p=0.1),
    ])

def get_val_augmentation() -> A.Compose:
    return A.Compose([])