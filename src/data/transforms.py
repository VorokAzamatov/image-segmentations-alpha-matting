import albumentations as A

from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=512):
    """Transforms for train"""
    return A.Compose([
        A.Resize(img_size, img_size),

        A.HorizontalFlip(p=0.5),                     
        A.Affine(
            translate_percent=(0.05, 0.05), scale=(0.9, 1.1), 
            rotate=(-15, 15), p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.7)], p=0.5),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_val_transforms(img_size=512):
    """Transforms for validation/test"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])