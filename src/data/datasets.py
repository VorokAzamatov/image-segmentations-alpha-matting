import cv2


import os

from torch.utils.data import Dataset




# -----------------------------
# DUTS Dataset
# -----------------------------
class DUTSdataset(Dataset):
    """
    DUTS dataset for segmentation training.
    data_path: path to DUTS
    transforms: transfroms
    pairs: list of (img_path, mask_path), if None, collects pairs automatically
    """
    def __init__(self, data_path, transforms=None, pairs=None):
        self.data_path = data_path
        self.transforms = transforms
        self.all_pairs = pairs if pairs is not None else self.collect_pairs()

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, index):
        img_path, mask_path = self.all_pairs[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        mask = mask.float() / 255.0
        mask = mask.unsqueeze(0)  # (1, H, W)

        return {'img': img, 'mask': mask}

    def collect_pairs(self):
        all_pairs = []
        image_dir = os.path.join(self.data_path, "DUTS-TR/DUTS-TR-Image")
        mask_dir = os.path.join(self.data_path, "DUTS-TR/DUTS-TR-Mask")

        for image_filename in sorted(os.listdir(image_dir)):
            img_path = os.path.join(image_dir, image_filename)
            mask_path = os.path.join(mask_dir, os.path.splitext(image_filename)[0] + ".png")

            if os.path.exists(mask_path):
                all_pairs.append((img_path, mask_path))

        return all_pairs


# -----------------------------
# AIM500 Dataset 
# -----------------------------
class AIM500_dataset(Dataset):
    """
    AIM500 dataset for fine-tuning.
    Structure: data_path/original (image), data_path/mask (mask)
    """
    def __init__(self, data_path, transforms=None, pairs=None):
        self.data_path = data_path
        self.transforms = transforms
        self.all_pairs = pairs if pairs is not None else self.collect_pairs()

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, index):
        img_path, mask_path = self.all_pairs[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        mask = mask.float() / 255.0
        mask = mask.unsqueeze(0)  # (1, H, W)

        return {'img': img, 'mask': mask}

    def collect_pairs(self):
        all_pairs = []
        image_dir = os.path.join(self.data_path, "original")
        mask_dir = os.path.join(self.data_path, "mask")

        for image_filename in sorted(os.listdir(image_dir)):
            img_path = os.path.join(image_dir, image_filename)
            mask_path = os.path.join(mask_dir, os.path.splitext(image_filename)[0] + ".png")

            if os.path.exists(mask_path):
                all_pairs.append((img_path, mask_path))

        return all_pairs