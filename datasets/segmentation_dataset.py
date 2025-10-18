import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment

        # Filter missing masks
        valid_image_paths, valid_mask_paths = [], []
        for img, msk in zip(self.image_paths, self.mask_paths):
            if os.path.exists(msk):
                valid_image_paths.append(img)
                valid_mask_paths.append(msk)
            else:
                print(f"⚠️ Skipping {img}, mask not found: {msk}")

        self.image_paths = valid_image_paths
        self.mask_paths = valid_mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        if self.augment:
            augmented = self.augment(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        else:
            image = ToTensorV2()(image=image)["image"]
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


def get_train_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.8, 1.2), translate_percent=(0.0, 0.1), rotate=(-30, 30), p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])
