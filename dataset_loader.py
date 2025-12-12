from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FlexibleDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        self.image_paths = {p.stem: p for p in self.image_dir.iterdir() if p.is_file()}
        self.mask_paths = {p.stem: p for p in self.mask_dir.iterdir() if p.is_file()}
        self.common_stems = sorted(list(set(self.image_paths.keys()) & set(self.mask_paths.keys())))

    def __len__(self):
        return len(self.common_stems)

    def __getitem__(self, idx):
        stem = self.common_stems[idx]
        image_path = self.image_paths[stem]
        mask_path = self.mask_paths[stem]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)  # biner
        mask = np.expand_dims(mask, axis=0)

        if self.transform:
            augmented = self.transform(image=image, mask=mask[0])
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)

        return image, mask

def get_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])