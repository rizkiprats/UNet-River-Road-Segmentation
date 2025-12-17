from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

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

class MultiLabelFlexibleDataset(Dataset):
    def __init__(self, base_dir, class_names=('road', 'river'), split='train', transform=None):
        self.base_dir = Path(base_dir)
        self.class_names = class_names
        self.transform = transform
        self.samples = []

        # Buat list semua gambar di semua kelas
        for cls_idx, cls in enumerate(class_names):
            image_dir = self.base_dir / cls / split / "images"
            mask_dir  = self.base_dir / cls / split / "labels"

            for img_path in sorted(image_dir.glob("*")):
                mask_path = mask_dir / img_path.name
                if mask_path.exists():
                    self.samples.append({
                        "image_path": str(img_path),
                        "mask_path": str(mask_path),
                        "class_idx": cls_idx
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_single = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
        mask_single = (mask_single > 0).astype(np.float32)

        H, W = mask_single.shape
        mask_multi = np.zeros((len(self.class_names), H, W), dtype=np.float32)
        mask_multi[sample["class_idx"]] = mask_single

        # Apply augmentasi
        if self.transform:
            augmented = self.transform(image=image, mask=np.moveaxis(mask_multi, 0, -1))
            image = augmented["image"]
            mask = augmented["mask"].permute(2, 0, 1)
        else:
            mask = torch.from_numpy(mask_multi)

        return image, mask
    
def get_transforms(img_resize=512):
    return A.Compose([
        A.Resize(img_resize, img_resize),
        # A.HorizontalFlip(p=0.5),
        # A.LongestMaxSize(max_size=img_resize),
        # A.PadIfNeeded(img_resize, img_resize, border_mode=0, value=0),
        A.Normalize(),
        ToTensorV2()
    ])