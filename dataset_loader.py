import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FlexibleDataset(Dataset):
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(
        self,
        image_dir,
        mask_dir,
        transform=None,
        multilabel=False,
        classes_path=None
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.multilabel = multilabel
        self.class_names = self._load_class_names(classes_path) if multilabel else None

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory does not exist: {self.mask_dir}")

        self.image_paths = self._index_files(self.image_dir, "image")
        self.mask_paths = self._index_files(self.mask_dir, "mask")
        missing_masks = sorted(set(self.image_paths) - set(self.mask_paths))
        missing_images = sorted(set(self.mask_paths) - set(self.image_paths))
        if missing_masks or missing_images:
            details = []
            if missing_masks:
                details.append(f"images without masks: {missing_masks[:5]}")
            if missing_images:
                details.append(f"masks without images: {missing_images[:5]}")
            raise ValueError(f"Unpaired dataset files in {self.image_dir.parent}: {'; '.join(details)}")

        self.common_stems = sorted(self.image_paths)
        if not self.common_stems:
            raise ValueError(f"No paired image-mask files found in {self.image_dir.parent}")

    def _load_class_names(self, classes_path):
        if classes_path is None:
            classes_path = self.mask_dir.parent.parent / "classes.json"
        classes_path = Path(classes_path)
        if not classes_path.is_file():
            raise FileNotFoundError(
                f"Multilabel dataset requires class metadata: {classes_path}"
            )

        with classes_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid class metadata in {classes_path}")
        class_mapping = metadata.get("class_to_channel", metadata)
        if not isinstance(class_mapping, dict) or not class_mapping:
            raise ValueError(f"Invalid class mapping in {classes_path}")

        channel_to_class = {}
        for class_name, channel in class_mapping.items():
            if not isinstance(channel, int) or isinstance(channel, bool) or channel < 0:
                raise ValueError(f"Invalid channel index for class '{class_name}'")
            if channel in channel_to_class:
                raise ValueError(f"Duplicate channel index {channel} in {classes_path}")
            channel_to_class[channel] = class_name

        if sorted(channel_to_class) != list(range(len(channel_to_class))):
            raise ValueError(f"Class channel indexes must be contiguous from zero in {classes_path}")
        return [channel_to_class[index] for index in range(len(channel_to_class))]

    @staticmethod
    def _index_files(directory, kind):
        files = {}
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in FlexibleDataset.VALID_EXTENSIONS:
                continue
            if path.stem in files:
                raise ValueError(
                    f"Multiple {kind} files share the same stem '{path.stem}' in {directory}"
                )
            files[path.stem] = path
        return files

    def __len__(self):
        return len(self.common_stems)

    def __getitem__(self, idx):
        stem = self.common_stems[idx]
        image_path = self.image_paths[stem]
        mask_path = self.mask_paths[stem]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask_path.suffix.lower() in {".tif", ".tiff"}:
            import rasterio

            with rasterio.open(mask_path) as src:
                mask_data = src.read()
                mask_band_names = src.descriptions
        else:
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask_image is None:
                raise ValueError(f"Unable to read mask: {mask_path}")
            mask_band_names = None
            if mask_image.ndim == 3:
                if self.multilabel:
                    mask_data = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
                    mask_data = np.moveaxis(mask_data, -1, 0)
                else:
                    mask_data = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)[np.newaxis, ...]
            else:
                mask_data = mask_image[np.newaxis, ...]

        if mask_data.ndim != 3 or mask_data.shape[0] == 0:
            raise ValueError(f"Invalid mask dimensions in {mask_path}: {mask_data.shape}")
        if image.shape[:2] != mask_data.shape[1:]:
            raise ValueError(
                f"Image and mask dimensions differ for '{stem}': "
                f"{image.shape[:2]} vs {mask_data.shape[1:]}"
            )

        if self.multilabel:
            if mask_data.shape[0] != len(self.class_names):
                raise ValueError(
                    f"Mask '{mask_path}' has {mask_data.shape[0]} bands, but "
                    f"{len(self.class_names)} classes are defined in metadata."
                )
            if mask_band_names and all(mask_band_names) and list(mask_band_names) != self.class_names:
                raise ValueError(
                    f"Mask band order in '{mask_path}' does not match class metadata: "
                    f"{mask_band_names} != {self.class_names}"
                )
            mask = np.ascontiguousarray(
                (mask_data > 0).astype(np.float32).transpose(1, 2, 0)
            )
        else:
            if mask_data.shape[0] != 1:
                raise ValueError(
                    f"Binary dataset mask must have one band; found "
                    f"{mask_data.shape[0]} in {mask_path}"
                )
            mask = (mask_data[0] > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            image_height, image_width = image.shape[-2:]
            if self.multilabel:
                channel_count = len(self.class_names)
                if mask.ndim != 3:
                    raise ValueError(
                        f"Expected a three-dimensional multilabel mask after transforms, "
                        f"got shape {tuple(mask.shape)} for '{mask_path}'."
                    )
                if tuple(mask.shape) == (
                    image_height,
                    image_width,
                    channel_count
                ):
                    mask = mask.permute(2, 0, 1)
                elif tuple(mask.shape) != (
                    channel_count,
                    image_height,
                    image_width
                ):
                    raise ValueError(
                        f"Transformed mask shape {tuple(mask.shape)} does not match "
                        f"image shape {(image_height, image_width)} and "
                        f"{channel_count} class channels for '{mask_path}'."
                    )
                mask = mask.contiguous()
            else:
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                if tuple(mask.shape) != (1, image_height, image_width):
                    raise ValueError(
                        f"Transformed binary mask has unexpected shape "
                        f"{tuple(mask.shape)} for '{mask_path}'."
                    )
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            if mask.ndim == 2:
                mask = mask[np.newaxis, ...]
            else:
                mask = mask.transpose(2, 0, 1)
            mask = torch.from_numpy(mask)

        return image, mask

def get_transforms(training=True):
    transforms = [A.Resize(256, 256)]
    if training:
        transforms.append(A.HorizontalFlip(p=0.5))
    transforms.extend([
        A.Normalize(),
        ToTensorV2()
    ])
    return A.Compose(transforms)