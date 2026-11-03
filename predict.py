import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2


def _save_prediction(mask, image_path, save_path, class_names, multilabel):
    output_path = Path(save_path)
    if multilabel and output_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Multilabel predictions must be saved as a multi-band .tif file.")

    if not multilabel:
        if mask.ndim != 2:
            raise ValueError("Binary predictions must have a two-dimensional mask.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() in {".tif", ".tiff"}:
            import rasterio

            profile = {
                "driver": "GTiff",
                "height": mask.shape[0],
                "width": mask.shape[1],
                "count": 1,
                "dtype": "uint8",
                "compress": "deflate",
            }
            if Path(image_path).suffix.lower() in {".tif", ".tiff"}:
                with rasterio.open(image_path) as source:
                    if source.crs is not None:
                        profile["crs"] = source.crs
                    if source.transform is not None:
                        profile["transform"] = source.transform
            with rasterio.open(output_path, "w", **profile) as destination:
                destination.write(mask, 1)
        elif not cv2.imwrite(str(output_path), mask):
            raise OSError(f"Unable to save predicted mask: {output_path}")
    else:
        import rasterio

        profile = {
            "driver": "GTiff",
            "height": mask.shape[0],
            "width": mask.shape[1],
            "count": mask.shape[2],
            "dtype": "uint8",
            "compress": "deflate",
        }
        if Path(image_path).suffix.lower() in {".tif", ".tiff"}:
            with rasterio.open(image_path) as source:
                if source.crs is not None:
                    profile["crs"] = source.crs
                if source.transform is not None:
                    profile["transform"] = source.transform
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as destination:
            destination.write(mask.transpose(2, 0, 1))
            for index, class_name in enumerate(class_names, start=1):
                destination.set_band_description(index, class_name)
    print(f"Saved predicted mask to {output_path}")


def predict(
    model,
    image_path,
    device,
    save_path=None,
    show_visualization=False,
    save_visualization=False,
    threshold=0.5,
    multilabel=None,
    class_names=None
):
    model.eval()

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Unable to read image: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    transform = Compose([
        Resize(256, 256),
        Normalize(),
        ToTensorV2()
    ])
    input_tensor = transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        if output.ndim != 4 or output.shape[0] != 1:
            raise ValueError(f"Expected model output shaped [1, classes, height, width], got {output.shape}")
        probabilities = torch.sigmoid(output).cpu().numpy()[0]

    channel_count = probabilities.shape[0]
    if multilabel is None:
        multilabel = channel_count > 1
    if multilabel and channel_count < 1:
        raise ValueError("A multilabel model must produce at least one output channel.")
    if not multilabel and channel_count != 1:
        raise ValueError(
            f"Binary prediction requires one output channel; model produced {channel_count}."
        )
    if class_names is None:
        class_names = [f"class_{index}" for index in range(channel_count)]
    if len(class_names) != channel_count:
        raise ValueError(
            f"Received {len(class_names)} class names for {channel_count} model output channels."
        )

    probability_masks = np.stack([
        cv2.resize(probability, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        for probability in probabilities
    ], axis=-1)
    smoothed_masks = np.stack([
        cv2.GaussianBlur(probability_masks[:, :, index], (5, 5), 0)
        for index in range(channel_count)
    ], axis=-1)
    binary_masks = (smoothed_masks > threshold).astype(np.uint8) * 255

    if not multilabel:
        binary_masks = binary_masks[:, :, 0]

    if save_path:
        _save_prediction(binary_masks, image_path, save_path, class_names, multilabel)

    if show_visualization or save_visualization:
        colors = plt.get_cmap("tab10")(np.arange(channel_count))[:, :3]
        overlay = image.astype(np.float32).copy()
        masks_for_overlay = binary_masks[:, :, np.newaxis] if binary_masks.ndim == 2 else binary_masks
        for index, color in enumerate(colors):
            selected = masks_for_overlay[:, :, index] > 0
            overlay[selected] = (
                0.55 * overlay[selected] + 0.45 * (color * 255)
            )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        figure, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(image)
        axes[0].set_title("Original image")
        axes[1].imshow(overlay)
        axes[1].set_title("Predicted classes: " + ", ".join(class_names))
        for axis in axes:
            axis.axis("off")

        if save_visualization:
            if save_path is None:
                plt.close(figure)
                raise ValueError("save_path is required when save_visualization=True.")
            visualization_path = f"{os.path.splitext(save_path)[0]}_visualization_overlay.png"
            figure.savefig(visualization_path, bbox_inches="tight")
            print(f"Saved visualization overlay to {visualization_path}")
        if show_visualization:
            plt.show()
        plt.close(figure)

    return binary_masks
