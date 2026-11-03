import os
import json
from pathlib import Path

import torch

from model import get_model
from predict import predict
from train import load_model_with_fallback, run_training


if __name__ == "__main__":
    dataset_dir = r"Datasets/Building_Tree_Dataset_Multilabel"

    train_model = True
    save_dir = "Model/Multilabel"
    save_path = os.path.join(save_dir, "unet_building_tree.pth")
    
    if train_model:
        os.makedirs(save_dir, exist_ok=True)
        run_training(
            dataset_dir=dataset_dir,
            multilabel=True,
            batch_size=8,
            num_epochs=100,
            lr=1e-4,
            save_path=save_path
        )

    with open(os.path.join(dataset_dir, "classes.json"), "r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError("classes.json must contain a JSON object.")

    class_mapping = metadata.get("class_to_channel", metadata)
    if not isinstance(class_mapping, dict) or not class_mapping:
        raise ValueError("classes.json must define at least one class and channel.")
    if any(
        not isinstance(class_name, str)
        or not isinstance(channel, int)
        or isinstance(channel, bool)
        or channel < 0
        for class_name, channel in class_mapping.items()
    ):
        raise ValueError("Class names must map to non-negative integer channels.")
    channel_indexes = list(class_mapping.values())
    if (
        len(set(channel_indexes)) != len(channel_indexes)
        or sorted(channel_indexes) != list(range(len(channel_indexes)))
    ):
        raise ValueError("Class channels must be unique and contiguous from zero.")

    class_names = [
        name for name, _ in sorted(class_mapping.items(), key=lambda item: item[1])
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=len(class_names)).to(device)
    load_model_with_fallback(
        model,
        save_path,
        device,
        multilabel=True,
        class_names=class_names
    )

    test_predict_dir = Path("Test Image/building_tree")
    if not test_predict_dir.is_dir():
        raise FileNotFoundError(
            f"Prediction input directory does not exist: {test_predict_dir}"
        )

    image_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    for image_path in sorted(test_predict_dir.iterdir()):
        if (
            image_path.is_file()
            and image_path.suffix.lower() in image_extensions
            and not any(
                marker in image_path.stem
                for marker in ("_predicted_mask", "_visualization_overlay")
            )
        ):
            predict(
                model,
                image_path=str(image_path),
                device=device,
                save_path=str(test_predict_dir / f"{image_path.stem}_predicted_mask.tif"),
                show_visualization=False,
                save_visualization=True,
                multilabel=True,
                class_names=class_names
            )