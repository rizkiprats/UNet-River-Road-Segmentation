import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_loss():
    return nn.BCEWithLogitsLoss()


def evaluate_model(model, loader, loss_fn, device, threshold=0.5, class_names=None):
    if len(loader.dataset) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    model.eval()
    total_loss = 0.0
    total_samples = 0
    true_positive = None
    false_positive = None
    false_negative = None

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            batch_size = images.shape[0]
            total_loss += loss_fn(logits, masks).item() * batch_size
            total_samples += batch_size

            predictions = torch.sigmoid(logits) >= threshold
            targets = masks >= 0.5
            batch_counts = (
                torch.logical_and(predictions, targets).sum(dim=(0, 2, 3)).cpu().numpy(),
                torch.logical_and(predictions, ~targets).sum(dim=(0, 2, 3)).cpu().numpy(),
                torch.logical_and(~predictions, targets).sum(dim=(0, 2, 3)).cpu().numpy(),
            )
            if true_positive is None:
                true_positive, false_positive, false_negative = (
                    np.zeros_like(counts, dtype=np.int64) for counts in batch_counts
                )
            true_positive += batch_counts[0]
            false_positive += batch_counts[1]
            false_negative += batch_counts[2]

    channel_count = len(true_positive)
    if class_names is None:
        class_names = [f"class_{index}" for index in range(channel_count)]
    if len(class_names) != channel_count:
        raise ValueError(
            f"Model produced {channel_count} channels but received "
            f"{len(class_names)} class names."
        )

    per_class = {}
    for index, class_name in enumerate(class_names):
        tp = int(true_positive[index])
        fp = int(false_positive[index])
        fn = int(false_negative[index])
        dice_denominator = 2 * tp + fp + fn
        iou_denominator = tp + fp + fn
        precision_denominator = tp + fp
        recall_denominator = tp + fn
        per_class[class_name] = {
            "dice": 1.0 if dice_denominator == 0 else 2 * tp / dice_denominator,
            "iou": 1.0 if iou_denominator == 0 else tp / iou_denominator,
            "precision": 0.0 if precision_denominator == 0 else tp / precision_denominator,
            "recall": 0.0 if recall_denominator == 0 else tp / recall_denominator,
        }

    metrics = {
        metric: float(np.mean([values[metric] for values in per_class.values()]))
        for metric in ("dice", "iou", "precision", "recall")
    }
    metrics["per_class"] = per_class
    return total_loss / total_samples, metrics


def train_one_epoch(
    model,
    loader,
    validation_loader,
    optimizer,
    loss_fn,
    device,
    epoch_num=None,
    total_epochs=None,
    class_names=None
):
    if len(loader.dataset) == 0:
        raise ValueError("Cannot train on an empty dataset.")

    model.train()
    all_train_losses = []
    num_batches = len(loader)

    start_time = time.time()
    progress_bar = tqdm(loader, desc=f"[Epoch {epoch_num}/{total_epochs}]", ncols=150)

    for batch_idx, (images, masks) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        all_train_losses.append(loss.item())
        avg_loss = np.mean(all_train_losses)

        elapsed = time.time() - start_time
        eta = (elapsed / (batch_idx + 1)) * (num_batches - batch_idx - 1)

        progress_bar.set_postfix({
            "Batch": f"{batch_idx+1}/{num_batches}",
            "Loss": f"{loss.item():.4f}",
            "AvgLoss": f"{avg_loss:.4f}",
            "ETA": f"{int(eta)}s"
        })

    avg_val_loss, val_metrics = evaluate_model(
        model,
        validation_loader,
        loss_fn,
        device,
        class_names=class_names
    )

    end_time = time.time()
    epoch_time = int(end_time - start_time)
    print(f"Epoch {epoch_num}/{total_epochs} completed in {epoch_time}s")

    return avg_loss, avg_val_loss, val_metrics


def run_training(
    dataset_dir, 
    batch_size=4, num_epochs=20, lr=1e-4,
    save_path="unet_seg.pth",
    resume_mode=None,   
    resume_path=None,
    num_classes=None,
    patience=5,
    model_type="unet",
    multilabel=False
):
    """Train a binary or multi-label model from a dataset with train/val splits."""
    from dataset_loader import FlexibleDataset, get_transforms
    from model import get_model, get_model_upernet

    if model_type not in {"unet", "upernet"}:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose 'unet' or 'upernet'.")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be greater than zero.")

    train_data = os.path.join(dataset_dir, "train")
    train_image_dir = os.path.join(train_data, "images")
    train_mask_dir = os.path.join(train_data, "labels")
    
    val_data = os.path.join(dataset_dir, "val")
    val_image_dir = os.path.join(val_data, "images")
    val_mask_dir = os.path.join(val_data, "labels")

    loss_fn = get_loss()

    classes_path = Path(dataset_dir) / "classes.json" if multilabel else None
    train_dataset = FlexibleDataset(
        train_image_dir,
        train_mask_dir,
        get_transforms(training=True),
        multilabel=multilabel,
        classes_path=classes_path
    )
    val_dataset = FlexibleDataset(
        val_image_dir,
        val_mask_dir,
        get_transforms(training=False),
        multilabel=multilabel,
        classes_path=classes_path
    )
    class_names = train_dataset.class_names if multilabel else ["foreground"]
    output_channels = len(class_names)
    if num_classes is not None and num_classes != output_channels:
        raise ValueError(
            f"num_classes={num_classes}, but dataset requires {output_channels} "
            f"output channel(s): {class_names}"
        )
    if not multilabel and output_channels != 1:
        raise ValueError("Binary mode must have exactly one output channel.")

    model_factory = get_model if model_type == "unet" else get_model_upernet
    model = model_factory(num_classes=output_channels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = None
    test_image_dir = Path(dataset_dir) / "test" / "images"
    test_mask_dir = Path(dataset_dir) / "test" / "labels"
    test_images_exist = test_image_dir.is_dir() and any(
        path.is_file() for path in test_image_dir.iterdir()
    )
    test_masks_exist = test_mask_dir.is_dir() and any(
        path.is_file() for path in test_mask_dir.iterdir()
    )
    if test_images_exist or test_masks_exist:
        test_dataset = FlexibleDataset(
            test_image_dir,
            test_mask_dir,
            get_transforms(training=False),
            multilabel=multilabel,
            classes_path=classes_path
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    train_losses, val_losses = [], []
    epoch_listed = []

    patience_counter = 0

    # Resume / Fine-tune
    if resume_mode in ["resume", "finetune"] and resume_path and os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("multilabel", False) != multilabel:
            raise ValueError("Checkpoint binary/multilabel mode does not match the dataset.")
        checkpoint_classes = checkpoint.get("class_names")
        if checkpoint_classes is not None and checkpoint_classes != class_names:
            raise ValueError(
                f"Checkpoint class order {checkpoint_classes} does not match "
                f"dataset class order {class_names}."
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        if resume_mode == "resume":
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_loss = checkpoint.get("best_loss", float("inf"))
            print(f"Resuming training from epoch {start_epoch + 1}...")

        elif resume_mode == "finetune":
            optimizer = optim.Adam(model.parameters(), lr=lr)
            print("Fine-tuning from pretrained weights...")
    else:
        start_epoch = 0
        best_val_loss = float("inf")
        print("Starting training from scratch...")

    # === Training Loop ===
    for epoch in range(start_epoch, num_epochs):
        print(f"\n Epoch {epoch+1}/{num_epochs}")

        avg_train_loss, avg_val_loss, val_metrics = train_one_epoch(
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_fn,
            device,
            epoch+1,
            num_epochs,
            class_names=class_names
        )

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epoch_listed.append(epoch+1)
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(
            "Val metrics: "
            f"Dice={val_metrics['dice']:.4f}, IoU={val_metrics['iou']:.4f}, "
            f"Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}"
        )
        for class_name, class_metrics in val_metrics["per_class"].items():
            print(
                f"  {class_name}: Dice={class_metrics['dice']:.4f}, "
                f"IoU={class_metrics['iou']:.4f}, "
                f"Precision={class_metrics['precision']:.4f}, "
                f"Recall={class_metrics['recall']:.4f}"
            )

        # ---- Early Stopping ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_val_loss,
                "val_metrics": val_metrics,
                "threshold": 0.5,
                "multilabel": multilabel,
                "class_names": class_names
            }, save_path)
            print(f"Model saved (best so far, Val Loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    if test_loader is not None:
        test_loss, test_metrics = evaluate_model(
            model,
            test_loader,
            loss_fn,
            device,
            class_names=class_names
        )
        print(f"\nHeld-out test loss: {test_loss:.4f}")
        print(
            "Test metrics: "
            f"Dice={test_metrics['dice']:.4f}, IoU={test_metrics['iou']:.4f}, "
            f"Precision={test_metrics['precision']:.4f}, Recall={test_metrics['recall']:.4f}"
        )
        for class_name, class_metrics in test_metrics["per_class"].items():
            print(
                f"  {class_name}: Dice={class_metrics['dice']:.4f}, "
                f"IoU={class_metrics['iou']:.4f}, "
                f"Precision={class_metrics['precision']:.4f}, "
                f"Recall={class_metrics['recall']:.4f}"
            )
    else:
        print("\nNo held-out test samples found; test metrics were not calculated.")

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_listed, train_losses, label="Train Loss", marker="o")
    plt.plot(epoch_listed, val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    save_path_dir = os.path.dirname(save_path)
    curve_name = f"{Path(save_path).stem}_learning_curve.png"
    curve_path = os.path.join(save_path_dir, curve_name)
    plt.savefig(curve_path)
    print(f"Learning curve saved to {curve_path}")


def run_training_for_classes(dataset_root, class_names="all", **training_options):
    """Train one binary model per selected class dataset.

    ``class_names`` may be ``"all"``, one class name, or a list of class names.
    Each class receives its own checkpoint and learning curve.
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if "dataset_dir" in training_options:
        raise ValueError("Pass the dataset root as the first argument, not in training_options.")

    available_classes = sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir()
        and (entry / "train").is_dir()
        and (entry / "val").is_dir()
    )
    if not available_classes:
        raise ValueError(f"No class datasets with train/ and val/ splits found in {root}")

    if isinstance(class_names, str):
        selected_classes = available_classes if class_names.lower() == "all" else [class_names]
    else:
        selected_classes = list(dict.fromkeys(class_names))
        if not selected_classes:
            raise ValueError("Select 'all' or at least one class name.")

    unknown_classes = sorted(set(selected_classes) - set(available_classes))
    if unknown_classes:
        raise ValueError(
            f"Unknown class dataset(s): {unknown_classes}. "
            f"Available classes: {available_classes}"
        )

    resume_path = training_options.get("resume_path")
    if resume_path and len(selected_classes) > 1:
        raise ValueError(
            "resume_path can be used for one class at a time; select one class "
            "or resume each class separately."
        )

    base_save_path = Path(training_options.get("save_path", "unet_seg.pth"))
    results = {}
    for class_name in selected_classes:
        class_save_path = base_save_path.with_name(
            f"{base_save_path.stem}_{class_name}{base_save_path.suffix}"
        )
        options = dict(training_options)
        options["save_path"] = str(class_save_path)
        print(f"\nTraining class '{class_name}' from {root / class_name}")
        run_training(
            dataset_dir=str(root / class_name),
            **options
        )
        results[class_name] = str(class_save_path)

    return results


def load_model_with_fallback(
    model,
    path,
    device,
    multilabel=None,
    class_names=None
):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format in '{path}'.")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if multilabel is not None and checkpoint.get("multilabel") is not None:
            if checkpoint["multilabel"] != multilabel:
                raise ValueError(
                    "Checkpoint binary/multilabel mode does not match the requested model."
                )
        checkpoint_classes = checkpoint.get("class_names")
        if (
            class_names is not None
            and checkpoint_classes is not None
            and checkpoint_classes != class_names
        ):
            raise ValueError(
                f"Checkpoint class order {checkpoint_classes} does not match "
                f"requested class order {class_names}."
            )
    else:
        state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) for key in state_dict
    ):
        raise ValueError(f"No valid model state_dict found in '{path}'.")

    model.load_state_dict(state_dict)
    print(f"Loaded model weights from {path}.")
    return model