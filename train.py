import torch.nn as nn
import segmentation_models_pytorch as smp

def get_loss():
    return nn.BCEWithLogitsLoss()

from tqdm import tqdm
import time
import numpy as np

def train_one_epoch(model, loader, validation_loader, optimizer, loss_fn, device, epoch_num=None, total_epochs=None):
    # === Training Loop ===
    model.train()
    # total_loss = 0
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
        # optimizer.step()
        
        # grad norm print occasionally
        if batch_idx % 100 == 0:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # total_loss += loss.item()
        # avg_loss = total_loss / (batch_idx + 1)

        all_train_losses.append(loss.item())
        avg_loss = np.mean(all_train_losses)

        elapsed = time.time() - start_time
        eta = (elapsed / (batch_idx + 1)) * (num_batches - batch_idx - 1)

        progress_bar.set_postfix({
            "Batch": f"{batch_idx+1}/{num_batches}",
            "BatchLoss": f"{loss.item():.4f}",
            "GradNorm": f"{total_norm:.4f}",
            "AvgLoss": f"{avg_loss:.4f}",
            "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
            "ETA": f"{int(eta)}s",
        })

    # === Validation Loop ===
    model.eval()
    # total_validation_loss = 0
    all_validation_losses = []
    num_batches_val = len(validation_loader)
    
    progress_bar_val = tqdm(validation_loader, desc=f"[Validation]", ncols=150)

    for val_batch_idx, (val_images, val_masks) in enumerate(progress_bar_val):
        val_images = val_images.to(device)
        val_masks = val_masks.to(device)

        with torch.no_grad():
            val_outputs = model(val_images)
            val_loss = loss_fn(val_outputs, val_masks)

        # total_validation_loss += val_loss.item()
        # avg_val_loss = total_validation_loss / (val_batch_idx + 1)

        all_validation_losses.append(val_loss.item())
        avg_val_loss = np.mean(all_validation_losses)

        elapsed = time.time() - start_time
        eta = (elapsed / (val_batch_idx + 1)) * (num_batches_val - val_batch_idx - 1)

        progress_bar_val.set_postfix({
            "Batch": f"{val_batch_idx+1}/{num_batches_val}",
            "BatchValLoss": f"{val_loss.item():.4f}",
            "AvgValLoss": f"{avg_val_loss:.4f}",
            "ETA": f"{int(eta)}s"
        })

    end_time = time.time()
    epoch_time = int(end_time - start_time)
    print(f"🟩 Epoch {epoch_num}/{total_epochs} completed in {epoch_time}s")

    return avg_loss, avg_val_loss

from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch
import os
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def run_training(
    dataset_dir, 
    batch_size=4, num_epochs=20, lr=1e-4,
    img_resize=512,
    save_path="unet_seg.pth",
    resume_mode=None,   
    resume_path=None,
    classes=None,     
    patience=5,
    model_type="unet",
    is_multilabel=False          
):
    """_summary_

    Args:
        model_type (str, optional):Defaults to "unet". Options: "unet", "upernet"
    """
    from dataset_loader import FlexibleDataset, MultiLabelFlexibleDataset, get_transforms
    from model import get_model, get_model_upernet
    from loss_custom import get_loss_v2 as custom_loss_v2
    from loss_custom import get_loss_v2_multilabel as custom_loss_v2_multilabel
    # from loss_custom import get_loss_v2_multiclass as custom_loss_v2_multiclass

    transform = get_transforms(img_resize=img_resize)
    
    # Dataset & DataLoader
    if is_multilabel:
        if classes is None:
            classes = []
            for path in os.listdir(dataset_dir):
                if os.path.isdir(os.path.join(dataset_dir, path)):
                    classes.append(path)
        
        num_features = len(classes) + 1 # +1 for background
        
        # Init model
        model = get_model(num_classes=len(classes)) if model_type == "unet" else get_model_upernet(num_classes=len(classes))
        loss_fn = custom_loss_v2_multilabel()
        transform = get_transforms(img_resize=img_resize)
        
        train_dataset = MultiLabelFlexibleDataset(base_dir=dataset_dir, class_names=classes, split="train", transform = transform)
        val_dataset = MultiLabelFlexibleDataset(base_dir=dataset_dir, class_names=classes, split="val", transform = transform)
    else:
        # === Split Dataset ===
        train_data = os.path.join(dataset_dir, "train")
        train_image_dir = os.path.join(train_data, "images")
        train_mask_dir = os.path.join(train_data, "labels")
        
        val_data = os.path.join(dataset_dir, "val")
        val_image_dir = os.path.join(val_data, "images")
        val_mask_dir = os.path.join(val_data, "labels")
        
        if classes is None:
            classes = ["Unknown"]
        
        num_features = len(classes) + 1 # +1 for background

        # Init model
        model = get_model() if model_type == "unet" else get_model_upernet()
        # loss_fn = get_loss()
        loss_fn = custom_loss_v2()
        transform = get_transforms(img_resize=img_resize)

        # Dataset & DataLoader
        train_dataset = FlexibleDataset(train_image_dir, train_mask_dir, transform)
        val_dataset = FlexibleDataset(val_image_dir, val_mask_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    train_losses, val_losses = [], []
    epoch_listed = []

    patience_counter = 0

    # Resume / Fine-tune
    if resume_mode in ["resume", "finetune"] and resume_path and os.path.exists(resume_path):
        print(f"📂 Loading checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        if resume_mode == "resume":
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_loss = checkpoint.get("best_loss", float("inf"))
            print(f"🔄 Resuming training from epoch {start_epoch + 1}...")

        elif resume_mode == "finetune":
            last_layer = model.out_conv
            out_features = last_layer.out_channels
            if out_features != num_features:
                print(f"⚠️ Output classes differ ({out_features} → {num_features}), replacing last layer...")
                import torch.nn as nn
                model.out_conv = nn.Conv2d(last_layer.in_channels, num_features, kernel_size=1)
                model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)  # reset optimizer
            print("🎯 Fine-tuning from pretrained weights...")
    else:
        start_epoch = 0
        best_val_loss = float("inf")
        print("🚀 Starting training from scratch...")

    # === Training Loop ===
    for epoch in range(start_epoch, num_epochs):
        print(f"\n🚀 Epoch {epoch+1}/{num_epochs}")

        avg_train_loss, avg_val_loss = train_one_epoch(
            model, train_loader, val_loader, optimizer, loss_fn, device, epoch+1, num_epochs
        )

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epoch_listed.append(epoch+1)
        print(f"📉 Train Loss: {avg_train_loss:.4f}")
        print(f"📉 Val Loss: {avg_val_loss:.4f}")
        
        # scheduler step (ReduceLROnPlateau)
        scheduler.step(avg_val_loss)

        # ---- Early Stopping ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_val_loss,
                "classes": classes,
                "model_type": model_type
            }, save_path)
            print(f"💾 Model saved (Val Loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("🛑 Early stopping triggered.")
                break
    
    # === Plot Learning Curve ===
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_listed, train_losses, label="Train Loss", marker="o")
    plt.plot(epoch_listed, val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    save_path_dir = os.path.dirname(save_path)
    plt.savefig(os.path.join(save_path_dir, "learning_curve.png"))
    print("📊 Learning curve saved")
            
def load_model_with_fallback(model, path, device):
    try:
        print(f"📂 Trying to load state_dict from {path}...")
        state_dict = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print("✅ Loaded using pure state_dict.")
        return model

    except Exception as e:
        print(f"⚠ Failed to load as state_dict")
        print("🔄 Trying to load as checkpoint...")

        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            print("✅ Loaded using checkpoint.")
            return model
        except Exception as e2:
            print(f"❌ Failed to load as checkpoint")
            raise RuntimeError("Model could not be loaded in either format.")
        
def load_model(path, device):
    print("🔄 Trying to load model path as checkpoint...")
    from model import get_model, get_model_upernet
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model_type = checkpoint.get("model_type", None)
    model_classes = checkpoint.get("classes", None)
    
    print("model classes: ", model_classes)
    if model_type == "unet" or model_type == None:
        if(model_classes == None):
            model = get_model()
        else:
            model = get_model(len(model_classes))
    elif model_type == "upernet":
        if(model_classes == None):
            model = get_model_upernet()
        else:
            model = get_model_upernet(len(model_classes))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✅ Loaded using checkpoint.")
        return model, checkpoint
    except Exception as e2:
        print(f"❌ Failed to load as checkpoint")
        raise RuntimeError("Model could not be loaded in either format.")