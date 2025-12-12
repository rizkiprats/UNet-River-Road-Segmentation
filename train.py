import torch.nn as nn

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
        optimizer.step()

        # total_loss += loss.item()
        # avg_loss = total_loss / (batch_idx + 1)

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
            "ValLoss": f"{val_loss.item():.4f}",
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
    save_path="unet_seg.pth",
    resume_mode=None,   
    resume_path=None,
    num_classes=2,     
    patience=5,
    model_type="unet"          
):
    """_summary_

    Args:
        model_type (str, optional):Defaults to "unet". Options: "unet", "upernet"
    """
    from dataset_loader import FlexibleDataset, get_transforms
    from model import get_model, get_model_upernet

    # === Split Dataset ===
    train_data = os.path.join(dataset_dir, "train")
    train_image_dir = os.path.join(train_data, "images")
    train_mask_dir = os.path.join(train_data, "labels")
    
    val_data = os.path.join(dataset_dir, "val")
    val_image_dir = os.path.join(val_data, "images")
    val_mask_dir = os.path.join(val_data, "labels")

    # Init model
    model = get_model() if model_type == "unet" else get_model_upernet()
    loss_fn = get_loss()
    transform = get_transforms()

    # Dataset & DataLoader
    train_dataset = FlexibleDataset(train_image_dir, train_mask_dir, transform)
    val_dataset = FlexibleDataset(val_image_dir, val_mask_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
            if out_features != num_classes:
                print(f"⚠️ Output classes differ ({out_features} → {num_classes}), replacing last layer...")
                import torch.nn as nn
                model.out_conv = nn.Conv2d(last_layer.in_channels, num_classes, kernel_size=1)
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

        # # ---- Train ----
        # model.train()
        # epoch_train_losses = []
        # for imgs, masks in train_loader:
        #     imgs, masks = imgs.to(device), masks.to(device)
        #     optimizer.zero_grad()
        #     outputs = model(imgs)
        #     loss = loss_fn(outputs, masks)
        #     loss.backward()
        #     optimizer.step()
        #     epoch_train_losses.append(loss.item())

        # avg_train_loss = np.mean(epoch_train_losses)
        # train_losses.append(avg_train_loss)
        # print(f"📉 Train Loss: {avg_train_loss:.4f}")

        # # ---- Validation ----
        # model.eval()
        # epoch_val_losses = []
        # with torch.no_grad():
        #     for imgs, masks in val_loader:
        #         imgs, masks = imgs.to(device), masks.to(device)
        #         outputs = model(imgs)
        #         loss = loss_fn(outputs, masks)
        #         epoch_val_losses.append(loss.item())

        # avg_val_loss = np.mean(epoch_val_losses)
        # val_losses.append(avg_val_loss)
        # print(f"✅ Val Loss: {avg_val_loss:.4f}")

        avg_train_loss, avg_val_loss = train_one_epoch(
            model, train_loader, val_loader, optimizer, loss_fn, device, epoch+1, num_epochs
        )

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epoch_listed.append(epoch+1)
        print(f"📉 Train Loss: {avg_train_loss:.4f}")
        print(f"📉 Val Loss: {avg_val_loss:.4f}")

        # ---- Early Stopping ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_val_loss
            }, save_path)
            print(f"💾 Model saved (best so far, Val Loss: {avg_val_loss:.4f})")
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