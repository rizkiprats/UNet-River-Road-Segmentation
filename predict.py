import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def predict(model, image_path, device, save_path=None, show_visualization=False, save_visualization=False, threshold=0.5):
    model.eval()
    
    # Baca gambar
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Simpan ukuran asli
    orig_h, orig_w = image.shape[:2]

    # Transform sama seperti training
    from albumentations import Compose, Resize, Normalize
    from albumentations.pytorch import ToTensorV2
    transform = Compose([
        Resize(256, 256),
        Normalize(),
        ToTensorV2()
    ])
    augmented = transform(image=image, mask=np.zeros((orig_h, orig_w)))
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    # Prediksi
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0]  # 256x256

    # Threshold
    # pred_mask_bin = (pred_mask > threshold).astype(np.uint8) * 255

    # Resize probability mask dan mask biner ke ukuran asli
    pred_mask_resized = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    # pred_mask_bin_resized = cv2.resize(pred_mask_bin, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) 
    
    # Threshold Ulang
    pred_mask_bin_resized = (pred_mask_resized > threshold).astype(np.uint8) * 255
    
    kernel = np.ones((3, 3), np.uint8)
    pred_mask_bin_resized = cv2.morphologyEx(pred_mask_bin_resized, cv2.MORPH_OPEN, kernel)
    pred_mask_bin_resized = cv2.morphologyEx(pred_mask_bin_resized, cv2.MORPH_CLOSE, kernel)
    
    blurred = cv2.GaussianBlur(pred_mask_resized, (5,5), 0)
    pred_mask_bin_resized = (blurred > threshold).astype(np.uint8) * 255

    # Simpan mask ke file
    if save_path:
        cv2.imwrite(save_path, pred_mask_bin_resized)
        print(f"✅ Saved predicted mask to {save_path}")

    # Visualisasi hasil
    if show_visualization:
        plt.figure(figsize=(12,5))
        plt.subplot(1,3,1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(pred_mask_resized, cmap="gray")  # sudah ukuran asli
        plt.title("Predicted Probability")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(image)
        plt.imshow(pred_mask_bin_resized, cmap="Reds", alpha=0.5)  # overlay ukuran pas
        plt.title("Overlay Mask")
        plt.axis("off")

        plt.show()
        
    if save_visualization:
        plt.figure(figsize=(16,9))
        plt.imshow(image)
        plt.imshow(pred_mask_bin_resized, cmap="Reds", alpha=0.5)  # overlay ukuran pas
        plt.title("Overlay Mask")
        plt.axis("off")
        extension_save = os.path.splitext(save_path)[1]
        basename_save = os.path.splitext(save_path)[0]
        save_viz_path = f"{basename_save}_visualization_overlay{extension_save}"
        plt.savefig(save_viz_path)
        print(f"✅ Saved visualization overlay to {save_viz_path}")

    return pred_mask_bin_resized