import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def predict(model, image_path, device, save_path=None, show_visualization=False, save_visualization=False, threshold=0.5, img_resize=512):
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
        Resize(img_resize, img_resize),
        # A.LongestMaxSize(max_size=img_resize),
        # A.PadIfNeeded(img_resize, img_resize, border_mode=0, value=0),
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
        plt.close()

    return pred_mask_bin_resized


import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from matplotlib import colors

def predict_multilabel(model, image_path, device, class_names, save_path=None,
                       show_visualization=False, save_visualization=False,
                       img_resize=512, threshold=0.5):
    """
    Multi-label segmentation prediction for SMP model (e.g. U-Net)
    --------------------------------------------------------------
    Args:
        model : trained segmentation model
        image_path : path ke gambar input
        device : torch device ('cuda' atau 'cpu')
        class_names : list nama kelas (ex: ['road','river'])
        save_path : path untuk simpan hasil mask
        show_visualization : tampilkan hasil visualisasi
        save_visualization : simpan hasil overlay
        img_resize : ukuran resize input (harus sama seperti training)
        threshold : ambang batas aktivasi sigmoid (default=0.5)
    """

    import torch
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    from albumentations import Compose, Resize, Normalize
    from albumentations.pytorch import ToTensorV2
    import cv2, os
    from matplotlib import colors

    model.eval()

    # === 1️⃣ Baca gambar ===
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    # === 2️⃣ Transformasi seperti di training ===
    transform = Compose([
        Resize(img_resize, img_resize),
        Normalize(),
        ToTensorV2()
    ])
    augmented = transform(image=image)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    # === 3️⃣ Prediksi ===
    with torch.no_grad():
        output = model(input_tensor)           # [B, C, H, W]
        prob = torch.sigmoid(output)[0]        # [C, H, W], per channel sigmoid

    # === 4️⃣ Resize & threshold per kelas ===
    pred_masks_resized = []
    for class_idx in range(len(class_names)):
        prob_resized = cv2.resize(prob[class_idx].cpu().numpy(), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        pred_mask = (prob_resized > threshold).astype(np.uint8)
        pred_masks_resized.append(pred_mask)

    # === 5️⃣ Gabungkan semua mask menjadi satu visualisasi multi-channel ===
    combined_mask = np.zeros((orig_h, orig_w, len(class_names)), dtype=np.uint8)
    for i, mask in enumerate(pred_masks_resized):
        combined_mask[..., i] = mask

    # === 6️⃣ Simpan mask keseluruhan (jika diminta) ===
    if save_path:
        combined_mask_uint8 = (np.sum(combined_mask * (np.arange(1, len(class_names)+1)), axis=-1)).astype(np.uint8)
        cv2.imwrite(save_path, combined_mask_uint8 * 50)  # tambahkan warna skala
        print(f"✅ Saved multilabel combined mask to {save_path}")

    # === 7️⃣ Visualisasi utama ===
    if show_visualization or save_visualization:
        plt.figure(figsize=(10, 4))
        plt.subplot(1,3,1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(np.sum(combined_mask * np.arange(1, len(class_names)+1), axis=-1),
                   cmap="viridis", vmin=0, vmax=len(class_names))
        plt.title("Predicted MultiLabel Mask")
        plt.axis("off")

        plt.subplot(1,3,3)
        overlay = image.copy()
        for i, mask in enumerate(pred_masks_resized):
            color = np.array(plt.cm.tab10(i)[:3]) * 255
            overlay[mask.astype(bool)] = overlay[mask.astype(bool)] * 0.5 + color * 0.5
        plt.imshow(overlay.astype(np.uint8))
        plt.title("Overlay Prediction")
        plt.axis("off")

        if show_visualization:
            plt.show()

        if save_visualization:
            basename_save, ext = os.path.splitext(save_path)
            save_viz_path = f"{basename_save}_overlay{ext}"
            plt.savefig(save_viz_path, bbox_inches='tight', pad_inches=0)
            print(f"✅ Saved overlay visualization to {save_viz_path}")
            plt.close()

    # === 8️⃣ Visualisasi per kelas ===
    if show_visualization or save_visualization:
        for class_idx, class_name in enumerate(class_names):
            class_mask = pred_masks_resized[class_idx]

            if np.sum(class_mask) == 0:
                continue  # skip jika kelas ini tidak muncul

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(class_mask, cmap="gray")
            plt.title(f"Mask - {class_name}")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(image)
            plt.imshow(class_mask, cmap="spring", alpha=0.5)
            plt.title(f"Overlay - {class_name}")
            plt.axis("off")

            if show_visualization:
                plt.show()

            if save_visualization:
                class_viz_path = f"{basename_save}_{class_name}_overlay{ext}"
                plt.savefig(class_viz_path, bbox_inches='tight', pad_inches=0)
                print(f"✅ Saved class '{class_name}' overlay to {class_viz_path}")
                plt.close()

    return pred_masks_resized
