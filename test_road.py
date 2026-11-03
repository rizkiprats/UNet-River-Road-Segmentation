from train import run_training_for_classes, load_model_with_fallback
from model import get_model
from predict import predict
import torch
import os

if __name__ == "__main__":
    dataset_root = r"Datasets/River_Road_Dataset"

    train_model = True
    save_dir = "Model/Singlelabel"
    save_path = os.path.join(save_dir, "unet_road_road.pth")
    
    if train_model:
        os.makedirs(save_dir, exist_ok=True)
        model_paths = run_training_for_classes(
            dataset_root=dataset_root,
            class_names="road",
            batch_size=8,
            num_epochs=100,
            lr=1e-4,
            save_path=os.path.join(save_dir, "unet_road.pth"),
            patience=10
        )
        save_path = model_paths["road"]

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model = load_model_with_fallback(model, save_path, device)
    model.to(device)
    
    test_predict_dir = "Test Image/river_road"

    for file in os.listdir(test_predict_dir):
        if "predicted_mask" not in file and file.endswith((".tif", ".png", ".jpg", ".jpeg", ".tiff")):
        
            # Test pada gambar
            predicted_mask = predict(
                model,
                image_path=os.path.join(test_predict_dir, file), 
                device=device,
                save_path=os.path.join(
                    test_predict_dir,
                    f"{os.path.splitext(file)[0]}_predicted_mask.tif"
                ),
                show_visualization=False,
                save_visualization=True
            )