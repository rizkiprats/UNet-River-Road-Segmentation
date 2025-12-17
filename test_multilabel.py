from train import run_training, load_model_with_fallback, load_model
from geojson_to_mask import batch_generate_datasets_from_geojson_tif
from model import get_model
from predict import predict_multilabel
import torch
import os

if __name__ == "__main__":
    input_folder = "geotiff_geojson_train_datasets/road_dataset"
    road_output_dir = "road"
    is_augmented = False
    num_augmented = 10
    
    batch_generate_datasets_from_geojson_tif(
        input_folder, 
        road_output_dir, 
        target_crs="EPSG:4326", 
        train_ratio=0.7, 
        val_ratio=0.3, 
        test_ratio=0, 
        dataset_name='dataset', 
        augmented=is_augmented, 
        num_augmented=num_augmented
    )
    
    input_folder = "geotiff_geojson_train_datasets/river_dataset"
    river_output_dir = "river"
    is_augmented = False
    num_augmented = 10
    
    batch_generate_datasets_from_geojson_tif(
        input_folder, 
        river_output_dir, 
        target_crs="EPSG:4326", 
        train_ratio=0.7, 
        val_ratio=0.3, 
        test_ratio=0, 
        dataset_name='dataset', 
        augmented=is_augmented, 
        num_augmented=num_augmented
    )
    
    import shutil    
    dataset_folder = "Dataset_Multilabel"
    if os.path.exists(dataset_folder):
        shutil.rmtree(dataset_folder)
    os.makedirs(dataset_folder, exist_ok=True)
    
    shutil.move(road_output_dir, os.path.join(dataset_folder, road_output_dir))
    shutil.move(river_output_dir, os.path.join(dataset_folder, river_output_dir))

    classes = []
    for path in os.listdir(dataset_folder):
            if os.path.isdir(os.path.join(dataset_folder, path)):
                classes.append(path)
    
    save_dir = "Multilabel_Model"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "multilabel_model_road_river.pth")
    
    run_training(
        dataset_dir=dataset_folder,
        batch_size=8,
        img_resize=256,
        num_epochs=50,
        lr=1e-4,
        save_path=save_path,
        patience=10,
        classes=classes,
        is_multiplabel=True
    )
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(save_path, device)
    model.to(device)
    classes = checkpoint.get("classes", None)
    
    test_predict_dir = "Test Image"

    for file in os.listdir(test_predict_dir):
        if "predicted_mask" not in file and file.endswith((".tif", ".png", ".jpg", ".jpeg", ".tiff")):
        
            # Test pada gambar
            predicted_mask = predict_multilabel(
                model,
                image_path=os.path.join(test_predict_dir, file), 
                device=device,
                img_resize=512, 
                save_path=os.path.join(test_predict_dir, f"{file}_predicted_mask.png"),
                show_visualization=False,
                save_visualization=True,
                class_names=classes
            )