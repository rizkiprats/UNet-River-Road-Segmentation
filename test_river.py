from train import run_training, load_model_with_fallback
from geojson_to_mask import batch_generate_datasets_from_geojson_tif
from model import get_model
from predict import predict
import torch
import os

if __name__ == "__main__":
    input_folder = "River_tiff_geojson"
    output_dir = "River_tiff_dataset"
    is_augmented = True
    num_augmented = 2
    
    batch_generate_datasets_from_geojson_tif(
        input_folder, 
        output_dir, 
        target_crs="EPSG:4326", 
        train_ratio=0.7, 
        val_ratio=0.2, 
        test_ratio=0.1, 
        dataset_name='dataset', 
        augmented=is_augmented, 
        num_augmented=num_augmented
    )
    
    save_dir = "RiverModel"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "unet_river_segmentation.pth")
    
    # run_training(
    #     dataset_dir=output_dir,
    #     batch_size=8,
    #     num_epochs=100,
    #     lr=1e-4,
    #     save_path=save_path,
    #     patience=10
    # )

    # run_training(
    #     dataset_dir=output_dir,
    #     batch_size=8,
    #     num_epochs=100,
    #     lr=1e-4,
    #     save_path=save_path,
    #     resume_mode="resume",
    #     resume_path=save_path,
    #     patience=10
    # )

    # # Load model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = get_model()
    # model = load_model_with_fallback(model, save_path, device)
    # model.to(device)

    # test_predict_dir = "Test Image"
    # test_file = "River Sawit (1).tif"
    
    # for file in os.listdir(test_predict_dir):
    #     if "predicted_mask" not in file and file.endswith((".tif", ".png", ".jpg", ".jpeg", ".tiff")):
        
    #         # Test pada gambar
    #         predicted_mask = predict(
    #             model,
    #             image_path=os.path.join(test_predict_dir, file), 
    #             device=device, 
    #             save_path=os.path.join(test_predict_dir, f"{file}_predicted_mask.png"),
    #             show_visualization=False,
    #             save_visualization=True,
    #         )