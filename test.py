from train import run_training, load_model_with_fallback
from geojson_to_mask import batch_generate_datasets_from_geojson_tif
from model import get_model
from predict import predict
import torch
import os

if __name__ == "__main__":
    input_folder = "River Geojson"
    output_dir = "River Dataset"
    is_augmented = True
    num_augmented = 10
    
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

    save_dir = "UnetModel"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "unet_segmentation.pth")
    
    run_training(
        dataset_dir=output_dir,
        batch_size=8,
        num_epochs=100,
        lr=1e-4,
        save_path=save_path
    )

    # run_training(
    #     dataset_dir=output_dir,
    #     batch_size=8,
    #     num_epochs=40,
    #     lr=1e-4,
    #     save_path=save_path,
    #     resume_mode="resume",
    #     resume_path=save_path
    # )

    # run_training(
    #     dataset_dir=output_dir,
    #     batch_size=8,
    #     num_epochs=100,
    #     lr=1e-4,
    #     save_path=save_path,
    #     resume_mode="finetune",
    #     resume_path=save_path,
    #     num_classes=3  # contoh kalau mau menambah segmentasi misal background + sungai + tanah
    # )

    # # Load model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = get_model()
    # model = load_model_with_fallback(model, save_path, device)
    # model.to(device)

    # # Test pada gambar
    # predicted_mask = predict(
    #     model, 
    #     image_path="Test Image/River/river_20250806_195316.tif", 
    #     device=device, 
    #     save_path="Test Image/River/river_20250806_195316_predicted_mask.png",
    #     show_visualization=True
    # )

    # import rasterio
    # from rasterio.features import shapes
    # from shapely.geometry import shape
    # import geopandas as gpd

    # def mask_to_geojson(mask, reference_image_path, geojson_path="output.geojson"):
    #     # Buka file referensi untuk ambil transform & CRS
    #     with rasterio.open(reference_image_path) as src:
    #         transform = src.transform
    #         crs = src.crs

    #     polygons = []
    #     for geom, value in shapes(mask, mask=mask > 0, transform=transform):
    #         if value > 0:
    #             polygons.append(shape(geom))

    #     gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    #     gdf.to_file(geojson_path, driver="GeoJSON")
    #     print(f"🌍 GeoJSON saved to {geojson_path}")

    # # Convert ke GeoJSON
    # mask_to_geojson(
    #     predicted_mask, 
    #     reference_image_path="Test Image/River/river_20250806_195316.tif", 
    #     geojson_path="Test Image/River/river_20250806_195316.geojson"
    # )