# UNet River & Road Segmentation

A concise, practical toolkit to convert geospatial annotations into training masks, train UNet-style segmentation models, and run inference for river and road extraction from satellite/geospatial imagery.

Overview
- Purpose: Train and evaluate semantic segmentation models for river and road detection using GeoTIFF imagery and vector annotations (GeoJSON or AnyLabel JSON).
- Typical workflow: convert annotations → prepare dataset → train model → run inference → evaluate/visualize results.

Repository layout (key files)
- `anylabeling_to_mask.py` — convert AnyLabel JSON annotations to raster masks.
- `geojson_to_mask.py` — convert GeoJSON vector annotations to raster masks.
- `dataset_loader.py` — dataset classes and PyTorch dataloader helpers.
- `model.py` — UNet model architecture used for training and inference.
- `train.py` — training script (CLI options in the header of the file).
- `predict.py` — inference script to run the trained model on new imagery.
- `loss_custom.py` — custom loss implementations used by the training loop.
- `requirements.txt` / `environtment_conda.yml` — dependency specifications.
- `geotiff_geojson_train_datasets/` — expected structure for prepared training datasets with `classes.txt` in each dataset.
- `Multilabel_Model/` — example or saved trained model files.

Requirements
- Python 3.8+ recommended.
- Core Python dependencies are listed in `requirements.txt`. For geospatial raster/vector processing you may need `rasterio`, `shapely`, and `fiona`.
- On Windows, prefer using the provided conda environment file to install GDAL/rasterio without build issues.

Quick start
1) Create environment (recommended):

	 conda env create -f environtment_conda.yml -n unet-seg
	 conda activate unet-seg

2) Prepare datasets
- Each dataset should follow this pattern:

	geotiff_geojson_train_datasets/
	├─ river_dataset/
	│  ├─ classes.txt
	│  └─ river/ (GeoJSON files or converted masks)
	└─ road_dataset/
		 ├─ classes.txt
		 └─ road/

- `classes.txt` should list target class names (one per line) in the order masks will be encoded (background first if used).

3) Convert annotations to masks (if needed)
- Convert GeoJSON to mask images using:

	python geojson_to_mask.py --input-dir path/to/geojson_folder --output-dir path/to/mask_folder --classes path/to/classes.txt

- Convert AnyLabel exports using:

	python anylabeling_to_mask.py --input path/to/anylabel.json --output path/to/mask_folder --classes path/to/classes.txt

Refer to the script headers for detailed CLI arguments.

Training
- Example training command (adjust flags per `train.py`):

	python train.py --data-dir geotiff_geojson_train_datasets/ --epochs 50 --batch-size 4 --lr 1e-4 --save-dir models/

- `train.py` contains augmentation, checkpoint and optimizer settings — inspect the top of the file for full options.

Inference
- Run inference on a single image or folder using `predict.py`:

	python predict.py --model Multilabel_Model/multilabel_model_road_river.pth --input path/to/image.tif --output path/to/output_mask.png

- Outputs are saved as raster mask images aligned to the input imagery.

Testing & utilities
- Small test scripts exist such as `test.py`, `test_road.py`, and `test_river.py` to validate functions or demo the pipeline.
- Example satellite / sample data is available under `geotiff_geojson_collected_datasets/` and `Test Image/`.

Tips & troubleshooting
- GDAL / rasterio errors on Windows: use conda-forge builds (`conda install -c conda-forge rasterio gdal fiona shapely`).
- If masks appear misaligned, verify coordinate reference systems (CRS) and transform metadata for input GeoTIFFs and generated masks.
- Ensure `classes.txt` ordering matches mask encoding expected by the training script.

Contributing
- Improve data conversion scripts, add evaluation metrics (IoU/F1), or provide visualization tools and notebooks.