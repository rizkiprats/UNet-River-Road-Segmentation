# UNet River & Road Segmentation

A compact pipeline for semantic segmentation of rivers and roads from geo-referenced imagery using a UNet-style model. The repository includes tools to convert vector annotations (GeoJSON or AnyLabel) to raster masks, dataset loading utilities, training and inference scripts, and example datasets.

**Features**
- Convert GeoJSON and AnyLabel exports into segmentation masks.
- Support for single-label and multi-label segmentation (river, road).
- UNet model implementation, training loop, and prediction script.
- Example datasets and sample trained model included.

**Repository layout**
- `model.py` — UNet model architecture and helpers.
- `train.py` — Training entrypoint and CLI options.
- `predict.py` — Run model inference and export masks.
- `geojson_to_mask.py` — Convert GeoJSON vector annotations to raster masks.
- `anylabeling_to_mask.py` — Convert AnyLabel JSON to raster masks.
- `dataset_loader.py` — Dataset and dataloader utilities.
- `loss_custom.py` — Custom loss functions used by training.
- `environtment_conda.yml` — Conda environment spec (recommended).
- `requirements.txt` — Python dependencies for pip installs.
- `geotiff_geojson_train_datasets/` — Example dataset layout used for training.
- `Multilabel_Model/` — Example trained model weights.

Requirements
- Python 3.8+ recommended.
- Common Python packages listed in `requirements.txt`.
- Optional: use `conda` to install GDAL/rasterio binaries on Windows for easier setup.

Installation
1. (Recommended) Create and activate the conda environment:

	 conda env create -f environtment_conda.yml -n unet-seg
	 conda activate unet-seg

2. Or install packages with pip:

	 pip install -r requirements.txt

Prepare your data
- Mirror the example dataset structure found in `geotiff_geojson_train_datasets/`.
- Each dataset should include the imagery and a folder with corresponding GeoJSON annotations and a `classes.txt` listing classes (one per line).

Convert annotations to masks
- Use `geojson_to_mask.py` for GeoJSON; the script rasterizes vector annotations into mask images aligned with the source imagery.
- Use `anylabeling_to_mask.py` to convert AnyLabel JSON exports into masks.
- Check each converter's `--help` or header comments for expected input/output paths and options.

Training
- Launch training with `train.py`. Example:

	python train.py --data-dir geotiff_geojson_train_datasets/ --epochs 50 --batch-size 4 --save-dir models/

- Inspect `train.py` for options (learning rate, augmentations, checkpointing).

Inference
- Run inference using `predict.py`. Example:

	python predict.py --model Multilabel_Model/multilabel_model_road_river.pth --input path/to/image.tif --output path/to/output_mask.png

- `predict.py` will save predicted mask images; open them in any image viewer or GIS tool that supports PNG/TIFF.

Testing
- Basic tests and examples are in `test.py`, `test_road.py`, and `test_river.py`.

Notes & troubleshooting
- On Windows, prefer `conda` to install `gdal` / `rasterio` to avoid build issues.
- Ensure `classes.txt` ordering matches mask generation and model label mapping.

Contributing
- Contributions are welcome: suggested improvements include clearer CLI docs, evaluation scripts (IoU/F1), and visualization utilities.