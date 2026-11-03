import os
import albumentations as A
import cv2
import numpy as np
import shutil
import random
import re
import json
import tempfile
from pathlib import Path

def reproject_raster(input_tif, output_tif, target_crs="EPSG:4326"):
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    with rasterio.open(input_tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_tif, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest)
                
def compress_raster(input_tif, output_tif, compression="lzw", dtype=None, quality=None):
    """
    Kompres file raster agar ukuran lebih kecil.
    
    Parameters:
    -----------
    input_tif : str
        Path file raster input (.tif)
    output_tif : str
        Path file raster output (.tif) hasil kompresi
    compression : str
        Metode kompresi (pilihan: "lzw", "deflate", "zstd", "jpeg")
    dtype : numpy dtype (optional)
        Jika ingin mengubah tipe data (misal float32 -> uint8)
    quality : int (optional)
        Hanya untuk compression="jpeg" (range 1-100)
    """
    import rasterio

    with rasterio.open(input_tif) as src:
        meta = src.meta.copy()

        # Ubah dtype jika diperlukan
        if dtype is not None:
            meta["dtype"] = dtype

        # Set kompresi
        meta.update({
            "compress": compression,
            "driver": "GTiff"
        })

        # Tambahkan quality jika JPEG
        if compression.lower() == "jpeg" and quality is not None:
            meta.update({"jpeg_quality": quality})

        with rasterio.open(output_tif, "w", **meta) as dst:
            for i in range(1, src.count + 1):
                data = src.read(i)
                if dtype is not None:
                    data = data.astype(dtype)
                dst.write(data, i)

    print(f"Raster dikompres dan disimpan di {output_tif}")
                
def convert_geojson_to_mask_geopandas(geojson_path, geotiff_path, output_file=None, return_mask=False):
    import rasterio
    from rasterio.features import rasterize
    import geopandas as gpd
    from shapely.geometry import Polygon, Point, shape
    import fiona
    
    with rasterio.open(geotiff_path) as src:
        raster_crs = src.crs
        transform = src.transform
        width, height = src.width, src.height
        profile = src.profile.copy()

        print(f"Raster CRS: {raster_crs}")

    # Buka GeoJSON dengan Fiona
    with fiona.open(geojson_path, 'r') as src:
        # Konversi ke GeoDataFrame manual
        features = list(src)
        geometries = [shape(feat["geometry"]) for feat in features]
        properties = [feat["properties"] for feat in features]
        crs = src.crs
    # Buat GeoDataFrame
    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs=crs)
    # print(f"GeoJSON CRS: {gdf.crs}")

    if gdf.crs is None:
        print("CRS GeoJSON tidak ada. Menetapkan EPSG:4326 sebagai default.")
        gdf.set_crs("EPSG:4326", inplace=True)

    if raster_crs is None:
        raise ValueError("GeoTIFF tidak memiliki CRS!")
    
    # Tambahkan ini untuk mengatasi CRS lokal
    if not raster_crs.to_epsg():
        print("CRS raster tidak standar. Menggunakan EPSG:3857 sebagai fallback.")
        raster_crs = "EPSG:3857"
        
    # Ubah ke CRS raster agar koordinat cocok dengan ukuran gambar
    gdf = gdf.to_crs(raster_crs)
    
    # Rasterisasi
    mask = rasterize(
        ((geom, 1) for geom in gdf.geometry),
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype='uint8'
    )

    # Ubah mask 1 → 255 untuk keperluan visualisasi
    mask_visual = mask * 255

    if return_mask:
        profile.update(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="uint8",
            crs=raster_crs,
            transform=transform
        )
        return mask_visual, profile

    if output_file is None:
        raise ValueError("output_file is required unless return_mask=True.")

    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='uint8',
        crs=raster_crs,
        transform=transform
    ) as dst:
        dst.write(mask_visual, 1)

    print(f"Mask berhasil disimpan ke {output_file}")


def _write_multilabel_mask(mask, output_file, profile, class_names):
    import rasterio

    if mask.ndim != 3:
        raise ValueError("A multilabel mask must have shape (height, width, classes).")
    if mask.shape[2] != len(class_names):
        raise ValueError("Mask channel count must match the class name count.")
    output_profile = {
        "driver": "GTiff",
        "height": profile["height"],
        "width": profile["width"],
        "count": mask.shape[2],
        "dtype": "uint8",
        "crs": profile["crs"],
        "transform": profile["transform"],
        "compress": "deflate"
    }
    with rasterio.open(output_file, "w", **output_profile) as dst:
        dst.write(mask.transpose(2, 0, 1))
        for channel, class_name in enumerate(class_names, start=1):
            dst.set_band_description(channel, class_name)

            
def save_images_and_labels_from_geojson(image_path, geojson_path, output_dir, index_class):
    import shutil
    extension = os.path.splitext(image_path)[1].lower()
    
    # Buat output directory
    os.makedirs(output_dir, exist_ok=True)
    
    output_dir_labels = os.path.join(output_dir, "labels")
    output_dir_images = os.path.join(output_dir, "images")
    
    os.makedirs(output_dir_labels, exist_ok=True)
    os.makedirs(output_dir_images, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(geojson_path))[0]
    output_file_labels = os.path.join(output_dir_labels, f"{base_name}{extension}")
    
    convert_geojson_to_mask_geopandas(geojson_path, image_path, output_file_labels)
    
    image_path_compress = os.path.join(os.path.dirname(image_path), f"{base_name}_compress{extension}")
    
    compress_raster(image_path, image_path_compress, compression="deflate")
    os.remove(image_path)
    print(f"Removed Compressed image: {image_path}")
    
    # Salin gambar ke directory images
    shutil.move(image_path_compress, os.path.join(output_dir_images, f"{base_name}{extension}"))
    print(f"Moved image: {image_path_compress}")    

def get_random_images_and_mask(image_dir, label_dir):
    image_files = [
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        and (label_dir / path.name).is_file()
    ]
    if not image_files:
        raise ValueError(f"No paired images and masks found in {image_dir}")
    random_img_file = random.choice(image_files)
    
    random_image = cv2.imread(str(random_img_file))
    random_mask_file = label_dir / random_img_file.name
    random_mask = _read_geojson_mask(random_mask_file)
    return random_image, random_mask


def _read_geojson_mask(mask_path):
    if Path(mask_path).suffix.lower() in {".tif", ".tiff"}:
        import rasterio

        with rasterio.open(mask_path) as src:
            mask = src.read()
        if mask.shape[0] == 1:
            return mask[0]
        return mask.transpose(1, 2, 0)
    return cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)


def _write_geojson_mask(mask, mask_path):
    import rasterio
    from rasterio.transform import from_origin

    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]
    height, width, channels = mask.shape
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=channels,
        dtype=mask.dtype,
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(mask.transpose(2, 0, 1))
    
def augment_dataset(image_dir, label_dir, output_dir, augment_count):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    output_img_dir = output_dir / 'images'
    output_label_dir = output_dir / 'labels'
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    transforms = [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Blur(p=0.3),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.MotionBlur(p=0.2),  # Motion blur
        A.Affine(shear=(-15, 15), p=0.3),  # Shear transformation
    ]
    mosaic = A.Mosaic(
            grid_yx=(2, 2),
            target_size=(256, 256),
            cell_shape=(256, 256),
            fit_mode='cover',
            p=0.5
        )
        
    # Process only original images; do not recursively augment generated outputs.
    for img_file in list(image_dir.glob('*')):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                image = cv2.imread(str(img_file))
                if image is None:
                    print(f"Failed to read image: {img_file}")
                    continue

                # Read corresponding mask
                mask_file = label_dir / img_file.name
                if not mask_file.exists():
                    print(f"Mask not found for: {img_file.name}")
                    continue

                mask = _read_geojson_mask(mask_file)
                if mask is None:
                    print(f"Failed to read mask: {mask_file}")
                    continue

                is_multilabel_mask = mask.ndim == 3
                transform = A.Compose(transforms + ([] if is_multilabel_mask else [mosaic]))
                mosaic_metadata = None
                if not is_multilabel_mask:
                    random_image1, random_mask1 = get_random_images_and_mask(image_dir, label_dir)
                    random_image2, random_mask2 = get_random_images_and_mask(image_dir, label_dir)
                    random_image3, random_mask3 = get_random_images_and_mask(image_dir, label_dir)
                    random_image4, random_mask4 = get_random_images_and_mask(image_dir, label_dir)
                    mosaic_metadata = [
                        {"image": random_image1, "mask": random_mask1},
                        {"image": random_image2, "mask": random_mask2},
                        {"image": random_image3, "mask": random_mask3},
                        {"image": random_image4, "mask": random_mask4},
                    ]

                for i in range(augment_count):
                    if mosaic_metadata is None:
                        augmented = transform(image=image, mask=mask)
                    else:
                        augmented = transform(
                            image=image,
                            mask=mask,
                            mosaic_metadata=mosaic_metadata
                        )
                    aug_img = augmented['image']
                    aug_mask = augmented['mask']

                    # Save image
                    aug_img_path = output_img_dir / f"{img_file.stem}_aug_{i}{img_file.suffix}"
                    cv2.imwrite(str(aug_img_path), aug_img)

                    # Save mask (keep same format)
                    aug_mask_path = output_label_dir / f"{img_file.stem}_aug_{i}{mask_file.suffix}"
                    if is_multilabel_mask:
                        _write_geojson_mask(aug_mask, str(aug_mask_path))
                    else:
                        cv2.imwrite(str(aug_mask_path), aug_mask)

                    print (f"Finished augmentation image and mask for {img_file.name} - augment count {i+1}/{augment_count}")
    
def split_dataset(output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42, num_augmented=0, is_augmented=False):
    """
    Membagi dataset menjadi train, validation, dan test set
    
    Args:
        output_dir (str): Directory yang berisi dataset
        train_ratio (float): Persentase data training (default: 0.7)
        val_ratio (float): Persentase data validation (default: 0.2)
        test_ratio (float): Persentase data test (default: 0.1)
        seed (int): Seed untuk random generator
    """
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 or ratio > 1 for ratio in ratios):
        raise ValueError("Dataset split ratios must be between 0 and 1.")
    if not abs(sum(ratios) - 1.0) < 1e-8:
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.")
    
    # Buat directory untuk setiap set
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
        
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)
    
    # Index image-mask pairs by stem, then keep augmented siblings in the same split.
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    image_paths = {}
    for name in os.listdir(images_dir):
        path = os.path.join(images_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in valid_extensions:
            stem = os.path.splitext(name)[0]
            if stem in image_paths:
                raise ValueError(f"Multiple image files share the stem '{stem}'.")
            image_paths[stem] = name

    label_paths = {}
    for name in os.listdir(labels_dir):
        path = os.path.join(labels_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in valid_extensions:
            stem = os.path.splitext(name)[0]
            if stem in label_paths:
                raise ValueError(f"Multiple mask files share the stem '{stem}'.")
            label_paths[stem] = name

    missing_masks = sorted(set(image_paths) - set(label_paths))
    missing_images = sorted(set(label_paths) - set(image_paths))
    if missing_masks or missing_images:
        raise ValueError(
            "Dataset contains unpaired files. "
            f"Images without masks: {missing_masks[:5]}; masks without images: {missing_images[:5]}"
        )
    if not image_paths:
        raise ValueError(f"No supported image-mask pairs found in {output_dir}.")

    groups = {}
    for stem in image_paths:
        source_stem = re.sub(r'_aug_\d+$', '', stem)
        groups.setdefault(source_stem, []).append(stem)

    group_names = sorted(groups)
    random.Random(seed).shuffle(group_names)
    n_groups = len(group_names)
    split_ratios = (train_ratio, val_ratio, test_ratio)
    split_counts = [int(n_groups * ratio) for ratio in split_ratios]
    remainder = n_groups - sum(split_counts)
    ranked_splits = sorted(
        range(3),
        key=lambda index: (n_groups * split_ratios[index] - split_counts[index], -index),
        reverse=True
    )
    for index in ranked_splits[:remainder]:
        split_counts[index] += 1
    n_train, n_val, _ = split_counts
    split_groups = {
        train_dir: group_names[:n_train],
        val_dir: group_names[n_train:n_train + n_val],
        test_dir: group_names[n_train + n_val:],
    }

    # Fungsi untuk menyalin file ke directory tujuan
    def copy_files(group_names, target_dir):
        for group_name in group_names:
            for stem in groups[group_name]:
                img_file = image_paths[stem]
                mask_file = label_paths[stem]
                shutil.copy2(
                    os.path.join(images_dir, img_file),
                    os.path.join(target_dir, 'images', img_file)
                )
                shutil.copy2(
                    os.path.join(labels_dir, mask_file),
                    os.path.join(target_dir, 'labels', mask_file)
                )

    for target_dir, assigned_groups in split_groups.items():
        copy_files(assigned_groups, target_dir)

    # Augment only after splitting so derived samples cannot leak into validation or test.
    if is_augmented:
        augment_dataset(
            image_dir=os.path.join(train_dir, 'images'),
            label_dir=os.path.join(train_dir, 'labels'),
            output_dir=train_dir,
            augment_count=num_augmented
        )

    counts = {
        "train": sum(len(groups[group]) for group in split_groups[train_dir]),
        "val": sum(len(groups[group]) for group in split_groups[val_dir]),
        "test": sum(len(groups[group]) for group in split_groups[test_dir]),
    }
    print("\nDataset split complete:")
    print(f"Total source groups: {n_groups}; total image-mask pairs: {len(image_paths)}")
    print(f"Training set: {counts['train']} files ({train_ratio*100:.1f}%)")
    print(f"Validation set: {counts['val']} files ({val_ratio*100:.1f}%)")
    print(f"Test set: {counts['test']} files ({test_ratio*100:.1f}%)")

def batch_generate_datasets_from_geojson(
    input_folder,
    output_dir,
    target_crs="EPSG:4326",
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    dataset_name='dataset',
    is_augmented=False,
    num_augmented=0,
    multilabel=False
):
    import shutil

    input_path = Path(input_folder)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_path}")

    output_path = Path(output_dir)
    output_resolved = output_path.resolve()
    input_resolved = input_path.resolve()
    if (
        output_resolved == input_resolved
        or input_resolved in output_resolved.parents
        or output_resolved in input_resolved.parents
    ):
        raise ValueError("Input and output directories must not contain one another.")

    classes_list = sorted(
        entry.name
        for entry in input_path.iterdir()
        if entry.is_dir() and entry.resolve() != output_resolved
    )
    if not classes_list:
        raise ValueError(f"No class subfolders found in {input_path}")

    if multilabel:
        class_scenes = {}
        for class_name in classes_list:
            class_folder = input_path / class_name
            scenes = {}
            for tif_path in class_folder.iterdir():
                if tif_path.is_file() and tif_path.suffix.lower() in {".tif", ".tiff"}:
                    geojson_path = tif_path.with_suffix(".geojson")
                    if not geojson_path.is_file():
                        raise FileNotFoundError(
                            f"Missing GeoJSON annotation for {tif_path}"
                        )
                    if tif_path.stem in scenes:
                        raise ValueError(
                            f"Multiple raster files share scene name '{tif_path.stem}' "
                            f"in class folder {class_folder}"
                        )
                    scenes[tif_path.stem] = (tif_path, geojson_path)
            if not scenes:
                raise ValueError(f"No GeoTIFF/GeoJSON pairs found in {class_folder}")
            class_scenes[class_name] = scenes

        scene_names = set.union(*(set(scenes) for scenes in class_scenes.values()))
        for class_name, scenes in class_scenes.items():
            missing_scenes = sorted(scene_names - set(scenes))
            if missing_scenes:
                raise ValueError(
                    f"Multilabel annotations are incomplete for class '{class_name}'. "
                    f"Missing scenes: {missing_scenes[:5]}. Add an empty GeoJSON for "
                    "scenes where that class is absent, or use multilabel=False."
                )

    # Remove only the generated output directory, never the source data.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    print("Class folders:", classes_list)

    if multilabel:
        class_mapping = {
            class_name: channel
            for channel, class_name in enumerate(classes_list)
        }
        os.makedirs(output_path / "images", exist_ok=True)
        os.makedirs(output_path / "labels", exist_ok=True)
        with open(output_path / "classes.json", "w", encoding="utf-8") as file:
            json.dump({"class_to_channel": class_mapping}, file, indent=2)

        with tempfile.TemporaryDirectory(prefix="geojson_multilabel_") as temp_dir:
            for scene_name in sorted(scene_names):
                band_masks = []
                reference_profile = None
                reference_grid = None
                reference_image_path = None

                for class_name in classes_list:
                    source_tif, geojson_path = class_scenes[class_name][scene_name]
                    reprojected_tif = Path(temp_dir) / f"{len(band_masks)}.tif"
                    reproject_raster(
                        str(source_tif),
                        str(reprojected_tif),
                        target_crs=target_crs
                    )
                    band_mask, mask_profile = convert_geojson_to_mask_geopandas(
                        str(geojson_path),
                        str(reprojected_tif),
                        return_mask=True
                    )

                    grid = (
                        mask_profile["height"],
                        mask_profile["width"],
                        mask_profile["transform"],
                        mask_profile["crs"]
                    )
                    if reference_grid is not None and grid != reference_grid:
                        raise ValueError(
                            f"Class rasters for scene '{scene_name}' do not share the "
                            "same dimensions, CRS, and geotransform."
                        )
                    reference_grid = grid
                    reference_profile = mask_profile
                    if reference_image_path is None:
                        reference_image_path = str(reprojected_tif)
                    band_masks.append(band_mask)

                image_output = output_path / "images" / f"{scene_name}.tif"
                compress_raster(
                    reference_image_path,
                    str(image_output),
                    compression="deflate"
                )
                mask_output = output_path / "labels" / f"{scene_name}.tif"
                _write_multilabel_mask(
                    np.stack(band_masks, axis=-1),
                    str(mask_output),
                    reference_profile,
                    classes_list
                )
                print(f"Generated multilabel image and mask for {scene_name}")

        split_dataset(
            str(output_path),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            is_augmented=is_augmented,
            num_augmented=num_augmented
        )
        print(f"Multilabel classes (zero-based band order): {class_mapping}")
        return output_dir

    generated_classes = []

    for class_name in classes_list:
        class_folder = input_path / class_name
        print(f"Found folder class: {class_name}")
        class_output_dir = output_path / class_name

        tif_files = [f for f in os.listdir(class_folder) if f.lower().endswith('.tif')]
        matching_pairs = []
        for tif_file in tif_files:
            base_name = os.path.splitext(tif_file)[0]
            geojson_file = base_name + ".geojson"
            tif_path = os.path.join(class_folder, tif_file)
            geojson_path = os.path.join(class_folder, geojson_file)
            if os.path.exists(geojson_path):
                matching_pairs.append((tif_path, geojson_path))
            else:
                print(f"GeoJSON not found for {tif_file}, skipping.")

        if not matching_pairs:
            print(f"No matching GeoTIFF/GeoJSON pairs found for class '{class_name}', skipping.")
            continue

        for tif_path, geojson_path in matching_pairs:
            tif_name = Path(tif_path)
            reprojected_image_path = str(tif_name.with_name(f"{tif_name.stem}_reprojected.tif"))
            print(f"Processing class '{class_name}': {tif_name.name} & {Path(geojson_path).name}")
            reproject_raster(tif_path, reprojected_image_path, target_crs=target_crs)
            save_images_and_labels_from_geojson(
                reprojected_image_path,
                geojson_path,
                str(class_output_dir),
                index_class=0
            )

        split_dataset(
            str(class_output_dir),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            is_augmented=is_augmented,
            num_augmented=num_augmented
        )
        generated_classes.append(class_name)

    if not generated_classes:
        raise ValueError(f"No matching GeoTIFF/GeoJSON pairs found in class folders under {input_path}.")

    print(f"Generated class datasets: {generated_classes}")
    return output_dir