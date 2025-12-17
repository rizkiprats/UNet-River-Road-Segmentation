import os
import albumentations as A
import cv2
import shutil
import random
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

    print(f"✅ Raster dikompres dan disimpan di {output_tif}")
                
def convert_geojson_to_mask_geopandas(geojson_path, geotiff_path, output_file):
    import rasterio
    from rasterio.features import rasterize
    import geopandas as gpd
    from shapely.geometry import Polygon, Point, shape
    import fiona
    
    with rasterio.open(geotiff_path) as src:
        raster_crs = src.crs
        transform = src.transform
        width, height = src.width, src.height

        print(f"Raster CRS: {raster_crs}")

    # gdf = gpd.read_file(geojson_path) #<- error fiona
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
        print("⚠️ CRS GeoJSON tidak ada. Menetapkan EPSG:4326 sebagai default.")
        gdf.set_crs("EPSG:4326", inplace=True)

    if raster_crs is None:
        raise ValueError("GeoTIFF tidak memiliki CRS!")
    
    # Tambahkan ini untuk mengatasi CRS lokal
    if not raster_crs.to_epsg():
        print("⚠️ CRS raster tidak standar. Menggunakan EPSG:3857 sebagai fallback.")
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

    print(f"✅ Mask berhasil disimpan ke {output_file}")

            
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
    random_img_file = random.choice(list(image_dir.glob('*')))
    
    if random_img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        random_image = cv2.imread(str(random_img_file))
        random_mask_file = label_dir / random_img_file.name
        random_mask = cv2.imread(str(random_mask_file), cv2.IMREAD_UNCHANGED)
    
    return random_image, random_mask
    
def augment_dataset(image_dir, label_dir, output_dir, augment_count):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    output_img_dir = output_dir / 'images'
    output_label_dir = output_dir / 'labels'
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    # # Define augmentation pipeline    
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Blur(p=0.3),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.MotionBlur(p=0.2),  # Motion blur
        # A.RandomBrightnessContrast(p=0.5),
        # A.HueSaturationValue(p=0.5),
        # A.CLAHE(p=0.3),  # Contrast Limited Adaptive Histogram Equalization
        # A.RandomGamma(p=0.3),  # Random gamma correction
        # A.GaussNoise(std_range=(0.2, 0.44), p=0.3),  # Gaussian noise
        # A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),  # RGB channel shift
        # A.ToGray(p=0.1),  # Convert to grayscale,
        A.Affine(shear=(-15, 15), p=0.3),  # Shear transformation
        A.Mosaic(
            grid_yx=(2, 2),
            target_size=(256, 256),
            cell_shape=(256, 256),
            fit_mode='cover',
            p=0.5
        ),
    ])
        
    # Process images
    for img_file in image_dir.glob('*'):
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

                mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    print(f"Failed to read mask: {mask_file}")
                    continue
                
                random_image1, random_mask1 = get_random_images_and_mask(image_dir, label_dir)
                random_image2, random_mask2 = get_random_images_and_mask(image_dir, label_dir)
                random_image3, random_mask3 = get_random_images_and_mask(image_dir, label_dir)
                random_image4, random_mask4 = get_random_images_and_mask(image_dir, label_dir)
                
                mosaic_metadata = [
                    {
                        "image": random_image1,
                        "mask": random_mask1,
                    },
                    {
                        "image": random_image2,
                        "mask": random_mask2,
                    },
                    {
                        "image": random_image3,
                        "mask": random_mask3,
                    },
                    {
                        "image": random_image4,
                        "mask": random_mask4,
                    },
                ]

                for i in range(augment_count):
                    augmented = transform(image=image, mask=mask, mosaic_metadata=mosaic_metadata)
                    aug_img = augmented['image']
                    aug_mask = augmented['mask']

                    # Save image
                    aug_img_path = output_img_dir / f"{img_file.stem}_aug_{i}{img_file.suffix}"
                    cv2.imwrite(str(aug_img_path), aug_img)

                    # Save mask (keep same format)
                    aug_mask_path = output_label_dir / f"{img_file.stem}_aug_{i}{mask_file.suffix}"
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
    # Set random seed
    random.seed(seed)
    
    # Buat directory untuk setiap set
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
        
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)
    
    # Dapatkan semua file gambar
    images_dir = os.path.join(output_dir, 'images')
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif'))]
    
    labels_dir = os.path.join(output_dir, 'labels')
    
    # Acak urutan file
    random.shuffle(image_files)
    
    # Hitung jumlah file untuk setiap set
    n_files = len(image_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    
    # Bagi file ke dalam set
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # Fungsi untuk menyalin file ke directory tujuan
    def copy_files(files, target_dir):
        for img_file in files:
            # Salin file gambar
            shutil.copy2(
                os.path.join(images_dir, img_file),
                os.path.join(target_dir, 'images', img_file)
            )
            
            # Salin file anotasi
            base_name = os.path.splitext(img_file)[0]
            extension = os.path.splitext(img_file)[1].lower()
            mask_file = f"{base_name}{extension}"
            if os.path.exists(os.path.join(labels_dir, mask_file)):
                shutil.copy2(
                    os.path.join(labels_dir, mask_file),
                    os.path.join(target_dir, 'labels', mask_file)
                )
    
    # Salin file ke masing-masing set
    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)
    
    
    # Augmentasi hanya untuk data train
    if is_augmented:
        train_images_dir = os.path.join(train_dir,'images')
        train_labels_dir = os.path.join(train_dir,'labels')
        augment_dataset(image_dir=train_images_dir, label_dir=train_labels_dir, output_dir=train_dir, augment_count=num_augmented)
    
    # Print statistik
    print(f"\nDataset split complete:")
    print(f"Total files: {n_files}")
    print(f"Training set: {len(train_files)} files ({train_ratio*100:.1f}%)")
    print(f"Validation set: {len(val_files)} files ({val_ratio*100:.1f}%)")
    print(f"Test set: {len(test_files)} files ({test_ratio*100:.1f}%)")

def batch_generate_datasets_from_geojson_tif(input_folder, output_dir, target_crs="EPSG:4326", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, dataset_name='dataset', augmented=False, num_augmented=0):
    import shutil
    
    # Hapus output directory jika sudah ada
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Baca classes.txt untuk mendapatkan nama kelas
    classes_list = []
    classes_file = os.path.join(input_folder, 'classes.txt')
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
            classes_list = [name for name in classes if name]

    print("classes_list", classes_list)

    for i, class_name in enumerate(classes_list):
        if os.path.exists(os.path.join(input_folder, class_name)):
            class_folder = os.path.join(input_folder, class_name)
            print(f"Found folder class: {class_name}")

            tif_files = [f for f in os.listdir(class_folder) if f.lower().endswith('.tif')]
            for tif_file in tif_files:
                base_name = os.path.splitext(tif_file)[0]
                geojson_file = base_name + ".geojson"
                tif_path = os.path.join(class_folder, tif_file)
                geojson_path = os.path.join(class_folder, geojson_file)
                if os.path.exists(geojson_path):
                    print(f"Processing: {tif_file} & {geojson_file}")
                    reprojected_image_path = tif_path.replace(".tif", "_reprojected.tif")
                    reproject_raster(tif_path, reprojected_image_path, target_crs=target_crs)
                    save_images_and_labels_from_geojson(reprojected_image_path, geojson_path, output_dir, index_class=i)
                else:
                    print(f"GeoJSON not found for {tif_file}, skipping.")
            
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    if augmented == True:
        augment_dataset(image_dir=images_dir, label_dir=labels_dir, output_dir=output_dir, augment_count=num_augmented)
        
    split_dataset(output_dir, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, is_augmented=False, num_augmented=num_augmented)
    
    return output_dir