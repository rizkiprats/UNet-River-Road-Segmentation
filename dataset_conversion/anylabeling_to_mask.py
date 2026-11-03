import json
import os
import numpy as np
from pathlib import Path
import shutil
import random
import re
import cv2
import albumentations as A


def _read_mask(mask_path):
    if Path(mask_path).suffix.lower() in {".tif", ".tiff"}:
        import rasterio

        with rasterio.open(mask_path) as src:
            mask = src.read()
        if mask.shape[0] == 1:
            return mask[0]
        return np.moveaxis(mask, 0, -1)

    return cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)


def _write_multilabel_mask(mask, mask_path):
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
        transform=from_origin(0, 0, 1, 1)
    ) as dst:
        dst.write(np.moveaxis(mask, -1, 0))


def convert_anylabeling_to_mask(
    image_path,
    json_file,
    output_dir,
    class_mapping=None,
    multilabel=False
):
    """
    Mengkonversi file anotasi X-AnyLabeling ke format Mask per kelas (folder per kelas)
    
    Args:
        json_file (str): Path ke file JSON X-AnyLabeling
        output_dir (str): Directory output untuk file Mask
        class_mapping (dict): Mapping dari label ke class ID
    """
    # Buat output directory jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Baca file JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Jika class_mapping tidak diberikan, buat dari label yang ada
    if class_mapping is None:
        unique_labels = set(shape['label'] for shape in data['shapes'])
        class_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    h, w = image.shape[:2]
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    ext = os.path.splitext(os.path.basename(image_path))[1].lower()

    if multilabel:
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        mask = np.zeros((h, w, len(class_mapping)), dtype=np.uint8)
        for shape_data in data["shapes"]:
            label = shape_data["label"]
            if label not in class_mapping:
                raise ValueError(f"Unknown label '{label}' in {json_file}")
            points = np.array(shape_data["points"], np.int32)
            channel_mask = mask[:, :, class_mapping[label]].copy()
            cv2.fillPoly(channel_mask, [points], 255)
            mask[:, :, class_mapping[label]] = channel_mask

        image_path_out = os.path.join(images_dir, f"{base_name}.tif")
        if not cv2.imwrite(image_path_out, image):
            raise OSError(f"Unable to save image: {image_path_out}")
        mask_path = os.path.join(labels_dir, f"{base_name}.tif")
        _write_multilabel_mask(mask, mask_path)
        return [mask_path]

    # Buat folder class
    for label in class_mapping.keys():
        class_dir = os.path.join(output_dir, label)
        os.makedirs(class_dir, exist_ok=True)

    output_dir_list = []

    for label in class_mapping.keys():
        # Buat directory untuk gambar
        images_dir = os.path.join(output_dir, label, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Buat directory untuk labels
        labels_dir = os.path.join(output_dir, label, 'labels')
        os.makedirs(labels_dir, exist_ok=True)
        
        #buat mask kosong sesuai dimensi gambar
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Loop setiap shape
        for shape in data['shapes']:
            label_shape = shape['label']
            
            if label == label_shape:
                points = np.array(shape['points'], np.int32)
                cv2.fillPoly(mask, [points], 255)
            
        # Simpan mask ke folder labels
        mask_path = os.path.join(labels_dir, f"{base_name}{ext}")
        cv2.imwrite(mask_path, mask)
        print("Masks created.")  
        
        # Salin gambar ke directory images
        shutil.copy2(image_path, os.path.join(images_dir, f"{base_name}{ext}"))
        
        output_dir_list.append(str(os.path.join(output_dir, label)))
        
    return output_dir_list

def split_dataset(
    output_dir,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42,
    is_augmented=False,
    num_augmented=0
):
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
    
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
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

    for target_dir, assigned_groups in split_groups.items():
        for group_name in assigned_groups:
            for stem in groups[group_name]:
                image_name = image_paths[stem]
                mask_name = label_paths[stem]
                shutil.copy2(
                    os.path.join(images_dir, image_name),
                    os.path.join(target_dir, 'images', image_name)
                )
                shutil.copy2(
                    os.path.join(labels_dir, mask_name),
                    os.path.join(target_dir, 'labels', mask_name)
                )

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
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.CLAHE(p=0.3),  # Contrast Limited Adaptive Histogram Equalization
        A.RandomGamma(p=0.3),  # Random gamma correction
        A.GaussNoise(std_range=(0.2, 0.44), p=0.3),  # Gaussian noise
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),  # RGB channel shift
        A.ToGray(p=0.1),  # Convert to grayscale,
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

                mask = _read_mask(mask_file)
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
                    if aug_mask.ndim == 3:
                        _write_multilabel_mask(aug_mask, str(aug_mask_path))
                    else:
                        cv2.imwrite(str(aug_mask_path), aug_mask)

                    print (f"Finished augmentation image and mask for {img_file.name} - augment count {i+1}/{augment_count}")

def batch_generate_datasets_from_anylabeling(
    input_dir,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    is_augmented=False,
    num_augmented=5,
    multilabel=False
):
    """
    Proses batch untuk mengkonversi semua file JSON di directory input menjadi dataset Mask per kelas, lalu membagi dataset menjadi train, validation, dan test set.
    
    Args:
        input_dir (str): Directory input yang berisi file JSON berserta file gambar yang sesuai
        output_dir (str): Directory output Dataset
        train_ratio (float): Persentase data training
        val_ratio (float): Persentase data validation
        test_ratio (float): Persentase data test
        dataset_name (str): Nama dataset untuk file YAML
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    output_path = Path(output_dir)
    input_resolved = input_path.resolve()
    output_resolved = output_path.resolve()
    if (
        input_resolved == output_resolved
        or input_resolved in output_resolved.parents
        or output_resolved in input_resolved.parents
    ):
        raise ValueError("Input and output directories must not contain one another.")

    json_files = sorted(Path(input_dir).glob('*.json'))
    if not json_files:
        raise ValueError(f"No AnyLabeling JSON files found in {input_path}")

    if multilabel:
        labels = set()
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
            labels.update(shape["label"] for shape in data.get("shapes", []))
        if not labels:
            raise ValueError(f"No annotation labels found in {input_dir}")

        class_mapping = {label: index for index, label in enumerate(sorted(labels))}

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if multilabel:
        with open(os.path.join(output_dir, "classes.json"), "w", encoding="utf-8") as file:
            json.dump(class_mapping, file, indent=2)

        for json_file in json_files:
            base_name = json_file.stem
            image_path = next(
                (
                    os.path.join(input_dir, f"{base_name}{ext}")
                    for ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
                    if os.path.exists(os.path.join(input_dir, f"{base_name}{ext}"))
                ),
                None
            )
            if image_path is None:
                print(f"Matching image not found for {json_file}, skipping.")
                continue
            convert_anylabeling_to_mask(
                image_path,
                str(json_file),
                output_dir,
                class_mapping=class_mapping,
                multilabel=True
            )
            print(f"Processed {json_file}")

        split_dataset(
            output_dir,
            train_ratio,
            val_ratio,
            test_ratio,
            is_augmented=is_augmented,
            num_augmented=num_augmented
        )
        print(f"Multilabel classes (mask band order): {class_mapping}")
        return output_dir

    list_output_dir = []
    for json_file in json_files:
        base_name = json_file.stem
        image_path = next(
            (
                os.path.join(input_dir, f"{base_name}{ext}")
                for ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
                if os.path.exists(os.path.join(input_dir, f"{base_name}{ext}"))
            ),
            None
        )
        if image_path is None:
            print(f"Matching image not found for {json_file}, skipping.")
            continue
        output_list_mask_dir = convert_anylabeling_to_mask(image_path, str(json_file), output_dir)
        for list_dir in output_list_mask_dir:
            if list_dir not in list_output_dir:
                list_output_dir.append(list_dir)
        print(f"Processed {json_file}")
        
    print("list output dir:", list_output_dir)
    
    # Bagi dataset
    for dir in list_output_dir:
        split_dataset(
            dir,
            train_ratio,
            val_ratio,
            test_ratio,
            is_augmented=is_augmented,
            num_augmented=num_augmented
        )

    return output_dir