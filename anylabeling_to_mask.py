import json
import os
import numpy as np
from pathlib import Path
import shutil
import random
import cv2
import albumentations as A

def convert_anylabeling_to_mask(image_path, json_file, output_dir, class_mapping=None):
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
    
    # Buat folder class
    for label in class_mapping.keys():
        class_dir = os.path.join(output_dir, label)
        os.makedirs(class_dir, exist_ok=True)

    # Ambil dimensi gambar
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Base name untuk mask
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    ext = os.path.splitext(os.path.basename(image_path))[1].lower()
    
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

def split_dataset(output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
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
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    
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
            ext_file = os.path.splitext(img_file)[1].lower()
            mask_file = f"{base_name}{ext_file}"
            if os.path.exists(os.path.join(labels_dir, mask_file)):
                shutil.copy2(
                    os.path.join(labels_dir, mask_file),
                    os.path.join(target_dir, 'labels', mask_file)
                )
    
    # Salin file ke masing-masing set
    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)
    
    # Print statistik
    print(f"\nDataset split complete:")
    print(f"Total files: {n_files}")
    print(f"Training set: {len(train_files)} files ({train_ratio*100:.1f}%)")
    print(f"Validation set: {len(val_files)} files ({val_ratio*100:.1f}%)")
    print(f"Test set: {len(test_files)} files ({test_ratio*100:.1f}%)")
    
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

def process_directory(input_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, is_augmented=False, num_augmented=5):
    """
    Proses semua file JSON dalam directory, salin gambar yang sesuai, dan bagi dataset
    
    Args:
        input_dir (str): Directory input yang berisi file JSON
        output_dir (str): Directory output untuk file YOLO OBB
        train_ratio (float): Persentase data training
        val_ratio (float): Persentase data validation
        test_ratio (float): Persentase data test
        dataset_name (str): Nama dataset untuk file YAML
    """
    # Hapus output directory jika sudah ada
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Buat output directory
    os.makedirs(output_dir, exist_ok=True)
    
    list_output_dir = []
    
    # Proses semua file JSON
    for json_file in Path(input_dir).glob('*.json'):
        # Salin file gambar yang sesuai
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        # Coba beberapa ekstensi gambar yang umum
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            img_file = os.path.join(input_dir, f"{base_name}{ext}")
            if os.path.exists(img_file):
                # Proses file JSON
                output_list_mask_dir = convert_anylabeling_to_mask(img_file, str(json_file), output_dir)
                for list_dir in output_list_mask_dir:
                    if(list_dir not in list_output_dir):
                        list_output_dir.append(list_dir)
                break
        
        print(f"Processed {json_file}")
        
    print("list output dir:", list_output_dir)
    
    # Bagi dataset
    for output_dir in list_output_dir:
        if is_augmented:
            augment_dataset(
                image_dir=os.path.join(output_dir, 'images'),
                label_dir=os.path.join(output_dir, 'labels'),
                output_dir=output_dir,
                augment_count=num_augmented
            )
        
        split_dataset(output_dir, train_ratio, val_ratio, test_ratio)

if __name__ == "__main__":
    # Contoh penggunaan
    input_dir = r"Test Annotation Anylabeling/Annotation"
    output_dir = r"Datasets Maks Anylabeling"
    is_augmented = True
    num_augmented = 5
    
    # Bagi dataset dengan rasio 70% training, 20% validation, 10% test
    process_directory(
        input_dir, 
        output_dir, 
        train_ratio=0.7, 
        val_ratio=0.2, 
        test_ratio=0.1,
        is_augmented=is_augmented,
        num_augmented=num_augmented
    ) 