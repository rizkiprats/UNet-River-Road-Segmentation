from dataset_conversion.anylabeling_to_mask import batch_generate_datasets_from_anylabeling
from dataset_conversion.geojson_to_mask import batch_generate_datasets_from_geojson

if __name__ == "__main__":
    input_dir = r"Annotation_Data_Collection/Anylabeling_format/building_tree"
    output_dir = r"Datasets/Building_Tree_Dataset"
    is_augmented = True
    num_augmented = 5
    multilabel = False
    
    batch_generate_datasets_from_anylabeling(
        input_dir, 
        output_dir, 
        train_ratio=0.7, 
        val_ratio=0.2, 
        test_ratio=0.1,
        is_augmented=is_augmented,
        num_augmented=num_augmented,
        multilabel=multilabel
    )

    output_dir = r"Datasets/Building_Tree_Dataset_Multilabel"
    multilabel = True

    batch_generate_datasets_from_anylabeling(
        input_dir, 
        output_dir, 
        train_ratio=0.7, 
        val_ratio=0.2, 
        test_ratio=0.1,
        is_augmented=is_augmented,
        num_augmented=num_augmented,
        multilabel=multilabel
    )

    input_dir = r"Annotation_Data_Collection/Geojson_format/river_road"
    output_dir = r"Datasets/River_Road_Dataset"
    is_augmented = True
    num_augmented = 5
    multilabel = False

    batch_generate_datasets_from_geojson(
        input_dir, 
        output_dir, 
        train_ratio=0.7, 
        val_ratio=0.2, 
        test_ratio=0.1,
        is_augmented=is_augmented,
        num_augmented=num_augmented,
        multilabel=multilabel
    )