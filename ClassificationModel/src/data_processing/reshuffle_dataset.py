import os
import shutil
import random
from pathlib import Path
from constants.classification.datasets_constants import DatasetConstants
# --- CONFIGURATION ---
# The path to your CURRENT (problematic) merged dataset
SOURCE_DATASET_DIR = DatasetConstants.UNIFIED_DATASET_DIR 

# The path where the NEW, balanced dataset will be created
OUTPUT_DATASET_DIR = DatasetConstants.DATASETS_DIR / 'unified_dataset_v2'

# The split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Seed for reproducibility - ensures you get the same split every time you run this
RANDOM_SEED = 42

def create_reshuffled_dataset():
    # 1. Verification
    if not SOURCE_DATASET_DIR.exists():
        print(f"Error: Source directory not found at {SOURCE_DATASET_DIR}")
        return

    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-5:
        print("Error: Ratios must sum to 1.0")
        return

    # 2. Prepare Output Directory
    if OUTPUT_DATASET_DIR.exists():
        print(f"Removing existing output directory: {OUTPUT_DATASET_DIR}")
        shutil.rmtree(OUTPUT_DATASET_DIR)
    
    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for splits
    for split in ['train', 'validation', 'test']:
        (OUTPUT_DATASET_DIR / split).mkdir()

    print(f"Scanning source data from: {SOURCE_DATASET_DIR}...")

    # 3. Identify Classes
    # We look into the 'train' folder of the source to find class names
    # Assuming structure: source / train / class_name
    class_names = [d.name for d in (SOURCE_DATASET_DIR / 'train').iterdir() if d.is_dir()]
    print(f"Found classes: {class_names}")

    # 4. Process each class
    random.seed(RANDOM_SEED)
    
    for class_name in class_names:
        print(f"\nProcessing class: {class_name}...")
        
        # Collect ALL images for this class from existing train/val/test folders
        all_images = []
        for split in ['train', 'validation', 'test']:
            source_class_dir = SOURCE_DATASET_DIR / split / class_name
            if source_class_dir.exists():
                images = list(source_class_dir.glob('*.*')) # Grab all files
                # Filter for valid image extensions if needed
                images = [img for img in images if img.suffix.lower() in ['.png', '.jpg', '.jpeg']]
                all_images.extend(images)
        
        total_images = len(all_images)
        print(f"  - Found {total_images} images total.")
        
        if total_images == 0:
            print(f"  - Warning: No images found for {class_name}, skipping.")
            continue

        # Shuffle the images randomly
        random.shuffle(all_images)

        # Calculate split indices
        train_count = int(total_images * TRAIN_RATIO)
        val_count = int(total_images * VAL_RATIO)
        # Test gets the remainder to ensure no image is lost due to rounding
        
        train_imgs = all_images[:train_count]
        val_imgs = all_images[train_count : train_count + val_count]
        test_imgs = all_images[train_count + val_count :]
        
        print(f"  - Splitting: Train ({len(train_imgs)}), Val ({len(val_imgs)}), Test ({len(test_imgs)})")

        # Function to copy files
        def copy_files(file_list, destination_split):
            dest_dir = OUTPUT_DATASET_DIR / destination_split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img_path in file_list:
                shutil.copy2(img_path, dest_dir / img_path.name)

        # Copy the files to the new destination
        copy_files(train_imgs, 'train')
        copy_files(val_imgs, 'validation')
        copy_files(test_imgs, 'test')

    print("\n" + "="*50)
    print("✅ DATASET RESHUFFLING COMPLETE")
    print(f"New dataset is located at: {OUTPUT_DATASET_DIR}")
    print("="*50)

if __name__ == "__main__":
    create_reshuffled_dataset()