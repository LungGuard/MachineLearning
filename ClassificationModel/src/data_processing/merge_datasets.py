from pathlib import Path
import shutil
from PIL import Image
from datasets import load_dataset
from constants.classification.datasets_constants import HuggingFaceDataSetFields as HuggingFaceFields


class DatasetMerger:
    
    def __init__(self, figshare_dir, huggingface_cache, output_dir):
        self.figshare_dir = Path(figshare_dir)
        self.huggingface_cache = Path(huggingface_cache)
        self.output_dir = Path(output_dir)
    
    def _base_type(self, name):
        return name.split("_")[0]
    
    def _copy_figshare(self, split):
        split_names = {"train": "training images", "validation": "validation images", "test": "test images"}
        split_dir = self.figshare_dir / split_names[split]
        
        if not split_dir.exists():
            return 0
        
        count = 0
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = self._base_type(class_dir.name)
            output_class_dir = self.output_dir / split / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            for img in class_dir.rglob("*"):
                if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    shutil.copy2(img, output_class_dir / f"fig_{img.name}")
                    count += 1
        
        return count
    
    def _copy_huggingface(self, split):
        dataset = load_dataset(HuggingFaceFields.DATASET_NAME, cache_dir=str(self.huggingface_cache))
        
        if split not in dataset:
            return 0
        
        split_data = dataset[split]
        class_names = dataset["train"].features["label"].names
        
        count = 0
        for idx, (image, label) in enumerate(zip(split_data["image"], split_data["label"])):
            class_name = self._base_type(class_names[label])
            output_class_dir = self.output_dir / split / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            if isinstance(image, str):
                image = Image.open(image)
            
            image.save(output_class_dir / f"hf_{split}_{idx}.png")
            count += 1
        
        return count
    
    def merge(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Merging datasets...")
        total = 0
        
        for split in ["train", "validation", "test"]:
            fig_count = self._copy_figshare(split)
            hf_count = self._copy_huggingface(split)
            split_total = fig_count + hf_count
            total += split_total
            print(f"{split}: {split_total} images ({fig_count} Figshare + {hf_count} HuggingFace)")
        
        print(f"\nTotal: {total} images merged to {self.output_dir}")


def main():
    """Main execution function."""
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    
    FIGSHARE_DIR = PROJECT_ROOT / "ClassificationModel" / "datasets" / "figshare_dataset"
    HUGGINGFACE_CACHE = PROJECT_ROOT / "ClassificationModel" / "datasets" / "hugging_face_dataset"
    OUTPUT_DIR = PROJECT_ROOT / "ClassificationModel" / "datasets" / "merged_dataset"
    
    merger = DatasetMerger(
        figshare_dir=FIGSHARE_DIR,
        huggingface_cache=HUGGINGFACE_CACHE,
        output_dir=OUTPUT_DIR,
        merge_variants=True
    )
    
    merger.merge()
    
    print("\n" + "="*70)
    print("✅ MERGE COMPLETE")
    print("="*70)
    print(f"Merged dataset available at: {OUTPUT_DIR}")




if __name__ == "__main__":
    main()


