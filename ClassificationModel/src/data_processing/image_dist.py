from pathlib import Path
import sys

# Add src directory to Python's module search path so it can find 'common' package
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from constants.classification.datasets_constants import DatasetConstants


class DatasetCalculator:
    
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
    
    def count_images_in_split(self, split_name):
        split_dir = self.dataset_dir / split_name
        if not split_dir.exists():
            return {}
        
        counts = {}
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                image_count = len([
                    f for f in class_dir.rglob("*")
                    if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ])
                counts[class_dir.name] = image_count
        
        return counts
    
    def calculate_split_stats(self, split_name):
        counts = self.count_images_in_split(split_name)
        total = sum(counts.values())
        
        classes = []
        for class_name in sorted(counts.keys()):
            count = counts[class_name]
            percentage = (count / total * 100) if total else 0.0
            classes.append({
                "name": class_name,
                "count": count,
                "percentage": percentage,
            })
        
        return {"total": total, "classes": classes}
    
    def print_split_statistics(self, split_name, summary):
        print(f"\n--- {split_name.upper()} ---")
        for entry in summary["classes"]:
            friendly_name = normalize_class_name(entry["name"])
            print(f"{friendly_name: <30}: {entry['count']} images ({entry['percentage']:.1f}%)")
        print(f"Total: {summary['total']} images")
    
    def print_dataset_statistics(self, splits=None):
        if splits is None:
            splits = [DatasetConstants.TRAIN_SPLIT_NAME, DatasetConstants.VAL_SPLIT_NAME, DatasetConstants.TEST_SPLIT_NAME]
        
        print(f"\n{'='*60}")
        print(f"Dataset Statistics")
        print(f"{'='*60}")
        
        all_summaries = []
        for split in splits:
            summary = self.calculate_split_stats(split)
            if summary["total"] > 0:
                self.print_split_statistics(split, summary)
                all_summaries.append(summary)
        
        if all_summaries:
            combined = self._merge_summaries(all_summaries)
            print(f"\n{'='*60}")
            print("TOTAL (ALL SPLITS)")
            print(f"{'='*60}")
            for entry in combined["classes"]:
                friendly_name = normalize_class_name(entry["name"])
                print(f"{friendly_name: <30}: {entry['count']} images ({entry['percentage']:.1f}%)")
            print(f"Total: {combined['total']} images")
    
    def _merge_summaries(self, summaries):
        total_images = 0
        class_totals = {}
        
        for summary in summaries:
            total_images += summary["total"]
            for entry in summary["classes"]:
                class_name = entry["name"]
                if class_name not in class_totals:
                    class_totals[class_name] = 0
                class_totals[class_name] += entry["count"]
        
        classes = []
        for class_name in sorted(class_totals.keys()):
            count = class_totals[class_name]
            percentage = (count / total_images * 100) if total_images else 0.0
            classes.append({
                "name": class_name,
                "count": count,
                "percentage": percentage,
            })
        
        return {"total": total_images, "classes": classes}


def normalize_class_name(class_name):
    mappings = {
        "adenocarcinoma": "Adenocarcinoma",
        "large.cell.carcinoma": "Large Cell Carcinoma",
        "squamous.cell.carcinoma": "Squamous Cell Carcinoma",
        "normal": "Normal",
    }
    return mappings.get(class_name, class_name)


def main():
    calculator = DatasetCalculator(DatasetConstants.DATASETS_DIR / 'unified_dataset_v2')
    calculator.print_dataset_statistics()


if __name__ == "__main__":
    main()