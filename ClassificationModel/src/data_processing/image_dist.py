from pathlib import Path


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
            splits = ["train", "validation", "test"]
        
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
    """Load and display statistics for all datasets."""
    # HuggingFace dataset
    hf_dataset = load_dataset(HuggingFaceFields.DATASET_NAME, cache_dir=HuggingFaceFields.CACHE_DIR)
    hf_class_names = hf_dataset["train"].features["label"].names
    hf_calculator = DatasetCalculator(hf_class_names)
    hf_calculator.print_dataset_statistics("HuggingFace - Lung Cancer", hf_dataset)

    # Figshare dataset
    figshare_dir = Path(__file__).parent.parent / "datasets" / "figshare_dataset"
    figshare_calculator = FigshareDatasetCalculator(figshare_dir)
    figshare_calculator.print_dataset_statistics("Figshare - Lung Cancer")

    # Combined statistics
    hf_summaries = []
    for split in hf_dataset.keys():
        labels = hf_dataset[split]["label"]
        summary = hf_calculator.calculate_split_stats(labels)
        hf_summaries.append(summary)

    figshare_summaries = []
    for split in ["training", "validation", "test"]:
        try:
            summary = figshare_calculator.calculate_split_stats(split)
            figshare_summaries.append(summary)
        except FileNotFoundError:
            pass

    all_summaries = hf_summaries + figshare_summaries
    combined_summary = CombinedDatasetCalculator.merge_summaries(all_summaries)
    CombinedDatasetCalculator.print_summary("Combined Statistics (All Data)", combined_summary)


def main():
    merged_dataset_dir = Path(__file__).parent.parent.parent.parent / "ClassificationModel" / "datasets" / "merged_dataset"
    
    calculator = DatasetCalculator(merged_dataset_dir)
    calculator.print_dataset_statistics()


if __name__ == "__main__":
    main()