from collections import Counter
from pathlib import Path
from datasets import load_dataset
from common.constants.datasets_constants import HuggingFaceDataSetFields as HuggingFaceFields


class DatasetCalculator:
    def __init__(self, class_names):
        self.class_names = class_names

    def calculate_split_stats(self, labels):
        """Calculate per-class counts and percentages for a label sequence."""
        counts = Counter(labels)
        total = len(labels)

        classes = []
        for label_id, count in sorted(counts.items()):
            percentage = (count / total * 100) if total else 0.0
            classes.append({
                "name": self.class_names[label_id],
                "count": count,
                "percentage": percentage,
            })

        return {"total": total, "classes": classes}

    def print_split_statistics(self, split_name, summary):
        """Pretty-print class distribution stats for a split."""
        print(f"\n--- Statistics for {split_name} ---")
        for entry in summary["classes"]:
            friendly_name = normalize_class_name_display(entry["name"])
            print(f"{friendly_name: <35}: {entry['count']} images ({entry['percentage']:.1f}%)")
        print(f"Total images in {split_name}: {summary['total']}")

    def print_dataset_statistics(self, dataset_name, dataset, splits=None):
        """Process and print statistics for all or specific splits in a dataset."""
        if splits is None:
            splits = dataset.keys()

        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*50}")

        for split in splits:
            labels = dataset[split]["label"]
            summary = self.calculate_split_stats(labels)
            self.print_split_statistics(split, summary)


class FigshareDatasetCalculator:
    """Handles statistics for figshare dataset with folder-based structure."""

    def __init__(self, figshare_dir):
        self.figshare_dir = Path(figshare_dir)
        self.class_names = self._extract_class_names()

    def _extract_class_names(self):
        """Extract class names from the training_images directory."""
        training_dir = self.figshare_dir / "training images"
        if not training_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {training_dir}")

        class_names = sorted([
            d.name for d in training_dir.iterdir()
            if d.is_dir()
        ])
        return class_names

    def _normalize_class_name(self, class_name):
        """Extract base cancer type from class name (handles variants with metadata)."""
        # Remove metadata suffixes that start with underscore followed by location/staging info
        # e.g., "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib" -> "adenocarcinoma"
        base_name = class_name.split("_")[0]
        return base_name

    def count_images_in_split(self, split_name):
        """Count images per base class type in a given split."""
        split_dir = self.figshare_dir / f"{split_name} images"
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        counts = {}
        for class_name in self.class_names:
            class_dir = split_dir / class_name
            if class_dir.exists():
                image_count = len([
                    f for f in class_dir.rglob("*")
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ])
                # Group by normalized (base) class name
                base_class = self._normalize_class_name(class_name)
                if base_class not in counts:
                    counts[base_class] = 0
                counts[base_class] += image_count

        return counts

    def calculate_split_stats(self, split_name):
        """Calculate statistics for a figshare split with grouped cancer types."""
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
        """Pretty-print class distribution stats for a split."""
        print(f"\n--- Statistics for {split_name} ---")
        for entry in summary["classes"]:
            print(f"{entry['name']: <25}: {entry['count']} images ({entry['percentage']:.1f}%)")

        print(f"Total images in {split_name}: {summary['total']}")

    def print_dataset_statistics(self, dataset_name, splits=None):
        """Process and print statistics for all or specific splits."""
        if splits is None:
            splits = ["training", "validation", "test"]

        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*50}")

        for split in splits:
            try:
                summary = self.calculate_split_stats(split)
                self.print_split_statistics(split, summary)
            except FileNotFoundError:
                print(f"\nSplit '{split}' not found, skipping...")


def normalize_class_name_display(class_name):
    """Make class names more readable."""
    mappings = {
        "adenocarcinoma": "Adenocarcinoma",
        "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib": "Adenocarcinoma (T2 N0 M0)",
        "large.cell.carcinoma": "Large Cell Carcinoma",
        "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa": "Large Cell Carcinoma (T2 N2 M0)",
        "squamous.cell.carcinoma": "Squamous Cell Carcinoma",
        "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa": "Squamous Cell Carcinoma (T1 N2 M0)",
        "normal": "Normal",
    }
    return mappings.get(class_name, class_name)


class CombinedDatasetCalculator:

    @staticmethod
    def merge_summaries(summaries_list):
        total_images = 0
        class_totals = {}

        for summary in summaries_list:
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

    @staticmethod
    def print_summary(title, summary):
        """Pretty-print combined statistics."""
        print(f"\n{'='*50}")
        print(title)
        print(f"{'='*50}")
        for entry in summary["classes"]:
            friendly_name = normalize_class_name_display(entry["name"])
            print(f"{friendly_name: <35}: {entry['count']} images ({entry['percentage']:.1f}%)")
        print(f"Total images: {summary['total']}")


def normalize_class_name_display(class_name):
    """Convert class names to readable format."""
    mappings = {
        "adenocarcinoma": "Adenocarcinoma",
        "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib": "Adenocarcinoma (T2 N0 M0)",
        "large.cell.carcinoma": "Large Cell Carcinoma",
        "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa": "Large Cell Carcinoma (T2 N2 M0)",
        "squamous.cell.carcinoma": "Squamous Cell Carcinoma",
        "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa": "Squamous Cell Carcinoma (T1 N2 M0)",
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


if __name__ == "__main__":
    main()