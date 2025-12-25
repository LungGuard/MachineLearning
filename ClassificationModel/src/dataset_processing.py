from collections import Counter
from datasets import load_dataset
from common.constants.datasets_constants import HuggingFaceDataSetFields as HuggingFaceFields

def load_lung_cancer_dataset(cache_dir=HuggingFaceFields.CACHE_DIR):
    return load_dataset(HuggingFaceFields.DATASET_NAME, cache_dir=cache_dir)


def get_class_names(dataset):

    return dataset["train"].features["label"].names


def summarize_split(labels, class_names):
    """calculate per-class counts and percentages for a label sequence."""

    counts = Counter(labels)
    total = len(labels)

    classes = []
    for label_id, count in sorted(counts.items()):
        percentage = (count / total * 100) if total else 0.0
        classes.append({
            "name": class_names[label_id],
            "count": count,
            "percentage": percentage,
        })

    return {"total": total, "classes": classes}


def print_split_statistics(split_name, summary):

    print(f"\n--- Statistics for {split_name} ---")
    for entry in summary["classes"]:
        print(f"{entry['name']: <25}: {entry['count']} images ({entry['percentage']:.1f}%)")

    print(f"Total images in {split_name}: {summary['total']}")


def main(cache_dir=HuggingFaceFields.CACHE_DIR):
    dataset = load_lung_cancer_dataset(cache_dir=cache_dir)
    class_names = get_class_names(dataset)

    for split in dataset.keys():
        labels = dataset[split]["label"]
        summary = summarize_split(labels, class_names)
        print_split_statistics(split, summary)


if __name__ == "__main__":
    main()