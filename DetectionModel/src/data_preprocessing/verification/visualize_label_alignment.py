"""
Alignment verification: draw YOLO bboxes from .txt labels onto paired images.

Flags boxes whose interior region is near-black (letterbox padding) or
low-variance (smooth non-tissue), which would indicate a coordinate-translation
bug in the crop/pad → normalize pipeline chain.

Usage:
    python -m DetectionModel.src.data_preprocessing.verification.visualize_label_alignment
    python -m DetectionModel.src.data_preprocessing.verification.visualize_label_alignment \
        --dataset-dir /path/to/dataset --splits train val --sample 100 \
        --out-dir /path/to/output --black-threshold 8 --variance-threshold 3.0
"""

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from paths import ProjectPaths

logger = logging.getLogger(__name__)

DEFAULT_BLACK_THRESHOLD: float = 8.0
DEFAULT_VARIANCE_THRESHOLD: float = 3.0
BOX_OUTLINE_COLOR: tuple = (255, 0, 0)
BOX_LABEL_COLOR: tuple = (255, 255, 0)
BOX_WIDTH: int = 2
STATUS_OK = "OK"
STATUS_SUSPICIOUS = "SUSPICIOUS"
STATUS_MALFORMED = "MALFORMED"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
CSV_FIELDNAMES = ["split", "image", "class_id", "cx", "cy", "w", "h", "bbox_mean", "bbox_std", "status"]


def _find_image_for_label(label_path: Path, images_dir: Path) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / (label_path.stem + ext)
        if candidate.exists():
            return candidate
    return None


def _parse_label_line(line: str) -> Optional[tuple]:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return None
    if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
        return None
    return class_id, cx, cy, w, h


def _box_stats(pixels: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(pixels, dtype=np.float32)
    return float(arr.mean()), float(arr.std())


def _classify_box(mean: float, std: float, black_threshold: float, variance_threshold: float) -> str:
    if mean < black_threshold or std < variance_threshold:
        return STATUS_SUSPICIOUS
    return STATUS_OK


def _denormalize_box(cx: float, cy: float, w: float, h: float,
                     img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x1 = max(0, int((cx - w / 2) * img_w))
    y1 = max(0, int((cy - h / 2) * img_h))
    x2 = min(img_w - 1, int((cx + w / 2) * img_w))
    y2 = min(img_h - 1, int((cy + h / 2) * img_h))
    return x1, y1, x2, y2


def process_label_file(
    label_path: Path,
    images_dir: Path,
    split: str,
    out_split_dir: Path,
    black_threshold: float,
    variance_threshold: float,
) -> list[dict]:
    image_path = _find_image_for_label(label_path, images_dir)
    if image_path is None:
        logger.warning(f"No paired image found for label: {label_path.name}")
        return []

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    draw = ImageDraw.Draw(image)

    rows = []
    raw_lines = label_path.read_text().strip().splitlines()

    for line in raw_lines:
        parsed = _parse_label_line(line)

        if parsed is None:
            rows.append({
                "split": split, "image": label_path.stem,
                "class_id": "", "cx": "", "cy": "", "w": "", "h": "",
                "bbox_mean": "", "bbox_std": "", "status": STATUS_MALFORMED,
            })
            continue

        class_id, cx, cy, bw, bh = parsed
        x1, y1, x2, y2 = _denormalize_box(cx, cy, bw, bh, img_w, img_h)

        if x2 <= x1 or y2 <= y1:
            interior_mean, interior_std = 0.0, 0.0
        else:
            interior = image.crop((x1, y1, x2, y2))
            interior_mean, interior_std = _box_stats(interior)

        status = _classify_box(interior_mean, interior_std, black_threshold, variance_threshold)

        draw.rectangle([x1, y1, x2, y2], outline=BOX_OUTLINE_COLOR, width=BOX_WIDTH)
        draw.text((x1 + 2, y1 + 2), str(class_id), fill=BOX_LABEL_COLOR)

        rows.append({
            "split": split, "image": label_path.stem,
            "class_id": class_id, "cx": cx, "cy": cy, "w": bw, "h": bh,
            "bbox_mean": round(interior_mean, 3), "bbox_std": round(interior_std, 3),
            "status": status,
        })

    annotated_path = out_split_dir / f"{label_path.stem}_annotated.png"
    image.save(annotated_path)
    return rows


def collect_label_paths(labels_dir: Path, sample: Optional[int]) -> list[Path]:
    all_labels = sorted(labels_dir.glob("*.txt"))
    if sample is not None and sample < len(all_labels):
        return random.sample(all_labels, sample)
    return all_labels


def run_verification(
    dataset_dir: Path,
    splits: list[str],
    sample: Optional[int],
    out_dir: Path,
    black_threshold: float,
    variance_threshold: float,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for split in splits:
        labels_dir = dataset_dir / split / "labels"
        images_dir = dataset_dir / split / "images"

        if not labels_dir.exists():
            logger.warning(f"Labels dir not found, skipping split '{split}': {labels_dir}")
            continue

        out_split_dir = out_dir / split
        out_split_dir.mkdir(parents=True, exist_ok=True)

        label_paths = collect_label_paths(labels_dir, sample)
        logger.info(f"[{split}] Processing {len(label_paths)} label files …")

        for label_path in label_paths:
            rows = process_label_file(
                label_path, images_dir, split, out_split_dir,
                black_threshold, variance_threshold,
            )
            all_rows.extend(rows)

    report_path = out_dir / "alignment_report.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    return _print_summary(all_rows, report_path)


def _print_summary(rows: list[dict], report_path: Path) -> int:
    total = len(rows)
    ok_count = sum(1 for r in rows if r["status"] == STATUS_OK)
    suspicious_count = sum(1 for r in rows if r["status"] == STATUS_SUSPICIOUS)
    malformed_count = sum(1 for r in rows if r["status"] == STATUS_MALFORMED)

    print(f"\n{'='*60}")
    print(f"  Alignment Report: {report_path}")
    print(f"{'='*60}")
    print(f"  Total boxes  : {total}")
    print(f"  OK           : {ok_count}")
    print(f"  SUSPICIOUS   : {suspicious_count}")
    print(f"  MALFORMED    : {malformed_count}")

    worst = [r for r in rows if r["status"] == STATUS_SUSPICIOUS]
    worst.sort(key=lambda r: float(r["bbox_mean"]) if r["bbox_mean"] != "" else 0.0)
    if worst:
        print(f"\n  Worst SUSPICIOUS boxes (lowest bbox_mean):")
        for r in worst[:10]:
            print(f"    [{r['split']}] {r['image']}  cx={r['cx']:.4f} cy={r['cy']:.4f}  mean={r['bbox_mean']}  std={r['bbox_std']}")

    print(f"{'='*60}\n")
    return 0 if suspicious_count == 0 and malformed_count == 0 else 1


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize YOLO label alignment on training images."
    )
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=ProjectPaths.DETECTION_DATASETS_DIR,
        help="Root dataset directory (default: ProjectPaths.DETECTION_DATASETS_DIR)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="Dataset splits to check (default: train val test)",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Random sample N label files per split (default: all)",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=ProjectPaths.LOGS_DIR / "alignment_check",
        help="Output directory for overlays and report CSV",
    )
    parser.add_argument(
        "--black-threshold", type=float, default=DEFAULT_BLACK_THRESHOLD,
        help=f"bbox mean pixel value below which a box is SUSPICIOUS (default: {DEFAULT_BLACK_THRESHOLD})",
    )
    parser.add_argument(
        "--variance-threshold", type=float, default=DEFAULT_VARIANCE_THRESHOLD,
        help=f"bbox pixel std below which a box is SUSPICIOUS (default: {DEFAULT_VARIANCE_THRESHOLD})",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args()
    exit_code = run_verification(
        dataset_dir=args.dataset_dir,
        splits=args.splits,
        sample=args.sample,
        out_dir=args.out_dir,
        black_threshold=args.black_threshold,
        variance_threshold=args.variance_threshold,
    )
    sys.exit(exit_code)
