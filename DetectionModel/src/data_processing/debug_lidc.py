"""
Diagnostic Script: Identify Problematic CT Slice Images

This script analyzes all generated images and identifies which ones are:
1. Completely blank (uniform/no variation)
2. Too dark (no lung tissue visible)
3. Too bright/washed out (grey screen issue)
4. Missing proper lung structure

A valid lung CT slice should have:
- Black background (air outside body)
- Dark grey lung regions
- Light grey/white soft tissue and bone
- Significant pixel variation (std > threshold)

Output: CSV report with problematic images and their scan info
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ImageAnalysis:
    """Analysis results for a single image."""
    filepath: str
    patient_id: str
    nodule_idx: int
    slice_idx: int
    
    # Basic statistics
    mean: float
    std: float
    min_val: int
    max_val: int
    
    # Histogram analysis
    dark_pixel_ratio: float      # pixels < 50 (should be high for valid CT)
    mid_pixel_ratio: float       # pixels 50-200
    bright_pixel_ratio: float    # pixels > 200
    
    # Problem flags
    is_uniform: bool             # std < threshold (blank image)
    is_too_dark: bool            # almost all black
    is_too_bright: bool          # almost all bright (grey screen)
    is_missing_contrast: bool    # no proper dark/light regions
    
    # Overall verdict
    is_problematic: bool
    problem_type: str


def parse_filename(filename: str) -> Tuple[str, int, int]:
    """
    Extract patient_id, nodule_idx, slice_idx from filename.
    Format: {patient_id}_n{nodule_idx:02d}_z{slice_idx:04d}.jpg
    Example: LIDC-IDRI-0001_n00_z0125.jpg
    """
    # Remove extension
    name = Path(filename).stem
    
    # Pattern: everything before _n is patient_id, then nodule and slice
    pattern = r'^(.+)_n(\d+)_z(\d+)$'
    match = re.match(pattern, name)
    
    if match:
        patient_id = match.group(1)
        nodule_idx = int(match.group(2))
        slice_idx = int(match.group(3))
        return patient_id, nodule_idx, slice_idx
    
    # Fallback
    return name, -1, -1


def analyze_image(filepath: Path) -> ImageAnalysis:
    """Analyze a single image for quality issues."""
    
    # Read image
    img = cv2.imread(str(filepath))
    
    if img is None:
        patient_id, nodule_idx, slice_idx = parse_filename(filepath.name)
        return ImageAnalysis(
            filepath=str(filepath),
            patient_id=patient_id,
            nodule_idx=nodule_idx,
            slice_idx=slice_idx,
            mean=0, std=0, min_val=0, max_val=0,
            dark_pixel_ratio=0, mid_pixel_ratio=0, bright_pixel_ratio=0,
            is_uniform=True, is_too_dark=True, is_too_bright=False,
            is_missing_contrast=True,
            is_problematic=True,
            problem_type="UNREADABLE"
        )
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Basic statistics
    mean_val = float(gray.mean())
    std_val = float(gray.std())
    min_val = int(gray.min())
    max_val = int(gray.max())
    
    # Pixel distribution analysis
    total_pixels = gray.size
    dark_pixels = np.sum(gray < 50)
    mid_pixels = np.sum((gray >= 50) & (gray <= 200))
    bright_pixels = np.sum(gray > 200)
    
    dark_ratio = dark_pixels / total_pixels
    mid_ratio = mid_pixels / total_pixels
    bright_ratio = bright_pixels / total_pixels
    
    # Problem detection thresholds
    UNIFORM_STD_THRESHOLD = 10.0        # Very low variation = blank
    TOO_DARK_THRESHOLD = 0.95           # >95% dark pixels = mostly black
    TOO_BRIGHT_MEAN_THRESHOLD = 180     # Mean > 180 = grey screen
    MIN_DARK_RATIO = 0.20               # Valid CT should have >20% dark (air/background)
    MIN_CONTRAST_RANGE = 100            # max - min should be > 100
    
    # Detect problems
    is_uniform = std_val < UNIFORM_STD_THRESHOLD
    is_too_dark = dark_ratio > TOO_DARK_THRESHOLD
    is_too_bright = mean_val > TOO_BRIGHT_MEAN_THRESHOLD
    is_missing_contrast = (max_val - min_val) < MIN_CONTRAST_RANGE
    
    # Additional check: valid lung CT should have dark background
    # If dark_ratio is very low, it might be grey screen
    has_no_dark_background = dark_ratio < MIN_DARK_RATIO and mean_val > 100
    
    # Determine problem type
    problem_types = []
    is_problematic = False
    
    if is_uniform:
        problem_types.append("UNIFORM/BLANK")
        is_problematic = True
    
    if is_too_dark:
        problem_types.append("TOO_DARK")
        is_problematic = True
    
    if is_too_bright:
        problem_types.append("TOO_BRIGHT/GREY_SCREEN")
        is_problematic = True
    
    if is_missing_contrast and not is_uniform:
        problem_types.append("LOW_CONTRAST")
        is_problematic = True
    
    if has_no_dark_background and not is_too_bright:
        problem_types.append("NO_DARK_BACKGROUND")
        is_problematic = True
    
    problem_type = "; ".join(problem_types) if problem_types else "OK"
    
    # Parse filename
    patient_id, nodule_idx, slice_idx = parse_filename(filepath.name)
    
    return ImageAnalysis(
        filepath=str(filepath),
        patient_id=patient_id,
        nodule_idx=nodule_idx,
        slice_idx=slice_idx,
        mean=round(mean_val, 2),
        std=round(std_val, 2),
        min_val=min_val,
        max_val=max_val,
        dark_pixel_ratio=round(dark_ratio, 4),
        mid_pixel_ratio=round(mid_ratio, 4),
        bright_pixel_ratio=round(bright_ratio, 4),
        is_uniform=is_uniform,
        is_too_dark=is_too_dark,
        is_too_bright=is_too_bright,
        is_missing_contrast=is_missing_contrast,
        is_problematic=is_problematic,
        problem_type=problem_type
    )


def analyze_directory(image_dir: Path) -> List[ImageAnalysis]:
    """Analyze all images in a directory."""
    results = []
    
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    total = len(image_files)
    
    logger.info(f"Analyzing {total} images in {image_dir}...")
    
    for idx, filepath in enumerate(image_files):
        if (idx + 1) % 100 == 0:
            logger.info(f"  Progress: {idx + 1}/{total}")
        
        analysis = analyze_image(filepath)
        results.append(analysis)
    
    return results


def generate_report(results: List[ImageAnalysis], output_dir: Path) -> Dict:
    """Generate analysis report and save to CSV."""
    
    # Convert to DataFrame
    df = pd.DataFrame([vars(r) for r in results])
    
    # Summary statistics
    total_images = len(df)
    problematic_images = df[df['is_problematic']]
    num_problematic = len(problematic_images)
    
    # Group by problem type
    problem_counts = problematic_images['problem_type'].value_counts().to_dict()
    
    # Group by patient
    problems_by_patient = problematic_images.groupby('patient_id').size().sort_values(ascending=False)
    
    # Patients with ALL images problematic vs SOME
    patient_stats = df.groupby('patient_id').agg({
        'is_problematic': ['sum', 'count']
    })
    patient_stats.columns = ['problematic_count', 'total_count']
    patient_stats['problematic_ratio'] = patient_stats['problematic_count'] / patient_stats['total_count']
    
    all_problematic_patients = patient_stats[patient_stats['problematic_ratio'] == 1.0].index.tolist()
    some_problematic_patients = patient_stats[
        (patient_stats['problematic_ratio'] > 0) & (patient_stats['problematic_ratio'] < 1.0)
    ].index.tolist()
    
    # Per-split statistics
    split_stats = {}
    if 'split' in df.columns:
        for split_name in df['split'].unique():
            split_df = df[df['split'] == split_name]
            split_problematic = split_df[split_df['is_problematic']]
            split_stats[split_name] = {
                'total': len(split_df),
                'problematic': len(split_problematic),
                'ratio': round(len(split_problematic) / len(split_df), 4) if len(split_df) > 0 else 0
            }
    
    # Save full results
    full_csv_path = output_dir / "image_analysis_full.csv"
    df.to_csv(full_csv_path, index=False)
    
    # Save only problematic images
    problematic_csv_path = output_dir / "problematic_images.csv"
    problematic_images.to_csv(problematic_csv_path, index=False)
    
    # Save patient-level summary
    patient_summary_path = output_dir / "patient_summary.csv"
    patient_stats.to_csv(patient_summary_path)
    
    # Generate summary report
    summary = {
        "total_images": total_images,
        "problematic_images": num_problematic,
        "problematic_ratio": round(num_problematic / total_images, 4) if total_images > 0 else 0,
        "problem_counts": problem_counts,
        "split_stats": split_stats,
        "patients_all_problematic": all_problematic_patients,
        "patients_some_problematic": some_problematic_patients,
        "output_files": {
            "full_analysis": str(full_csv_path),
            "problematic_only": str(problematic_csv_path),
            "patient_summary": str(patient_summary_path)
        }
    }
    
    return summary


def print_summary(summary: Dict):
    """Print formatted summary to console."""
    print("\n" + "=" * 70)
    print("IMAGE QUALITY ANALYSIS REPORT")
    print("=" * 70)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total images analyzed: {summary['total_images']}")
    print(f"   Problematic images: {summary['problematic_images']} ({summary['problematic_ratio']:.1%})")
    
    # Per-split statistics
    if summary.get('split_stats'):
        print(f"\n📈 PER-SPLIT BREAKDOWN:")
        for split_name in sorted(summary['split_stats'].keys()):
            stats = summary['split_stats'][split_name]
            print(f"   {split_name.capitalize():5s}: {stats['problematic']:4d}/{stats['total']:4d} problematic ({stats['ratio']:.1%})")
    
    print(f"\n🔴 PROBLEM BREAKDOWN:")
    for problem_type, count in summary['problem_counts'].items():
        print(f"   {problem_type}: {count}")
    
    print(f"\n👤 PATIENT ANALYSIS:")
    all_prob = summary['patients_all_problematic']
    some_prob = summary['patients_some_problematic']
    
    print(f"   Patients with ALL images problematic: {len(all_prob)}")
    if all_prob:
        print(f"      Examples: {all_prob[:10]}{'...' if len(all_prob) > 10 else ''}")
    
    print(f"   Patients with SOME images problematic: {len(some_prob)}")
    if some_prob:
        print(f"      Examples: {some_prob[:10]}{'...' if len(some_prob) > 10 else ''}")
    
    print(f"\n📁 OUTPUT FILES:")
    for name, path in summary['output_files'].items():
        print(f"   {name}: {path}")
    
    print("\n" + "=" * 70)
    
    # Diagnosis
    print("\n🔍 DIAGNOSIS:")
    
    if len(all_prob) > 0 and len(some_prob) == 0:
        print("   → ALL problematic images come from specific patients")
        print("   → This suggests a SCAN-LEVEL issue (offset, corrupted data)")
        print("   → Check these patients' raw DICOM data")
    
    elif len(some_prob) > len(all_prob):
        print("   → Problems occur randomly across patients")
        print("   → This suggests a SLICE-LEVEL issue (coordinate transform, slice selection)")
        print("   → Check the slice index calculation logic")
    
    elif summary['problematic_ratio'] > 0.5:
        print("   → More than half of images are problematic")
        print("   → This suggests a PIPELINE-WIDE issue (windowing, preprocessing)")
        print("   → Check the volume preprocessing steps")
    
    print("=" * 70 + "\n")


def main(dataset_dir: str, output_dir: str = None):
    """
    Main entry point.
    
    Args:
        dataset_dir: Path to the YOLO dataset directory (containing train/val/test subdirs)
        output_dir: Where to save the reports (defaults to dataset_dir/analysis)
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir) if output_dir else dataset_path / "analysis"
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Analyze each split
    for split in ['train', 'val', 'test']:
        image_dir = dataset_path / split / 'images'
        
        if image_dir.exists():
            logger.info(f"\n{'='*50}")
            logger.info(f"Analyzing {split} split...")
            logger.info(f"{'='*50}")
            
            results = analyze_directory(image_dir)
            
            # Add split info
            for r in results:
                r.split = split
            
            all_results.extend(results)
            
            # Quick summary for this split
            problematic = sum(1 for r in results if r.is_problematic)
            logger.info(f"  {split}: {problematic}/{len(results)} problematic")
        else:
            logger.warning(f"  {split}/images directory not found, skipping")
    
    if not all_results:
        logger.error("No images found to analyze!")
        return
    
    # Generate full report
    logger.info("\nGenerating report...")
    summary = generate_report(all_results, output_path)
    
    print_summary(summary)
    
    return summary


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnose_images.py <dataset_dir> [output_dir]")
        print("Example: python diagnose_images.py ./DetectionModel/datasets")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(dataset_dir, output_dir)