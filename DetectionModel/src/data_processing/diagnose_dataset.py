import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AnalysisThresholds:
    """Thresholds for image quality analysis."""
    uniform_std: float = 10.0
    too_dark_ratio: float = 0.95
    too_bright_mean: float = 180.0
    min_dark_ratio: float = 0.20
    min_contrast_range: int = 100

@dataclass
class ImageAnalysisResult:
    """Analysis results for a single image."""
    filepath: str
    filename: str
    split: str = "unknown"
    metadata: Dict = field(default_factory=dict)
    
    # Statistics
    mean: float = 0.0
    std: float = 0.0
    min_val: int = 0
    max_val: int = 0
    
    # Ratios
    dark_pixel_ratio: float = 0.0
    mid_pixel_ratio: float = 0.0
    bright_pixel_ratio: float = 0.0
    
    # Diagnosis
    is_uniform: bool = False
    is_too_dark: bool = False
    is_too_bright: bool = False
    is_missing_contrast: bool = False
    has_no_dark_background: bool = False
    
    # Verdict
    is_problematic: bool = False
    problem_type: str = "OK"


class DatasetDiagnoser:
    """
    Analyzes image quality in datasets and provides filtering capabilities.
    """
    
    def __init__(self, dataset_dir: str, thresholds: AnalysisThresholds = AnalysisThresholds()):
        self.dataset_dir = Path(dataset_dir)
        self.thresholds = thresholds
        self.results: List[ImageAnalysisResult] = []
        self._cached_dfs: Dict[str, pd.DataFrame] = {}

    def analyze(self) -> None:
        """Run full analysis on the dataset directory structure."""
        self.results = []
        self._cached_dfs = {} 

        # Support standard dataset structure: split/images
        for split in ['train', 'val', 'test']:
            image_dir = self.dataset_dir / split / 'images'
            if image_dir.exists():
                self._analyze_directory(image_dir, split)
            else:
                # Fallback: check if dataset_dir itself contains images
                if split == 'train': 
                     self._analyze_directory(self.dataset_dir, "root")

    def get_results_dataframe(self) -> pd.DataFrame:
        """Returns the full analysis results as a DataFrame."""
        if 'full' not in self._cached_dfs:
            if not self.results:
                return pd.DataFrame()
            
            data = []
            for r in self.results:
                row = vars(r).copy()
                if row['metadata']:
                    for k, v in row['metadata'].items():
                        row[k] = v
                del row['metadata']
                data.append(row)
            
            self._cached_dfs['full'] = pd.DataFrame(data)
            
        return self._cached_dfs['full']

    def get_problematic_images(self) -> pd.DataFrame:
        """Returns only images flagged as problematic."""
        df = self.get_results_dataframe()
        if df.empty: return df
        return df[df['is_problematic']].copy()

    def get_summary_report(self) -> Dict:
        """Generates a dictionary containing summary statistics."""
        df = self.get_results_dataframe()
        if df.empty: return {"error": "No data analyzed"}

        total_images = len(df)
        problematic_df = df[df['is_problematic']]
        num_problematic = len(problematic_df)
        
        summary = {
            "total_images": total_images,
            "problematic_count": num_problematic,
            "problematic_ratio": num_problematic / total_images if total_images > 0 else 0,
            "problem_breakdown": problematic_df['problem_type'].value_counts().to_dict(),
            "split_breakdown": {}
        }

        if 'split' in df.columns:
            for split in df['split'].unique():
                split_df = df[df['split'] == split]
                prob_count = len(split_df[split_df['is_problematic']])
                summary['split_breakdown'][split] = {
                    "total": len(split_df),
                    "problematic": prob_count,
                    "ratio": prob_count / len(split_df) if len(split_df) > 0 else 0
                }
        
        return summary

    def save_reports_to_disk(self, output_dir: str = None) -> None:
        """Saves all analysis dataframes to CSV files."""
        save_path = Path(output_dir) if output_dir else self.dataset_dir / "analysis"
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving analysis reports to {save_path}...")
        
        self.get_results_dataframe().to_csv(save_path / "image_analysis_full.csv", index=False)
        self.get_problematic_images().to_csv(save_path / "problematic_images.csv", index=False)
        
        # Save summary text
        summary = self.get_summary_report()
        with open(save_path / "summary_report.txt", "w") as f:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")
        
        logger.info("Reports saved successfully.")

    def filter_dataset_metadata(self, metadata_csv_path: str, output_csv_path: str) -> pd.DataFrame:
        """
        Filters the dataset metadata CSV to keep ONLY nodules where ALL slices are valid.
        
        Args:
            metadata_csv_path: Path to the original regression_dataset.csv
            output_csv_path: Path to save the cleaned CSV
            
        Returns:
            The cleaned DataFrame
        """
        df_results = self.get_results_dataframe()
        if df_results.empty:
            logger.warning("No analysis results found. Cannot filter.")
            return pd.DataFrame()

        # 1. Identify problematic nodules (PatientID + NoduleIdx)
        # Group by nodule and check if ANY slice is problematic
        problematic_nodules = df_results[df_results['is_problematic']]
        bad_nodule_keys: Set[Tuple[str, int]] = set()
        
        for _, row in problematic_nodules.iterrows():
            bad_nodule_keys.add((str(row['patient_id']), int(row['nodule_idx'])))
            
        logger.info(f"Found {len(bad_nodule_keys)} nodules with at least one bad slice.")

        # 2. Load original metadata
        meta_path = Path(metadata_csv_path)
        if not meta_path.exists():
            logger.error(f"Metadata file not found: {meta_path}")
            return pd.DataFrame()
            
        df_meta = pd.read_csv(meta_path)
        original_count = len(df_meta)
        
        # 3. Filter the metadata DataFrame
        # We define a helper function to check if a row belongs to a bad nodule
        def is_valid_nodule(row):
            key = (str(row['patient_id']), int(row['nodule_index']))
            return key not in bad_nodule_keys

        df_clean = df_meta[df_meta.apply(is_valid_nodule, axis=1)]
        clean_count = len(df_clean)
        
        # 4. Save
        df_clean.to_csv(output_csv_path, index=False)
        
        logger.info(f"Filtering Complete: {original_count} -> {clean_count} samples.")
        logger.info(f"Removed {original_count - clean_count} samples belonging to problematic nodules.")
        logger.info(f"Clean dataset saved to: {output_csv_path}")
        
        return df_clean

    def _analyze_directory(self, directory: Path, split: str):
        images = list(directory.glob("*.jpg")) + list(directory.glob("*.png"))
        logger.info(f"Analyzing {len(images)} images in {split}...")
        
        for idx, filepath in enumerate(images):
            if (idx + 1) % 500 == 0:
                logger.info(f"Progress: {idx + 1}/{len(images)}")
            
            result = self._analyze_single_image(filepath, split)
            self.results.append(result)

    def _analyze_single_image(self, filepath: Path, split: str) -> ImageAnalysisResult:
        img = cv2.imread(str(filepath))
        metadata = self._extract_metadata_from_filename(filepath.name)

        if img is None:
            return ImageAnalysisResult(
                filepath=str(filepath), filename=filepath.name, split=split, metadata=metadata,
                is_problematic=True, problem_type="UNREADABLE"
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        mean_val = float(gray.mean())
        std_val = float(gray.std())
        min_val = int(gray.min())
        max_val = int(gray.max())
        
        total_pixels = gray.size
        dark_pixels = np.sum(gray < 50)
        mid_pixels = np.sum((gray >= 50) & (gray <= 200))
        bright_pixels = np.sum(gray > 200)
        
        dark_ratio = dark_pixels / total_pixels
        mid_ratio = mid_pixels / total_pixels
        bright_ratio = bright_pixels / total_pixels
        
        is_uniform = std_val < self.thresholds.uniform_std
        is_too_dark = dark_ratio > self.thresholds.too_dark_ratio
        is_too_bright = mean_val > self.thresholds.too_bright_mean
        is_missing_contrast = (max_val - min_val) < self.thresholds.min_contrast_range
        has_no_dark_background = dark_ratio < self.thresholds.min_dark_ratio and mean_val > 100
        
        problems = []
        if is_uniform: problems.append("UNIFORM")
        if is_too_dark: problems.append("TOO_DARK")
        if is_too_bright: problems.append("TOO_BRIGHT")
        if is_missing_contrast and not is_uniform: problems.append("LOW_CONTRAST")
        if has_no_dark_background and not is_too_bright: problems.append("NO_BG")
        
        return ImageAnalysisResult(
            filepath=str(filepath), filename=filepath.name, split=split, metadata=metadata,
            mean=round(mean_val, 2), std=round(std_val, 2),
            min_val=min_val, max_val=max_val,
            dark_pixel_ratio=round(dark_ratio, 3), mid_pixel_ratio=round(mid_ratio, 3), bright_pixel_ratio=round(bright_ratio, 3),
            is_uniform=is_uniform, is_too_dark=is_too_dark, is_too_bright=is_too_bright,
            is_missing_contrast=is_missing_contrast, has_no_dark_background=has_no_dark_background,
            is_problematic=len(problems) > 0, problem_type="; ".join(problems) if problems else "OK"
        )

    def _extract_metadata_from_filename(self, filename: str) -> Dict:
        name = Path(filename).stem
        match = re.match(r'^(.+)_n(\d+)_z(\d+)$', name)
        if match:
            return {
                "patient_id": match.group(1),
                "nodule_idx": int(match.group(2)),
                "slice_idx": int(match.group(3))
            }
        return {}

# Usage Example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnose_class.py <dataset_dir>")
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    
    # Initialize
    diagnoser = DatasetDiagnoser(dataset_path)
    
    # Run
    diagnoser.analyze()
    
    # Get Data
    df_all = diagnoser.get_results_dataframe()
    df_problems = diagnoser.get_problematic_images()
    summary = diagnoser.get_summary_report()
    
    # Print Summary
    print("\n--- Analysis Summary ---")
    print(f"Total Images: {summary['total_images']}")
    print(f"Problematic: {summary['problematic_count']} ({summary['problematic_ratio']:.1%})")
    print("\nBreakdown by Issue:")
    for issue, count in summary.get('problem_breakdown', {}).items():
        print(f"  - {issue}: {count}")
    
    # Example: Accessing the DataFrame directly
    if not df_problems.empty:
        print("\nSample problematic files:")
        print(df_problems[['filename', 'problem_type']].head())