import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
import re
import logging
import shutil

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
    Analyzes image quality in datasets and provides filtering/cleaning capabilities.
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
        """Saves all analysis dataframes (Full, Nodule, Patient, Summary) to CSV files."""
        save_path = Path(output_dir) if output_dir else self.dataset_dir / "analysis"
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving analysis reports to {save_path}...")
        
        df = self.get_results_dataframe()
        if df.empty:
            logger.warning("No results to save.")
            return

        # 1. Full Image Analysis
        df.to_csv(save_path / "image_analysis_full.csv", index=False)
        
        # 2. Problematic Only
        self.get_problematic_images().to_csv(save_path / "problematic_images.csv", index=False)
        
        # --- NEW: Nodule Analysis (Aggregation) ---
        # Group by Patient + NoduleIdx
        if 'nodule_idx' in df.columns:
            nodule_stats = df.groupby(['patient_id', 'nodule_idx']).agg(
                total_images=('is_problematic', 'count'),
                bad_images=('is_problematic', 'sum')
            ).reset_index()
            
            # A nodule is valid ONLY if it has 0 bad images
            nodule_stats['is_valid'] = nodule_stats['bad_images'] == 0
            nodule_stats.to_csv(save_path / "nodule_analysis.csv", index=False)
        
        # --- NEW: Patient Summary (Aggregation) ---
        if 'patient_id' in df.columns:
            patient_stats = df.groupby('patient_id').agg(
                problematic_count=('is_problematic', 'sum'),
                total_count=('is_problematic', 'count')
            ).reset_index()
            
            patient_stats['problematic_ratio'] = patient_stats['problematic_count'] / patient_stats['total_count']
            patient_stats.to_csv(save_path / "patient_summary.csv", index=False)

        # 5. Text Summary
        summary = self.get_summary_report()
        with open(save_path / "summary_report.txt", "w") as f:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")
        
        logger.info(f"Reports saved successfully to: {save_path}")

    def export_clean_dataset(self, output_dir: Optional[str] = None, overwrite_existing: bool = False) -> None:
        """
        Creates a clean dataset containing ONLY valid nodules.
        """
        if not self.results:
            logger.warning("No analysis results available. Run analyze() first.")
            return

        if not overwrite_existing and not output_dir:
            logger.error("Must provide either 'output_dir' or set 'overwrite_existing=True'.")
            return

        # 1. Identify Bad Nodules
        df = self.get_results_dataframe()
        problematic_nodules = df[df['is_problematic']]
        bad_nodule_keys: Set[Tuple[str, int]] = set()
        
        for _, row in problematic_nodules.iterrows():
            bad_nodule_keys.add((str(row['patient_id']), int(row['nodule_idx'])))
            
        logger.info(f"Identified {len(bad_nodule_keys)} problematic nodules.")

        # 2. Process Files
        operations_count = 0
        
        for result in self.results:
            key = (str(result.metadata.get('patient_id')), int(result.metadata.get('nodule_idx', -1)))
            is_valid_nodule = key not in bad_nodule_keys
            
            img_path = Path(result.filepath)
            label_path = img_path.parent.parent / 'labels' / (img_path.stem + ".txt")
            
            # MODE: Create New Dataset
            if not overwrite_existing and output_dir:
                if is_valid_nodule:
                    rel_path_img = img_path.relative_to(self.dataset_dir)
                    dest_img = Path(output_dir) / rel_path_img
                    
                    try:
                        rel_path_lbl = label_path.relative_to(self.dataset_dir)
                        dest_lbl = Path(output_dir) / rel_path_lbl
                    except ValueError:
                        dest_lbl = Path(output_dir) / result.split / 'labels' / label_path.name

                    dest_img.parent.mkdir(parents=True, exist_ok=True)
                    dest_lbl.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(img_path, dest_img)
                    if label_path.exists():
                        shutil.copy2(label_path, dest_lbl)
                    
                    operations_count += 1

            # MODE: Clean In-Place
            elif overwrite_existing:
                if not is_valid_nodule:
                    try:
                        if img_path.exists(): img_path.unlink()
                        if label_path.exists(): label_path.unlink()
                        operations_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete {img_path.name}: {e}")

        if overwrite_existing:
            logger.info(f"Cleanup Complete: Deleted {operations_count} invalid files.")
        else:
            logger.info(f"Export Complete: Copied {operations_count} valid files to {output_dir}.")

        # 3. Handle Metadata CSV
        self._update_metadata_csv(output_dir, overwrite_existing, bad_nodule_keys)

    def _update_metadata_csv(self, output_dir, overwrite, bad_keys):
        """Helper to filter and save the metadata CSV."""
        meta_path = self.dataset_dir / "metadata" / "regression_dataset.csv"
        if not meta_path.exists():
            return

        df_meta = pd.read_csv(meta_path)
        
        def is_valid(row):
            n_idx = row.get('nodule_index', row.get('nodule_idx', -1))
            return (str(row['patient_id']), int(n_idx)) not in bad_keys

        df_clean = df_meta[df_meta.apply(is_valid, axis=1)]
        
        if overwrite:
            df_clean.to_csv(meta_path, index=False)
            logger.info(f"Updated metadata CSV in-place: {meta_path}")
        elif output_dir:
            dest_meta_dir = Path(output_dir) / "metadata"
            dest_meta_dir.mkdir(parents=True, exist_ok=True)
            df_clean.to_csv(dest_meta_dir / "regression_dataset.csv", index=False)
            logger.info(f"Saved clean metadata CSV to: {dest_meta_dir}")

    def filter_dataset_metadata(self, metadata_csv_path: str, output_csv_path: str) -> pd.DataFrame:
        """Keeps compatibility with the previous API call."""
        return self._update_metadata_csv(None, False, self._get_bad_keys_from_results()) or pd.DataFrame()

    def _get_bad_keys_from_results(self):
        df = self.get_results_dataframe()
        problematic = df[df['is_problematic']]
        bad_keys = set()
        for _, row in problematic.iterrows():
            bad_keys.add((str(row['patient_id']), int(row['nodule_idx'])))
        return bad_keys

    def _analyze_directory(self, directory: Path, split: str):
        images = list(directory.glob("*.jpg")) + list(directory.glob("*.png"))
        logger.info(f"Analyzing {len(images)} images in {split}...")
        for idx, filepath in enumerate(images):
            if (idx + 1) % 500 == 0:
                logger.info(f"Progress: {idx + 1}/{len(images)}")
            self.results.append(self._analyze_single_image(filepath, split))

    def _analyze_single_image(self, filepath: Path, split: str) -> ImageAnalysisResult:
        img = cv2.imread(str(filepath))
        metadata = self._extract_metadata_from_filename(filepath.name)

        if img is None:
            return ImageAnalysisResult(filepath=str(filepath), filename=filepath.name, split=split, metadata=metadata, is_problematic=True, problem_type="UNREADABLE")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_val = float(gray.mean())
        std_val = float(gray.std())
        min_val = int(gray.min())
        max_val = int(gray.max())
        
        total = gray.size
        dark = np.sum(gray < 50)
        mid = np.sum((gray >= 50) & (gray <= 200))
        bright = np.sum(gray > 200)
        
        dark_ratio = dark / total
        mid_ratio = mid / total
        bright_ratio = bright / total
        
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
            mean=round(mean_val, 2), std=round(std_val, 2), min_val=min_val, max_val=max_val,
            dark_pixel_ratio=round(dark_ratio, 3), mid_pixel_ratio=round(mid_ratio, 3), bright_pixel_ratio=round(bright_ratio, 3),
            is_uniform=is_uniform, is_too_dark=is_too_dark, is_too_bright=is_too_bright,
            is_missing_contrast=is_missing_contrast, has_no_dark_background=has_no_dark_background,
            is_problematic=len(problems) > 0, problem_type="; ".join(problems) if problems else "OK"
        )

    def _extract_metadata_from_filename(self, filename: str) -> Dict:
        name = Path(filename).stem
        match = re.match(r'^(.+)_n(\d+)_z(\d+)$', name)
        if match:
            return {"patient_id": match.group(1), "nodule_idx": int(match.group(2)), "slice_idx": int(match.group(3))}
        return {}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python diagnose_class.py <dataset_dir>")
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    diagnoser = DatasetDiagnoser(dataset_path)
    diagnoser.analyze()
    diagnoser.save_reports_to_disk()