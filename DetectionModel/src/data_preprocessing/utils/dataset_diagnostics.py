import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import re
import logging
import shutil
import time

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich import box

from common.constants.emums import Color, Decoration

console = Console()

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

@dataclass
class AnalysisThresholds:
    """Thresholds for image quality analysis."""
    # Generic quality
    uniform_std: float = 20.0
    too_dark_ratio: float = 0.88
    too_bright_mean: float = 180.0
    min_dark_ratio: float = 0.20
    min_contrast_range: int = 100

    # Lung content
    body_intensity_floor: int = 20
    lung_intensity_range: Tuple[int, int] = (10, 90)
    morph_kernel_size: int = 15
    min_lung_contour_area: int = 500
    min_lung_body_ratio: float = 0.20
    min_body_ratio: float = 0.30
    min_tissue_ratio: float = 0.05

    # Spatial / geometry
    min_body_aspect_ratio: float = 0.5
    min_content_coverage: float = 0.5


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

    # Pixel distribution
    dark_pixel_ratio: float = 0.0
    mid_pixel_ratio: float = 0.0
    bright_pixel_ratio: float = 0.0

    # Lung content metrics
    body_ratio: float = 0.0
    lung_body_ratio: float = 0.0
    lung_region_count: int = 0

    # Spatial / geometry metrics
    body_aspect_ratio: float = 0.0
    content_width_ratio: float = 0.0
    content_height_ratio: float = 0.0
    content_coverage: float = 0.0

    # Diagnosis flags
    is_uniform: bool = False
    is_too_dark: bool = False
    is_too_bright: bool = False
    is_missing_contrast: bool = False
    has_no_dark_background: bool = False
    has_insufficient_lung: bool = False
    has_insufficient_tissue: bool = False
    has_bad_aspect_ratio: bool = False
    has_low_content_coverage: bool = False

    # Verdict
    is_problematic: bool = False
    problem_type: str = "OK"


# ──────────────────────────────────────────────
# Terminal Display
# ──────────────────────────────────────────────

class DiagnoserDisplay:
    """Handles all rich terminal output for the diagnoser."""

    PROBLEM_STYLES = {
        "UNIFORM": ("⬜", "dim"),
        "TOO_DARK": ("🌑", "bright_black"),
        "TOO_BRIGHT": ("☀️", "yellow"),
        "LOW_CONTRAST": ("🔲", "grey50"),
        "NO_BG": ("🖼️", "magenta"),
        "INSUFFICIENT_LUNG": ("🫁", "red"),
        "NO_LUNG_TISSUE": ("🔬", "dim red"),
        "BAD_GEOMETRY": ("📐", "bright_magenta"),
        "NARROW_CONTENT": ("▮", "bright_yellow"),
        "UNREADABLE": ("❌", "bold red"),
    }

    @staticmethod
    def print_banner() -> None:
        banner = Text()
        banner.append("╔══════════════════════════════════════════════╗\n", style="cyan")
        banner.append("║       ", style="cyan")
        banner.append("🫁  LungGuard Dataset Diagnoser", style="bold white")
        banner.append("       ║\n", style="cyan")
        banner.append("║  ", style="cyan")
        banner.append("   CT Image Quality & Lung Content Analyzer  ", style="dim white")
        banner.append("║\n", style="cyan")
        banner.append("╚══════════════════════════════════════════════╝", style="cyan")
        console.print(banner)
        console.print()

    @staticmethod
    def print_thresholds(t: AnalysisThresholds) -> None:
        table = Table(
            title="⚙️  Active Thresholds",
            box=box.ROUNDED, title_style="bold cyan",
            show_header=True, header_style="bold",
            padding=(0, 1),
        )
        table.add_column("Parameter", style="white")
        table.add_column("Value", style="green", justify="right")
        table.add_column("Description", style="dim")

        # Generic quality
        table.add_row("uniform_std", f"{t.uniform_std:.1f}", "Max std allowed (filters blurry, blank, or smooth tissue like liver).")
        table.add_row("too_dark_ratio", f"{t.too_dark_ratio:.2f}", "Max pure black ratio (filters empty space or scans outside the body).")
        table.add_row("too_bright_mean", f"{t.too_bright_mean:.1f}", "Max mean brightness (filters overexposed scans or dense bone).")
        table.add_row("min_dark_ratio", f"{t.min_dark_ratio:.2f}", "Min dark pixel ratio (detects missing background, e.g., cropped scans).")
        table.add_row("min_contrast_range", f"{t.min_contrast_range}", "Min pixel diff (ensures image sharpness, avoids grey fog).")
        table.add_row("min_tissue_ratio", f"{t.min_tissue_ratio:.2f}", "Min % of mid-gray tissue (filters black/white artifact smears).")

        table.add_section()

        # Lung content
        table.add_row("body_intensity_floor", f"{t.body_intensity_floor}", "Min pixel value to be considered patient body vs. background air.")
        table.add_row("lung_intensity_range", f"{t.lung_intensity_range}", "Pixel intensity range representing spongy lung tissue.")
        table.add_row("morph_kernel_size", f"{t.morph_kernel_size}", "Kernel size for morphological ops (closes holes in body mask).")
        table.add_row("min_lung_contour_area", f"{t.min_lung_contour_area}", "Min area for a valid lung region (ignores small noise artifacts).")
        table.add_row("min_lung_body_ratio", f"{t.min_lung_body_ratio:.2f}", "Min lung-to-body area (filters slices with too little lung, e.g., neck/abdomen).")
        table.add_row("min_body_ratio", f"{t.min_body_ratio:.2f}", "Min body-to-image area (ensures the patient is actually visible).")

        table.add_section()

        # Spatial / geometry
        table.add_row("min_body_aspect_ratio", f"{t.min_body_aspect_ratio:.2f}", "Min body bounding-box aspect ratio (filters non-axial sagittal/coronal slices).")
        table.add_row("min_content_coverage", f"{t.min_content_coverage:.2f}", "Min content spread per axis (filters narrow-strip padding images).")

        console.print(table)
        console.print()

    @staticmethod
    def create_progress() -> Progress:
        return Progress(
            SpinnerColumn("dots", style="cyan"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, style="cyan", complete_style="green", finished_style="bold green"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
        )

    @classmethod
    def print_summary(cls, summary: Dict, df: pd.DataFrame) -> None:
        total = summary["total_images"]
        prob = summary["problematic_count"]
        ratio = summary["problematic_ratio"]
        clean = total - prob

        health = cls._compute_health_grade(ratio)

        stats_text = Text()
        stats_text.append(f"  Total Images:       ", style="dim")
        stats_text.append(f"{total:,}\n", style="bold white")
        stats_text.append(f"  ✅ Clean:            ", style="dim")
        stats_text.append(f"{clean:,}\n", style="bold green")
        stats_text.append(f"  ❌ Problematic:      ", style="dim")
        stats_text.append(f"{prob:,}", style="bold red")
        stats_text.append(f"  ({ratio:.1%})\n", style="dim red")
        stats_text.append(f"\n  Dataset Health:     ", style="dim")
        stats_text.append(f"{health}\n", style="bold")

        console.print(Panel(stats_text, title="📊 Analysis Summary", border_style="cyan", box=box.ROUNDED))

        # Lung stats
        lung = summary.get("lung_stats", {})
        lung_table = Table(box=box.SIMPLE_HEAD, show_edge=False, padding=(0, 2))
        lung_table.add_column("Metric", style="white")
        lung_table.add_column("Value", style="cyan", justify="right")

        lung_table.add_row("Mean Lung/Body Ratio", f"{lung.get('mean_lung_body_ratio', 0):.3f}")
        lung_table.add_row("Median Lung/Body Ratio", f"{lung.get('median_lung_body_ratio', 0):.3f}")
        lung_table.add_row("Min Lung/Body Ratio", f"{lung.get('min_lung_body_ratio', 0):.3f}")
        lung_table.add_row("Flagged Insufficient Lung", f"{lung.get('flagged_insufficient_lung', 0):,}")

        # Geometry stats
        geo = summary.get("geometry_stats", {})
        lung_table.add_row("", "")
        lung_table.add_row("Mean Body Aspect Ratio", f"{geo.get('mean_body_aspect_ratio', 0):.3f}")
        lung_table.add_row("Mean Content Coverage", f"{geo.get('mean_content_coverage', 0):.3f}")
        lung_table.add_row("Flagged Bad Geometry", f"{geo.get('flagged_bad_geometry', 0):,}")
        lung_table.add_row("Flagged Narrow Content", f"{geo.get('flagged_narrow_content', 0):,}")

        console.print(Panel(lung_table, title="🫁 Lung & Geometry Analysis", border_style="blue", box=box.ROUNDED))

        breakdown = summary.get("problem_breakdown", {})
        cls._print_problem_breakdown(breakdown) if breakdown else None

        splits = summary.get("split_breakdown", {})
        cls._print_split_breakdown(splits) if splits else None

    @classmethod
    def _print_problem_breakdown(cls, breakdown: Dict) -> None:
        table = Table(
            title="🔍 Problem Breakdown",
            box=box.ROUNDED, title_style="bold yellow",
            show_header=True, header_style="bold",
        )
        table.add_column("", width=3)
        table.add_column("Problem Type", style="white")
        table.add_column("Count", justify="right", style="red")
        table.add_column("Bar", width=25)

        max_count = max(breakdown.values()) if breakdown else 1
        for problem_type, count in sorted(breakdown.items(), key=lambda x: -x[1]):
            first_type = problem_type.split(";")[0].strip()
            icon, _ = cls.PROBLEM_STYLES.get(first_type, ("⚠️", "yellow"))
            bar_len = int((count / max_count) * 20)
            bar = f"{Color.RED('█' * bar_len)}{Decoration.DIM('░' * (20 - bar_len))}"
            table.add_row(icon, problem_type, str(count), bar)

        console.print(table)

    @staticmethod
    def _print_split_breakdown(splits: Dict) -> None:
        table = Table(
            title="📂 Split Breakdown",
            box=box.ROUNDED, title_style="bold green",
            show_header=True, header_style="bold",
        )
        table.add_column("Split", style="cyan")
        table.add_column("Total", justify="right")
        table.add_column("Problematic", justify="right", style="red")
        table.add_column("Clean", justify="right", style="green")
        table.add_column("Health", justify="right")

        for split_name, info in splits.items():
            total = info["total"]
            prob = info["problematic"]
            clean_count = total - prob
            ratio = info["ratio"]
            health_icon = "🟢" if ratio < 0.05 else ("🟡" if ratio < 0.15 else "🔴")
            table.add_row(split_name, str(total), str(prob), str(clean_count), f"{health_icon} {1 - ratio:.1%}")

        console.print(table)

    @staticmethod
    def print_verification(report: Dict) -> None:
        is_clean = report.get("is_clean", False)
        status_text = "✅ PASS — Dataset is clean!" if is_clean else "❌ FAIL — Problematic images remain"

        text = Text()
        text.append(f"  Path:             ", style="dim")
        text.append(f"{report['dataset_path']}\n", style="white")
        text.append(f"  Total Images:     ", style="dim")
        text.append(f"{report['total_images']:,}\n", style="white")
        text.append(f"  Remaining Issues: ", style="dim")
        text.append(f"{report['problematic_remaining']:,}\n", style="red" if report['problematic_remaining'] else "green")
        text.append(f"\n  Result:           ", style="dim")
        text.append(status_text, style="bold green" if is_clean else "bold red")
        text.append("\n")

        integrity = report.get("nodule_integrity", {})
        nod_status = integrity.get("status", "SKIP")
        nod_style = {"PASS": "green", "FAIL": "red", "SKIP": "yellow"}.get(nod_status, "yellow")

        text.append(f"\n  Nodule Integrity: ", style="dim")
        text.append(f"{nod_status}", style=f"bold {nod_style}")

        total_nod = integrity.get("total_nodules", "?")
        correct_nod = integrity.get("nodules_with_3_slices", "?")
        wrong_nod = integrity.get("nodules_with_wrong_count", 0)

        nod_detail = f" ({correct_nod}/{total_nod} nodules with 3 slices"
        text.append(nod_detail, style="dim") if nod_status != "SKIP" else None
        wrong_suffix = f", {wrong_nod} incorrect)\n" if wrong_nod > 0 else ")\n"
        wrong_style = "dim red" if wrong_nod > 0 else "dim"
        text.append(wrong_suffix, style=wrong_style) if nod_status != "SKIP" else text.append("\n")

        border_style = "green" if is_clean else "red"
        console.print(Panel(text, title="🔎 Verification Report", border_style=border_style, box=box.DOUBLE))

    @staticmethod
    def print_export_result(mode: str, count: int, path: str) -> None:
        icon = "🗑️" if mode == "delete" else "📦"
        verb = "Deleted" if mode == "delete" else "Copied"
        console.print(f"\n  {icon} {verb} {Decoration.BOLD(f'{count:,}')} files", style=Color.GREEN)
        console.print(f"  📁 Location: {Color.CYAN(path)}\n") if mode != "delete" else console.print()

    @staticmethod
    def _compute_health_grade(ratio: float) -> str:
        grades = [
            (0.01, "🟢 Excellent (A+)"),
            (0.05, "🟢 Good (A)"),
            (0.10, "🟡 Fair (B)"),
            (0.20, "🟠 Needs Attention (C)"),
            (0.40, "🔴 Poor (D)"),
            (1.01, "🔴 Critical (F)"),
        ]
        return next(label for threshold, label in grades if ratio < threshold)


# ──────────────────────────────────────────────
# Core Diagnoser
# ──────────────────────────────────────────────

class DatasetDiagnoser:
    """
    Analyzes CT image quality with emphasis on lung content visibility.
    """

    def __init__(self, dataset_dir: str, thresholds: AnalysisThresholds = None):
        self.dataset_dir = Path(dataset_dir)
        self.thresholds = thresholds or AnalysisThresholds()
        self.results: List[ImageAnalysisResult] = []
        self._cached_dfs: Dict[str, pd.DataFrame] = {}
        self.display = DiagnoserDisplay()

    # ── Public API ────────────────────────────

    def analyze(self, show_progress: bool = True) -> None:
        """Run full analysis on the dataset directory structure."""
        self.results = []
        self._cached_dfs = {}

        splits = ['train', 'val', 'test']
        found_any = False

        work_items: List[Tuple[Path, str]] = []
        for split in splits:
            image_dir = self.dataset_dir / split / 'images'
            found_any = found_any or image_dir.exists()
            work_items.append((image_dir, split)) if image_dir.exists() else None

        work_items.append((self.dataset_dir, "root")) if not found_any else None

        all_images: List[Tuple[Path, str]] = []
        for directory, split in work_items:
            images = sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.png"))
            all_images.extend([(fp, split) for fp in images])

        total = len(all_images)
        console.print(f"  📁 Dataset: {Color.CYAN(self.dataset_dir)}")
        console.print(f"  🖼️  Found {Decoration.BOLD(f'{total:,}')} images across {Decoration.BOLD(len(work_items))} split(s)\n")

        self._analyze_with_progress(all_images) if show_progress else self._analyze_silent(all_images)

    def get_results_dataframe(self) -> pd.DataFrame:
        cache_key = 'full'
        cached = self._cached_dfs.get(cache_key)
        result = cached if cached is not None else self._build_results_df(cache_key)
        return result

    def get_problematic_images(self) -> pd.DataFrame:
        df = self.get_results_dataframe()
        return df[df['is_problematic']].copy() if not df.empty else df

    def get_summary_report(self) -> Dict:
        df = self.get_results_dataframe()
        empty_result = {"error": "No data analyzed"}
        result = self._build_summary(df) if not df.empty else empty_result
        return result

    def print_summary(self) -> None:
        """Display rich summary in terminal."""
        summary = self.get_summary_report()
        df = self.get_results_dataframe()
        self.display.print_summary(summary, df) if "error" not in summary else console.print(Color.RED("No data to display."))

    def save_reports_to_disk(self, output_dir: str = None) -> None:
        save_path = Path(output_dir) if output_dir else self.dataset_dir / "analysis"
        save_path.mkdir(parents=True, exist_ok=True)

        df = self.get_results_dataframe()
        if df.empty:
            console.print(Color.YELLOW("⚠️  No results to save."))
            return

        with console.status(Color.CYAN("Saving reports..."), spinner="dots"):
            df.to_csv(save_path / "image_analysis_full.csv", index=False)
            self.get_problematic_images().to_csv(save_path / "problematic_images.csv", index=False)
            self._save_nodule_analysis(df, save_path)
            self._save_patient_summary(df, save_path)

            summary = self.get_summary_report()
            with open(save_path / "summary_report.txt", "w") as f:
                for k, v in summary.items():
                    f.write(f"{k}: {v}\n")

        console.print(f"  💾 Reports saved to: {Color.CYAN(save_path)}\n")

    def export_clean_dataset(self, output_dir: Optional[str] = None, overwrite_existing: bool = False) -> None:
        if not self.results:
            console.print(Color.YELLOW("⚠️  No analysis results. Run analyze() first."))
            return

        if not overwrite_existing and not output_dir:
            console.print(Color.RED("Must provide output_dir or set overwrite_existing=True."))
            return

        bad_nodule_keys = self._get_bad_nodule_keys()
        console.print(f"  🔍 Found {Color.RED(len(bad_nodule_keys), Decoration.BOLD)} problematic nodules to remove\n")

        with console.status(Color.CYAN("Processing files..."), spinner="dots"):
            operations_count = self._process_files(output_dir, overwrite_existing, bad_nodule_keys)
            self._update_metadata_csv(output_dir, overwrite_existing, bad_nodule_keys)

        mode = "delete" if overwrite_existing else "copy"
        target = str(self.dataset_dir) if overwrite_existing else output_dir
        self.display.print_export_result(mode, operations_count, target)

    def verify_clean_dataset(self, clean_dir: str) -> Dict:
        console.print(f"\n  🔎 Verifying: {Color.CYAN(clean_dir)}\n")

        verifier = DatasetDiagnoser(clean_dir, self.thresholds)
        verifier.analyze(show_progress=True)

        summary = verifier.get_summary_report()

        verification = {
            "dataset_path": clean_dir,
            "total_images": summary.get("total_images", 0),
            "problematic_remaining": summary.get("problematic_count", 0),
            "is_clean": summary.get("problematic_count", 0) == 0,
            "problem_breakdown": summary.get("problem_breakdown", {}),
        }

        df = verifier.get_results_dataframe()
        verification["nodule_integrity"] = self._check_nodule_integrity(df)

        self.display.print_verification(verification)
        return verification

    # ── Analysis Engine ───────────────────────

    def _analyze_with_progress(self, all_images: List[Tuple[Path, str]]) -> None:
        progress = self.display.create_progress()
        with progress:
            task = progress.add_task("Analyzing images", total=len(all_images))
            for filepath, split in all_images:
                self.results.append(self._analyze_single_image(filepath, split))
                progress.advance(task)
        console.print()

    def _analyze_silent(self, all_images: List[Tuple[Path, str]]) -> None:
        for filepath, split in all_images:
            self.results.append(self._analyze_single_image(filepath, split))

    def _analyze_single_image(self, filepath: Path, split: str) -> ImageAnalysisResult:
        metadata = self._extract_metadata_from_filename(filepath.name)
        img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)

        unreadable = ImageAnalysisResult(
            filepath=str(filepath), filename=filepath.name, split=split,
            metadata=metadata, is_problematic=True, problem_type="UNREADABLE"
        )

        result = self._run_full_analysis(filepath, split, metadata, img) if img is not None else unreadable
        return result

    def _run_full_analysis(self, filepath: Path, split: str, metadata: Dict, gray: np.ndarray) -> ImageAnalysisResult:
        t = self.thresholds
        total = gray.size

        mean_val = float(gray.mean())
        std_val = float(gray.std())
        min_val = int(gray.min())
        max_val = int(gray.max())

        dark = int(np.sum(gray < 50))
        mid = int(np.sum((gray >= 50) & (gray <= 200)))
        bright = int(np.sum(gray > 200))

        dark_ratio = dark / total
        mid_ratio = mid / total
        bright_ratio = bright / total

        lung_metrics = self._compute_lung_metrics(gray)
        geo_metrics = self._compute_geometry_metrics(gray)

        is_uniform = std_val < t.uniform_std
        is_too_dark = dark_ratio > t.too_dark_ratio
        is_too_bright = mean_val > t.too_bright_mean
        is_missing_contrast = (max_val - min_val) < t.min_contrast_range
        has_no_dark_background = dark_ratio < t.min_dark_ratio and mean_val > 100
        has_insufficient_lung = lung_metrics['lung_body_ratio'] < t.min_lung_body_ratio
        has_insufficient_tissue = mid_ratio < t.min_tissue_ratio
        has_bad_aspect_ratio = geo_metrics['body_aspect_ratio'] < t.min_body_aspect_ratio
        has_low_content_coverage = geo_metrics['content_coverage'] < t.min_content_coverage

        problem_checks = [
            (has_insufficient_tissue, "NO_LUNG_TISSUE"),
            (is_uniform, "UNIFORM"),
            (is_too_dark, "TOO_DARK"),
            (is_too_bright, "TOO_BRIGHT"),
            (is_missing_contrast and not is_uniform, "LOW_CONTRAST"),
            (has_no_dark_background and not is_too_bright, "NO_BG"),
            (has_bad_aspect_ratio, "BAD_GEOMETRY"),
            (has_low_content_coverage and not has_bad_aspect_ratio, "NARROW_CONTENT"),
            (has_insufficient_lung, "INSUFFICIENT_LUNG"),
        ]
        problems = list(map(lambda p: p[1], filter(lambda p: p[0], problem_checks)))

        return ImageAnalysisResult(
            filepath=str(filepath), filename=filepath.name, split=split, metadata=metadata,
            mean=round(mean_val, 2), std=round(std_val, 2),
            min_val=min_val, max_val=max_val,
            dark_pixel_ratio=round(dark_ratio, 3),
            mid_pixel_ratio=round(mid_ratio, 3),
            bright_pixel_ratio=round(bright_ratio, 3),
            body_ratio=round(lung_metrics['body_ratio'], 3),
            lung_body_ratio=round(lung_metrics['lung_body_ratio'], 3),
            lung_region_count=lung_metrics['lung_region_count'],
            body_aspect_ratio=round(geo_metrics['body_aspect_ratio'], 3),
            content_width_ratio=round(geo_metrics['content_width_ratio'], 3),
            content_height_ratio=round(geo_metrics['content_height_ratio'], 3),
            content_coverage=round(geo_metrics['content_coverage'], 3),
            is_uniform=is_uniform, is_too_dark=is_too_dark, is_too_bright=is_too_bright,
            is_missing_contrast=is_missing_contrast,
            has_no_dark_background=has_no_dark_background,
            has_insufficient_lung=has_insufficient_lung,
            has_insufficient_tissue=has_insufficient_tissue,
            has_bad_aspect_ratio=has_bad_aspect_ratio,
            has_low_content_coverage=has_low_content_coverage,
            is_problematic=len(problems) > 0,
            problem_type="; ".join(problems) if problems else "OK"
        )

    def _compute_lung_metrics(self, gray: np.ndarray) -> Dict:
        t = self.thresholds
        total = gray.size

        body_mask = (gray > t.body_intensity_floor).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t.morph_kernel_size, t.morph_kernel_size))
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel)
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel)

        body_area = int(np.sum(body_mask > 0))
        body_ratio = body_area / total

        lo, hi = t.lung_intensity_range
        lung_candidate = ((gray >= lo) & (gray < hi)).astype(np.uint8) * 255
        lung_in_body = cv2.bitwise_and(lung_candidate, body_mask)

        lung_area = int(np.sum(lung_in_body > 0))
        lung_body_ratio = lung_area / body_area if body_area > 0 else 0.0

        contours, _ = cv2.findContours(lung_in_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant = list(filter(lambda c: cv2.contourArea(c) > t.min_lung_contour_area, contours))

        return {
            "body_ratio": body_ratio,
            "lung_body_ratio": lung_body_ratio,
            "lung_region_count": len(significant),
        }

    def _compute_geometry_metrics(self, gray: np.ndarray) -> Dict:
        """Compute spatial geometry metrics to detect non-axial / narrow-strip images."""
        t = self.thresholds
        h, w = gray.shape

        # ── Body aspect ratio via bounding rect of body contour ──
        body_mask = (gray > t.body_intensity_floor).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t.morph_kernel_size, t.morph_kernel_size))
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel)
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        body_aspect_ratio = 0.0
        largest = max(contours, key=cv2.contourArea, default=None) if contours else None
        _, _, bw, bh = cv2.boundingRect(largest) if largest is not None else (0, 0, 0, 0)
        body_aspect_ratio = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0.0

        # ── Content coverage: how much of each axis has non-padding pixels ──
        cols_with_content = np.any(gray > t.body_intensity_floor, axis=0)
        rows_with_content = np.any(gray > t.body_intensity_floor, axis=1)
        content_width_ratio = float(np.sum(cols_with_content)) / w if w > 0 else 0.0
        content_height_ratio = float(np.sum(rows_with_content)) / h if h > 0 else 0.0
        content_coverage = min(content_width_ratio, content_height_ratio)

        return {
            "body_aspect_ratio": body_aspect_ratio,
            "content_width_ratio": content_width_ratio,
            "content_height_ratio": content_height_ratio,
            "content_coverage": content_coverage,
        }

    # ── Metadata & Filename Parsing ───────────

    @staticmethod
    def _extract_metadata_from_filename(filename: str) -> Dict:
        name = Path(filename).stem
        match = re.match(r'^(.+)_n(\d+)_z(\d+)$', name)
        result = (
            {"patient_id": match.group(1), "nodule_idx": int(match.group(2)), "slice_idx": int(match.group(3))}
            if match
            else {}
        )
        return result

    # ── DataFrame Builders ────────────────────

    def _build_results_df(self, cache_key: str) -> pd.DataFrame:
        empty = pd.DataFrame()
        if not self.results:
            return empty

        data = []
        for r in self.results:
            row = vars(r).copy()
            meta = row.pop('metadata', {})
            row.update(meta)
            data.append(row)

        df = pd.DataFrame(data)
        self._cached_dfs[cache_key] = df
        return df

    def _build_summary(self, df: pd.DataFrame) -> Dict:
        total_images = len(df)
        problematic_df = df[df['is_problematic']]
        num_problematic = len(problematic_df)

        summary = {
            "total_images": total_images,
            "problematic_count": num_problematic,
            "problematic_ratio": round(num_problematic / total_images, 4) if total_images > 0 else 0,
            "problem_breakdown": problematic_df['problem_type'].value_counts().to_dict(),
            "lung_stats": {
                "mean_lung_body_ratio": round(df['lung_body_ratio'].mean(), 3),
                "median_lung_body_ratio": round(df['lung_body_ratio'].median(), 3),
                "min_lung_body_ratio": round(df['lung_body_ratio'].min(), 3),
                "flagged_insufficient_lung": int(df['has_insufficient_lung'].sum()),
            },
            "geometry_stats": {
                "mean_body_aspect_ratio": round(df['body_aspect_ratio'].mean(), 3),
                "mean_content_coverage": round(df['content_coverage'].mean(), 3),
                "flagged_bad_geometry": int(df['has_bad_aspect_ratio'].sum()),
                "flagged_narrow_content": int(df['has_low_content_coverage'].sum()),
            },
            "split_breakdown": {},
        }

        splits = df['split'].unique() if 'split' in df.columns else []
        for split in splits:
            split_df = df[df['split'] == split]
            prob_count = int(split_df['is_problematic'].sum())
            summary['split_breakdown'][split] = {
                "total": len(split_df),
                "problematic": prob_count,
                "ratio": round(prob_count / len(split_df), 4) if len(split_df) > 0 else 0,
            }

        return summary

    # ── Report Saving Helpers ─────────────────

    def _save_nodule_analysis(self, df: pd.DataFrame, save_path: Path) -> None:
        has_col = 'nodule_idx' in df.columns
        if not has_col:
            return

        nodule_stats = df.groupby(['patient_id', 'nodule_idx']).agg(
            total_images=('is_problematic', 'count'),
            bad_images=('is_problematic', 'sum'),
            mean_lung_ratio=('lung_body_ratio', 'mean'),
            mean_body_aspect=('body_aspect_ratio', 'mean'),
            mean_content_coverage=('content_coverage', 'mean'),
        ).reset_index()

        nodule_stats['is_valid'] = nodule_stats['bad_images'] == 0
        nodule_stats.to_csv(save_path / "nodule_analysis.csv", index=False)

    def _save_patient_summary(self, df: pd.DataFrame, save_path: Path) -> None:
        has_col = 'patient_id' in df.columns
        if not has_col:
            return

        patient_stats = df.groupby('patient_id').agg(
            problematic_count=('is_problematic', 'sum'),
            total_count=('is_problematic', 'count'),
            mean_lung_ratio=('lung_body_ratio', 'mean'),
            mean_body_aspect=('body_aspect_ratio', 'mean'),
            mean_content_coverage=('content_coverage', 'mean'),
        ).reset_index()

        patient_stats['problematic_ratio'] = patient_stats['problematic_count'] / patient_stats['total_count']
        patient_stats.to_csv(save_path / "patient_summary.csv", index=False)

    # ── Nodule Integrity ──────────────────────

    @staticmethod
    def _check_nodule_integrity(df: pd.DataFrame) -> Dict:
        result = {"status": "SKIP", "details": "No nodule metadata"}

        has_cols = 'patient_id' in df.columns and 'nodule_idx' in df.columns
        if has_cols:
            counts = df.groupby(['patient_id', 'nodule_idx']).size().reset_index(name='slice_count')
            non_three = counts[counts['slice_count'] != 3]
            result = {
                "status": "PASS" if non_three.empty else "FAIL",
                "total_nodules": len(counts),
                "nodules_with_3_slices": int((counts['slice_count'] == 3).sum()),
                "nodules_with_wrong_count": len(non_three),
                "wrong_count_details": non_three.to_dict('records') if not non_three.empty else [],
            }

        return result

    # ── Clean Dataset Export ──────────────────

    def _get_bad_nodule_keys(self) -> Set[Tuple[str, int]]:
        df = self.get_results_dataframe()
        problematic = df[df['is_problematic']]
        bad_keys = set()
        for _, row in problematic.iterrows():
            bad_keys.add((str(row['patient_id']), int(row['nodule_idx'])))
        return bad_keys

    def _process_files(self, output_dir: Optional[str], overwrite: bool, bad_keys: Set) -> int:
        ops = 0
        for result in self.results:
            key = (str(result.metadata.get('patient_id')), int(result.metadata.get('nodule_idx', -1)))
            is_valid = key not in bad_keys

            img_path = Path(result.filepath)
            label_path = img_path.parent.parent / 'labels' / (img_path.stem + ".txt")

            ops += self._copy_valid_file(result, img_path, label_path, output_dir) if (not overwrite and output_dir and is_valid) else 0
            ops += self._delete_invalid_file(img_path, label_path) if (overwrite and not is_valid) else 0

        return ops

    def _copy_valid_file(self, result: ImageAnalysisResult, img_path: Path, label_path: Path, output_dir: str) -> int:
        rel_img = img_path.relative_to(self.dataset_dir)
        dest_img = Path(output_dir) / rel_img

        try:
            rel_lbl = label_path.relative_to(self.dataset_dir)
            dest_lbl = Path(output_dir) / rel_lbl
        except ValueError:
            dest_lbl = Path(output_dir) / result.split / 'labels' / label_path.name

        dest_img.parent.mkdir(parents=True, exist_ok=True)
        dest_lbl.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img_path, dest_img)
        shutil.copy2(label_path, dest_lbl) if label_path.exists() else None

        return 1

    @staticmethod
    def _delete_invalid_file(img_path: Path, label_path: Path) -> int:
        try:
            img_path.unlink() if img_path.exists() else None
            label_path.unlink() if label_path.exists() else None
            return 1
        except Exception as e:
            logger.error(f"Failed to delete {img_path.name}: {e}")
            return 0

    def _update_metadata_csv(self, output_dir: Optional[str], overwrite: bool, bad_keys: Set) -> None:
        meta_path = self.dataset_dir / "metadata" / "regression_dataset.csv"
        if not meta_path.exists():
            return

        df_meta = pd.read_csv(meta_path)

        def is_valid(row):
            n_idx = row.get('nodule_index', row.get('nodule_idx', -1))
            return (str(row['patient_id']), int(n_idx)) not in bad_keys

        df_clean = df_meta[df_meta.apply(is_valid, axis=1)]

        save_actions = {
            True: lambda: df_clean.to_csv(meta_path, index=False) or logger.info(f"Updated metadata in-place: {meta_path}"),
            False: lambda: self._save_clean_metadata(df_clean, output_dir),
        }
        save_actions[overwrite]()

    @staticmethod
    def _save_clean_metadata(df_clean: pd.DataFrame, output_dir: Optional[str]) -> None:
        if not output_dir:
            return
        dest = Path(output_dir) / "metadata"
        dest.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(dest / "regression_dataset.csv", index=False)


# ──────────────────────────────────────────────
# Interactive CLI
# ──────────────────────────────────────────────

def edit_thresholds_interactive(thresholds: AnalysisThresholds) -> AnalysisThresholds:
    """Interactive threshold editor."""
    console.print(f"\n  {Decoration.BOLD_CYAN('Threshold Editor')}")
    console.print(f"  {Decoration.DIM('Select parameters to modify (or press Enter to skip)')}\n")
    
    # Define threshold groups with their fields
    threshold_options = {
        "1": ("uniform_std", "Max std for uniform images", float),
        "2": ("too_dark_ratio", "Max dark pixel ratio", float),
        "3": ("too_bright_mean", "Max mean brightness", float),
        "4": ("min_dark_ratio", "Min dark pixel ratio", float),
        "5": ("min_contrast_range", "Min contrast range", int),
        "6": ("min_tissue_ratio", "Min tissue ratio", float),
        "7": ("body_intensity_floor", "Body intensity floor", int),
        "8": ("lung_intensity_range", "Lung intensity range (min,max)", tuple),
        "9": ("morph_kernel_size", "Morphology kernel size", int),
        "10": ("min_lung_contour_area", "Min lung contour area", int),
        "11": ("min_lung_body_ratio", "Min lung/body ratio", float),
        "12": ("min_body_ratio", "Min body ratio", float),
        "13": ("min_body_aspect_ratio", "Min body aspect ratio", float),
        "14": ("min_content_coverage", "Min content coverage", float),
    }
    
    # Display options
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("No.", style="cyan", width=4)
    table.add_column("Parameter", style="white")
    table.add_column("Current Value", style="green", justify="right")
    
    for num, (field, desc, _) in threshold_options.items():
        current_val = getattr(thresholds, field)
        table.add_row(num, desc, str(current_val))
    
    console.print(table)
    console.print()
    
    # Get selection
    selection = Prompt.ask(
        "  Enter parameter numbers to edit (comma-separated, or 'all')",
        default="none"
    ).strip().lower()
    
    if selection == "none" or not selection:
        console.print(f"  {Decoration.DIM('No changes made')}\n")
        return thresholds
    
    # Parse selection
    if selection == "all":
        to_edit = list(threshold_options.keys())
    else:
        to_edit = [s.strip() for s in selection.split(",") if s.strip() in threshold_options]
    
    if not to_edit:
        console.print(f"  {Color.YELLOW('No valid selections')}\n")
        return thresholds
    
    console.print()
    
    # Edit selected thresholds
    for num in to_edit:
        field, desc, val_type = threshold_options[num]
        current_val = getattr(thresholds, field)
        
        if val_type == tuple:
            # Special handling for lung_intensity_range
            prompt_text = f"  {desc} {Decoration.DIM(f'(current: {current_val})')}"
            new_val_str = Prompt.ask(prompt_text, default=f"{current_val[0]},{current_val[1]}")
            try:
                parts = new_val_str.split(",")
                new_val = (int(parts[0].strip()), int(parts[1].strip()))
                setattr(thresholds, field, new_val)
                console.print(f"    ✓ Updated to {new_val}", style=Color.GREEN)
            except (ValueError, IndexError):
                console.print(f"    ✗ Invalid format, keeping {current_val}", style=Color.RED)
        else:
            prompt_text = f"  {desc} {Decoration.DIM(f'(current: {current_val})')}"
            new_val_str = Prompt.ask(prompt_text, default=str(current_val))
            try:
                new_val = val_type(new_val_str)
                setattr(thresholds, field, new_val)
                console.print(f"    ✓ Updated to {new_val}", style=Color.GREEN)
            except ValueError:
                console.print(f"    ✗ Invalid value, keeping {current_val}", style=Color.RED)
    
    console.print()
    return thresholds


def run_interactive() -> None:
    """Interactive terminal interface for the diagnoser."""
    DiagnoserDisplay.print_banner()

    # Step 1: Get dataset path
    dataset_path = Prompt.ask("  📂 Enter dataset directory path").strip()
    dataset = Path(dataset_path)

    if not dataset.exists():
        console.print(f"\n  {Decoration.BOLD_RED(f'❌ Path does not exist:')} {dataset_path}")
        return

    console.print()

    # Step 2: Initialize and show thresholds
    thresholds = AnalysisThresholds()
    diagnoser = DatasetDiagnoser(dataset_path, thresholds)
    diagnoser.display.print_thresholds(diagnoser.thresholds)
    
    # Step 2.5: Ask if user wants to modify thresholds
    modify = Confirm.ask("  ✏️  Would you like to modify any thresholds?", default=False)
    if modify:
        thresholds = edit_thresholds_interactive(thresholds)
        diagnoser = DatasetDiagnoser(dataset_path, thresholds)
        console.print(f"  {Color.GREEN('✓', Decoration.BOLD)} Updated thresholds:\n")
        diagnoser.display.print_thresholds(thresholds)

    # Step 3: Analyze
    console.rule(Decoration.BOLD_CYAN("Phase 1: Analysis"), style=Color.CYAN)
    console.print()

    start = time.time()
    diagnoser.analyze()
    elapsed = time.time() - start

    console.print(f"  ⏱️  Analysis completed in {Decoration.BOLD(f'{elapsed:.1f}s')}\n")

    # Step 4: Display summary
    console.rule(Decoration.BOLD_CYAN("Results"), style=Color.CYAN)
    console.print()
    diagnoser.print_summary()

    # Step 5: Save reports
    console.print()
    save = Confirm.ask("  💾 Save reports to disk?", default=True)
    if save:
        default_path = str(dataset / "analysis")
        save_dir = Prompt.ask("     Save location", default=default_path).strip()
        diagnoser.save_reports_to_disk(save_dir)

    # Step 6: Ask about cleaning
    summary = diagnoser.get_summary_report()
    prob_count = summary.get("problematic_count", 0)

    if prob_count == 0:
        console.print(f"\n  {Decoration.BOLD_GREEN('✨ Dataset is already clean — no action needed!')}\n")
        return

    console.print()
    console.rule(Decoration.BOLD_YELLOW("Phase 2: Cleaning"), style=Color.YELLOW)
    console.print()
    console.print(f"  Found {Color.RED(prob_count, Decoration.BOLD)} problematic images.")
    console.print()

    clean = Confirm.ask("  🧹 Would you like to create a clean dataset?", default=True)
    if not clean:
        console.print(f"\n  {Decoration.DIM('Skipping cleaning. You can re-run anytime.')}\n")
        return

    console.print()
    mode_choice = Prompt.ask(
        "     Clean mode",
        choices=["copy", "overwrite"],
        default="copy"
    )

    output_dir = None
    overwrite = mode_choice == "overwrite"

    if overwrite:
        console.print(f"\n  {Decoration.BOLD_RED('⚠️  WARNING: This will permanently delete files from the original dataset!')}")
        confirmed = Confirm.ask("     Are you sure?", default=False)
        if not confirmed:
            console.print(f"\n  {Decoration.DIM('Cancelled.')}\n")
            return
    else:
        default_clean = str(dataset.parent / (dataset.name + "_clean"))
        output_dir = Prompt.ask("     Output directory for clean dataset", default=default_clean).strip()

    console.print()
    diagnoser.export_clean_dataset(output_dir=output_dir, overwrite_existing=overwrite)

    # Step 7: Verify
    console.rule(Decoration.BOLD_GREEN("Phase 3: Verification"), style=Color.GREEN)

    verify_path = output_dir if output_dir else str(dataset)
    do_verify = Confirm.ask(f"\n  🔎 Verify the cleaned dataset at {Color.CYAN(verify_path)}?", default=True)

    if do_verify:
        diagnoser.verify_clean_dataset(verify_path)

    console.print()
    console.print(Panel(
        f"  {Decoration.BOLD_GREEN('Done!')} Your dataset is ready for training. 🚀  ",
        border_style=Color.GREEN, box=box.DOUBLE
    ))
    console.print()


if __name__ == "__main__":
    run_interactive()