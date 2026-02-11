"""
Terminal UI utilities for the LungGuard Data Preparation Pipeline.

Pure display layer — Rich-based visual components: banners, progress bars,
styled logging, tables, status messages. Contains zero business logic.

Dual output strategy:
  * Rich handler  -> pretty stderr  (never interrupted by warnings)
  * File handler  -> rotating plain-text log  (10 MB x 3 backups)
"""

import logging
import os
import sys
import warnings
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.tree import Tree

# ──────────────────────────────────────────────────────────
# Warning Suppression (applied at import time)
# ──────────────────────────────────────────────────────────

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", module="pkg_resources")
warnings.filterwarnings("ignore", module="google")
warnings.filterwarnings("ignore", module="tensorflow")
warnings.filterwarnings("ignore", module="numpy")

# ──────────────────────────────────────────────────────────
# Theme & Console
# ──────────────────────────────────────────────────────────

LUNGGUARD_THEME = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "phase": "bold magenta",
        "metric": "bold white",
        "highlight": "bold cyan",
        "muted": "dim white",
        "accent": "bold blue",
        "dim_rule": "dim cyan",
        "banner_border": "bright_cyan",
        "table_border": "dim cyan",
        "wizard": "bold bright_magenta",
        "key": "bold yellow",
        "value": "white",
    }
)

console = Console(theme=LUNGGUARD_THEME, stderr=True)


# ──────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────


class PipelineMode(Enum):
    """Execution modes for the data preparation pipeline."""
    SERIAL = "Serial"
    PARALLEL = "Parallel"


class CleanupChoice(Enum):
    """Options for the interactive dataset cleanup."""
    EXPORT_NEW = "1"
    CLEAN_IN_PLACE = "2"
    SKIP = "3"


class PipelineCommand(Enum):
    """Commands the user can issue during a live run."""
    NONE = "none"
    PAUSE = "pause"
    RESUME = "resume"
    SKIP = "skip"
    ABORT = "abort"


# ──────────────────────────────────────────────────────────
# Logging Setup — Rich console + rotating file
# ──────────────────────────────────────────────────────────

_LOG_DIR = Path("logs")
_LOG_FILENAME = "lungguard_pipeline.log"

_NOISY_LOGGERS: Tuple[str, ...] = (
    "pylidc", "PIL", "matplotlib", "urllib3",
    "tensorflow", "absl", "h5py", "numba",
    "google", "charset_normalizer", "filelock",
)


def setup_rich_logging(
    debug: bool = False,
    log_dir: Optional[Path] = None,
) -> Path:
    """
    Configure dual logging.
    Returns the resolved path to the log file.
    """
    log_level = logging.DEBUG if debug else logging.INFO
    resolved_log_dir = Path(log_dir) if log_dir else _LOG_DIR
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = resolved_log_dir / _LOG_FILENAME

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)

    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=debug,
        log_time_format="[%H:%M:%S]",
    )
    rich_handler.setLevel(log_level)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    root.addHandler(rich_handler)
    root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    return log_file


# ──────────────────────────────────────────────────────────
# ASCII Banner
# ──────────────────────────────────────────────────────────

_LOGO_LINES = [
    "██╗     ██╗   ██╗███╗   ██╗ ██████╗  ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗ ",
    "██║     ██║   ██║████╗  ██║██╔════╝ ██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗",
    "██║     ██║   ██║██╔██╗ ██║██║  ███╗██║  ███╗██║   ██║███████║██████╔╝██║  ██║",
    "██║     ██║   ██║██║╚██╗██║██║   ██║██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║",
    "███████╗╚██████╔╝██║ ╚████║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝",
    "╚══════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ",
]


def print_pipeline_banner(
    mode: Optional[PipelineMode] = None,
    num_workers: Optional[int] = None,
) -> None:
    """Display the startup banner with ASCII logo."""
    logo = Text("\n".join(_LOGO_LINES), style="bright_cyan", justify="center")

    subtitle_parts = [f"Started: [highlight]{datetime.now():%Y-%m-%d %H:%M}[/highlight]"]
    if mode is not None:
        subtitle_parts.insert(0, f"Mode: [highlight]{mode.value}[/highlight]")
    if mode == PipelineMode.PARALLEL and num_workers is not None:
        subtitle_parts.insert(1, f"Workers: [highlight]{num_workers}[/highlight]")

    banner = Panel(
        Group(
            logo,
            Text("Data Preparation Pipeline", style="bold white", justify="center"),
        ),
        subtitle=" │ ".join(subtitle_parts),
        border_style="banner_border",
        padding=(1, 2),
    )
    console.print()
    console.print(banner)
    console.print()


# ──────────────────────────────────────────────────────────
# Phase / Section Headers
# ──────────────────────────────────────────────────────────


def print_phase_header(phase_number: int, total_phases: int, description: str) -> None:
    console.print(f"\n  [phase]❯ [{phase_number}/{total_phases}][/phase] {description}")


def print_section_divider(title: str = "") -> None:
    styled = f"[bold]{title}[/bold]" if title else ""
    console.rule(styled, style="dim_rule")


# ──────────────────────────────────────────────────────────
# Progress Bars
# ──────────────────────────────────────────────────────────


def create_scan_progress() -> Progress:
    """Rich progress bar for the main scan-processing loop."""
    return Progress(
        SpinnerColumn("dots"),
        TextColumn("[accent]{task.description}[/accent]"),
        BarColumn(bar_width=45, style="cyan", complete_style="green", finished_style="bright_green"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        expand=False,
    )


def create_filter_progress() -> Progress:
    """Lighter progress bar for filtering / analysis steps."""
    return Progress(
        SpinnerColumn("arc"),
        TextColumn("[muted]{task.description}[/muted]"),
        BarColumn(bar_width=30, style="dim cyan", complete_style="cyan"),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    )


# ──────────────────────────────────────────────────────────
# Live Dashboard Components
# ──────────────────────────────────────────────────────────


def build_dashboard_layout() -> Layout:
    """Create a 3-row layout: header, progress+stats, controls."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="progress", ratio=3),
        Layout(name="stats", ratio=2),
    )
    return layout


def render_live_stats(
    successful: int,
    failed: int,
    images: int,
    elapsed: float,
    paused: bool = False,
) -> Panel:
    """Render the right-side stats panel for the live dashboard."""
    mins, secs = divmod(int(elapsed), 60)
    throughput = successful / (elapsed / 60) if elapsed > 0 else 0.0

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("k", style="muted", width=14)
    table.add_column("v", justify="right")

    table.add_row("Processed", f"[metric]{successful:,}[/metric]")
    table.add_row("Failed", f"[error]{failed:,}[/error]" if failed else "[success]0[/success]")
    table.add_row("Images", f"[metric]{images:,}[/metric]")
    table.add_row("Elapsed", f"[highlight]{mins}m {secs}s[/highlight]")
    table.add_row("Throughput", f"[muted]{throughput:.1f}/min[/muted]")

    status_line = "[warning]⏸  PAUSED[/warning]" if paused else "[success]● Running[/success]"
    title = f"Live Stats  {status_line}"

    return Panel(table, title=title, border_style="table_border", padding=(0, 1))


def render_controls_bar(paused: bool = False) -> Panel:
    """Render the bottom controls hint bar."""
    parts = [
        "[key]P[/key] [muted]Pause[/muted]" if not paused else "[key]R[/key] [muted]Resume[/muted]",
        "[key]S[/key] [muted]Skip scan[/muted]",
        "[key]Q[/key] [muted]Abort[/muted]",
    ]
    return Panel(
        Text.from_markup("    ".join(parts)),
        border_style="dim",
        padding=(0, 1),
    )


# ──────────────────────────────────────────────────────────
# Table Factory
# ──────────────────────────────────────────────────────────


def _styled_table(title: str, **kwargs) -> Table:
    defaults = dict(
        border_style="table_border",
        show_header=True,
        header_style="bold",
        padding=(0, 2),
        title_style="bold cyan",
    )
    defaults.update(kwargs)
    return Table(title=title, **defaults)


# ──────────────────────────────────────────────────────────
# Summary Tables
# ──────────────────────────────────────────────────────────


def print_split_summary(splits: Dict[str, list]) -> None:
    table = _styled_table("Patient Splits")
    table.add_column("Split", style="cyan")
    table.add_column("Patients", justify="right", style="metric")
    table.add_column("Share", justify="right", style="muted")

    total = sum(len(splits.get(s, [])) for s in ("train", "val", "test"))
    for name in ("train", "val", "test"):
        count = len(splits.get(name, []))
        pct = f"{count / total:.0%}" if total else "–"
        table.add_row(name.capitalize(), str(count), pct)

    table.add_section()
    table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]", "100%")
    console.print(table)


def print_diagnosis_summary(summary: Dict) -> None:
    total = summary.get("total_images", 0)
    problematic = summary.get("problematic_count", 0)
    ratio = summary.get("problematic_ratio", 0.0)

    color = "success" if ratio < 0.05 else ("warning" if ratio < 0.20 else "error")
    health = "Healthy" if ratio < 0.05 else ("Needs Review" if ratio < 0.20 else "Unhealthy")

    table = _styled_table("Dataset Diagnosis", show_header=False)
    table.add_column("Metric", style="muted")
    table.add_column("Value", justify="right")

    table.add_row("Total Images", f"[metric]{total:,}[/metric]")
    table.add_row("Problematic", f"[{color}]{problematic:,}[/{color}]")
    table.add_row("Ratio", f"[{color}]{ratio:.1%}[/{color}]")
    table.add_section()
    table.add_row("Health", f"[{color}]{health}[/{color}]")
    console.print(table)


def print_finalization_summary(csv_path: Path, config_path: Path, yaml_path: Path, total_records: int) -> None:
    tree = Tree("[bold cyan]Pipeline Artifacts[/bold cyan]", guide_style="dim cyan")
    tree.add(f"[muted]CSV  [/muted] {csv_path}")
    tree.add(f"[muted]JSON [/muted] {config_path}")
    tree.add(f"[muted]YAML [/muted] {yaml_path}")

    panel = Panel(
        Group(tree, Text(f"\nTotal Records: {total_records:,}", style="bold white")),
        title="[bold]Pipeline Output[/bold]",
        border_style="table_border",
        padding=(1, 2),
    )
    console.print(panel)


def print_processing_stats(
    total_scans: int, successful: int, failed: int,
    total_images: int, elapsed_seconds: float,
) -> None:
    mins, secs = divmod(int(elapsed_seconds), 60)
    throughput = successful / (elapsed_seconds / 60) if elapsed_seconds > 0 else 0

    table = _styled_table("Processing Statistics", show_header=False)
    table.add_column("", style="muted", width=20)
    table.add_column("", justify="right")

    table.add_row("Scans Processed", f"[metric]{successful:,}[/metric]")
    table.add_row("Scans Failed", f"[error]{failed:,}[/error]" if failed else "[success]0[/success]")
    table.add_row("Images Generated", f"[metric]{total_images:,}[/metric]")
    table.add_row("Elapsed Time", f"[highlight]{mins}m {secs}s[/highlight]")
    table.add_row("Throughput", f"[muted]{throughput:.1f} scans/min[/muted]")
    console.print(table)


def print_config_review(config_dict: Dict) -> None:
    """Print a review table of the resolved configuration before running."""
    table = _styled_table("Configuration Review")
    table.add_column("Parameter", style="key")
    table.add_column("Value", style="value")

    display_order = [
        ("Mode", "mode"), ("DICOM Path", "data_path"), ("Output Dir", "output_dir"),
        ("Workers", "num_workers"),
        ("Train / Val / Test", "split_display"),
        ("Nodule ⌀ Range", "diameter_display"),
        ("Slices per Nodule", "slices_per_nodule"),
        ("Seed", "random_seed"), ("Debug", "debug"),
    ]
    for label, key in display_order:
        val = config_dict.get(key)
        if val is not None:
            table.add_row(label, str(val))

    console.print(table)


# ──────────────────────────────────────────────────────────
# Interactive Prompts
# ──────────────────────────────────────────────────────────


def prompt_save_reports() -> bool:
    return Confirm.ask("  [bold]Save detailed analysis CSV reports?[/bold]", console=console, default=False)


def prompt_cleanup_choice() -> CleanupChoice:
    print_section_divider("Clean Dataset Export")
    table = Table(show_header=False, border_style="dim", padding=(0, 2), box=None)
    table.add_column("Opt", style="bold cyan", width=4)
    table.add_column("Description")
    table.add_row("1", "Create [bold]new[/bold] clean dataset (copy valid files)")
    table.add_row("2", "Clean [bold]in-place[/bold] (delete invalid files)")
    table.add_row("3", "Skip cleanup")
    console.print(table)
    console.print()
    raw = Prompt.ask("  Select an option", choices=["1", "2", "3"], default="3", console=console)
    return CleanupChoice(raw)


def prompt_export_path() -> Optional[Path]:
    raw = Prompt.ask("  [bold]Enter path for the new dataset folder[/bold]", console=console, default="")
    return Path(raw).resolve() if raw.strip() else None


def prompt_destructive_confirm() -> bool:
    return Confirm.ask(
        "  [error]⚠  This will PERMANENTLY DELETE invalid files. Proceed?[/error]",
        console=console, default=False,
    )


# ──────────────────────────────────────────────────────────
# Status Messages
# ──────────────────────────────────────────────────────────


def print_success(msg: str) -> None:
    console.print(f"  [success]✓[/success] {msg}")

def print_warning(msg: str) -> None:
    console.print(f"  [warning]⚠[/warning] {msg}")

def print_error(msg: str) -> None:
    console.print(f"  [error]✗[/error] {msg}")

def print_info(msg: str) -> None:
    console.print(f"  [info]ℹ[/info] {msg}")


# ──────────────────────────────────────────────────────────
# Completion Banner
# ──────────────────────────────────────────────────────────


def print_completion_banner(log_file: Optional[Path] = None) -> None:
    parts = [Text("✓ Pipeline execution finished", style="success")]
    if log_file:
        parts.append(Text(f"  Full log → {log_file}", style="muted"))
    console.print()
    console.print(Panel(Group(*parts), border_style="green", padding=(1, 2)))
    console.print()