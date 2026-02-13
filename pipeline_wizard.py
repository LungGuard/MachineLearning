"""
LungGuard Interactive Pipeline Wizard.

Three-phase interactive TUI:
  Phase 1 — Startup Wizard:   prompts for mode, paths, ratios, workers
  Phase 2 — Live Dashboard:   real-time progress with pause/resume/skip/abort
  Phase 3 — Post-Processing:  diagnosis, reports, cleanup

Keyboard controls during processing:
  [P] Pause   [R] Resume   [S] Skip current scan   [Q] Abort
"""


import contextlib
import logging
import os
import select
import sys
import threading
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

from rich.live import Live
from rich.prompt import Confirm, Prompt
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Group

from terminal_ui import (
    CleanupChoice,
    PipelineCommand,
    PipelineMode,
    build_dashboard_layout,
    console,
    create_filter_progress,
    create_scan_progress,
    print_completion_banner,
    print_config_review,
    print_diagnosis_summary,
    print_error,
    print_finalization_summary,
    print_info,
    print_phase_header,
    print_pipeline_banner,
    print_processing_stats,
    print_section_divider,
    print_split_summary,
    print_success,
    print_warning,
    prompt_cleanup_choice,
    prompt_destructive_confirm,
    prompt_export_path,
    prompt_save_reports,
    render_controls_bar,
    render_live_stats,
    setup_rich_logging,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  PHASE 1 — STARTUP WIZARD
# ══════════════════════════════════════════════════════════

_DEFAULT_DATA_PATH = r"E:\FinalsProject\Datasets\CancerDetection\images\manifest-1600709154662\LIDC-IDRI"
_DEFAULT_OUTPUT_DIR = r".\DetectionModel\datasets_monai"


def _ask_mode() -> PipelineMode:
    print_section_divider("Pipeline Mode")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Opt", style="bold cyan", width=4)
    table.add_column("Description")
    table.add_row("1", "[bold]Serial[/bold]   — single-threaded, easier to debug")
    table.add_row("2", "[bold]Parallel[/bold] — multi-process, much faster")
    console.print(table)
    console.print()
    raw = Prompt.ask("  Select mode", choices=["1", "2"], default="2", console=console)
    return PipelineMode.SERIAL if raw == "1" else PipelineMode.PARALLEL


def _ask_workers() -> int:
    max_w = max(1, cpu_count() - 2)
    default = max(1, cpu_count() // 2)
    raw = Prompt.ask(
        f"  [bold]Number of workers[/bold] [muted](1–{max_w}, 0 = auto)[/muted]",
        default=str(default),
        console=console,
    )
    parsed = int(raw) if raw.strip().isdigit() else default
    return max(1, min(parsed, max_w)) if parsed != 0 else max_w


def _ask_path(label: str, default: str) -> str:
    raw = Prompt.ask(f"  [bold]{label}[/bold]", default=default, console=console)
    return raw.strip() or default


def _ask_ratios() -> Tuple[float, float, float]:
    raw = Prompt.ask(
        "  [bold]Train / Val / Test ratios[/bold] [muted](space-separated, e.g. 0.70 0.15 0.15)[/muted]",
        default="0.70 0.15 0.15",
        console=console,
    )
    parts = raw.strip().split()
    result = (0.70, 0.15, 0.15)
    if len(parts) == 3:
        try:
            parsed = tuple(float(p) for p in parts)
            total = sum(parsed)
            if abs(total - 1.0) < 0.01:
                result = parsed
            else:
                print_warning(f"Ratios sum to {total:.2f}, not 1.0 — using defaults")
        except ValueError:
            print_warning("Could not parse ratios — using defaults")
    return result


def _ask_advanced() -> Dict[str, Any]:
    """Optional advanced settings."""
    advanced: Dict[str, Any] = {}
    if Confirm.ask("  [bold]Configure advanced settings?[/bold]", default=False, console=console):
        console.print()
        raw_min = Prompt.ask("    Min nodule diameter (mm)", default="3.0", console=console)
        raw_max = Prompt.ask("    Max nodule diameter (mm)", default="100.0", console=console)
        raw_slices = Prompt.ask("    Slices per nodule", default="3", console=console)
        raw_seed = Prompt.ask("    Random seed", default="42", console=console)
        debug = Confirm.ask("    Enable debug logging?", default=False, console=console)

        advanced["min_nodule_diameter"] = float(raw_min)
        advanced["max_nodule_diameter"] = float(raw_max)
        advanced["slices_per_nodule"] = int(raw_slices)
        advanced["random_seed"] = int(raw_seed)
        advanced["debug"] = debug

    return advanced


def run_wizard() -> Dict[str, Any]:
    """
    Interactive startup wizard. Returns a resolved config dict ready for
    DataPrepConfig construction + pipeline dispatch.
    """
    print_pipeline_banner()
    print_section_divider("Configuration Wizard")

    mode = _ask_mode()
    console.print()

    num_workers = _ask_workers() if mode == PipelineMode.PARALLEL else None
    console.print()

    data_path = _ask_path("DICOM data path", _DEFAULT_DATA_PATH)
    output_dir = _ask_path("Output directory", _DEFAULT_OUTPUT_DIR)
    console.print()

    train, val, test = _ask_ratios()
    console.print()

    advanced = _ask_advanced()

    config_dict = {
        "mode": mode.value,
        "data_path": data_path,
        "output_dir": output_dir,
        "num_workers": num_workers,
        "train_ratio": train,
        "val_ratio": val,
        "test_ratio": test,
        "split_display": f"{train:.0%} / {val:.0%} / {test:.0%}",
        "min_nodule_diameter": advanced.get("min_nodule_diameter", 3.0),
        "max_nodule_diameter": advanced.get("max_nodule_diameter", 100.0),
        "diameter_display": f"{advanced.get('min_nodule_diameter', 3.0)}–{advanced.get('max_nodule_diameter', 100.0)} mm",
        "slices_per_nodule": advanced.get("slices_per_nodule", 3),
        "random_seed": advanced.get("random_seed", 42),
        "debug": advanced.get("debug", False),
        "log_freq": 5,
    }

    console.print()
    print_config_review(config_dict)
    console.print()

    confirmed = Confirm.ask("  [bold]Confirm and start pipeline?[/bold]", default=True, console=console)
    if not confirmed:
        print_info("Aborted by user.")
        sys.exit(0)

    return config_dict


# ══════════════════════════════════════════════════════════
#  PHASE 2 — LIVE DASHBOARD WITH INTERACTIVE CONTROLS
# ══════════════════════════════════════════════════════════


class _KeyboardListener(threading.Thread):
    """
    Non-blocking keyboard listener running in a background thread.
    Sets command flags that the processing loop checks.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.command = PipelineCommand.NONE
        self._stop_event = threading.Event()

    def run(self) -> None:
        """Listen for single-character commands on stdin."""
        with contextlib.suppress(Exception):
            # Windows: use msvcrt
            if sys.platform == "win32":
                import msvcrt
                while not self._stop_event.is_set():
                    if msvcrt.kbhit():
                        ch = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                        self._handle_key(ch)
                    time.sleep(0.1)
            else:
                # Unix: use select on stdin
                import tty
                import termios
                old = termios.tcgetattr(sys.stdin)
                try:
                    tty.setcbreak(sys.stdin.fileno())
                    while not self._stop_event.is_set():
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            ch = sys.stdin.read(1).lower()
                            self._handle_key(ch)
                finally:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)

    def _handle_key(self, ch: str) -> None:
        key_map = {
            "p": PipelineCommand.PAUSE,
            "r": PipelineCommand.RESUME,
            "s": PipelineCommand.SKIP,
            "q": PipelineCommand.ABORT,
        }
        mapped = key_map.get(ch, PipelineCommand.NONE)
        if mapped != PipelineCommand.NONE:
            self.command = mapped

    def consume(self) -> PipelineCommand:
        """Read and reset the current command."""
        cmd = self.command
        self.command = PipelineCommand.NONE
        return cmd

    def stop(self) -> None:
        self._stop_event.set()


class LiveDashboard:
    """
    Wraps a Rich Live display that shows progress + stats + controls,
    and exposes pause/resume/skip/abort from keyboard input.
    """

    def __init__(self, total_scans: int):
        self.total = total_scans
        self.successful = 0
        self.failed = 0
        self.images = 0
        self.paused = False
        self.aborted = False
        self.skip_current = False
        self._start_time = time.time()
        self._progress = create_scan_progress()
        self._task_id = self._progress.add_task("Processing CT scans", total=total_scans)
        self._listener = _KeyboardListener()

    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time

    def _render(self) -> Panel:
        """Build the composite dashboard panel."""
        stats = render_live_stats(
            self.successful, self.failed, self.images, self.elapsed, self.paused,
        )
        controls = render_controls_bar(self.paused)

        # Build a group: progress bar on top, stats below, controls at bottom
        return Panel(
            Group(self._progress, Text(""), stats, controls),
            title="[bold cyan]LungGuard Live Dashboard[/bold cyan]",
            border_style="banner_border",
            padding=(1, 1),
        )

    def start(self) -> "LiveDashboard":
        self._live = Live(
            self._render(),
            console=console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()
        self._listener.start()
        return self

    def stop(self) -> None:
        self._listener.stop()
        self._live.stop()

    def poll_commands(self) -> None:
        """Check keyboard input and update state accordingly."""
        cmd = self._listener.consume()
        if cmd == PipelineCommand.PAUSE:
            self.paused = True
        elif cmd == PipelineCommand.RESUME:
            self.paused = False
        elif cmd == PipelineCommand.SKIP:
            self.skip_current = True
        elif cmd == PipelineCommand.ABORT:
            self.aborted = True

    def wait_while_paused(self) -> None:
        """Block the processing loop while paused, still polling for resume/abort."""
        while self.paused and not self.aborted:
            self.poll_commands()
            self._live.update(self._render())
            time.sleep(0.2)

    def advance(self, scan_images: int = 0, was_error: bool = False) -> None:
        """Record one scan completed and refresh the display."""
        if was_error:
            self.failed += 1
        else:
            self.successful += 1
        self.images += scan_images
        self.skip_current = False
        self._progress.advance(self._task_id)
        self._live.update(self._render())

    def refresh(self) -> None:
        self._live.update(self._render())


# ══════════════════════════════════════════════════════════
#  PHASE 3 — POST-PROCESSING CLEANUP  (shared)
# ══════════════════════════════════════════════════════════


def run_interactive_cleanup(output_dir: str, diagnoser_cls: Any) -> None:
    """
    Post-processing flow: diagnose → optionally save reports → cleanup.
    Uses Rich prompts throughout.
    """
    print_section_divider("Post-Processing: Diagnosis & Cleanup")

    diagnoser = diagnoser_cls(output_dir)
    print_info("Running analysis on generated images…")

    with create_filter_progress() as progress:
        task = progress.add_task("Analysing images", total=None)
        diagnoser.analyze()
        progress.update(task, completed=1, total=1)

    summary = diagnoser.get_summary_report()
    print_diagnosis_summary(summary)
    console.print()

    if prompt_save_reports():
        diagnoser.save_reports_to_disk()
        print_success("Reports saved")

    choice = prompt_cleanup_choice()
    _dispatch_cleanup(choice, diagnoser)


def _dispatch_cleanup(choice: CleanupChoice, diagnoser: Any) -> None:
    actions = {
        CleanupChoice.EXPORT_NEW: _cleanup_export_new,
        CleanupChoice.CLEAN_IN_PLACE: _cleanup_in_place,
        CleanupChoice.SKIP: lambda _d: print_info("Cleanup skipped"),
    }
    actions[choice](diagnoser)


def _cleanup_export_new(diagnoser: Any) -> None:
    new_path = prompt_export_path()
    if new_path is None:
        print_warning("No path provided — skipping export")
        return
    print_info(f"Exporting clean dataset to [muted]{new_path}[/muted]…")
    diagnoser.export_clean_dataset(output_dir=str(new_path), overwrite_existing=False)
    print_success("Clean dataset exported")


def _cleanup_in_place(diagnoser: Any) -> None:
    confirmed = prompt_destructive_confirm()
    if not confirmed:
        print_info("Operation cancelled")
        return
    print_info("Cleaning dataset in-place…")
    diagnoser.export_clean_dataset(overwrite_existing=True)
    print_success("In-place cleanup complete")