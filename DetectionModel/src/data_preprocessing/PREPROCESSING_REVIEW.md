# Preprocessing Package Review

> Generated: 2026-03-22
> Scope: `DetectionModel/src/data_preprocessing/`
> MONAI version: 1.5.2 | pylidc version: 0.2.3

---

## 1. Package Structure

```
data_preprocessing/
├── config.py                          # Pipeline hyperparameters
├── __init__.py                        # Public API exports
├── __main__.py                        # Interactive CLI entry point
├── core/
│   ├── coordinate_transformer.py      # Coordinate space math
│   ├── scan_protocols.py              # Data contracts (Protocol + dataclasses)
│   └── pylidc_config.py              # pylidc setup / path wiring
├── sources/
│   ├── scan_adapters.py              # Adapters: PyLIDC → protocol, DICOM → protocol
│   └── annotation_processor.py       # Radiologist annotation aggregation
├── preprocessing/
│   ├── volume_processor.py           # 3D volume clean + resample + window
│   ├── slice_processor.py            # 2D slice utilities (resize, crop, 2.5D)
│   ├── bbox_converter.py             # Coordinate → YOLO bbox math
│   └── slice_quality_gate.py         # CLAHE + 4-check quality filter
├── pipelines/
│   ├── scan_processor.py             # Unified per-scan orchestrator
│   ├── batch_preparation.py          # Serial pipeline with live dashboard
│   ├── parallel_preparation.py       # Multiprocessing pipeline
│   └── inference_processor.py        # In-memory inference slice prep
├── io/
│   ├── atomic_io.py                  # Atomic image+label save / rollback
│   └── dataset_writer.py             # CSV / JSON / YOLO YAML finalization
└── utils/
    ├── patient_splitter.py           # Patient-level train/val/test split
    └── dataset_diagnostics.py        # Full quality audit + clean export
```

---

## 2. Module-by-Module Explanation

### 2.1 `config.py` — Pipeline Hyperparameters
Single `DataPrepConfig` dataclass holding all knobs: data paths, split ratios,
target voxel spacing `(1,1,1)` mm, lung window center/width `(-600, 1500)`,
nodule diameter bounds `(3–100 mm)`, bbox padding factor `(1.5×)`,
output image size `(512×512)`, and random seed.

**Issue:** `data_path` is hardcoded to a Windows path (`E:\FinalsProject\...`).
Must be overridden at runtime; no env-variable fallback.

---

### 2.2 `core/scan_protocols.py` — Data Contracts
Defines the frozen dataclasses that flow through the entire system:

| Class | Used in | Contains |
|-------|---------|----------|
| `VolumeData` | data-prep + inference | numpy volume + (z,y,x) spacing |
| `NoduleData` | data-prep | centroid, features dict, slice indices |
| `YOLODetection` | inference | slice index, normalized bbox, confidence |
| `ProcessedSlice` | inference | 2.5D enhanced array, quality flag |
| `NoduleCropResult` | inference | 2.5D crop + single-channel crop |

`ScanSource` is a `Protocol` enforcing two methods — `load_volume()` and
`extract_nodules()` — on any data adapter.

---

### 2.3 `core/coordinate_transformer.py` — Coordinate Math
Static class with three methods:
- `transform_coordinates_to_resampled(zyx, orig_spacing, target_spacing)` — scales
  (z,y,x) coords proportionally when volume is resampled
- `transform_slice_to_resampled_space(idx, z_orig, z_target)` — scales a single z-index
- `is_slice_within_volume(idx, depth)` — bounds check

Simple scale-factor arithmetic (`orig / target × coord`). No error handling for
zero spacing or negative indices.

---

### 2.4 `core/pylidc_config.py` — pylidc Setup
Detects OS, writes the `[dicom] path = ...` INI file to `~/pylidcrc` (Unix) or
`~/pylidc.conf` (Windows), sets an environment variable as a fallback, and
validates that the LIDC directory contains `LIDC-IDRI-*` patient folders.
Proceeds (with a warning) even if validation fails.

---

### 2.5 `sources/scan_adapters.py` — Adapter Layer

#### `PyLIDCScanSource`
Wraps a `pylidc.Scan` for training data:
1. `load_volume()` — calls `scan.to_volume()`, extracts `pixel_spacing` + `slice_spacing`
2. `extract_nodules()` — calls `scan.cluster_annotations()`, delegates feature/centroid
   extraction to `NoduleAnnotationProcessor`, returns `List[NoduleData]`

#### `DICOMScanSource`
Wraps a raw DICOM directory for inference:
1. `load_volume()` — calls MONAI's `LoadImage` on the directory, extracts spacing
   from metadata `pixdim[3,1,2]` (brittle index) or affine fallback
2. `extract_nodules()` — returns `[]` (no ground truth at inference time)

**Issue:** DICOM spacing extraction has a hardcoded `pixdim` index `[3,1,2]` that
will silently return wrong values if the metadata shape changes.

---

### 2.6 `sources/annotation_processor.py` — Annotation Aggregation
Static class that:
1. Averages all 9 radiologist feature scores (malignancy, spiculation, etc.)
   across 1–4 annotations per cluster
2. Averages centroids in original space, transforms to resampled space
3. Collects contour slice indices across all annotations, transforms to resampled
   space, deduplicates

Uses `contextlib.suppress(Exception)` everywhere — a silent swallower that can
mask legitimate data corruption.

---

### 2.7 `preprocessing/volume_processor.py` — 3D Volume Pipeline
`VolumePreprocessingPipeline.preprocess()` runs in sequence:
1. Cast to float32
2. Replace NaN → -1000 HU
3. **Offset detection** — if P5 of center-slice valid pixels > -100 HU,
   subtract 1024 (scanner stored air at 0 instead of -1000)
4. Clamp HU range [-1000, 3000]
5. **Resample** to target spacing using `MONAI.Zoom`
6. **Lung window** + normalize to [0,1] using `MONAI.ScaleIntensityRange`

Already uses MONAI for steps 5–6. The offset correction (step 3) is a
domain-specific heuristic with no equivalent in standard libraries.

---

### 2.8 `preprocessing/slice_processor.py` — 2D Slice Utilities
Static utilities for the 2D stage:
- `clean_and_fix_volume()` — **duplicate** of the offset detection logic in
  `volume_processor.py` (same code, different class)
- `resample_volume()` — uses `scipy.ndimage.zoom` (redundant given step 5 above)
- `apply_windowing()` — uses `np.interp` (redundant given step 6 above)
- `resize_slice_to_target()` — resize with aspect-ratio preservation + padding
- `center_crop_slice()` — scale-then-crop, returns `crop_info` dict for bbox adjustment
- `create_25d_sandwich()` — stacks `[z-1, z, z+1]` into a 3-channel RGB image

The 2.5D sandwich has no MONAI equivalent and belongs here.
The volume-level methods (`clean_and_fix_volume`, `resample_volume`, `apply_windowing`)
are duplicates of work already done upstream.

---

### 2.9 `preprocessing/bbox_converter.py` — YOLO Bbox Math
Static class:
1. `compute_diameter(bbox, spacing)` — Euclidean distance in x,y from pylidc bbox
2. `convert_to_yolo_format(centroid, diameter, img_size)` — normalized `(cx, cy, w, h)`
3. `compute_nodule_bbox_yolo()` — wrapper that clamps to [0.001, 1.0]
4. `adjust_bbox_for_resize()` — scales bbox after aspect-ratio resize + padding
5. `adjust_bbox_for_center_crop()` — adjusts bbox after scale-then-crop;
   returns `None` if nodule center is cropped out

The center-crop offset arithmetic (positive = crop, negative = pad) is subtle
and the most error-prone part of the package.

---

### 2.10 `preprocessing/slice_quality_gate.py` — Quality Filter
`SliceQualityGate.validate_and_enhance(slice)`:
1. Applies CLAHE (clipLimit=2.5, gridSize=8×8) per channel
2. Runs 4 checks on the center slice:
   - LOW_CONTRAST: contrast range < 80
   - NO_BG: dark pixel ratio < 0.20 **and** mean > 100 (detects no-background slices)
   - TOO_BRIGHT: mean > 180
   - INSUFFICIENT_LUNG: lung/body area ratio < 0.12
3. Returns `(enhanced_image, passed: bool, reason: str)`

Lung/body ratio computed via morphological close+open (15×15 ellipse) —
same logic duplicated in `dataset_diagnostics.py`.

---

### 2.11 `pipelines/inference_processor.py` — Shared Inference Core
`InferencePipeline` is the **shared core** used by both the data-prep pipeline
and the FastAPI inference server:
- `prepare_slice_image()` — sandwich → CLAHE → quality gate → returns
  `(enhanced_25d, crop_info, quality_passed, reason)`
- `prepare_slices_for_yolo()` — batch version, filters to passing slices
- `extract_nodule_crops()` — converts YOLO detections → pixel crops

Design is clean: all slice logic lives here, no I/O.

---

### 2.12 `pipelines/scan_processor.py` — Per-Scan Orchestrator
`CTScanProcessor.process_scan(source, split)`:
1. Loads and preprocesses volume (`VolumePreprocessingPipeline`)
2. Filters nodules by diameter bounds
3. For each valid nodule: selects candidate slices (evenly distributed),
   calls `_process_single_slice()` for each
4. **Nodule integrity guarantee**: if a nodule doesn't produce exactly
   `slices_per_nodule` slices, all its files are deleted (all-or-nothing)

`_process_single_slice()` calls `InferencePipeline.prepare_slice_image()`,
computes YOLO bbox, and atomically saves image+label.

---

### 2.13 `pipelines/batch_preparation.py` — Serial Pipeline
`DataPreparationPipeline`:
- `setup()` — configures pylidc, creates YOLO directory tree, loads/creates
  patient splits, displays summary
- `run_serial()` — iterates scans inside a `LiveDashboard` (pause/resume/skip/abort)
- `finalize()` — saves metadata CSV, config JSON, YOLO `dataset.yaml`

Patient splits are saved as `patient_splits.json` so runs can be resumed.
Existing split directories are cleaned but metadata is preserved.

---

### 2.14 `pipelines/parallel_preparation.py` — Parallel Pipeline
`run_parallel_pipeline(config, num_workers)`:
- Shares `setup()` and `finalize()` from `DataPreparationPipeline`
- Worker processes initialized with `_init_worker()`: suppresses all warnings,
  redirects logs to file only (keeps Rich console clean), configures own pylidc connection
- Uses `multiprocessing.Pool` with `imap_unordered` + chunkSize=1
- Abort clears all_metadata to prevent partial saves

**Issue:** Each worker re-queries pylidc independently. On a large dataset this
causes significant startup overhead per worker.

---

### 2.15 `io/atomic_io.py` — Atomic File Saves
`atomic_save_image_and_label(image, bbox, img_path, lbl_path)`:
1. Save image (RGB→BGR for OpenCV)
2. If image fails → return failure
3. Save label (`class_id cx cy w h` with 6-decimal precision)
4. If label fails → **delete image** (rollback) → return failure
5. Both succeed → return `AtomicSaveResult(success=True, ...)`

Guarantees no orphaned image files without matching label files.

---

### 2.16 `io/dataset_writer.py` — Output Finalization
Three functions called at the end of every run:
- `save_metadata_csv()` → `metadata.csv` (all slice rows)
- `save_config_json()` → `config.json` (config + sample counts + timestamp)
- `save_yolo_yaml()` → `dataset.yaml` (train/val/test paths, nc=1)

---

### 2.17 `utils/patient_splitter.py` — No-Leakage Split
Two-step sklearn `train_test_split` ensuring all scans from one patient always
land in the same split. Lookup is O(n) reverse-mapping from the splits dict.

---

### 2.18 `utils/dataset_diagnostics.py` — Quality Audit
`DatasetDiagnoser.analyze()` scans the output directory, runs 9 checks per image
(uniform, too-dark, too-bright, low-contrast, no-background, insufficient-lung,
no-lung-tissue, bad-aspect-ratio, narrow-content), and produces:
- Rich terminal summary with per-split and per-problem breakdowns
- Nodule integrity report (expects exactly 3 slices per nodule)
- Clean export: copy valid → new dir, or delete invalid in-place

---

## 3. Logic Complexity: What's Unnecessarily Complex

### 3.1 Duplicated Volume Cleaning Logic
`VolumePreprocessingPipeline._clean_with_offset_detection()` and
`SlicePreprocessor.clean_and_fix_volume()` are nearly identical. The inference
path calls the latter even though the volume has already been cleaned upstream.
**One version should own this logic.**

### 3.2 Duplicated Lung/Body Morphology
`SliceQualityGate._compute_lung_ratio()` and
`DatasetDiagnoser._compute_lung_metrics()` both implement morphological
close+open body-mask + lung-candidate filtering with similar thresholds that
are defined separately in two different config objects (`SliceQualityConfig`,
`AnalysisThresholds`). **Shared helper function needed.**

### 3.3 `SlicePreprocessor.resample_volume()` is Dead Code in the Normal Path
`volume_processor.py` already resamples with MONAI `Zoom`. `slice_processor.py`
duplicates this with `scipy.ndimage.zoom`. In the normal data-prep flow the
scipy version is never called; it's only reachable if someone calls
`SlicePreprocessor` directly. It should be removed or made internal.

### 3.4 `CoordinateTransformer` is Manual Bookkeeping of What MONAI Tracks Automatically
When MONAI's `Spacingd` resamples a volume it updates the affine matrix.
All coordinate transformations from world space to resampled voxel space are
then a matrix multiply — no custom class needed. The current implementation
manually computes scale factors that are already encoded in the affine.

### 3.5 `BoundingBoxConverter.adjust_bbox_for_center_crop()` — Fragile Offset Arithmetic
The bbox adjustment for center-crop has a subtle sign convention: a positive
`crop_info['crop_x']` means "pixels cropped from the left" and must be subtracted,
while a negative value means "pixels padded on the left" and must also be subtracted
(padding shifts the content right). This produces correct results but is
non-obvious and undocumented. A regression test is the only safety net.

### 3.6 `annotation_processor.py` Uses `contextlib.suppress(Exception)` on Everything
Silent exception suppression means a malformed annotation (e.g., corrupted DICOM
metadata) produces a silently-defaulted centroid of `(0, 0, 0)` instead of a
visible error. This is very hard to debug when a nodule appears at the top-left
corner of every scan.

### 3.7 Parallel Pipeline Worker Query Overhead
Each worker calls `pylidc.query(Scan).filter(patient_id=pid).first()` at startup.
For 1010 LIDC-IDRI patients with 8 workers this is 1010 serial round-trips to
SQLite. Pre-fetching all `(patient_id, scan_id)` pairs in the main process and
passing scan IDs (not patient IDs) to workers would eliminate this.

---

## 4. What MONAI Can Replace

### 4.1 HIGH — Replace `scipy.ndimage.zoom` with `Spacingd`
**Current** (`slice_processor.py`, `volume_processor.py`):
```python
zoom_factors = [o/t for o, t in zip(original_spacing, target_spacing)]
resampled = zoom(volume, zoom=zoom_factors, order=1)
```
**MONAI replacement:**
```python
from monai.transforms import Spacingd
transform = Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear")
resampled_dict = transform({"image": volume_tensor, "image_meta_dict": meta})
```
`Spacingd` automatically updates the affine matrix, which eliminates the need
for `CoordinateTransformer` entirely — bbox coordinates in the resampled volume
are computed by applying the inverse affine to the original world coordinates.

### 4.2 HIGH — Replace Manual Windowing in `slice_processor.py` with `ScaleIntensityRanged`
**Current:**
```python
windowed = np.interp(volume, [-1350, 150], [0, 1])
```
**MONAI (already used in `volume_processor.py`, should be the single path):**
```python
from monai.transforms import ScaleIntensityRange
windowed = ScaleIntensityRange(a_min=-1350, a_max=150, b_min=0.0, b_max=1.0, clip=True)(volume)
```
`volume_processor.py` already does this correctly; `slice_processor.py`
should be updated to call the volume-level result instead of re-windowing.

### 4.3 MEDIUM — Use MONAI `CacheDataset` for Training Data Loading
**Current:** custom CSV → file-path lookup → `cv2.imread` per batch.
**MONAI replacement:**
```python
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged

train_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0),
])
train_ds = CacheDataset(data=train_file_dicts, transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=16, num_workers=4, shuffle=True)
```
`CacheDataset` with `cache_rate=1.0` loads and preprocesses the entire dataset
once and serves it from RAM — a significant speedup for repeated epoch training.

### 4.4 MEDIUM — Use MONAI Augmentation Suite for Training Robustness
No augmentation is currently applied. Adding MONAI's medical-imaging-aware
augmentations would improve generalization:
```python
from monai.transforms import (
    RandRotate90d, RandFlipd, RandGaussianNoised,
    RandScaleIntensityd, RandAffined
)
augment = Compose([
    RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    RandGaussianNoised(keys=["image"], prob=0.2, std=0.01),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
])
```

### 4.5 LOW — Replace `DICOMScanSource` Spacing Extraction
Current extraction uses brittle `pixdim[3,1,2]` index. MONAI's `LoadImaged`
always returns spacing as `image_meta_dict["pixdim"][1:4]` in a predictable order.
Switching to the dict-based API removes the magic index.

---

## 5. What pylidc Already Provides (and what we could use more of)

### 5.1 Already Used Correctly
- `scan.to_volume()` — raw HU volume from DICOM ✓
- `scan.cluster_annotations()` — groups radiologists by physical nodule ✓
- `scan.pixel_spacing` + `scan.slice_spacing` — in-plane and axial spacing ✓
- `annotation.centroid` — 3D centroid ✓
- `annotation.contour_slice_indices` — slice presence ✓
- All 9 feature properties (malignancy, spiculation, etc.) ✓

### 5.2 Not Used — `annotation.bbox(pad=None)`
pylidc can return the 3D bounding box as Python slices directly, with optional
mm-based padding:
```python
bbox_slices = ann.bbox(pad=3.0)  # 3mm padding in all dimensions
z_slice, y_slice, x_slice = bbox_slices
```
`BoundingBoxConverter.compute_diameter()` re-derives this manually from raw
annotation data. Using `ann.bbox()` would be simpler and less error-prone.

### 5.3 Not Used — `annotation.diameter`
```python
diameter_mm = ann.diameter  # Computed from longest axis of contour
```
Currently the code computes diameter from the bbox extents, which is an
approximation. `ann.diameter` uses the actual annotation contour.

### 5.4 Not Used — Mask Generation from Contours
pylidc can generate binary masks per slice from annotation contour data.
This is not needed for YOLO bbox detection but would be useful if the project
ever adds segmentation head training.

---

## 6. Recommended Refactoring Priorities

| Priority | Change | Files Affected | Benefit |
|----------|--------|---------------|---------|
| **1 — High** | Delete `SlicePreprocessor.resample_volume()` and `apply_windowing()`; unify all volume-level preprocessing in `VolumePreprocessingPipeline` | `slice_processor.py`, `volume_processor.py` | Eliminate dead code + duplication |
| **2 — High** | Replace `scipy.ndimage.zoom` with `Spacingd`; use MONAI affine to derive resampled coordinates; delete `CoordinateTransformer` | `volume_processor.py`, `coordinate_transformer.py`, `annotation_processor.py` | ~80 lines removed, auto-correct coordinate math |
| **3 — High** | Use `ann.bbox()` and `ann.diameter` in `annotation_processor.py` | `annotation_processor.py`, `bbox_converter.py` | Removes manual bbox arithmetic |
| **4 — Medium** | Extract shared lung/body morphology helper; share between `SliceQualityGate` and `DatasetDiagnoser` | `slice_quality_gate.py`, `dataset_diagnostics.py` | Single source of truth for thresholds |
| **5 — Medium** | Replace `contextlib.suppress(Exception)` in annotation processor with targeted exception types and explicit logging | `annotation_processor.py` | Surfaced data errors |
| **6 — Medium** | Pre-fetch scan IDs in main process and pass to parallel workers | `parallel_preparation.py` | Eliminates per-worker SQLite overhead |
| **7 — Medium** | Add MONAI augmentation pipeline for training (rotations, flips, noise) | New `augmentation.py` in preprocessing/ | Training robustness |
| **8 — Low** | Switch training data loading to `CacheDataset` | Training scripts | Faster epoch iteration |
| **9 — Low** | Fix `DICOMScanSource` brittle `pixdim` index | `scan_adapters.py` | Inference robustness |

---

## 7. What Should Stay as Custom Code

| Component | Reason to Keep |
|-----------|----------------|
| Offset detection (P5 heuristic) | Domain-specific scanner artifact; no library equivalent |
| 2.5D sandwich creation | MONAI is 3D-first; this 2D-centric operation is correct as-is |
| `atomic_io.py` | MONAI has no atomic paired image+label save; this is a correct solution |
| `slice_quality_gate.py` checks | Domain thresholds specific to LIDC/lung window preprocessing |
| YOLO bbox format conversion | MONAI detection is 3D RetinaNet; 2D YOLO format is project-specific |
| `patient_splitter.py` | sklearn two-step split with patient-level grouping is correct and simple |
| `DatasetDiagnoser` | No MONAI equivalent for post-hoc nodule-integrity audit |

---

## 8. Summary Scorecard

| Dimension | Current State | After Recommended Changes |
|-----------|--------------|--------------------------|
| Duplicate code blocks | 4 (offset detection ×2, morphology ×2, windowing ×2) | 0 |
| Manual coordinate math | `CoordinateTransformer` (50 lines) | Eliminated via MONAI affine |
| MONAI utilization | Partial (Zoom + ScaleIntensity only) | Full (Spacingd + affine pipeline) |
| pylidc utilization | 80% (missing bbox, diameter) | 100% |
| Silent error swallowing | `contextlib.suppress(Exception)` in 4 methods | Targeted exception handling |
| Training augmentation | None | MONAI augmentation suite |
| Data loading at train time | Manual CSV + cv2 | `CacheDataset` |
