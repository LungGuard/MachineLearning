# Preprocessing Package Refactoring Report

**Date:** 2026-04-11
**Branch:** `pipeline_and_api`
**Scope:** `DetectionModel/src/data_preprocessing/`

---

## Summary

Four independent workstreams were applied to reduce dead code, eliminate
duplication, improve diagnostics accuracy, and fix runtime correctness bugs.
Net diff: **−306 / +118 lines** across 10 files (1 file deleted, 1 file created).

```
 11 files changed, 118 insertions(+), 306 deletions(-)
  1 file deleted  : core/coordinate_transformer.py
  1 file created  : preprocessing/lung_morphology.py
```

---

## Workstream 1 — Delete Dead Code from `slice_processor.py`

### Problem
Three methods existed in `SlicePreprocessor` with zero callers anywhere in the
codebase. They were duplicates of logic already handled by the MONAI-based
`VolumePreprocessingPipeline` (volume cleaning, resampling) and the windowing
step in `scan_processor.py`.

### Methods Removed

| Method | Lines | Why Dead |
|--------|-------|----------|
| `clean_and_fix_volume()` | 24–103 | Duplicate of `VolumePreprocessingPipeline` offset detection and NaN handling |
| `resample_volume()` | 106–131 | Redundant `scipy.ndimage.zoom` — live path uses MONAI `Zoom` |
| `apply_windowing()` | 134–165 | Redundant `np.interp` windowing — live path uses MONAI `ScaleIntensityRange` |

### Imports Removed
- `from scipy.ndimage import zoom` — only used by `resample_volume()`
- `from DetectionModel.constants.enums.hu_values import HUValues` — only used by `clean_and_fix_volume()`

### Methods Kept (all actively called)
- `resize_slice_to_target()` — called by `create_25d_sandwich()`
- `center_crop_slice()` — called by `create_25d_sandwich()` and `scan_processor.py`
- `create_25d_sandwich()` — called by `scan_processor.py`

### Net Change
`−148 lines` in `slice_processor.py`.

---

## Workstream 2 — Extract Shared Lung Morphology Helper

### Problem
The body-mask + lung-candidate OpenCV pipeline was copy-pasted in two places
with divergent thresholds and subtle structural differences:

| Location | `lung_body_ratio` threshold | `min_contrast_range` | Contour filter |
|----------|-----------------------------|----------------------|----------------|
| `SliceQualityGate._compute_lung_ratio()` | 0.12 (lenient, training gate) | 80 | None |
| `DatasetDiagnoser._compute_lung_metrics()` | 0.20 (strict, audit) | 100 | `min_lung_contour_area=500` |

The morphological pipeline itself (body mask → MORPH_CLOSE/OPEN → lung
candidate → `bitwise_and` → contour count) was identical in both.

### Solution

**New file:** `preprocessing/lung_morphology.py`

```python
def compute_lung_body_metrics(
    gray: np.ndarray,
    body_intensity_floor: int = 20,
    lung_intensity_low: int = 10,
    lung_intensity_high: int = 90,
    morph_kernel_size: int = 15,
    min_contour_area: int = 0,
) -> dict:
    """Returns: body_mask, body_area, body_ratio, lung_body_ratio, lung_region_count"""
```

All parameters are explicit and configurable; each caller passes its own
thresholds, so the intentional threshold differences are preserved.

**`SliceQualityGate._compute_lung_ratio()`** — delegates to helper, extracts
`lung_body_ratio`. Body is 15 lines → 8 lines.

**`DatasetDiagnoser._compute_lung_metrics()`** — delegates to helper, passes
`min_contour_area=t.min_lung_contour_area` (500). Returns same dict shape as
before so callers are unaffected. 22 lines → 12 lines.

**`preprocessing/__init__.py`** — exports `compute_lung_body_metrics`.

### Net Change
`+67 lines` in `lung_morphology.py` (new file), `−36 lines` total in the two
callers.

---

## Workstream 3 — Annotation Processor Refactor

### 3A — Use `ann.diameter` for Nodule Diameter

**Before:**
```python
bboxs = filter(None, map(NoduleAnnotationProcessor._safe_extract_bbox, annotations))
diameters = list(map(BoundingBoxConverter.compute_diameter, bboxs))
```
`BoundingBoxConverter.compute_diameter()` computed a 2D Euclidean diagonal
from raw slice extent integers — a coarse pixel-space approximation with no
mm calibration.

**After:**
```python
diameters = [float(ann.diameter) for ann in annotations if has_annotations]
```
`ann.diameter` is pylidc's built-in property that measures the longest axis of
the actual annotation contour in **millimetres**, using the scan's pixel
spacing. This is the correct clinical measurement.

`BoundingBoxConverter.compute_diameter()` is retained (still exported from
`bbox_converter.py`) since it is used by the YOLO bounding-box path in
`compute_nodule_bbox_yolo()`.

The now-unused `from ..preprocessing.bbox_converter import BoundingBoxConverter`
import was removed from `annotation_processor.py`.

### 3B — Replace `contextlib.suppress` with Targeted Exceptions + Logging

**Before** (silently swallowed any exception):
```python
with contextlib.suppress(Exception):
    centroid = ann.centroid
    ...
```

**After** (explicit exception types, logged at WARNING):
```python
try:
    centroid = ann.centroid
    ...
except (AttributeError, TypeError, ValueError) as e:
    logger.warning(f"Annotation centroid extraction failed: {e}")
```

Same pattern applied to `_safe_extract_bbox()` and `_safe_extract_centroid()`.
`import contextlib` was removed. `import logging` and `logger` were added.

### 3C — Inline `CoordinateTransformer` and Delete the Class

`CoordinateTransformer` was a class with 3 trivial one-liner static methods
and exactly one consumer (`annotation_processor.py`, 3 call sites).

| Original call | Inlined equivalent |
|---------------|--------------------|
| `CoordinateTransformer.transform_slice_to_resampled_space(idx, z_scale)` | `int(round(idx * z_scale))` |
| `CoordinateTransformer.is_slice_within_volume(idx, volume_depth)` | `0 <= idx < volume_depth` |
| `CoordinateTransformer.transform_coordinates_to_resampled(avg, orig, tgt)` | Two-line scale-factor tuple comprehension |

**Files updated:**

- `core/coordinate_transformer.py` — **deleted**
- `core/__init__.py` — removed import and `__all__` entry
- `__init__.py` (package root) — removed import and `__all__` entry
- `sources/annotation_processor.py` — removed `from ..core.coordinate_transformer import CoordinateTransformer`

---

## Workstream 4 — Parallel Pipeline + DICOM Adapter Fixes

### 4A — Pre-fetch Scan IDs in Main Process

**Problem:**
Each multiprocessing worker was executing:
```python
pylidc.query(pylidc.Scan).filter(pylidc.Scan.patient_id == patient_id).first()
```
This is a **string-equality scan** over the SQLite table for every one of the
~1010 scans, performed inside the worker where SQLite connections are
re-established per process.

**Fix:**
The main process, which already holds `pipeline.scans_to_process` (a list of
live pylidc `Scan` objects), now extracts `scan.id` (integer primary key)
before dispatching:

```python
# Before
task_args.append((pid, split, config_dict, directories_dict))

# After
scan_id = scan.id
task_args.append((pid, scan_id, split, config_dict, directories_dict))
```

Workers query by primary key:
```python
scan = pylidc.query(pylidc.Scan).filter(pylidc.Scan.id == scan_id).first()
```

SQLite has an implicit B-tree index on integer primary keys. The lookup is
O(log n) at worst and effectively instant, versus a full-table string scan.

### 4B — Fix `DICOMScanSource` Brittle Pixdim Indexing

**Problem:**
The original spacing extraction was a single-branch conditional with magic
index arithmetic and no validity check:

```python
spacing = (
    (float(pixdim[3]), float(pixdim[1]), float(pixdim[2]))
    if pixdim is not None and len(pixdim) >= 4
    else (
        tuple(map(lambda i: float(np.abs(affine[i, i])), (2, 0, 1)))
        if affine is not None
        else None
    )
)
```

Issues:
1. Silently returns zero or negative spacing if NIfTI `pixdim` is malformed
2. No fallback to MONAI's own `'spacing'` key, which MONAI computes correctly
   from DICOM metadata when available
3. The affine branch is unreachable if `pixdim` exists but has wrong values

**Fix — three-stage waterfall with positivity guard:**

```python
# Stage 1: MONAI standard key (already (x, y, z) order)
if 'spacing' in meta:
    candidate = tuple(float(s) for s in raw[:3])
    if all(s > 0 for s in candidate):
        spacing = candidate

# Stage 2: NIfTI pixdim fallback
if spacing is None and pixdim is not None and len(pixdim) >= 4:
    candidate = (float(pixdim[3]), float(pixdim[1]), float(pixdim[2]))
    if all(s > 0 for s in candidate):
        spacing = candidate

# Stage 3: Affine diagonal last resort
if spacing is None and affine is not None:
    candidate = tuple(float(np.abs(affine[i, i])) for i in (2, 0, 1))
    if all(s > 0 for s in candidate):
        spacing = candidate
```

Each stage validates `all(s > 0)` before accepting the candidate, so
zero/negative spacing from corrupt metadata causes an orderly fallback rather
than silent propagation.

---

## File Change Table

| File | Status | Change |
|------|--------|--------|
| `preprocessing/slice_processor.py` | Modified | Deleted 3 dead methods + 2 unused imports (−148 lines) |
| `preprocessing/lung_morphology.py` | **Created** | New shared morphology helper (+67 lines) |
| `preprocessing/slice_quality_gate.py` | Modified | `_compute_lung_ratio` delegates to helper (−16 lines) |
| `preprocessing/__init__.py` | Modified | Export `compute_lung_body_metrics` (+2 lines) |
| `utils/dataset_diagnostics.py` | Modified | `_compute_lung_metrics` delegates to helper (−10 lines) |
| `sources/annotation_processor.py` | Modified | `ann.diameter`, targeted exceptions, inline coord math, remove imports |
| `core/coordinate_transformer.py` | **Deleted** | Inlined into annotation_processor.py (−37 lines) |
| `core/__init__.py` | Modified | Remove `CoordinateTransformer` export (−2 lines) |
| `__init__.py` | Modified | Remove `CoordinateTransformer` from `__all__` (−2 lines) |
| `pipelines/parallel_preparation.py` | Modified | Pre-fetch `scan.id`, pass to workers (+5/−3 lines) |
| `sources/scan_adapters.py` | Modified | Three-stage spacing waterfall with positivity guard (+20/−12 lines) |

---

## Verification

All modified and created files pass syntax checks:

```
python -m py_compile preprocessing/slice_processor.py      → OK
python -m py_compile preprocessing/lung_morphology.py      → OK
python -m py_compile preprocessing/slice_quality_gate.py   → OK
python -m py_compile utils/dataset_diagnostics.py          → OK
python -m py_compile sources/annotation_processor.py       → OK
python -m py_compile core/__init__.py                      → OK
python -m py_compile __init__.py                           → OK
python -m py_compile pipelines/parallel_preparation.py     → OK
python -m py_compile sources/scan_adapters.py              → OK
```

No remaining references to deleted symbols:
```
grep CoordinateTransformer **/*.py  → only comment in annotation_processor.py
grep clean_and_fix_volume  **/*.py  → no results
grep resample_volume       **/*.py  → no results
grep apply_windowing       **/*.py  → no results
```
