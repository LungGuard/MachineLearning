# Package Review: `data_preprocessing`

> Reviewed: 2026-04-16

---

## Package Structure

| Subpackage | Purpose |
|---|---|
| `core/` | Data contracts (frozen dataclasses + ScanSource protocol) and PyLIDC setup |
| `sources/` | Source adapters (PyLIDC, raw DICOM) + annotation feature extraction |
| `preprocessing/` | Per-slice and per-volume processing (resample, window, CLAHE, bbox) |
| `pipelines/` | Orchestration: serial batch, parallel batch, inference |
| `io/` | Atomic image/label save + CSV/JSON/YAML output |
| `utils/` | Patient splitting and dataset quality diagnostics |

### Module Map

```
data_preprocessing/
├── __init__.py
├── __main__.py
├── config.py                           DataPrepConfig dataclass
│
├── core/
│   ├── scan_protocols.py               VolumeData, NoduleData, YOLODetection, ScanSource protocol
│   └── pylidc_config.py                Cross-platform PyLIDC config setup
│
├── sources/
│   ├── scan_adapters.py                PyLIDCScanSource, DICOMScanSource
│   └── annotation_processor.py         NoduleAnnotationProcessor
│
├── preprocessing/
│   ├── volume_processor.py             NaN clean → offset detect → resample → lung window
│   ├── slice_processor.py              2.5D sandwich, aspect-preserving resize, center crop
│   ├── slice_quality_gate.py           CLAHE enhancement + pre-save validation
│   ├── bbox_converter.py               YOLO bbox conversion + resize/crop adjustment
│   └── lung_morphology.py              Shared morphological body/lung segmentation
│
├── pipelines/
│   ├── scan_processor.py               CTScanProcessor — main data-prep orchestrator
│   ├── batch_preparation.py            Serial batch pipeline with LiveDashboard
│   ├── parallel_preparation.py         Multiprocessing pool pipeline
│   └── inference_processor.py          In-memory inference pipeline (no disk I/O)
│
├── io/
│   ├── atomic_io.py                    Image + label save with rollback on failure
│   └── dataset_writer.py               CSV metadata, config JSON, YOLO YAML
│
└── utils/
    ├── patient_splitter.py             Train/val/test split
    └── dataset_diagnostics.py          DatasetDiagnoser — quality analysis + export
```

### Data Flow

**Data-prep pipeline:**
```
DataPreparationPipeline
  → CTScanProcessor.process_scan
    → VolumePreprocessingPipeline       (raw DICOM → float32 volume, resampled + windowed)
    → NoduleAnnotationProcessor         (pylidc annotations → features, centroid, slice indices)
    → For each nodule:
        → InferencePipeline.prepare_slice_image   (2.5D sandwich + CLAHE + quality gate)
        → BoundingBoxConverter                    (centroid + diameter → YOLO bbox)
        → atomic_save_image_and_label             (image + label to disk, atomic)
  → dataset_writer                      (CSV, config JSON, YOLO YAML)
```

**Inference pipeline:**
```
DICOMScanSource
  → VolumePreprocessingPipeline
  → InferencePipeline.prepare_slices_for_yolo    (batch 2.5D slices)
  → [External YOLO model]
  → InferencePipeline.extract_nodule_crops       (crop detections for CNN classifier)
```

---

## Actual Bugs

### Bug 1 — `DEFAULT_FEATURES` mutated across calls
**File:** `sources/annotation_processor.py:57-58`

```python
# CURRENT — default_features is a reference, not a copy
default_features = DEFAULT_FEATURES
default_features[Features.DIAMETER_MM] = fallback_diameter  # mutates the module constant!
```

Every call to `extract_nodule_features` with a non-default `fallback_diameter` permanently
changes the module-level dict. The next call gets the previous call's diameter as its default.

**Fix:**
```python
default_features = {**DEFAULT_FEATURES, Features.DIAMETER_MM: fallback_diameter}
```

---

### Bug 2 — Diameter comprehension uses filter instead of conditional
**File:** `sources/annotation_processor.py:76`

```python
# All other features use the correct conditional pattern:
malignancy_scores = [ann.malignancy for ann in annotations] if has_annotations else []

# But diameters uses a per-element filter instead:
diameters = [float(ann.diameter) for ann in annotations if has_annotations]
```

The `if has_annotations` here is a loop filter on a constant bool, not a conditional expression.
It happens to produce the same result (since `has_annotations = len(annotations) > 0`), but
it's misleading and inconsistent.

**Fix:**
```python
diameters = [float(ann.diameter) for ann in annotations] if has_annotations else []
```

---

## Code Style Anti-patterns

### Anti-pattern 1 — Ternary expressions used for side effects

Ternary (`x if cond else y`) is for expressions that produce a value. Using `expr() if cond else None`
to execute side effects conditionally is confusing and non-idiomatic. This pattern appears in at least
5 places:

**`volume_processor.py:110-112`**
```python
# current
volume = np.nan_to_num(volume, nan=float(HUValues.AIR_HU)) if nan_count > 0 else volume
self.logger.warning(f"[{patient_id}] Found {nan_count} NaN values...") if nan_count > 0 else None
# fix
if nan_count > 0:
    volume = np.nan_to_num(volume, nan=float(HUValues.AIR_HU))
    self.logger.warning(f"[{patient_id}] Found {nan_count} NaN values...")
```

**`volume_processor.py:134`** — no-op on the else branch:
```python
# current — the else branch assigns volume[mask] to itself (no-op)
volume[padding_mask] = float(HUValues.AIR_HU) if padding_count > 0 else volume[padding_mask]
# fix
if padding_count > 0:
    volume[padding_mask] = float(HUValues.AIR_HU)
```

**`scan_processor.py:148`**
```python
# current
accepted.append(metadata) if metadata is not None else None
# fix
if metadata is not None:
    accepted.append(metadata)
```

**`scan_processor.py:277-278`**
```python
# current
img_path.unlink() if img_path.exists() else None
lbl_path.unlink() if lbl_path.exists() else None
# fix
if img_path.exists():
    img_path.unlink()
if lbl_path.exists():
    lbl_path.unlink()
```

**`slice_quality_gate.py:61`**
```python
# current
logger.debug(f"[{patient_id}] {context} rejected: {reason}") if not passed else None
# fix
if not passed:
    logger.debug(f"[{patient_id}] {context} rejected: {reason}")
```

---

### Anti-pattern 2 — Nested ternary in `_select_candidate_slices`
**File:** `pipelines/scan_processor.py:283-300`

Three levels of nesting — takes real effort to parse:
```python
return (
    [] if total == 0 else (
        slice_indices if total <= num_candidates else (
            [slice_indices[int(i * (total - 1) / (num_candidates - 1))]
             for i in range(num_candidates)]
            if num_candidates > 1
            else [slice_indices[total // 2]]
        )
    )
)
```

Collapses to 5 readable lines with `np.linspace`:
```python
if total == 0:
    return []
if total <= num_candidates:
    return slice_indices
if num_candidates == 1:
    return [slice_indices[total // 2]]
return [slice_indices[i] for i in np.linspace(0, total - 1, num_candidates, dtype=int)]
```

---

### Anti-pattern 3 — `map(lambda ...)` instead of list comprehensions

**`slice_quality_gate.py:113-116`** — CLAHE per channel:
```python
# current
np.stack(
    list(map(lambda ch: clahe.apply(self._to_uint8(image[:, :, ch])), range(image.shape[2]))),
    axis=2
)
# fix
np.stack(
    [clahe.apply(self._to_uint8(image[:, :, ch])) for ch in range(image.shape[2])],
    axis=2
)
```

**`annotation_processor.py:131-136`** — slice indices:
```python
# current
all_valid_indices = map(
    lambda ann: NoduleAnnotationProcessor._get_valid_slice_indices_from_annotation(...),
    annotations
)
unique_indices = {idx for indices in all_valid_indices for idx in indices}

# fix — itertools.chain.from_iterable is the stdlib idiom for flattening
from itertools import chain
unique_indices = set(chain.from_iterable(
    NoduleAnnotationProcessor._get_valid_slice_indices_from_annotation(ann, z_scale, volume_depth)
    for ann in annotations
))
```

---

### Anti-pattern 4 — `list(filter(lambda ...))` instead of comprehension
**File:** `preprocessing/slice_quality_gate.py:90`

```python
# current
failures = list(filter(lambda chk: chk[0], checks))
# fix
failures = [chk for chk in checks if chk[0]]
```

---

## Manual Logic Replaceable With Libraries

### Improvement 1 — Centroid averaging via `np.mean`
**File:** `sources/annotation_processor.py:164-167`

```python
# current — manual sum and divide across each axis
avg_centroid = tuple(
    sum(c[i] for c in centroids) / len(centroids)
    for i in range(len(CENTROID))
)

# fix
avg_centroid = tuple(np.mean(centroids, axis=0))
```

---

### Improvement 2 — Unnecessary RGB branching in resize/crop
**File:** `preprocessing/slice_processor.py`

`cv2.resize` already handles `(H, W, C)` arrays natively. The only place the RGB check matters
is constructing the output canvas. Both branches can be unified using numpy shape inference:

```python
# current — duplicated for 2D and 3D
if is_rgb:
    output = np.full((target_h, target_w, slice_2d.shape[2]), pad_value, dtype=slice_2d.dtype)
else:
    output = np.full((target_h, target_w), pad_value, dtype=slice_2d.dtype)
# fix — shape[2:] is () for 2D, (C,) for 3D
out_shape = (target_h, target_w) + slice_2d.shape[2:]
output = np.full(out_shape, pad_value, dtype=slice_2d.dtype)
```

The copy assignments (`output[y0:y1, x0:x1] = resized[...]`) are identical for both dimensionalities,
so the `is_rgb` branches there can also be removed, leaving a single unified code path.

---

### Improvement 3 — Repetitive feature extraction pattern
**File:** `sources/annotation_processor.py:67-76`

Ten lines doing the exact same thing for each radiologist annotation attribute:
```python
malignancy_scores  = [ann.malignancy  for ann in annotations] if has_annotations else []
spiculation_scores = [ann.spiculation for ann in annotations] if has_annotations else []
lobulation_scores  = [ann.lobulation  for ann in annotations] if has_annotations else []
subtlety_scores    = [ann.subtlety    for ann in annotations] if has_annotations else []
sphericity_scores  = [ann.sphericity  for ann in annotations] if has_annotations else []
margin_scores      = [ann.margin      for ann in annotations] if has_annotations else []
texture_scores     = [ann.texture     for ann in annotations] if has_annotations else []
calcification_scores      = [ann.calcification      for ann in annotations] if has_annotations else []
internal_structure_scores = [ann.internalStructure  for ann in annotations] if has_annotations else []
diameters          = [float(ann.diameter) for ann in annotations] if has_annotations else []
```

Map `Features` to attribute names and collapse to a dict comprehension:
```python
_FEATURE_ATTR: dict[Features, str] = {
    Features.MALIGNANCY:         "malignancy",
    Features.SPICULATION:        "spiculation",
    Features.LOBULATION:         "lobulation",
    Features.SUBTLETY:           "subtlety",
    Features.SPHERICITY:         "sphericity",
    Features.MARGIN:             "margin",
    Features.TEXTURE:            "texture",
    Features.CALCIFICATION:      "calcification",
    Features.INTERNAL_STRUCTURE: "internalStructure",
}

scores = (
    {feat: [getattr(ann, attr) for ann in annotations] for feat, attr in _FEATURE_ATTR.items()}
    if has_annotations
    else {feat: [] for feat in _FEATURE_ATTR}
)
diameters = [float(ann.diameter) for ann in annotations] if has_annotations else []
```

Then the return dict becomes a single comprehension over `scores`.

---

### Improvement 4 — MONAI `Spacing` vs. manual zoom factor calculation
**File:** `preprocessing/volume_processor.py:77-86`

Currently:
```python
zoom_factors = [orig / target for orig, target in zip(original_spacing, self.config.target_spacing)]
zoomer = Zoom(zoom=zoom_factors, mode="bilinear", padding_mode="border", keep_size=False)
return zoomer(volume_tensor)
```

MONAI's `Spacing` transform does exactly this with built-in metadata/affine handling.
Not a bug as-is, but if you ever need metatensor support or multi-channel volumes, `Spacing`
is the correct tool:
```python
from monai.transforms import Spacing
spacer = Spacing(pixdim=self.config.target_spacing, mode="bilinear")
```

---

## Dead Code

### Dead code 1 — `BoundingBoxConverter.compute_diameter`
**File:** `preprocessing/bbox_converter.py:17-23`

```python
@staticmethod
def compute_diameter(bbox) -> float:
    """Compute nodule diameter from bounding box."""
    x_extent = bbox[2].stop - bbox[2].start
    y_extent = bbox[1].stop - bbox[1].start
    diameter = np.sqrt(x_extent**2 + y_extent**2)
    return float(diameter)
```

This method is never called anywhere in the package. The actual YOLO conversion uses
`ann.diameter` from pylidc (via `Features.DIAMETER_MM` in `NoduleData`). Can be deleted.

---

## Minor Notes

### Note 1 — Middle channel selection is a magic index
**File:** `preprocessing/slice_quality_gate.py:72`

```python
check_slice = image[:, :, image.shape[2] // 2] if len(image.shape) == 3 else image
```

`// 2` selects the green channel (index 1 of R=0, G=1, B=2), which is the actual current slice
in the 2.5D sandwich. Correct, but fragile — if channel ordering ever changes, this silently
uses the wrong slice. A named constant (`SANDWICH_CENTER_CHANNEL = 1`) or comment would
make the intent clear.

---

### Note 2 — `save_label` uses file existence as success signal
**File:** `io/atomic_io.py:59`

```python
with open(label_path, 'w') as f:
    f.write(label_content)
return label_path.exists()
```

`path.exists()` after a write is a weak success check — the file could exist from a previous run
even if this write silently failed. The `open + write` will raise on failure anyway, so the
`exists()` check is both redundant and misleading. The `try/except` around it already handles
errors. Just `return True` after a successful write is cleaner.

---

### Note 3 — `_process_nodule` early exit guard can be simplified
**File:** `pipelines/scan_processor.py:140-148`

```python
for slice_idx in candidates:
    remaining = required - len(accepted)
    metadata = (
        self._process_single_slice(...)
        if remaining > 0
        else None
    )
    accepted.append(metadata) if metadata is not None else None
```

Cleaner as:
```python
for slice_idx in candidates:
    if len(accepted) >= required:
        break
    metadata = self._process_single_slice(...)
    if metadata is not None:
        accepted.append(metadata)
```
The `break` is more honest about intent than computing `remaining` and faking a no-op.

---

## Priority Summary

| Priority | Item | File |
|---|---|---|
| **High** | Bug: `DEFAULT_FEATURES` mutation | `sources/annotation_processor.py:57` |
| **High** | Anti-pattern: ternary for side effects (5 locations) | Multiple |
| **High** | Anti-pattern: nested ternary in `_select_candidate_slices` | `pipelines/scan_processor.py:283` |
| **Medium** | Bug: diameter comprehension inconsistency | `sources/annotation_processor.py:76` |
| **Medium** | Improvement: collapse repetitive feature extraction | `sources/annotation_processor.py:67` |
| **Medium** | Improvement: unify RGB branching in resize/crop | `preprocessing/slice_processor.py` |
| **Medium** | Improvement: `np.mean` for centroid averaging | `sources/annotation_processor.py:164` |
| **Low** | Anti-pattern: `map(lambda)` vs comprehensions | `slice_quality_gate.py`, `annotation_processor.py` |
| **Low** | Dead code: `compute_diameter` | `preprocessing/bbox_converter.py:17` |
| **Low** | Note: magic channel index | `preprocessing/slice_quality_gate.py:72` |
| **Low** | Note: `exists()` as write success signal | `io/atomic_io.py:59` |
| **Low** | Note: `_process_nodule` loop guard | `pipelines/scan_processor.py:140` |
