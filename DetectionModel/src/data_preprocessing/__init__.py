"""Data Preprocessing Package - Organized structure for CT scan data preparation.

Package structure:
    core/           - Data contracts (VolumeData, NoduleData, etc.), coordinate transforms, PyLIDC setup
    sources/        - Data source adapters (PyLIDC, DICOM) and annotation processing
    preprocessing/  - Volume processing, slice processing (2.5D), quality gate, bbox conversion
    pipelines/      - Pipeline orchestration (serial, parallel, inference)
    io/             - Atomic file I/O and dataset metadata writers
    utils/          - Patient splitting and dataset diagnostics
    legacy/         - Deprecated code (for reference)
"""

from .config import DataPrepConfig

# Core exports
from .core import (
    ScanSource,
    VolumeData,
    NoduleData,
    YOLODetection,
    ProcessedSlice,
    NoduleCropResult,
    configure_pylidc,
    import_pylidc,
    CoordinateTransformer,
)

# Source adapters + annotation processing
from .sources import PyLIDCScanSource, DICOMScanSource, NoduleAnnotationProcessor

# Preprocessing
from .preprocessing import (
    VolumePreprocessingPipeline,
    SliceQualityGate,
    SliceQualityConfig,
    BoundingBoxConverter,
    SlicePreprocessor,
)

# Pipelines
from .pipelines import (
    CTScanProcessor,
    prepare_dataset,
    prepare_dataset_from_config_dict,
    prepare_dataset_parallel,
    InferencePipeline,
)

# I/O
from .io import (
    save_image,
    save_label,
    atomic_save_image_and_label,
    AtomicSaveResult,
    save_metadata_csv,
    save_config_json,
    save_yolo_yaml,
    log_summary_statistics,
)

# Utils
from .utils import (
    split_patients_by_id,
    get_patient_split,
    DatasetDiagnoser,
)

__all__ = [
    # Config
    'DataPrepConfig',
    # Core
    'ScanSource',
    'VolumeData',
    'NoduleData',
    'YOLODetection',
    'ProcessedSlice',
    'NoduleCropResult',
    'configure_pylidc',
    'import_pylidc',
    'CoordinateTransformer',
    # Sources
    'PyLIDCScanSource',
    'DICOMScanSource',
    'NoduleAnnotationProcessor',
    # Preprocessing
    'VolumePreprocessingPipeline',
    'SliceQualityGate',
    'SliceQualityConfig',
    'BoundingBoxConverter',
    'SlicePreprocessor',
    # Pipelines
    'CTScanProcessor',
    'prepare_dataset',
    'prepare_dataset_from_config_dict',
    'prepare_dataset_parallel',
    'InferencePipeline',
    # I/O
    'save_image',
    'save_label',
    'atomic_save_image_and_label',
    'AtomicSaveResult',
    'save_metadata_csv',
    'save_config_json',
    'save_yolo_yaml',
    'log_summary_statistics',
    # Utils
    'split_patients_by_id',
    'get_patient_split',
    'DatasetDiagnoser',
]
