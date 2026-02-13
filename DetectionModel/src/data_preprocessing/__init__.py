"""Data Preprocessing Package - Organized structure for CT scan data preparation.

Package structure:
    core/         - Core protocols and configurations
    sources/      - Data source adapters (PyLIDC, DICOM)
    preprocessing/ - Volume and slice preprocessing
    pipelines/    - Pipeline orchestration
    io/           - File I/O operations
    utils/        - Utilities (splitting, diagnostics)
    legacy/       - Deprecated code (for reference)
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
)

# Source adapters
from .sources import PyLIDCScanSource, DICOMScanSource

# Preprocessing
from .preprocessing import (
    VolumePreprocessingPipeline,
    SliceQualityGate,
    SliceQualityConfig,
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
    # Sources
    'PyLIDCScanSource',
    'DICOMScanSource',
    # Preprocessing
    'VolumePreprocessingPipeline',
    'SliceQualityGate',
    'SliceQualityConfig',
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
