"""PyLIDC configuration utilities."""

import os
import logging
import configparser
import platform
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def get_pylidc_config_path() -> Path:
    """Get pylidc config file path for current OS."""
    system = platform.system()
    
    config_filename = (
        "pylidc.conf"  
        if system == 'Windows'
        else ".pylidcrc"  
    )
    
    home_config = Path.home() / config_filename
    
    return home_config


def normalize_dicom_path(dicom_path: str) -> str:
    """Normalize DICOM path to absolute cross-platform format."""
    path_obj = Path(dicom_path)
    
    absolute_path = path_obj.resolve()
    
    normalized = str(absolute_path)
    
    # Log the normalization for debugging
    logger.debug(f"Path normalization: '{dicom_path}' -> '{normalized}'")
    
    return normalized


def validate_lidc_directory(dicom_path: str) -> Tuple[bool, str]:
    """Validate LIDC-IDRI directory structure."""
    path_obj = Path(dicom_path)
    
    exists = path_obj.exists()
    
    is_dir = path_obj.is_dir() if exists else False
    
    patient_dirs = (
        list(path_obj.glob("LIDC-IDRI-*"))
        if is_dir
        else []
    )
    
    has_patients = len(patient_dirs) > 0
    
    message = (
        f"Valid LIDC-IDRI directory with {len(patient_dirs)} patient folders"
        if exists and is_dir and has_patients
        else f"Path does not exist: {dicom_path}"
        if not exists
        else f"Path is not a directory: {dicom_path}"
        if not is_dir
        else f"No LIDC-IDRI-* patient folders found in: {dicom_path}"
    )
    
    is_valid = exists and is_dir and has_patients
    
    return (is_valid, message)


def configure_pylidc(dicom_path: str) -> bool:
    """Configure pylidc with custom DICOM directory (cross-platform)."""
    normalized_path = normalize_dicom_path(dicom_path)
    
    
    is_valid, validation_message = validate_lidc_directory(normalized_path)
    logger.info(f"LIDC directory validation: {validation_message}")
    
    
    if not is_valid:
        logger.warning(
            f"Directory validation failed - proceeding anyway. "
            f"Ensure path contains LIDC-IDRI-* patient folders."
        )
    
    
    config_path = get_pylidc_config_path()
    local_config_path = Path.cwd() / "pylidc.conf"
    
    
    config = configparser.ConfigParser()
    config['dicom'] = {'path': normalized_path}
    
    
    success = False
    used_path = None
    
    try:
        with open(config_path, 'w') as f:
            config.write(f)
        logger.info(f"PyLIDC config written to: {config_path}")
        success = True
        used_path = config_path
    except (PermissionError, OSError) as e:
        logger.warning(f"Could not write to {config_path}: {e}")
    
    if not success:
        try:
            with open(local_config_path, 'w') as f:
                config.write(f)
            logger.info(f"PyLIDC config written to fallback: {local_config_path}")
            success = True
            used_path = local_config_path
        except (PermissionError, OSError) as e:
            logger.warning(f"Could not write to {local_config_path}: {e}")
    
    os.environ['PYLIDC_DICOM_PATH'] = normalized_path
    
    if success:
        logger.info(
            f"PyLIDC configured successfully\n"
            f"  Config file: {used_path}\n"
            f"  DICOM path: {normalized_path}\n"
            f"  Platform: {platform.system()}"
        )
    else:
        logger.error(
            f"Failed to configure pylidc. "
            f"Please manually create ~/.pylidcrc with:\n"
            f"[dicom]\n"
            f"path = {normalized_path}"
        )
    
    return success


def import_pylidc():
    """Import pylidc module."""
    import pylidc as pl
    return pl
