"""
PyLIDC Configuration Module
============================
Handles pylidc library configuration for custom DICOM paths.

This module provides cross-platform compatible configuration
for the pylidc library to work with custom LIDC-IDRI data locations.

"""

import os
import logging
import configparser
import platform
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def get_pylidc_config_path() -> Path:
    """
    Get the pylidc configuration file path for the current operating system.
    
    Returns
    -------
    Path
        Path to the pylidc configuration file
    
    Notes
    -----
    Configuration file locations:
        - Windows: C:\\Users\\<username>\\pylidc.conf
        - macOS/Linux: /Users/<username>/.pylidcrc
    """
    system = platform.system()
    
    # PyLIDC uses different config file names per OS
    config_filename = (
        "pylidc.conf"   # Windows uses no dot prefix
        if system == 'Windows'
        else ".pylidcrc"  # Unix/macOS uses dot prefix
    )
    
    home_config = Path.home() / config_filename
    
    return home_config


def normalize_dicom_path(dicom_path: str) -> str:
    """
    Normalize DICOM path for cross-platform compatibility.
    
    Parameters
    ----------
    dicom_path : str
        User-provided path (may use Windows backslashes or Unix forward slashes)
    
    Returns
    -------
    str
        Normalized absolute path
    
    Notes
    -----
    Handles:
        - Windows paths: E:\\folder\\subfolder or E:/folder/subfolder
        - Unix paths: /home/user/data
        - Relative paths: ./data or ../data
        - Paths with spaces
    """
    path_obj = Path(dicom_path)
    
    # Resolve to absolute path
    absolute_path = path_obj.resolve()
    
    # Convert to string with OS-appropriate separators
    normalized = str(absolute_path)
    
    # Log the normalization for debugging
    logger.debug(f"Path normalization: '{dicom_path}' -> '{normalized}'")
    
    return normalized


def validate_lidc_directory(dicom_path: str) -> Tuple[bool, str]:
    """
    Validate that the provided path appears to be a valid LIDC-IDRI directory.
    
    Parameters
    ----------
    dicom_path : str
        Path to validate
    
    Returns
    -------
    Tuple[bool, str]
        (is_valid, message) - validation result and descriptive message
    
    Notes
    -----
    Expected structure:
        LIDC-IDRI/
        ├── LIDC-IDRI-0001/
        ├── LIDC-IDRI-0002/
        └── ...
    """
    path_obj = Path(dicom_path)
    
    # Check if path exists
    exists = path_obj.exists()
    
    # Check if it's a directory
    is_dir = path_obj.is_dir() if exists else False
    
    # Look for patient subdirectories (LIDC-IDRI-XXXX pattern)
    patient_dirs = (
        list(path_obj.glob("LIDC-IDRI-*"))
        if is_dir
        else []
    )
    
    has_patients = len(patient_dirs) > 0
    
    # Construct validation result
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
    """
    Dynamically configure pylidc to use a custom DICOM directory.
    
    Cross-platform compatible (Windows, macOS, Linux).
    
    Parameters
    ----------
    dicom_path : str
        Path to the LIDC-IDRI DICOM data directory
        Example Windows: E:\\FinalsProject\\Datasets\\...\\LIDC-IDRI
        Example Unix: /data/LIDC-IDRI
    
    Returns
    -------
    bool
        True if configuration was successful
    
    Notes
    -----
    The pylidc configuration file format:
        [dicom]
        path = /path/to/LIDC-IDRI
    
    PyLIDC searches for configuration in this order:
        1. ~/.pylidcrc (home directory)
        2. pylidc.conf (current working directory)
    
    On Windows, paths with backslashes are automatically handled.
    """
    # Normalize the path for the current OS
    normalized_path = normalize_dicom_path(dicom_path)
    
    # Validate the directory structure
    is_valid, validation_message = validate_lidc_directory(normalized_path)
    logger.info(f"LIDC directory validation: {validation_message}")
    
    # Warn but don't fail if validation fails (user might know what they're doing)
    if not is_valid:
        logger.warning(
            f"Directory validation failed - proceeding anyway. "
            f"Ensure path contains LIDC-IDRI-* patient folders."
        )
    
    # Get configuration file path
    config_path = get_pylidc_config_path()
    local_config_path = Path.cwd() / "pylidc.conf"
    
    # Create configuration content
    config = configparser.ConfigParser()
    config['dicom'] = {'path': normalized_path}
    
    # Attempt to write configuration
    success = False
    used_path = None
    
    # Try primary location first (home directory)
    try:
        with open(config_path, 'w') as f:
            config.write(f)
        logger.info(f"PyLIDC config written to: {config_path}")
        success = True
        used_path = config_path
    except (PermissionError, OSError) as e:
        logger.warning(f"Could not write to {config_path}: {e}")
    
    # Fallback to local directory if home directory failed
    if not success:
        try:
            with open(local_config_path, 'w') as f:
                config.write(f)
            logger.info(f"PyLIDC config written to fallback: {local_config_path}")
            success = True
            used_path = local_config_path
        except (PermissionError, OSError) as e:
            logger.warning(f"Could not write to {local_config_path}: {e}")
    
    # Also set environment variable as additional fallback
    os.environ['PYLIDC_DICOM_PATH'] = normalized_path
    
    # Log final status
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
    """
    Import pylidc after configuration.
    
    Returns
    -------
    module
        The pylidc module
    
    Notes
    -----
    This should be called after configure_pylidc() to ensure
    the library can find the DICOM data.
    """
    import pylidc as pl
    return pl
