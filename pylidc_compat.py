"""
Python 3.12+ / NumPy 2.0+ Compatibility Patches for PyLIDC.



Two issues this solves:
  1. configparser.SafeConfigParser was removed in Python 3.12
     → pylidc internally still uses it
  2. np.int / np.float / etc. aliases were removed in NumPy 2.0
     → pylidc and some DICOM libraries still use them
"""

import configparser
import numpy as np

_PATCHES_APPLIED = False


def apply_patches() -> None:
    """
    Apply all compatibility monkey-patches.
    Safe to call multiple times — patches are applied only once.
    """
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    # ── configparser: SafeConfigParser removed in Python 3.12 ──
    if not hasattr(configparser, "SafeConfigParser"):
        configparser.SafeConfigParser = configparser.ConfigParser

    # ── numpy: deprecated aliases removed in NumPy 2.0 ──
    _numpy_aliases = {
        "int": np.int64,
        "float": np.float64,
        "bool": np.bool_,
        "object": np.object_,
        "str": np.str_,
        "complex": np.complex128,
    }
    for alias, replacement in _numpy_aliases.items():
        if not hasattr(np, alias) or not callable(getattr(np, alias, None)):
            setattr(np, alias, replacement)

    _PATCHES_APPLIED = True


# Auto-apply on import so that any module importing this file
# gets the patches for free.
apply_patches()