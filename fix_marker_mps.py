"""
fix_marker_mps.py — Enable marker-pdf / surya-ocr MPS acceleration on macOS.

The forked surya-ocr allows TableRecEncoderDecoderModel to run on MPS with
float32 (instead of being blocked entirely).  This module sets the environment
variables needed for safe MPS operation.

Call apply_fix() **before** importing any marker or surya module.
"""

import os
import sys


def apply_fix():
    """Set environment variables for MPS-accelerated inference.

    Must be called before any marker.* or surya.* module is imported,
    because both libraries read TORCH_DEVICE via pydantic-settings
    singletons at module-import time.

    Raises
    ------
    RuntimeError
        If marker or surya modules have already been imported.
    """
    # Guard: no marker/surya modules may be loaded yet
    already_loaded = [
        name for name in sys.modules
        if name.startswith("marker.") or name.startswith("surya.")
    ]
    if already_loaded:
        raise RuntimeError(
            "apply_fix() must be called before importing marker/surya. "
            f"Already loaded: {already_loaded}"
        )

    # Duplicate-libomp workaround common on macOS with conda/brew torch
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Fallback: any unsupported MPS op silently runs on CPU instead of crashing
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def verify_fix():
    """Import both settings singletons and confirm they resolved to MPS.

    Returns
    -------
    dict
        Keys: marker_device, surya_device, surya_dtype, fix_verified
    """
    import torch
    from marker.settings import settings as marker_settings
    from surya.settings import settings as surya_settings

    result = {
        "marker_device": marker_settings.TORCH_DEVICE_MODEL,
        "surya_device": surya_settings.TORCH_DEVICE_MODEL,
        "surya_dtype": str(surya_settings.MODEL_DTYPE),
        "fix_verified": (
            marker_settings.TORCH_DEVICE_MODEL == "mps"
            and surya_settings.TORCH_DEVICE_MODEL == "mps"
        ),
    }
    return result


def get_status_report():
    """Return a human-readable multi-line status string."""
    import torch

    lines = [
        "=== marker-pdf MPS Acceleration Status ===",
        f"TORCH_DEVICE env var : {os.environ.get('TORCH_DEVICE', '<not set>')}",
        f"KMP_DUPLICATE_LIB_OK : {os.environ.get('KMP_DUPLICATE_LIB_OK', '<not set>')}",
        f"MPS_FALLBACK         : {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '<not set>')}",
        f"torch.backends.mps.is_available(): {torch.backends.mps.is_available()}",
    ]

    try:
        info = verify_fix()
        lines += [
            f"marker  TORCH_DEVICE_MODEL: {info['marker_device']}",
            f"surya   TORCH_DEVICE_MODEL: {info['surya_device']}",
            f"surya   MODEL_DTYPE       : {info['surya_dtype']}",
            f"MPS verified              : {info['fix_verified']}",
        ]
    except Exception as exc:
        lines.append(f"Could not verify settings: {exc}")

    return "\n".join(lines)
