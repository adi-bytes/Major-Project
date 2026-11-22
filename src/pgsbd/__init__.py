"""PG-SBD: Physics-Guided Surrogate Modeling and Inverse Design Framework."""

from __future__ import annotations

import ctypes
import os
import sys

# Prevent OpenMP / MKL thread contention and deadlocks on Windows CPU inference
if sys.platform == "win32":
    try:
        ctypes.cdll.msvcrt._putenv(b"OMP_NUM_THREADS=1")
        ctypes.cdll.msvcrt._putenv(b"MKL_NUM_THREADS=1")
        ctypes.cdll.msvcrt._putenv(b"OPENBLAS_NUM_THREADS=1")
        ctypes.cdll.msvcrt._putenv(b"VECLIB_MAXIMUM_THREADS=1")
        ctypes.cdll.msvcrt._putenv(b"NUMEXPR_NUM_THREADS=1")
        ctypes.cdll.msvcrt._putenv(b"KMP_DUPLICATE_LIB_OK=TRUE")
    except (AttributeError, OSError):
        pass

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch

    torch.set_num_threads(1)
except ImportError:
    pass

__version__ = "0.1.0"
