"""Schema-driven dataset loader and preprocessor for PG-SBD."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class DatasetSchemaError(ValueError):
    """Raised when the raw or processed dataset deviates from the expected schema."""


def load_dataset(
    csv_path: str | Path = "data/raw/new_dataset_GaO_WF+T.csv",
    v_start: float = 0.0,
    v_stop: float = 4.0,
    v_points: int = 52,
    noise_floor: float = 1.0e-14,
    negative_current_tolerance: float = 1.0e-10,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the raw TCAD dataset.

    Args:
        csv_path: Path to the raw CSV file.
        v_start: Starting voltage.
        v_stop: Ending voltage.
        v_points: Number of voltage points.
        noise_floor: Clamping floor for tiny numerical noise near 0 V.
        negative_current_tolerance: Threshold below which negative currents raise an error.

    Returns:
        tuple of:
            - df_cleaned: DataFrame containing cleaned currents and metadata.
            - v_grid: 1D array of voltages corresponding to the 52 points.
            - X: 2D array of parameters [WF, T] (or full parameter matrix).
            - Y: 2D array of currents (shape [N, 52]).
    """
    path = Path(csv_path)
    if not path.is_absolute():
        root = Path(__file__).resolve().parents[3]
        path = root / csv_path

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    # Separate integer-named columns (I-V curve currents) from parameter columns
    iv_cols = [c for c in df.columns if str(c).isdigit()]
    iv_cols_sorted = sorted(iv_cols, key=lambda x: int(x))

    if len(iv_cols_sorted) != v_points:
        raise DatasetSchemaError(
            f"Expected {v_points} I-V columns, but found {len(iv_cols_sorted)}: {iv_cols_sorted}"
        )

    # Verify expected parameter columns exist
    expected_param_cols = ["temperature", "WF", "t_Drift", "D_N", "D_C"]
    missing_params = [p for p in expected_param_cols if p not in df.columns]
    if missing_params:
        raise DatasetSchemaError(f"Missing parameter columns: {missing_params}")

    v_grid = np.linspace(v_start, v_stop, v_points)

    # Extract currents as float64
    raw_currents = df[iv_cols_sorted].to_numpy(dtype=np.float64)

    # Check for NaNs or Infs
    if not np.isfinite(raw_currents).all():
        raise DatasetSchemaError("Found NaN or Inf values in raw currents.")

    # Audit negative currents
    min_val = np.min(raw_currents)
    if min_val < -negative_current_tolerance:
        raise DatasetSchemaError(
            f"Substantial negative current detected: {min_val:.3e} A < -{negative_current_tolerance:.3e} A"
        )

    # Clean currents: floor tiny nonpositive numerical noise
    cleaned_currents = np.where(raw_currents < noise_floor, noise_floor, raw_currents)

    # Build cleaned DataFrame
    df_cleaned = df.copy()
    df_cleaned[iv_cols_sorted] = cleaned_currents

    X_params = df_cleaned[["WF", "temperature"]].to_numpy(dtype=np.float64)

    return df_cleaned, v_grid, X_params, cleaned_currents
