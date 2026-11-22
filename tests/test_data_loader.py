"""Unit tests for dataset loading and schema validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pgsbd.data.load import DatasetSchemaError, load_dataset


def test_load_dataset_success():
    df, v_grid, X, Y = load_dataset()
    assert len(df) == 2000
    assert len(v_grid) == 52
    assert v_grid[0] == 0.0
    assert v_grid[-1] == 4.0
    assert X.shape == (2000, 2)
    assert Y.shape == (2000, 52)
    assert np.all(np.isfinite(Y))
    assert np.all(Y >= 1.0e-14)  # Ensure floored at noise floor


def test_schema_error_missing_column(tmp_path: Path):
    dummy_csv = tmp_path / "bad_schema.csv"
    # Create CSV missing an I-V column (only 51 instead of 52)
    data = {str(i): [1e-10] for i in range(51)}
    data.update(
        {
            "temperature": [300],
            "WF": [5.0],
            "t_Drift": [9.0],
            "D_N": [4e16],
            "D_C": [3.4e16],
        }
    )
    pd.DataFrame(data).to_csv(dummy_csv, index=False)

    with pytest.raises(DatasetSchemaError, match="Expected 52 I-V columns"):
        load_dataset(csv_path=dummy_csv)


def test_schema_error_substantial_negative_current(tmp_path: Path):
    dummy_csv = tmp_path / "bad_neg_current.csv"
    data = {str(i): [1e-10] for i in range(52)}
    data["10"] = [-1.0e-5]  # Substantial negative current
    data.update(
        {
            "temperature": [300],
            "WF": [5.0],
            "t_Drift": [9.0],
            "D_N": [4e16],
            "D_C": [3.4e16],
        }
    )
    pd.DataFrame(data).to_csv(dummy_csv, index=False)

    with pytest.raises(DatasetSchemaError, match="Substantial negative current detected"):
        load_dataset(csv_path=dummy_csv)
