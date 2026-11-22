"""Unit tests for metric extraction on repo example I-V CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pgsbd.data.metrics import extract_metrics_numpy


@pytest.fixture
def example_iv_data() -> tuple[np.ndarray, np.ndarray]:
    csv_path = Path("data/raw/n9000_4e+16_3X4e+16_377_5014.csv")
    if not csv_path.exists():
        pytest.skip(f"Example CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    v = df.iloc[:, 0].to_numpy(dtype=np.float64)
    i = df.iloc[:, 1].to_numpy(dtype=np.float64)
    return v, i


def test_metric_extraction_values(example_iv_data: tuple[np.ndarray, np.ndarray]):
    v, i = example_iv_data
    metrics = extract_metrics_numpy(v, i, i_ref=1.0e-3, v_op=2.0)

    # Physical expectations from Section 2 & literature:
    # Von at 1 mA should be ~ 0.52 to 0.90 V
    assert 0.52 <= metrics["von"] <= 0.90, f"Von out of expected range: {metrics['von']}"

    # Vo at 10 uA should be positive and less than Von
    assert 0.30 <= metrics["v_o"] <= 0.70, f"Vo out of expected range: {metrics['v_o']}"
    assert metrics["v_o"] < metrics["von"]

    # Subthreshold slope SS should be ~ 60 - 120 mV/dec
    assert 60.0 <= metrics["ss_mV_dec"] <= 120.0, f"SS out of range: {metrics['ss_mV_dec']}"

    # Rdiff over last 5 points should be ~ 36 - 57 ohm
    assert 36.0 <= metrics["r_diff"] <= 57.0, f"Rdiff out of range: {metrics['r_diff']}"

    # Current at 2V should be positive and in mA range
    assert 0.01 <= metrics["i_at_vop"] <= 0.10, f"I@Vop out of range: {metrics['i_at_vop']}"


def test_metric_extraction_edge_cases():
    v = np.linspace(0.0, 4.0, 52)
    # Extremely low currents that never reach 1 mA
    tiny_i = np.full(52, 1e-12)
    m = extract_metrics_numpy(v, tiny_i, i_ref=1.0e-3)
    assert m["von"] == 4.0  # Should clamp to upper grid limit
