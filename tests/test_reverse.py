"""Tests for Tier A Reverse Mode parameter recovery (Phase 5)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pgsbd.data.load import load_dataset
from pgsbd.inverse.reverse import SBDReverseEngine


@pytest.fixture(scope="module")
def reverse_engine() -> SBDReverseEngine:
    return SBDReverseEngine()


@pytest.fixture(scope="module")
def held_out_data():
    _df, v_grid, X, Y = load_dataset()
    with open("artifacts/splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_idx = splits["indices"]["test"]
    return v_grid, X[test_idx], Y[test_idx]


def test_reverse_recovery_accuracy_held_out(reverse_engine, held_out_data):
    """Verify that parameter recovery on a held-out TCAD curve meets scientific tolerance."""
    v_grid, X_test, Y_test = held_out_data

    # Test on a representative held-out curve in the interior
    # Find a curve away from extreme boundaries
    interior_mask = (
        (X_test[:, 0] >= 5.03)
        & (X_test[:, 0] <= 5.15)
        & (X_test[:, 1] >= 340.0)
        & (X_test[:, 1] <= 460.0)
    )
    idx = int(np.where(interior_mask)[0][0])

    true_wf = float(X_test[idx, 0])
    true_t = float(X_test[idx, 1])
    target_curve = Y_test[idx]

    result = reverse_engine.recover_parameters(v_grid, target_curve)

    assert result["status"] == "met"
    rec = result["recovered"]
    rec_wf = rec["WF"]
    rec_t = rec["T"]

    # Advisory target: WF within 0.02 eV, T within 25 K
    assert abs(rec_wf - true_wf) < 0.025, f"WF error too high: {abs(rec_wf - true_wf):.4f} eV"
    assert abs(rec_t - true_t) < 30.0, f"T error too high: {abs(rec_t - true_t):.2f} K"

    # Fit residual checks
    assert result["fit_residual"]["mae_decades"] < 0.15
    assert result["fit_residual"]["rmse_decades"] < 0.25

    # Check uncertainty bounds contain the recovery
    assert rec["WF_uncertainty"] > 0
    assert rec["T_uncertainty"] > 0
    assert rec["WF_interval"][0] <= rec_wf <= rec["WF_interval"][1]
    assert rec["T_interval"][0] <= rec_t <= rec["T_interval"][1]


def test_reverse_recovery_arbitrary_voltage_grid(reverse_engine, held_out_data):
    """Verify that arbitrary, non-standard voltage grids are properly resampled via PCHIP."""
    v_grid, _X_test, Y_test = held_out_data
    target_curve = Y_test[0]

    # Create an arbitrary non-standard grid: 35 points between 0.0 and 3.8 V
    v_arbitrary = np.sort(np.concatenate([np.linspace(0.0, 1.5, 20), np.linspace(1.6, 3.8, 15)]))
    # Interpolate true curve onto arbitrary grid
    log_true = np.log10(np.maximum(target_curve, 1e-14))
    log_arb = np.interp(v_arbitrary, v_grid, log_true)
    i_arbitrary = 10.0**log_arb

    result = reverse_engine.recover_parameters(v_arbitrary, i_arbitrary)

    assert result["status"] == "met"
    assert np.isfinite(result["recovered"]["WF"])
    assert np.isfinite(result["recovered"]["T"])
    assert 4.99 <= result["recovered"]["WF"] <= 5.19
    assert 300.0 <= result["recovered"]["T"] <= 500.0
    assert len(result["reconstructed"]["V"]) == 52


def test_reverse_recovery_boundary_warning(reverse_engine):
    """Verify that boundary warnings fire when operating near the parameter envelope."""
    # Generate a curve near boundary: WF = 4.991, T = 302 K
    v_grid = np.linspace(0.0, 4.0, 52)
    boundary_sim = reverse_engine.ensemble.predict(4.992, 302.0)
    i_curve = np.array(boundary_sim["I_mean"])

    result = reverse_engine.recover_parameters(v_grid, i_curve)

    assert len(result["warnings"]) > 0
    warning_text = " ".join(result["warnings"])
    assert "boundary" in warning_text.lower()


def test_reverse_recovery_noise_robustness(reverse_engine, held_out_data):
    """Verify that reverse recovery does not crash and stays bounded with 5% noise."""
    _v_grid, _X_test, Y_test = held_out_data
    target_curve = Y_test[5]

    np.random.seed(123)
    noisy_curve = target_curve * (1.0 + np.random.normal(0.0, 0.05, size=len(target_curve)))
    noisy_curve = np.maximum(noisy_curve, 1e-14)

    v_grid = np.linspace(0.0, 4.0, 52)
    result = reverse_engine.recover_parameters(v_grid, noisy_curve)

    assert result["status"] == "met"
    assert 4.99 <= result["recovered"]["WF"] <= 5.19
    assert 300.0 <= result["recovered"]["T"] <= 500.0
    assert result["fit_residual"]["mae_decades"] < 0.25
