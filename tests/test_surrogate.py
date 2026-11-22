"""Unit tests and Hard Gates for PhysicsGuidedSurrogate and Ensemble (Phase 4).

Tests:
1. Bounded residual constraint: |log10(I) - log10(I_phys)| <= c
2. Off-grid cubic spline interpolation
3. Ensemble prediction, uncertainty quantification, and verifier flags
4. Hard Gate: 10,000-point random sweep with ZERO NaN / Inf
5. Hard Gate: Dense-grid monotonicity across voltage
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgsbd.ml.ensemble import EnsembleSurrogate
from pgsbd.ml.surrogate import PhysicsGuidedSurrogate
from pgsbd.physics.sbd_core import SBDPhysicsCore


@pytest.fixture
def mock_physics_core() -> SBDPhysicsCore:
    return SBDPhysicsCore()


def test_bounded_residual_formulation(mock_physics_core):
    """Verify that the neural residual correction is strictly bounded by residual_cap c."""
    c_cap = 0.7
    model = PhysicsGuidedSurrogate(
        physics_core=mock_physics_core,
        residual_cap=c_cap,
        use_physics=True,
        use_residual_cap=True,
    )
    # Feed arbitrary inputs, including out-of-distribution extremes
    x_t = torch.tensor([[5.05, 350.0], [4.90, 280.0], [5.30, 520.0]], dtype=torch.float32)
    log_pred, log_phys, _ = model(x_t, return_components=True)

    delta = torch.abs(log_pred - log_phys)
    # Numerical tolerance for float32 tanh
    assert torch.all(delta <= c_cap + 1e-5), (
        f"Residual exceeded cap of {c_cap}: max={torch.max(delta)}"
    )


def test_off_grid_cubic_spline_interpolation(mock_physics_core):
    """Verify that off-grid query voltages interpolate smoothly without NaN or negative values."""
    model = PhysicsGuidedSurrogate(physics_core=mock_physics_core)
    x_np = np.array([[5.05, 380.0], [5.12, 420.0]])

    # Dense 101-point voltage grid from 0 to 4 V
    v_dense = np.linspace(0.0, 4.0, 101)
    preds = model.predict_numpy(x_np, v_query=v_dense)

    assert preds.shape == (2, 101)
    assert np.all(np.isfinite(preds)), "Interpolation produced non-finite values!"
    assert np.all(preds >= 1e-14), "Interpolation produced negative or unfloored values!"


def test_ensemble_prediction_and_flags(mock_physics_core):
    """Verify ensemble prediction outputs and domain/uncertainty verifier flags."""
    ensemble = EnsembleSurrogate(physics_core=mock_physics_core, ensemble_size=2)

    # 1. Nominal in-bounds prediction
    res = ensemble.predict(wf=5.05, temperature_k=350.0)
    assert len(res["V"]) == 52
    assert len(res["I_mean"]) == 52
    assert len(res["I_std"]) == 52
    assert len(res["I_phys"]) == 52
    assert "outside_box" not in res["flags"]
    assert "physics_extrapolated" not in res["flags"]

    # 2. Outside box prediction (WF > 5.19 eV)
    res_out = ensemble.predict(wf=5.25, temperature_k=350.0)
    assert "outside_box" in res_out["flags"]

    # 3. Physics extrapolated prediction (t_D = 12 um != 9.0 um)
    res_ext = ensemble.predict(wf=5.05, temperature_k=350.0, t_d_um=12.0)
    assert "physics_extrapolated" in res_ext["flags"]


def test_10000_point_nan_sweep_surrogate(mock_physics_core):
    """Hard Gate: 10,000-point random sweep across bounds produces zero NaN or Inf."""
    model = PhysicsGuidedSurrogate(physics_core=mock_physics_core)
    n_points = 10000
    torch.manual_seed(42)

    wf = torch.empty(n_points, 1).uniform_(4.99, 5.19)
    temp = torch.empty(n_points, 1).uniform_(300.0, 500.0)
    x_batch = torch.cat([wf, temp], dim=1)

    model.eval()
    with torch.no_grad():
        preds = model(x_batch)

    assert torch.all(torch.isfinite(preds)), "10,000-point sweep produced NaN or Inf!"


def test_dense_grid_monotonicity(mock_physics_core):
    """Hard Gate: On a dense (WF, T) grid, predicted forward I-V curves are nondecreasing in V."""
    from pathlib import Path

    ensemble = EnsembleSurrogate(physics_core=mock_physics_core)
    ckpt_path = Path("artifacts/ensemble/model_0.pt")
    if ckpt_path.exists():
        ensemble.load_ensemble("artifacts/ensemble")

    wf_vals = np.linspace(4.99, 5.19, 10)
    t_vals = np.linspace(300.0, 500.0, 10)
    for w in wf_vals:
        for t in t_vals:
            res = ensemble.predict(w, t)
            i_arr = np.array(res["I_mean"])
            diffs = np.diff(i_arr)
            assert np.min(diffs) >= -1e-12, (
                f"Non-monotonic I-V curve detected at WF={w}, T={t}: min diff = {np.min(diffs):.4e}"
            )
