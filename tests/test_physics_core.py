"""Unit tests for differentiable SBD physics core, DeviceSpec, and calibration."""

from __future__ import annotations

import torch
from torch.autograd import gradcheck

from pgsbd.device.sbd import BetaGa2O3SBD
from pgsbd.device.spec import DeviceSpec
from pgsbd.physics.sbd_core import SBDPhysicsCore, StableLambertWLog


def test_stable_lambert_w_gradcheck():
    """Verify that the Lambert-W function passes torch.autograd.gradcheck in float64."""
    # Test across a range of log_z values: negative, zero, positive, large positive
    test_log_z = torch.tensor(
        [-5.0, -1.0, 0.0, 1.5, 3.0, 10.0, 25.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    # gradcheck tests numerical gradients vs analytical backward pass
    test_passed = gradcheck(StableLambertWLog.apply, (test_log_z,), eps=1e-6, atol=1e-4)
    assert test_passed, "Lambert-W failed gradcheck!"


def test_physics_core_finite_and_monotonic():
    """Verify finite outputs and monotonicity across the entire voltage range."""
    core = SBDPhysicsCore()
    v_grid = torch.linspace(0.0, 4.0, 52, dtype=torch.float64)

    # Test extreme bounds: (min_WF, min_T), (max_WF, max_T), (min_WF, max_T), (max_WF, min_T)
    test_wfs = torch.tensor([4.99, 5.19, 4.99, 5.19], dtype=torch.float64)
    test_temps = torch.tensor([300.0, 500.0, 500.0, 300.0], dtype=torch.float64)

    currents = core.forward(v_grid, test_wfs, test_temps)

    assert torch.all(torch.isfinite(currents)), "Currents contain NaN or Inf!"
    assert torch.all(currents >= 1e-14), "Currents below minimum noise floor!"

    # Monotonicity across V: diff >= 0
    diffs = torch.diff(currents, dim=1)
    assert torch.all(diffs >= -1e-14), "Non-monotonic forward current detected!"


def test_10000_point_random_sweep_zero_nan():
    """Hard Gate: 10,000-point random sweep across bounds produces zero NaN or Inf."""
    core = SBDPhysicsCore()
    n_points = 10000
    torch.manual_seed(42)

    # Sample random WF, T, and V
    wf = torch.empty(n_points, dtype=torch.float64).uniform_(4.99, 5.19)
    temp = torch.empty(n_points, dtype=torch.float64).uniform_(300.0, 500.0)
    v = torch.empty(n_points, dtype=torch.float64).uniform_(0.0, 4.0)

    # Forward evaluation
    currents = core.forward(v, wf, temp)

    assert torch.all(torch.isfinite(currents)), "10k sweep produced NaN or Inf!"
    assert torch.all(currents > 0), "10k sweep produced non-positive currents!"


def test_physics_monotonic_trends():
    """Verify physical trends:
    1. Higher barrier (WF) decreases current across all forward voltages (dI/dWF < 0).
    2. In turn-on regime (V = 0.5V), higher temperature exponentially increases current (thermionic emission).
    3. In high-bias regime (V = 3.0V), higher temperature decreases current due to mobility degradation.
    """
    core = SBDPhysicsCore()

    # 1. WF sensitivity at V = 1.0V: higher WF lowers current
    v_mid = torch.tensor([1.0], dtype=torch.float64)
    wf_low = torch.tensor([5.00], dtype=torch.float64)
    wf_high = torch.tensor([5.15], dtype=torch.float64)
    temp = torch.tensor([350.0], dtype=torch.float64)

    i_low_wf = core.forward(v_mid, wf_low, temp)
    i_high_wf = core.forward(v_mid, wf_high, temp)
    assert i_low_wf > i_high_wf, "Current did not decrease with higher work function!"

    # 2. Temperature in turn-on regime (V = 0.5V): thermionic current increases with T
    v_sub = torch.tensor([0.5], dtype=torch.float64)
    wf = torch.tensor([5.05], dtype=torch.float64)
    temp_low = torch.tensor([300.0], dtype=torch.float64)
    temp_high = torch.tensor([450.0], dtype=torch.float64)

    i_turnon_low_t = core.forward(v_sub, wf, temp_low)
    i_turnon_high_t = core.forward(v_sub, wf, temp_high)
    assert i_turnon_high_t > i_turnon_low_t, "Turn-on current did not increase with temperature!"

    # 3. Temperature in series resistance regime (V = 3.0V): mobility degradation lowers current
    v_high = torch.tensor([3.0], dtype=torch.float64)
    i_res_low_t = core.forward(v_high, wf, temp_low)
    i_res_high_t = core.forward(v_high, wf, temp_high)
    assert i_res_low_t > i_res_high_t, (
        "High-bias current did not decrease with temperature (mobility effect)!"
    )


def test_device_spec_compliance():
    """Verify BetaGa2O3SBD adheres to DeviceSpec abstraction."""
    device = BetaGa2O3SBD()
    assert isinstance(device, DeviceSpec)
    assert device.device_id == "beta_ga2o3_sbd"
    assert device.material == "β-Ga₂O₃"

    bounds = device.get_variable_bounds()
    assert "WF" in bounds and "temperature" in bounds
    assert bounds["WF"].lower == 4.99 and bounds["WF"].upper == 5.19

    ev_map = device.get_evidence_map()
    assert ev_map["WF"] == "tcad_validated"
    assert ev_map["t_D_um"] == "physics_extrapolated"
    assert ev_map["bv_est"] == "analytic_unvalidated"
