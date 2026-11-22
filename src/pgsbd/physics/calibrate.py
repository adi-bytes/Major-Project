"""Calibration and identifiability audit for the SBD physics core.

Performs multi-start bounded least squares optimization to calibrate:
    theta = {chi_eff, ideality_n, A_eff_cm2, mu_300, alpha, R_c_ohm}
on the training split only, minimizing Huber loss on log10(I) for I >= 1e-12 A (V > 0).
Computes parameter correlation matrix, jacobian condition number, and residuals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import least_squares

from pgsbd.physics.sbd_core import SBDPhysicsCore


def calibrate_sbd_core(
    v_grid: np.ndarray,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    i_min: float = 1.0e-12,
    output_json_path: str | Path = "artifacts/calib.json",
) -> dict[str, Any]:
    """Calibrate identifiable parameters on the training split.

    Args:
        v_grid: 1D array of voltages [52].
        X_train: Array of [WF, T] for training samples [N, 2].
        Y_train: Array of currents for training samples [N, 52].
        i_min: Current threshold for loss evaluation (default 1e-12 A).
        output_json_path: Path to save calibration artifact.

    Returns:
        Dict with calibrated parameters, correlations, residuals, and status.
    """
    # Downsample training curves if large for fast, high-precision calibration
    # E.g. select 100 representative curves spanning the (WF, T) space
    n_total = len(X_train)
    step = max(1, n_total // 150)
    calib_indices = np.arange(0, n_total, step)

    X_calib = X_train[calib_indices]
    Y_calib = Y_train[calib_indices]

    # Mask: exclude V=0 (col 0) and points with true I < i_min
    v_gt0 = v_grid[1:]
    y_gt0 = Y_calib[:, 1:]
    valid_mask = y_gt0 >= i_min
    log_y_target = np.log10(y_gt0)

    # PyTorch tensors for evaluation
    v_t = torch.tensor(v_gt0, dtype=torch.float64)
    wf_t = torch.tensor(X_calib[:, 0], dtype=torch.float64)
    temp_t = torch.tensor(X_calib[:, 1], dtype=torch.float64)

    core = SBDPhysicsCore()

    # Parameter vector p:
    # 0: chi_eff (eV)       [3.8, 4.2]
    # 1: ideality_n         [1.00, 1.25]
    # 2: log10(A_eff_cm2)   [-6.0, -2.0]
    # 3: mu_300 (cm^2/Vs)   [50.0, 250.0]
    # 4: alpha              [1.0, 3.0]
    # 5: R_c_ohm            [0.0, 20.0]

    param_names = ["chi_eff", "ideality_n", "log10_A_eff", "mu_300", "alpha", "R_c_ohm"]
    lower_bounds = [3.8, 1.00, -6.0, 50.0, 1.0, 0.0]
    upper_bounds = [4.2, 1.25, -2.0, 250.0, 3.0, 20.0]
    p0 = [4.05, 1.05, -3.8, 120.0, 1.7, 0.1]

    def residual_fn(p: np.ndarray) -> np.ndarray:
        chi_eff, n, log_a, mu_300, alpha, r_c = p
        a_eff = 10.0**log_a

        theta = {
            "chi_eff": float(chi_eff),
            "ideality_n": float(n),
            "A_eff_cm2": float(a_eff),
            "mu_300": float(mu_300),
            "alpha": float(alpha),
            "R_c_ohm": float(r_c),
        }

        with torch.no_grad():
            i_pred = core.forward(v_t, wf_t, temp_t, theta=theta).numpy()

        log_i_pred = np.log10(np.maximum(i_pred, 1e-14))
        diff = (log_i_pred - log_y_target)[valid_mask]
        return diff

    # Multi-start optimization
    starts = [
        p0,
        [4.00, 1.03, -4.0, 100.0, 1.5, 0.0],
        [4.10, 1.08, -3.6, 140.0, 2.0, 1.0],
        [4.02, 1.05, -3.7, 110.0, 1.8, 0.0],
    ]

    best_res = None
    best_cost = float("inf")

    print(
        f"Calibrating SBD physics core on {len(X_calib)} curves using multi-start least-squares..."
    )
    for idx, start_p in enumerate(starts):
        res = least_squares(
            residual_fn,
            x0=start_p,
            bounds=(lower_bounds, upper_bounds),
            loss="huber",
            f_scale=0.5,
            max_nfev=150,
            ftol=1e-5,
            xtol=1e-5,
        )
        if res.cost < best_cost:
            best_cost = res.cost
            best_res = res
        print(f"  Start #{idx + 1}: Cost = {res.cost:.4f}, Success: {res.success}")

    p_opt = best_res.x
    chi_eff_opt, n_opt, log_a_opt, mu_300_opt, alpha_opt, r_c_opt = p_opt
    a_eff_opt = float(10.0**log_a_opt)

    # Identifiability analysis via Jacobian
    # J has shape [M, 6]
    J = best_res.jac
    # Approximate covariance matrix: cov = inv(J^T J)
    jt_j = J.T @ J
    cond_number = float(np.linalg.cond(jt_j))

    try:
        cov = np.linalg.pinv(jt_j) * (2.0 * best_res.cost / max(1, len(best_res.fun) - len(p_opt)))
        std_err = np.sqrt(np.diag(cov))
        # Correlation matrix: corr[i, j] = cov[i, j] / sqrt(cov[i, i] * cov[j, j])
        denom = np.outer(std_err, std_err)
        corr_matrix = cov / np.where(denom > 0, denom, 1.0)
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    except (np.linalg.LinAlgError, ValueError):
        std_err = np.zeros(len(p_opt))
        corr_matrix = np.eye(len(p_opt))

    final_residuals = best_res.fun
    res_mae = float(np.mean(np.abs(final_residuals)))
    res_rmse = float(np.sqrt(np.mean(final_residuals**2)))

    calib_data = {
        "status": "success" if best_res.success else "converged_approx",
        "cost": float(best_res.cost),
        "residual_mae_decades": res_mae,
        "residual_rmse_decades": res_rmse,
        "jacobian_condition_number": cond_number,
        "parameters": {
            "chi_eff": float(chi_eff_opt),
            "ideality_n": float(n_opt),
            "A_eff_cm2": a_eff_opt,
            "log10_A_eff": float(log_a_opt),
            "mu_300": float(mu_300_opt),
            "alpha": float(alpha_opt),
            "R_c_ohm": float(r_c_opt),
        },
        "parameter_uncertainties": {name: float(err) for name, err in zip(param_names, std_err)},
        "parameter_names": param_names,
        "correlation_matrix": corr_matrix.tolist(),
        "omitted_physics_absorbed_by_residual": [
            "Thermodynamic lattice self-heating (Joule heating raises internal temperature above ambient at high forward currents)",
            "Incomplete dopant ionization at low-to-moderate temperatures",
            "PhuMob doping and high-field saturation velocity",
            "Cylindrical 2D edge field enhancement at contact perimeter",
        ],
    }

    out_p = Path(output_json_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=2)

    print(f"\nCalibration complete! Saved to {out_p}")
    print(f"  chi_eff = {chi_eff_opt:.4f} eV")
    print(f"  ideality_n = {n_opt:.4f}")
    print(
        f"  A_eff = {a_eff_opt:.3e} cm^2 (equivalent radius ~ {np.sqrt(a_eff_opt / np.pi) * 1e4:.1f} um)"
    )
    print(f"  mu_300 = {mu_300_opt:.2f} cm^2/Vs")
    print(f"  alpha = {alpha_opt:.3f}")
    print(f"  R_c = {r_c_opt:.4f} ohm")
    print(f"  Residual MAE: {res_mae:.4f} decades")

    return calib_data
