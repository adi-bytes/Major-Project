"""Script to calibrate the SBD physics core and log metrics (Phase 3)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pgsbd.data.load import load_dataset
from pgsbd.eval.validate import evaluate_predictions
from pgsbd.physics.calibrate import calibrate_sbd_core
from pgsbd.physics.sbd_core import SBDPhysicsCore


def main():
    print("Loading dataset and splits for calibration...")
    _, v_grid, X, Y = load_dataset()

    with open("artifacts/splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_idx = splits["indices"]["train"]
    test_idx = splits["indices"]["test"]
    extrap_idx = splits["indices"]["extrapolation"]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    X_extrap, Y_extrap = X[extrap_idx], Y[extrap_idx]

    # 1. Run multi-start calibration
    calib_data = calibrate_sbd_core(
        v_grid=v_grid,
        X_train=X_train,
        Y_train=Y_train,
        output_json_path="artifacts/calib.json",
    )

    params = calib_data["parameters"]

    # 2. Initialize core with calibrated parameters
    core = SBDPhysicsCore(
        chi_eff=params["chi_eff"],
        ideality_n=params["ideality_n"],
        A_eff_cm2=params["A_eff_cm2"],
        mu_300=params["mu_300"],
        alpha=params["alpha"],
        R_c_ohm=params["R_c_ohm"],
    )

    # 3. Evaluate physics-only model on Test and Extrapolation sets
    def predict_physics(x_arr: np.ndarray) -> np.ndarray:
        v_t = torch.tensor(v_grid, dtype=torch.float64)
        wf_t = torch.tensor(x_arr[:, 0], dtype=torch.float64)
        t_t = torch.tensor(x_arr[:, 1], dtype=torch.float64)
        with torch.no_grad():
            i_p = core.forward(v_t, wf_t, t_t).numpy()
        return np.log10(np.maximum(i_p, 1e-14))

    log_pred_test = predict_physics(X_test)
    log_pred_extrap = predict_physics(X_extrap)

    phys_test_metrics = evaluate_predictions(v_grid, Y_test, log_pred_test)
    phys_extrap_metrics = evaluate_predictions(v_grid, Y_extrap, log_pred_extrap)

    print("\n--- Physics-Only Performance ---")
    print(
        f"[Physics-Only Test] log10-MAE: {phys_test_metrics['log10_mae_all']:.4f} dec | "
        f"Vo-MAE: {phys_test_metrics['vo_mae_V'] * 1000:.1f} mV | "
        f"Rdiff-RelErr: {phys_test_metrics['rdiff_rel_err_pct']:.1f}% | "
        f"Von-MAE: {phys_test_metrics['von_mae_V'] * 1000:.1f} mV"
    )
    print(
        f"[Physics-Only Extrap] log10-MAE: {phys_extrap_metrics['log10_mae_all']:.4f} dec | "
        f"Vo-MAE: {phys_extrap_metrics['vo_mae_V'] * 1000:.1f} mV | "
        f"Rdiff-RelErr: {phys_extrap_metrics['rdiff_rel_err_pct']:.1f}% | "
        f"Von-MAE: {phys_extrap_metrics['von_mae_V'] * 1000:.1f} mV"
    )

    # Update artifacts/metrics.json with "Physics-Only" entry
    metrics_path = Path("artifacts/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {"train": {}, "val": {}, "test": {}, "extrapolation": {}}

    all_metrics["test"]["Physics-Only"] = phys_test_metrics
    all_metrics["extrapolation"]["Physics-Only"] = phys_extrap_metrics

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Updated {metrics_path} with Physics-Only baseline.")

    # 4. Generate 6-curve comparison plot (3 test curves, 3 extrapolation curves)
    _fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    # Pick 3 test curves across WF spectrum
    test_sort = np.argsort(X_test[:, 0])
    selected_test = [test_sort[0], test_sort[len(test_sort) // 2], test_sort[-1]]

    # Pick 3 extrap curves across T spectrum
    extrap_sort = np.argsort(X_extrap[:, 1])
    selected_extrap = [extrap_sort[0], extrap_sort[len(extrap_sort) // 2], extrap_sort[-1]]

    plot_samples = [("Test", X_test, Y_test, i) for i in selected_test] + [
        ("Extrap", X_extrap, Y_extrap, i) for i in selected_extrap
    ]

    for ax, (s_name, x_arr, y_arr, idx) in zip(axes, plot_samples):
        wf_val = x_arr[idx, 0]
        t_val = x_arr[idx, 1]
        tcad_y = y_arr[idx]

        v_t = torch.tensor(v_grid, dtype=torch.float64)
        wf_t = torch.tensor([wf_val], dtype=torch.float64)
        t_t = torch.tensor([t_val], dtype=torch.float64)
        with torch.no_grad():
            phys_y = core.forward(v_t, wf_t, t_t).numpy()[0]

        ax.semilogy(v_grid, tcad_y, "k-", label="TCAD (True)", linewidth=2.0)
        ax.semilogy(v_grid, phys_y, "r--", label="Analytic Physics Core", linewidth=1.8)
        ax.set_ylim(1e-12, 1.0)
        ax.set_xlim(0.0, 4.0)
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current (A)")
        ax.set_title(f"[{s_name}] WF = {wf_val:.3f} eV, T = {t_val:.1f} K")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right")

    plt.tight_layout()
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "physics_fits.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved 6-curve comparison plot to {fig_path}")


if __name__ == "__main__":
    main()
