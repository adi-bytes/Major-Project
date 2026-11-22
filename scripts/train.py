"""Comprehensive training and evaluation script for PG-SBD surrogates and ablations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pgsbd.data.load import load_dataset
from pgsbd.eval.ablation import (
    compute_finite_difference_sensitivity,
    generate_model_card_markdown,
    run_single_ablation_model,
)
from pgsbd.eval.validate import (
    evaluate_predictions,
    generate_validation_report_markdown,
    save_metrics_artifact,
)
from pgsbd.ml.baselines import (
    HistGradientBoostingBaseline,
    KNNBaseline,
    PlainMLPBaseline,
    RidgeBaseline,
    save_scaler_artifact,
)
from pgsbd.ml.ensemble import EnsembleSurrogate
from pgsbd.physics.sbd_core import SBDPhysicsCore


def main():
    print("=" * 65)
    print("PG-SBD UNIFIED TRAINING PIPELINE: BASELINES + HYBRID ENSEMBLE + ABLATIONS")
    print("=" * 65)

    dataset_path = "data/raw/new_dataset_GaO_WF+T.csv"
    splits_path = "artifacts/splits.json"
    metrics_path = "artifacts/metrics.json"
    report_path = "reports/validation_report.md"

    _, v_grid, X, Y = load_dataset(dataset_path)

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    indices = splits["indices"]
    train_idx = indices["train"]
    val_idx = indices["val"]
    test_idx = indices["test"]
    extrap_idx = indices["extrapolation"]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    X_extrap, Y_extrap = X[extrap_idx], Y[extrap_idx]

    log_Y_train = np.log10(np.maximum(Y_train, 1e-14))
    log_Y_val = np.log10(np.maximum(Y_val, 1e-14))

    # Load existing metrics if available
    metrics_file = Path(metrics_path)
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {"train": {}, "val": {}, "test": {}, "extrapolation": {}}

    # 1. Train Baselines (if not already recorded)
    if "PlainMLP" not in all_results.get("test", {}):
        print("\n--- Training ML Baselines ---")
        baselines = {
            "Ridge": RidgeBaseline(alpha=1.0),
            "HistGradientBoosting": HistGradientBoostingBaseline(max_iter=100, random_state=42),
            "KNN": KNNBaseline(n_neighbors=3),
            "PlainMLP": PlainMLPBaseline(
                hidden_dims=(128, 128), lr=2e-3, epochs=250, batch_size=64, seed=42
            ),
        }
        for name, model in baselines.items():
            print(f"Training {name} ...")
            model.fit(X_train, log_Y_train, X_val, log_Y_val)
            pred_test = model.predict(X_test)
            pred_extrap = model.predict(X_extrap)
            all_results["test"][name] = evaluate_predictions(v_grid, Y_test, pred_test)
            all_results["extrapolation"][name] = evaluate_predictions(v_grid, Y_extrap, pred_extrap)

        save_scaler_artifact(baselines["Ridge"].scaler, "artifacts/scalers.json")
        baselines["PlainMLP"].save_weights("artifacts/plain_mlp.pt")

    # 2. Train Physics-Guided Residual Ensemble (M=5)
    print("\n--- Training Hybrid Physics-Guided Residual Ensemble (M=5) ---")
    calib_file = Path("artifacts/calib.json")
    if not calib_file.exists():
        from pgsbd.physics.calibrate import calibrate_sbd_core

        calib_dict = calibrate_sbd_core(v_grid, X_train, Y_train)
    else:
        with open(calib_file, "r", encoding="utf-8") as f:
            calib_dict = json.load(f)

    params = calib_dict["parameters"]
    physics_core = SBDPhysicsCore(
        chi_eff=params["chi_eff"],
        ideality_n=params["ideality_n"],
        A_eff_cm2=params["A_eff_cm2"],
        mu_300=params["mu_300"],
        alpha=params["alpha"],
        R_c_ohm=params["R_c_ohm"],
    )

    ensemble = EnsembleSurrogate(
        physics_core=physics_core,
        ensemble_size=5,
        hidden_dims=(128, 128),
        residual_cap=0.7,
        use_physics=True,
        use_residual_cap=True,
    )

    # Train ensemble members
    ensemble.train_ensemble(
        X_train=X_train,
        log_y_train=log_Y_train,
        X_val=X_val,
        log_y_val=log_Y_val,
        epochs=150,
        batch_size=64,
        lr=2.0e-3,
        lambda_mono=0.1,
        lambda_smooth=0.01,
        lambda_res=0.01,
        base_seed=42,
    )
    ensemble.save_ensemble("artifacts/ensemble")

    # Evaluate ensemble mean
    def eval_ensemble_split(x_arr: np.ndarray) -> np.ndarray:
        log_preds = []
        for model in ensemble.models:
            model.eval()
            with torch.no_grad():
                lp = model(torch.tensor(x_arr, dtype=torch.float32)).numpy()
                log_preds.append(lp)
        return np.mean(log_preds, axis=0)

    log_pred_test_ens = eval_ensemble_split(X_test)
    log_pred_extrap_ens = eval_ensemble_split(X_extrap)

    ens_test_metrics = evaluate_predictions(v_grid, Y_test, log_pred_test_ens)
    ens_extrap_metrics = evaluate_predictions(v_grid, Y_extrap, log_pred_extrap_ens)

    all_results["test"]["Physics-Guided Residual Ensemble"] = ens_test_metrics
    all_results["extrapolation"]["Physics-Guided Residual Ensemble"] = ens_extrap_metrics

    print("\n--- Hybrid Ensemble Performance ---")
    print(
        f"[Ensemble Test] log10-MAE: {ens_test_metrics['log10_mae_all']:.4f} dec | "
        f"Vo-MAE: {ens_test_metrics['vo_mae_V'] * 1000:.2f} mV | "
        f"SS-MAE: {ens_test_metrics['ss_mae_mV_dec']:.2f} mV/dec | "
        f"Rdiff-RelErr: {ens_test_metrics['rdiff_rel_err_pct']:.2f}% | "
        f"Von-MAE: {ens_test_metrics['von_mae_V'] * 1000:.2f} mV"
    )
    print(
        f"[Ensemble Extrap] log10-MAE: {ens_extrap_metrics['log10_mae_all']:.4f} dec | "
        f"Vo-MAE: {ens_extrap_metrics['vo_mae_V'] * 1000:.2f} mV | "
        f"SS-MAE: {ens_extrap_metrics['ss_mae_mV_dec']:.2f} mV/dec | "
        f"Rdiff-RelErr: {ens_extrap_metrics['rdiff_rel_err_pct']:.2f}% | "
        f"Von-MAE: {ens_extrap_metrics['von_mae_V'] * 1000:.2f} mV"
    )

    # 3. Run Ablation Suite
    print("\n--- Running Ablation Suite ---")
    ablations_dict = {}

    # Ablation 1: Physics Core OFF (pure MLP)
    print("Ablation 1: Physics Core OFF (Pure MLP with constraints & cap)...")
    model_no_phys = run_single_ablation_model(
        X_train,
        log_Y_train,
        X_val,
        log_Y_val,
        use_physics=False,
        use_residual_cap=True,
        use_constraints=True,
        physics_core=physics_core,
        epochs=120,
        seed=101,
    )
    pred_test_np = model_no_phys(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
    pred_extrap_np = model_no_phys(torch.tensor(X_extrap, dtype=torch.float32)).detach().numpy()
    ablations_dict["Physics Core OFF (Pure MLP)"] = {
        "test": evaluate_predictions(v_grid, Y_test, pred_test_np),
        "extrapolation": evaluate_predictions(v_grid, Y_extrap, pred_extrap_np),
    }

    # Ablation 2: Constraints OFF (lambda_mono=0, lambda_smooth=0)
    print("Ablation 2: Physics Constraints OFF...")
    model_no_constr = run_single_ablation_model(
        X_train,
        log_Y_train,
        X_val,
        log_Y_val,
        use_physics=True,
        use_residual_cap=True,
        use_constraints=False,
        physics_core=physics_core,
        epochs=120,
        seed=102,
    )
    pred_test_nc = model_no_constr(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
    pred_extrap_nc = model_no_constr(torch.tensor(X_extrap, dtype=torch.float32)).detach().numpy()
    ablations_dict["Physics Constraints OFF"] = {
        "test": evaluate_predictions(v_grid, Y_test, pred_test_nc),
        "extrapolation": evaluate_predictions(v_grid, Y_extrap, pred_extrap_nc),
    }

    # Ablation 3: Residual Cap OFF (c -> infinity)
    print("Ablation 3: Residual Cap OFF...")
    model_no_cap = run_single_ablation_model(
        X_train,
        log_Y_train,
        X_val,
        log_Y_val,
        use_physics=True,
        use_residual_cap=False,
        use_constraints=True,
        physics_core=physics_core,
        epochs=120,
        seed=103,
    )
    pred_test_ncap = model_no_cap(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
    pred_extrap_ncap = model_no_cap(torch.tensor(X_extrap, dtype=torch.float32)).detach().numpy()
    ablations_dict["Residual Cap OFF"] = {
        "test": evaluate_predictions(v_grid, Y_test, pred_test_ncap),
        "extrapolation": evaluate_predictions(v_grid, Y_extrap, pred_extrap_ncap),
    }

    all_results["ablation"] = ablations_dict

    # 4. Sensitivity Analysis
    print("\n--- Finite Difference Sensitivity Analysis ---")

    def predict_ens_fn(w: float, t: float) -> np.ndarray:
        res = ensemble.predict(w, t)
        return np.array(res["I_mean"])

    sensitivity = compute_finite_difference_sensitivity(
        predict_fn=predict_ens_fn,
        nominal_wf=5.09,
        nominal_t=400.0,
        d_wf=0.01,
        d_t=5.0,
        v_grid=v_grid,
    )
    all_results["sensitivity"] = sensitivity
    print("Sensitivity results at (WF=5.09 eV, T=400 K):")
    print(f"  dVo/dWF = {sensitivity['dVo_dWF_V_per_eV']:.4f} V/eV")
    print(f"  dVo/dT  = {sensitivity['dVo_dT_V_per_K'] * 1000.0:.3f} mV/K")
    print(f"  dSS/dT  = {sensitivity['dSS_dT_mV_dec_per_K']:.4f} mV/dec/K")
    print(f"  dRdiff/dT = {sensitivity['dRdiff_dT_ohm_per_K']:.4f} ohm/K")

    # Save metrics artifact
    save_metrics_artifact(all_results, metrics_path)
    print(f"\nSaved updated {metrics_path}")

    # 5. Representative Curve Overlays
    curve_overlays = []
    test_wfs = X_test[:, 0]
    sorted_test_indices = np.argsort(test_wfs)
    rep_test_indices = [
        sorted_test_indices[0],
        sorted_test_indices[len(sorted_test_indices) // 2],
        sorted_test_indices[-1],
    ]

    for idx in rep_test_indices:
        real_idx = test_idx[idx]
        wf_v = float(X_test[idx, 0])
        t_v = float(X_test[idx, 1])
        tcad_curve = Y_test[idx]
        ens_pred = ensemble.predict(wf_v, t_v)
        curve_overlays.append(
            {
                "curve_index": real_idx,
                "split": "test",
                "WF": wf_v,
                "temperature": t_v,
                "v_grid": v_grid,
                "tcad_i": tcad_curve,
                "models": {
                    "Physics-Only": ens_pred["I_phys"],
                    "Hybrid-Ensemble": ens_pred["I_mean"],
                },
            }
        )

    for idx in [0, len(extrap_idx) - 1]:
        real_idx = extrap_idx[idx]
        wf_v = float(X_extrap[idx, 0])
        t_v = float(X_extrap[idx, 1])
        tcad_curve = Y_extrap[idx]
        ens_pred = ensemble.predict(wf_v, t_v)
        curve_overlays.append(
            {
                "curve_index": real_idx,
                "split": "extrapolation",
                "WF": wf_v,
                "temperature": t_v,
                "v_grid": v_grid,
                "tcad_i": tcad_curve,
                "models": {
                    "Physics-Only": ens_pred["I_phys"],
                    "Hybrid-Ensemble": ens_pred["I_mean"],
                },
            }
        )

    generate_validation_report_markdown(all_results, curve_overlays, report_path)
    print(f"Generated {report_path}")

    generate_model_card_markdown(all_results, calib_dict, "artifacts/MODEL_CARD.md")
    print("Auto-generated artifacts/MODEL_CARD.md")


if __name__ == "__main__":
    main()
