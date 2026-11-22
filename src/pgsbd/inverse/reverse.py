"""Tier A Reverse Mode: I-V curve -> (WF, T) parameter recovery.

Accepts an I-V curve on any voltage grid, resamples onto the standard grid,
and optimizes weighted log-current mismatch plus (Vo, SS) feature mismatch
strictly within verified (WF, T) bounds [4.99, 5.19] eV x [300, 500] K.

Returns:
- Estimated (WF, T)
- Ensemble-derived uncertainty intervals
- Reconstructed I-V curve with uncertainty band
- Fit residual metrics
- Nearest training TCAD curve lookup
- Ambiguity and boundary warnings
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

from pgsbd.data.load import load_dataset
from pgsbd.data.metrics import extract_metrics_numpy
from pgsbd.ml.ensemble import EnsembleSurrogate


class SBDReverseEngine:
    """Reverse Mode parameter recovery engine for Schottky Barrier Diodes."""

    def __init__(
        self,
        ensemble: EnsembleSurrogate | None = None,
        calib_path: str | Path = "artifacts/calib.json",
        ensemble_dir: str | Path = "artifacts/ensemble",
        dataset_path: str | Path = "data/raw/new_dataset_GaO_WF+T.csv",
        splits_path: str | Path = "artifacts/splits.json",
    ):
        if ensemble is None:
            self.ensemble = EnsembleSurrogate()
            ens_p = Path(ensemble_dir)
            if (ens_p / "model_0.pt").exists():
                self.ensemble.load_ensemble(ens_p)
        else:
            self.ensemble = ensemble

        self.v_grid = np.linspace(0.0, 4.0, 52)
        self.wf_bounds = (4.99, 5.19)
        self.t_bounds = (300.0, 500.0)

        # Load training database for nearest-TCAD lookup
        self.train_wfs: np.ndarray | None = None
        self.train_temps: np.ndarray | None = None
        self.train_indices: list[int] | None = None
        self._init_database(dataset_path, splits_path)

    def _init_database(self, dataset_path: str | Path, splits_path: str | Path) -> None:
        try:
            d_p = Path(dataset_path)
            s_p = Path(splits_path)
            if d_p.exists() and s_p.exists():
                df, _, _, _ = load_dataset(d_p)
                with open(s_p, "r", encoding="utf-8") as f:
                    splits = json.load(f)
                train_idx = splits["indices"]["train"]
                self.train_wfs = df.loc[train_idx, "WF"].to_numpy(dtype=float)
                self.train_temps = df.loc[train_idx, "temperature"].to_numpy(dtype=float)
                self.train_indices = train_idx
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            self.train_wfs = None
            self.train_temps = None
            self.train_indices = None

    def resample_iv_curve(
        self,
        v_input: np.ndarray,
        i_input: np.ndarray,
    ) -> np.ndarray:
        """Resample arbitrary input I-V curve onto the standard 52-point voltage grid."""
        v_in = np.asarray(v_input, dtype=np.float64)
        i_in = np.asarray(i_input, dtype=np.float64)

        # Sort by voltage
        sort_idx = np.argsort(v_in)
        v_sorted = v_in[sort_idx]
        i_sorted = i_in[sort_idx]

        # Deduplicate identical voltages if any
        _, unique_idx = np.unique(v_sorted, return_index=True)
        v_sorted = v_sorted[unique_idx]
        i_sorted = i_sorted[unique_idx]

        # Floor at noise level and work in log10 space
        safe_i = np.maximum(i_sorted, 1.0e-14)
        log_i = np.log10(safe_i)

        if len(v_sorted) == 52 and np.allclose(v_sorted, self.v_grid, atol=1e-4):
            return safe_i

        # Monotonic shape-preserving PCHIP interpolation in log10 space
        pchip = PchipInterpolator(v_sorted, log_i, extrapolate=True)
        resampled_log_i = pchip(self.v_grid)

        # Clamp endpoints physically
        resampled_log_i = np.clip(resampled_log_i, -14.0, 1.0)
        return 10.0**resampled_log_i

    def recover_parameters(
        self,
        v_input: np.ndarray,
        i_input: np.ndarray,
        w_fom: float = 1.0,
    ) -> dict[str, Any]:
        """Perform reverse recovery of (WF, T) from supplied I-V curve.

        Args:
            v_input: Input voltages.
            i_input: Input currents.
            w_fom: Relative weight on (Vo, SS) feature mismatch.

        Returns:
            Dictionary with recovered parameters, uncertainties, warnings, and reconstructed curve.
        """
        # 1. Resample onto standard 52-point grid
        target_currents = self.resample_iv_curve(v_input, i_input)
        log_target = np.log10(target_currents)

        # Extract target features (Vo at 10 uA, SS in mV/dec)
        target_fom = extract_metrics_numpy(self.v_grid, target_currents)
        vo_target = target_fom["v_o"]
        ss_target = target_fom["ss_mV_dec"]

        # Objective function in normalized coordinates u in [0, 1]^2
        # u[0] -> WF in [4.99, 5.19], u[1] -> T in [300, 500]
        weights = np.ones(51)
        turn_on_mask = (log_target[1:] >= -6.0) & (log_target[1:] <= -2.0)
        weights[turn_on_mask] = 2.0

        def u_to_params(u: np.ndarray) -> tuple[float, float]:
            wf = 4.99 + float(u[0]) * (5.19 - 4.99)
            t = 300.0 + float(u[1]) * (500.0 - 300.0)
            return wf, t

        def loss_fn_u(u: np.ndarray, model_idx: int | None = None) -> float:
            wf_val, t_val = u_to_params(u)

            x_t = torch.tensor([[wf_val, t_val]], dtype=torch.float32)
            if model_idx is None:
                # Ensemble mean prediction
                with torch.no_grad():
                    preds = [m(x_t).squeeze(0).numpy() for m in self.ensemble.models]
                log_pred = np.mean(preds, axis=0)
            else:
                m = self.ensemble.models[model_idx]
                m.eval()
                with torch.no_grad():
                    log_pred = m(x_t).squeeze(0).numpy()

            # 1. Log-current mismatch (V > 0)
            diff = log_pred[1:] - log_target[1:]
            mse_curr = float(np.mean(weights * (diff**2)))

            # 2. FOM feature mismatch
            pred_linear = 10.0**log_pred
            pred_fom = extract_metrics_numpy(self.v_grid, pred_linear)
            vo_err = (pred_fom["v_o"] - vo_target) / 0.02  # normalized by 20 mV
            ss_err = (pred_fom["ss_mV_dec"] - ss_target) / 10.0  # normalized by 10 mV/dec
            fom_loss = float(vo_err**2 + ss_err**2)

            return mse_curr + w_fom * 0.1 * fom_loss

        # 2. Multi-start search in normalized coordinates
        # 9 starting points across normalized domain
        u_starts = [[u0, u1] for u0 in [0.15, 0.50, 0.85] for u1 in [0.15, 0.50, 0.85]]

        best_res = None
        best_loss = float("inf")
        local_minima = []

        u_bounds = [(0.0, 1.0), (0.0, 1.0)]

        for s in u_starts:
            res = minimize(
                loss_fn_u,
                x0=s,
                bounds=u_bounds,
                method="L-BFGS-B",
                options={"maxiter": 60, "ftol": 1e-6, "eps": 1e-4},
            )
            local_minima.append(res)
            if res.fun < best_loss:
                best_loss = res.fun
                best_res = res

        recovered_wf, recovered_t = u_to_params(best_res.x)

        # 3. Ensemble-derived uncertainty intervals
        # Optimize over each of the M=5 models individually to get model variance
        member_recoveries = []
        for m_i in range(self.ensemble.ensemble_size):
            res_m = minimize(
                lambda u, m_idx=m_i: loss_fn_u(u, model_idx=m_idx),
                x0=best_res.x,
                bounds=u_bounds,
                method="L-BFGS-B",
                options={"maxiter": 40, "ftol": 1e-6, "eps": 1e-4},
            )
            wf_m, t_m = u_to_params(res_m.x)
            member_recoveries.append([wf_m, t_m])

        member_recoveries_arr = np.array(member_recoveries)
        wf_std = float(np.std(member_recoveries_arr[:, 0]))
        t_std = float(np.std(member_recoveries_arr[:, 1]))

        # Minimum honest uncertainty bounds
        wf_err_bound = max(wf_std * 2.0, 0.003)
        t_err_bound = max(t_std * 2.0, 3.0)

        # 4. Reconstruct forward curve
        reconstructed = self.ensemble.predict(recovered_wf, recovered_t)

        # Fit residual metrics
        pred_i = np.array(reconstructed["I_mean"])
        log_diff = np.abs(np.log10(np.maximum(pred_i[1:], 1e-14)) - log_target[1:])
        fit_mae_decades = float(np.mean(log_diff))
        fit_rmse_decades = float(np.sqrt(np.mean(log_diff**2)))

        # 5. Nearest TCAD lookup in training set
        nearest_tcad = {"index": -1, "WF": recovered_wf, "T": recovered_t, "distance": 0.0}
        if self.train_wfs is not None and self.train_temps is not None:
            # Normalized Euclidean distance in (WF, T)
            d_wf = (self.train_wfs - recovered_wf) / (5.19 - 4.99)
            d_t = (self.train_temps - recovered_t) / (500.0 - 300.0)
            dists = np.sqrt(d_wf**2 + d_t**2)
            nn_idx = int(np.argmin(dists))
            nearest_tcad = {
                "index": int(self.train_indices[nn_idx]),
                "WF": float(self.train_wfs[nn_idx]),
                "T": float(self.train_temps[nn_idx]),
                "distance": float(dists[nn_idx]),
            }

        # 6. Boundary & Ambiguity Warnings
        warnings: list[str] = []
        # Near boundary check
        if recovered_wf <= 4.995 or recovered_wf >= 5.185:
            warnings.append(
                f"Recovered WF ({recovered_wf:.3f} eV) is near dataset boundary [4.99, 5.19] eV."
            )
        if recovered_t <= 305.0 or recovered_t >= 495.0:
            warnings.append(
                f"Recovered temperature ({recovered_t:.1f} K) is near dataset boundary [300, 500] K."
            )

        # Ambiguity / multi-modal basin check:
        # Check if another local minimum exists with competitive loss but distinct parameters
        for lm in local_minima:
            other_wf, other_t = u_to_params(lm.x)
            is_distinct = abs(other_wf - recovered_wf) > 0.015 or abs(other_t - recovered_t) > 15.0
            is_close_loss = lm.fun <= best_loss * 1.10
            if is_distinct and is_close_loss:
                warnings.append(
                    f"Ambiguity detected: secondary minimum at WF={other_wf:.3f} eV, T={other_t:.1f} K "
                    f"with competitive loss ({lm.fun:.4f} vs {best_loss:.4f})."
                )
                break

        return {
            "recovered": {
                "WF": recovered_wf,
                "T": recovered_t,
                "WF_uncertainty": wf_err_bound,
                "T_uncertainty": t_err_bound,
                "WF_interval": [recovered_wf - wf_err_bound, recovered_wf + wf_err_bound],
                "T_interval": [recovered_t - t_err_bound, recovered_t + t_err_bound],
            },
            "fit_residual": {
                "mae_decades": fit_mae_decades,
                "rmse_decades": fit_rmse_decades,
                "objective_loss": best_loss,
            },
            "reconstructed": reconstructed,
            "nearest_tcad": nearest_tcad,
            "warnings": warnings,
            "status": "met",
        }
