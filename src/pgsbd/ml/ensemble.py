"""Ensemble surrogate model (M=5) with uncertainty quantification and verifier flags.

Computes:
- Ensemble mean current I_mean and log10 standard deviation sigma(V)
- Analytic physics-only baseline curve I_phys(V)
- Verification flags: 'outside_box', 'physics_extrapolated', 'high_uncertainty'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pgsbd.ml.constraints import SBDPhysicsLoss
from pgsbd.ml.surrogate import PhysicsGuidedSurrogate
from pgsbd.physics.sbd_core import SBDPhysicsCore


class EnsembleSurrogate:
    """M=5 multi-seed bootstrap ensemble of PhysicsGuidedSurrogate models."""

    def __init__(
        self,
        physics_core: SBDPhysicsCore | None = None,
        ensemble_size: int = 5,
        hidden_dims: tuple[int, ...] = (128, 128),
        residual_cap: float = 0.7,
        use_physics: bool = True,
        use_residual_cap: bool = True,
        uncertainty_threshold: float = 0.1,
    ):
        self.ensemble_size = ensemble_size
        self.uncertainty_threshold = uncertainty_threshold
        self.use_physics = use_physics
        self.use_residual_cap = use_residual_cap
        self.residual_cap = residual_cap

        if physics_core is None:
            # Load calibrated parameters from artifacts/calib.json if available
            calib_path = Path("artifacts/calib.json")
            if calib_path.exists():
                with open(calib_path, "r", encoding="utf-8") as f:
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
            else:
                physics_core = SBDPhysicsCore()

        self.physics_core = physics_core
        self.models: list[PhysicsGuidedSurrogate] = [
            PhysicsGuidedSurrogate(
                physics_core=self.physics_core,
                hidden_dims=hidden_dims,
                residual_cap=residual_cap,
                use_physics=use_physics,
                use_residual_cap=use_residual_cap,
            )
            for _ in range(ensemble_size)
        ]

    def train_ensemble(
        self,
        X_train: np.ndarray,
        log_y_train: np.ndarray,
        X_val: np.ndarray,
        log_y_val: np.ndarray,
        epochs: int = 250,
        batch_size: int = 64,
        lr: float = 2.0e-3,
        lambda_mono: float = 0.1,
        lambda_smooth: float = 0.01,
        lambda_res: float = 0.01,
        base_seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Train each ensemble member on bootstrap resamples with distinct seeds."""
        training_logs = []
        n_samples = len(X_train)

        val_x_t = torch.tensor(X_val, dtype=torch.float32)
        val_y_t = torch.tensor(log_y_val, dtype=torch.float32)

        for m_idx, model in enumerate(self.models):
            member_seed = base_seed + m_idx * 101
            torch.manual_seed(member_seed)
            np.random.seed(member_seed)

            # Bootstrap resampling
            boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_train[boot_indices]
            log_y_boot = log_y_train[boot_indices]

            ds = TensorDataset(
                torch.tensor(X_boot, dtype=torch.float32),
                torch.tensor(log_y_boot, dtype=torch.float32),
            )
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            optimizer = torch.optim.AdamW(
                model.residual_mlp.parameters(),
                lr=lr,
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-5
            )
            criterion = SBDPhysicsLoss(
                lambda_mono=lambda_mono,
                lambda_smooth=lambda_smooth,
                lambda_res=lambda_res,
            )

            best_val_loss = float("inf")
            best_state = None
            patience = 40
            patience_cnt = 0

            print(
                f"Training ensemble model {m_idx + 1}/{self.ensemble_size} (seed={member_seed})..."
            )

            for ep in range(epochs):
                model.train()
                for bx, by in loader:
                    optimizer.zero_grad()
                    log_pred, _, raw_res = model(bx, return_components=True)

                    # Random in-bounds collocation batch for monotonicity penalty
                    n_col = 32
                    c_wf = torch.empty(n_col, 1).uniform_(4.99, 5.19)
                    c_t = torch.empty(n_col, 1).uniform_(300.0, 500.0)
                    c_x = torch.cat([c_wf, c_t], dim=1)

                    c_pred = model(c_x)
                    c_wf_plus = model(torch.cat([c_wf + 0.01, c_t], dim=1))
                    c_t_plus = model(torch.cat([c_wf, c_t + 5.0], dim=1))

                    loss, _ = criterion(
                        log_pred=log_pred,
                        log_target=by,
                        raw_res=raw_res,
                        colloc_log_pred=c_pred,
                        colloc_log_pred_wf_plus=c_wf_plus,
                        colloc_log_pred_t_plus=c_t_plus,
                    )
                    loss.backward()
                    optimizer.step()

                scheduler.step()

                # Validation step
                model.eval()
                with torch.no_grad():
                    v_pred = model(val_x_t)
                    # Huber loss on validation (V > 0)
                    val_loss = float(torch.mean(torch.abs(v_pred[:, 1:] - val_y_t[:, 1:])).item())

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone() for k, v in model.residual_mlp.state_dict().items()
                    }
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience:
                        break

            if best_state is not None:
                model.residual_mlp.load_state_dict(best_state)

            training_logs.append(
                {"member": m_idx, "seed": member_seed, "best_val_mae": best_val_loss}
            )

        return training_logs

    def predict(
        self,
        wf: float,
        temperature_k: float,
        t_d_um: float = 9.0,
        n_net_cm3: float = 6.0e15,
        d_um: float = 200.0,
    ) -> dict[str, Any]:
        """Predict I-V curves, ensemble uncertainty, physics curve, and verification flags."""
        v_grid = self.models[0].v_grid
        v_grid_np = v_grid.cpu().numpy()

        x_t = torch.tensor([[wf, temperature_k]], dtype=torch.float32)

        # 1. Forward pass across all M models
        log_preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                lp = model(x_t).squeeze(0).cpu().numpy()
                log_preds.append(lp)

        log_preds_arr = np.array(log_preds)  # [M, 52]

        # 2. Mean and std across ensemble in log space
        mean_log = np.mean(log_preds_arr, axis=0)  # [52]
        std_log = np.std(log_preds_arr, axis=0) if self.ensemble_size > 1 else np.zeros(52)

        i_mean = 10.0**mean_log
        i_std = std_log

        # 3. Analytic physics-only curve
        with torch.no_grad():
            wf_t = torch.tensor([wf], dtype=torch.float64)
            temp_t = torch.tensor([temperature_k], dtype=torch.float64)
            t_d_t = torch.tensor([t_d_um], dtype=torch.float64)
            n_net_t = torch.tensor([n_net_cm3], dtype=torch.float64)
            d_t = torch.tensor([d_um], dtype=torch.float64)
            i_phys = (
                self.physics_core.forward(
                    v_grid.to(torch.float64),
                    wf_t,
                    temp_t,
                    t_d_um=t_d_t,
                    n_net_cm3=n_net_t,
                    d_um=d_t,
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )

        # Scale i_mean by active area if diameter d_um deviates from baseline 200 um
        if abs(d_um - 200.0) > 1e-3:
            area_scale = (d_um / 200.0) ** 2
            i_mean = i_mean * area_scale

        # 4. Domain & Uncertainty Flags
        flags: list[str] = []
        if wf < 4.99 or wf > 5.19 or temperature_k < 300.0 or temperature_k > 500.0:
            flags.append("outside_box")
        if abs(t_d_um - 9.0) > 1e-3 or abs(n_net_cm3 - 6.0e15) > 1e14:
            flags.append("physics_extrapolated")

        # Check for high uncertainty in turn-on region [1e-6, 1e-2] A
        turn_on_mask = (mean_log >= -6.0) & (mean_log <= -2.0)
        if np.any(turn_on_mask) and np.max(std_log[turn_on_mask]) > self.uncertainty_threshold:
            flags.append("high_uncertainty")

        return {
            "V": v_grid_np.tolist(),
            "I_mean": i_mean.tolist(),
            "I_std": i_std.tolist(),
            "I_phys": i_phys.tolist(),
            "flags": flags,
        }

    def save_ensemble(self, save_dir: str | Path = "artifacts/ensemble") -> None:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        for idx, model in enumerate(self.models):
            ckpt_path = p / f"model_{idx}.pt"
            torch.save(model.residual_mlp.state_dict(), ckpt_path)
        print(f"Saved {self.ensemble_size} ensemble checkpoints to {p}")

    def load_ensemble(self, save_dir: str | Path = "artifacts/ensemble") -> None:
        p = Path(save_dir)
        for idx, model in enumerate(self.models):
            ckpt_path = p / f"model_{idx}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu")
            model.residual_mlp.load_state_dict(state)
            model.eval()
        print(f"Successfully loaded {self.ensemble_size} ensemble checkpoints from {p}")
