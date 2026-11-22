"""Bounded Physics-Guided Residual Surrogate Network.

Architecture:
    log10(I(V)) = log10(I_phys(V; WF, T)) + c * tanh(g(WF, T) / c)

Where:
    - I_phys is the differentiable SBD analytic physics core
    - g(WF, T) is a 2 -> 128 -> 128 -> 52 feedforward network with SiLU activations
    - c is the residual cap (default 0.7 decades)
    - Off-grid voltages supported via cubic spline interpolation in log space
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch import nn

from pgsbd.physics.sbd_core import SBDPhysicsCore


class ResidualMLP(nn.Module):
    """Residual neural network g(WF, T) with SiLU activations."""

    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 52,
        hidden_dims: tuple[int, ...] = (128, 128),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.SiLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicsGuidedSurrogate(nn.Module):
    """Hybrid surrogate combining differentiable SBD physics core with a bounded residual MLP."""

    def __init__(
        self,
        physics_core: SBDPhysicsCore | None = None,
        hidden_dims: tuple[int, ...] = (128, 128),
        residual_cap: float = 0.7,
        use_physics: bool = True,
        use_residual_cap: bool = True,
        v_grid: torch.Tensor | None = None,
    ):
        super().__init__()
        if physics_core is None:
            physics_core = SBDPhysicsCore()
        self.physics_core = physics_core
        self.residual_mlp = ResidualMLP(in_features=2, out_features=52, hidden_dims=hidden_dims)

        self.residual_cap = float(residual_cap)
        self.use_physics = bool(use_physics)
        self.use_residual_cap = bool(use_residual_cap)

        if v_grid is None:
            v_grid = torch.linspace(0.0, 4.0, 52, dtype=torch.float32)
        self.register_buffer("v_grid", v_grid)

        # Normalization buffers for [WF, T] inputs
        self.register_buffer("wf_min", torch.tensor(4.99, dtype=torch.float32))
        self.register_buffer("wf_max", torch.tensor(5.19, dtype=torch.float32))
        self.register_buffer("t_min", torch.tensor(300.0, dtype=torch.float32))
        self.register_buffer("t_max", torch.tensor(500.0, dtype=torch.float32))

    def normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Min-Max scale (WF, T) into [0, 1]."""
        wf = x[:, 0:1]
        t = x[:, 1:2]
        wf_norm = (wf - self.wf_min) / (self.wf_max - self.wf_min)
        t_norm = (t - self.t_min) / (self.t_max - self.t_min)
        return torch.cat([wf_norm, t_norm], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute forward predicted log10(I) across the 52 voltage points.

        Args:
            x: Input Tensor of shape [batch, 2] containing [WF, T].
            return_components: If True, return (log_pred, log_phys, raw_res).

        Returns:
            log_pred: Predicted log10(currents) of shape [batch, 52].
        """
        # Ensure float32
        x = x.to(torch.float32)
        wf = x[:, 0]
        temp = x[:, 1]

        # 1. Physics Core prediction
        if self.use_physics:
            # Evaluate physics core
            with torch.set_grad_enabled(self.training):
                i_phys = self.physics_core.forward(self.v_grid, wf, temp)
                log_phys = torch.log10(torch.clamp(i_phys, min=1e-14))
        else:
            log_phys = torch.zeros(x.shape[0], 52, device=x.device, dtype=torch.float32)

        # 2. Residual MLP prediction
        x_norm = self.normalize_inputs(x)
        raw_res = self.residual_mlp(x_norm)

        # 3. Apply residual cap c * tanh(g / c)
        if self.use_residual_cap:
            c = self.residual_cap
            bounded_res = c * torch.tanh(raw_res / c)
        else:
            bounded_res = raw_res

        # 4. Combine: log10(I) = log10(I_phys) + bounded_res
        if self.use_physics:
            log_pred = log_phys + bounded_res
        else:
            # Baseline mode: pure MLP directly predicts log10(I)
            log_pred = raw_res

        if return_components:
            return log_pred, log_phys, raw_res
        return log_pred

    def predict_numpy(
        self,
        x_np: np.ndarray,
        v_query: np.ndarray | None = None,
    ) -> np.ndarray:
        """Inference for NumPy inputs, with optional off-grid cubic-spline interpolation.

        Args:
            x_np: Array of shape [N, 2] with [WF, T].
            v_query: Optional 1D array of query voltages.

        Returns:
            Array of shape [N, len(v_query)] with linear currents (A).
        """
        self.eval()
        with torch.no_grad():
            x_t = torch.tensor(x_np, dtype=torch.float32)
            log_pred = self.forward(x_t).cpu().numpy()

        # If on-grid (v_query is None or exactly matches v_grid)
        v_grid_np = self.v_grid.cpu().numpy()
        if v_query is None or (len(v_query) == 52 and np.allclose(v_query, v_grid_np)):
            return 10.0**log_pred

        # Off-grid interpolation via cubic spline in log space
        n_samples = len(x_np)
        interp_currents = np.empty((n_samples, len(v_query)), dtype=np.float64)
        for i in range(n_samples):
            cs = CubicSpline(v_grid_np, log_pred[i], bc_type="natural")
            interp_currents[i] = 10.0 ** cs(v_query)

        return np.maximum(interp_currents, 1e-14)
