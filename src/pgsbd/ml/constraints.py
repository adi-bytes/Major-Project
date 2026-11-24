"""Physics-informed loss constraints for SBD surrogate training.

Loss Terms:
1. L_data: Huber loss on log10(I), up-weighted in turn-on region [1e-6, 1e-2] A.
2. L_mono:
   - Voltage monotonicity: penalty on d(log_I)/dV < 0.
   - Parameter monotonicity (empirically & physically verified):
     * dVo / dWF > 0 (higher barrier shifts turn-on voltage higher)
     * dVo / dT < 0 (higher temperature shifts turn-on voltage lower)
3. L_smooth: Second-difference roughness penalty along voltage grid.
4. L_res: Regularization on raw residual network output norm.
"""

from __future__ import annotations

import torch
from torch import nn


class SBDPhysicsLoss(nn.Module):
    """Composite loss function with data fidelity and physical regularization."""

    def __init__(
        self,
        lambda_mono: float = 0.1,
        lambda_smooth: float = 0.01,
        lambda_res: float = 0.01,
        v_grid: torch.Tensor | None = None,
    ):
        super().__init__()
        self.lambda_mono = float(lambda_mono)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_res = float(lambda_res)
        self.huber = nn.HuberLoss(delta=1.0, reduction="none")

        if v_grid is None:
            v_grid = torch.linspace(0.0, 4.0, 52, dtype=torch.float32)
        self.register_buffer("v_grid", v_grid)

    def forward(
        self,
        log_pred: torch.Tensor,
        log_target: torch.Tensor,
        raw_res: torch.Tensor,
        colloc_log_pred: torch.Tensor | None = None,
        colloc_log_pred_wf_plus: torch.Tensor | None = None,
        colloc_log_pred_t_plus: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute composite loss and individual components.

        Args:
            log_pred: Predicted log10(currents) [batch, 52].
            log_target: Ground truth log10(currents) [batch, 52].
            raw_res: Raw MLP residual output g(WF, T) [batch, 52].
            colloc_log_pred: Log-currents for random collocation batch [N_colloc, 52].
            colloc_log_pred_wf_plus: Log-currents with WF + dWF [N_colloc, 52].
            colloc_log_pred_t_plus: Log-currents with T + dT [N_colloc, 52].

        Returns:
            total_loss: Scalar torch Tensor.
            loss_dict: Dictionary of float loss components.
        """
        # 1. L_data: Huber loss with turn-on weighting
        # Exclude col 0 (V=0)
        p_vgt0 = log_pred[:, 1:]
        t_vgt0 = log_target[:, 1:]

        # Weighting: 2.0x weight in turn-on/subthreshold region [-6.0, -2.0]
        weights = torch.ones_like(t_vgt0)
        turn_on_mask = (t_vgt0 >= -6.0) & (t_vgt0 <= -2.0)
        weights[turn_on_mask] = 2.0

        elem_loss = self.huber(p_vgt0, t_vgt0)
        l_data = torch.mean(weights * elem_loss)

        # 2. L_mono: Voltage monotonicity
        # Diff along voltage: log_pred[:, k+1] - log_pred[:, k] should be >= 0
        v_diffs = log_pred[:, 2:] - log_pred[:, 1:-1]
        # Penalize negative slope
        l_mono_v = torch.mean(torch.relu(-v_diffs) ** 2)

        # Parameter monotonicity via finite differences on collocation batch
        l_mono_param = torch.tensor(0.0, device=log_pred.device, dtype=log_pred.dtype)
        if (
            colloc_log_pred is not None
            and colloc_log_pred_wf_plus is not None
            and colloc_log_pred_t_plus is not None
        ):
            # As WF increases, barrier increases -> current at fixed V MUST decrease (d log_I / dWF < 0)
            diff_wf = colloc_log_pred_wf_plus[:, 1:] - colloc_log_pred[:, 1:]
            pen_wf = torch.mean(torch.relu(diff_wf) ** 2)

            # In turn-on region, as T increases, thermionic emission increases -> current increases (d log_I / dT > 0)
            mask_sub = (colloc_log_pred[:, 1:] >= -6.0) & (colloc_log_pred[:, 1:] <= -2.0)
            if torch.any(mask_sub):
                diff_t = colloc_log_pred_t_plus[:, 1:] - colloc_log_pred[:, 1:]
                pen_t = torch.mean(torch.relu(-diff_t[mask_sub]) ** 2)
            else:
                pen_t = torch.tensor(0.0, device=log_pred.device)

            l_mono_param = pen_wf + pen_t

        l_mono = l_mono_v + l_mono_param

        # 3. L_smooth: Second differences along voltage (roughness penalty)
        d2 = log_pred[:, 3:] - 2.0 * log_pred[:, 2:-1] + log_pred[:, 1:-2]
        l_smooth = torch.mean(d2**2)

        # 4. L_res: Norm of raw neural residual g
        l_res = torch.mean(raw_res**2)

        total_loss = (
            l_data
            + self.lambda_mono * l_mono
            + self.lambda_smooth * l_smooth
            + self.lambda_res * l_res
        )

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "l_data": float(l_data.item()),
            "l_mono": float(l_mono.item()),
            "l_smooth": float(l_smooth.item()),
            "l_res": float(l_res.item()),
        }

        return total_loss, loss_dict
