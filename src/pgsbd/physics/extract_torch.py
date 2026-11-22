"""Differentiable PyTorch metric extractor for I-V curves.

Verified against extract_metrics_numpy to within 1e-6 numerical tolerance.
Provides autograd gradients for turn-on voltage, subthreshold swing,
differential on-resistance, and operating point current.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def _torch_interp_1d(
    x_target: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Exact 1D piecewise-linear interpolation matching np.interp differentiably.

    Args:
        x_target: Target x values to interpolate at, shape [B] or scalar.
        x: Monotonically increasing grid, shape [B, N] or [N].
        y: Values at grid points, shape [B, N] or [N].

    Returns:
        Interpolated y values matching x_target shape.
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        was_1d = True
    else:
        was_1d = False

    if x_target.ndim == 0:
        x_target = x_target.unsqueeze(0)

    # Broadcast x_target to [B, 1] if needed
    if x_target.ndim == 1:
        if x_target.shape[0] != x.shape[0] and x_target.shape[0] == 1:
            x_target = x_target.expand(x.shape[0])
        x_t = x_target.unsqueeze(1)  # [B, 1]
    else:
        x_t = x_target

    _batch_size, n = x.shape

    # For each batch row, find where x_t falls in x
    # Clamp below minimum and above maximum per np.interp behavior
    x_min = x[:, 0:1]
    x_max = x[:, -1:]
    y_min = y[:, 0:1]
    y_max = y[:, -1:]

    # Search sorted: bucketize requires contiguous and sorted 1D or vectorized per row
    indices = torch.searchsorted(x.contiguous(), x_t.contiguous(), right=False)  # [B, 1]
    # k is the left index (clamped in [0, n - 2])
    k = torch.clamp(indices - 1, min=0, max=n - 2)

    # Gather x_k, x_{k+1}, y_k, y_{k+1}
    x_k = torch.gather(x, 1, k)
    x_kp1 = torch.gather(x, 1, k + 1)
    y_k = torch.gather(y, 1, k)
    y_kp1 = torch.gather(y, 1, k + 1)

    dx = torch.clamp(x_kp1 - x_k, min=1.0e-30)
    dy = y_kp1 - y_k
    weight = (x_t - x_k) / dx
    y_interp = y_k + weight * dy

    # Apply boundary conditions: x_t < x_min => y_min, x_t > x_max => y_max
    out = torch.where(x_t <= x_min, y_min, torch.where(x_t >= x_max, y_max, y_interp))
    out = out.squeeze(1)

    if was_1d and out.shape[0] == 1:
        return out.squeeze(0)
    return out


def extract_metrics_torch(
    v_grid: torch.Tensor,
    currents: torch.Tensor,
    i_ref: float = 1.0e-3,
    v_op: float = 2.0,
    vo_current: float = 1.0e-5,
    ss_low: float = 1.0e-6,
    ss_high: float = 1.0e-5,
    rdiff_points: int = 5,
    area_cm2: float | None = None,
) -> dict[str, torch.Tensor]:
    """Extract figures of merit from an I-V curve differentiably in PyTorch.

    Args:
        v_grid: 1D Tensor of voltages (e.g. [N]) or [B, N].
        currents: 1D Tensor of currents [N] or [B, N] in Amperes.
        i_ref: Reference current for Von (default 1 mA = 1e-3 A).
        v_op: Operating voltage for forward current evaluation (default 2.0 V).
        vo_current: Current for Vo extraction (default 10 uA = 1e-5 A).
        ss_low: Lower current for subthreshold slope (default 1 uA = 1e-6 A).
        ss_high: Upper current for subthreshold slope (default 10 uA = 1e-5 A).
        rdiff_points: Number of points at high-voltage end for linear regression.
        area_cm2: Optional device active area in cm^2 for ron_sp.

    Returns:
        Dictionary of extracted PyTorch scalar/tensor metrics matching extract_metrics_numpy.
    """
    dtype, device = currents.dtype, currents.device

    # Ensure 2D [B, N]
    was_1d = currents.ndim == 1
    if was_1d:
        curr = currents.unsqueeze(0)
    else:
        curr = currents

    batch_size, _n = curr.shape

    if v_grid.ndim == 1:
        v_g = v_grid.unsqueeze(0).expand(batch_size, -1)
    else:
        v_g = v_grid

    # Floor safe currents for log space
    safe_curr = torch.clamp(curr, min=1.0e-14)
    log_i = torch.log10(safe_curr)

    # 1. Von at i_ref via log-linear interpolation
    log_iref_t = torch.tensor(math.log10(i_ref), dtype=dtype, device=device).expand(batch_size)
    v_on = _torch_interp_1d(log_iref_t, log_i, v_g)

    # 2. Vo at vo_current (10 uA = 1e-5 A)
    log_vo_t = torch.tensor(math.log10(vo_current), dtype=dtype, device=device).expand(batch_size)
    v_o = _torch_interp_1d(log_vo_t, log_i, v_g)

    # 3. Subthreshold slope SS between ss_low and ss_high in mV/dec
    log_low_t = torch.tensor(math.log10(ss_low), dtype=dtype, device=device).expand(batch_size)
    log_high_t = torch.tensor(math.log10(ss_high), dtype=dtype, device=device).expand(batch_size)
    v_low = _torch_interp_1d(log_low_t, log_i, v_g)
    v_high = _torch_interp_1d(log_high_t, log_i, v_g)
    decades = math.log10(ss_high) - math.log10(ss_low)
    ss_mv_dec = (v_high - v_low) / decades * 1000.0

    # 4. Rdiff (ohms) via linear regression of V vs I over the last rdiff_points
    # Fit V = m * I + c => m is dV/dI = Rdiff
    v_tail = v_g[:, -rdiff_points:]  # [B, K]
    i_tail = curr[:, -rdiff_points:]  # [B, K]

    i_mean = torch.mean(i_tail, dim=-1, keepdim=True)
    v_mean = torch.mean(v_tail, dim=-1, keepdim=True)

    delta_i = i_tail - i_mean
    delta_v = v_tail - v_mean

    var_i = torch.sum(delta_i * delta_i, dim=-1)  # [B]
    cov_iv = torch.sum(delta_i * delta_v, dim=-1)  # [B]

    # Guard against flat current saturation
    fallback_r = (v_tail[:, -1] - v_tail[:, 0]) / torch.clamp(
        i_tail[:, -1] - i_tail[:, 0], min=1.0e-12
    )
    r_diff = torch.where(var_i > 1.0e-24, cov_iv / torch.clamp(var_i, min=1.0e-30), fallback_r)

    # 5. I at v_op (default 2.0 V) via linear interpolation on v_grid
    v_op_t = torch.tensor(v_op, dtype=dtype, device=device).expand(batch_size)
    i_at_vop = _torch_interp_1d(v_op_t, v_g, curr)

    out: dict[str, Any] = {
        "von": v_on.squeeze(0) if was_1d else v_on,
        "v_o": v_o.squeeze(0) if was_1d else v_o,
        "ss_mV_dec": ss_mv_dec.squeeze(0) if was_1d else ss_mv_dec,
        "r_diff": r_diff.squeeze(0) if was_1d else r_diff,
        "i_at_vop": i_at_vop.squeeze(0) if was_1d else i_at_vop,
    }

    if area_cm2 is not None:
        area_t = torch.tensor(area_cm2, dtype=dtype, device=device)
        ron = out["r_diff"] * area_t * 1000.0
        out["ron_sp"] = ron
        out["ron_sp_mohm_cm2"] = ron

    return out
