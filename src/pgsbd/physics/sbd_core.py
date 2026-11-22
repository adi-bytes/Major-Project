"""Differentiable analytic forward current model for Schottky Barrier Diodes.

Equations (Section 5.1):
- Thermal voltage: V_t = k_B * T / q
- Effective barrier: phi_B = WF - chi_eff
- Saturation current: I_s = A_eff * A_richardson * T^2 * exp(-phi_B / V_t)
- Mobility: mu(T, N_net) = mu_300 * (T / 300)^(-alpha) / (1 + N_net / N_ref)^beta
- Drift resistance: R_drift = (t_D / (q * mu * N_net)) / A_eff
- Series resistance: R_s = R_drift + R_c
- Implicit diode equation with series resistance:
    I = I_s * [exp((V - I * R_s) / (n * V_t)) - 1]
Solved stably via closed-form Lambert-W in log domain with analytical autograd backward pass.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class StableLambertWLog(torch.autograd.Function):
    """Numerically stable Lambert-W function for w * exp(w) = exp(log_z).

    Computes w = W_0(exp(log_z)) where log_z is the natural log of z.
    Works for log_z spanning from -100 to +300 without overflow.
    Gradient dw / d(log_z) = w / (1 + w) is strictly bounded in [0, 1).
    """

    @staticmethod
    def forward(ctx: Any, log_z: torch.Tensor) -> torch.Tensor:
        # Convert to float64 for maximum precision during root finding
        orig_dtype = log_z.dtype
        x = log_z.to(torch.float64)

        # Asymptotic initial guess:
        # For large log_z: w ~ log_z - ln(log_z)
        # For moderate/small log_z: w ~ exp(log_z)
        w = torch.where(
            x > 2.0,
            x - torch.log(torch.clamp(x, min=1.0)),
            torch.exp(torch.clamp(x, max=2.0)),
        )
        w = torch.clamp(w, min=1e-30)

        # 5 Halley iterations: f(w) = w + ln(w) - x = 0
        for _ in range(5):
            ln_w = torch.log(w)
            f = w + ln_w - x
            f_prime = 1.0 + 1.0 / w
            f_second = -1.0 / (w * w)
            # Halley step: delta = f / (f_prime - 0.5 * f * f_second / f_prime)
            denom = f_prime - 0.5 * f * f_second / f_prime
            w = w - f / denom
            w = torch.clamp(w, min=1e-30)

        w_out = w.to(orig_dtype)
        ctx.save_for_backward(w_out)
        return w_out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (w,) = ctx.saved_tensors
        # dw / d(log_z) = w / (1.0 + w)
        grad_log_z = grad_output * (w / (1.0 + w))
        return grad_log_z


def lambert_w_log(log_z: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper for StableLambertWLog."""
    return StableLambertWLog.apply(log_z)


class SBDPhysicsCore(nn.Module):
    """Differentiable Analytic SBD Physics Model."""

    def __init__(
        self,
        chi_eff: float = 4.0,
        ideality_n: float = 1.05,
        A_eff_cm2: float = 1.0e-4,
        mu_300: float = 120.0,
        alpha: float = 1.7,
        R_c_ohm: float = 5.0,
        q: float = 1.602176634e-19,
        kB: float = 1.380649e-23,
        A_richardson: float = 34.0,
        N_ref: float = 1.0e17,
        beta: float = 0.6,
        default_t_d_um: float = 9.0,
        default_n_net_cm3: float = 6.0e15,
    ):
        super().__init__()
        self.chi_eff = float(chi_eff)
        self.ideality_n = float(ideality_n)
        self.A_eff_cm2 = float(A_eff_cm2)
        self.mu_300 = float(mu_300)
        self.alpha = float(alpha)
        self.R_c_ohm = float(R_c_ohm)

        self.q = float(q)
        self.kB = float(kB)
        self.A_richardson = float(A_richardson)
        self.N_ref = float(N_ref)
        self.beta = float(beta)
        self.default_t_d_um = float(default_t_d_um)
        self.default_n_net_cm3 = float(default_n_net_cm3)

    def forward(
        self,
        v_grid: torch.Tensor,
        wf: torch.Tensor,
        t_k: torch.Tensor,
        t_d_um: torch.Tensor | None = None,
        n_net_cm3: torch.Tensor | None = None,
        d_um: torch.Tensor | None = None,
        theta: dict[str, float | torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute forward currents I(V) across v_grid.

        Args:
            v_grid: 1D Tensor of voltages [num_voltages] or [batch, num_voltages].
            wf: 1D Tensor of effective work functions [batch].
            t_k: 1D Tensor of temperatures in Kelvin [batch].
            t_d_um: Optional drift thickness in um [batch].
            n_net_cm3: Optional net donor concentration in cm^-3 [batch].
            d_um: Optional anode diameter in um [batch].
            theta: Optional parameter overrides for calibration.

        Returns:
            Currents Tensor of shape [batch, num_voltages] in Amperes.
        """
        # Parameter overrides if provided
        chi_eff = theta.get("chi_eff", self.chi_eff) if theta else self.chi_eff
        n = theta.get("ideality_n", self.ideality_n) if theta else self.ideality_n
        A_eff = theta.get("A_eff_cm2", self.A_eff_cm2) if theta else self.A_eff_cm2
        mu_300 = theta.get("mu_300", self.mu_300) if theta else self.mu_300
        alpha = theta.get("alpha", self.alpha) if theta else self.alpha
        R_c = theta.get("R_c_ohm", self.R_c_ohm) if theta else self.R_c_ohm

        # Ensure parameters are tensors matching wf
        dtype, device = wf.dtype, wf.device
        if not isinstance(chi_eff, torch.Tensor):
            chi_eff = torch.tensor(float(chi_eff), dtype=dtype, device=device)
        if not isinstance(n, torch.Tensor):
            n = torch.tensor(float(n), dtype=dtype, device=device)
        if not isinstance(A_eff, torch.Tensor):
            A_eff = torch.tensor(float(A_eff), dtype=dtype, device=device)
        if not isinstance(mu_300, torch.Tensor):
            mu_300 = torch.tensor(float(mu_300), dtype=dtype, device=device)
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(float(alpha), dtype=dtype, device=device)
        if not isinstance(R_c, torch.Tensor):
            R_c = torch.tensor(float(R_c), dtype=dtype, device=device)

        if t_d_um is None:
            t_d_um = torch.full_like(wf, self.default_t_d_um)
        if n_net_cm3 is None:
            n_net_cm3 = torch.full_like(wf, self.default_n_net_cm3)

        # Scale A_eff if diameter d_um is provided relative to baseline 200 um
        if d_um is not None:
            # Area scales as (d / 200)^2
            area_scale = (d_um / 200.0) ** 2
            A_eff_actual = A_eff * area_scale
        else:
            A_eff_actual = A_eff

        # 1. Thermal voltage V_t = kB * T / q
        v_t = (self.kB * t_k) / self.q  # [batch]

        # 2. Barrier height phi_B = WF - chi_eff
        phi_b = wf - chi_eff  # [batch]

        # 3. Saturation current I_s = A_eff * A** * T^2 * exp(-phi_B / V_t)
        # In log space: ln(I_s) = ln(A_eff * A**) + 2 * ln(T) - phi_b / V_t
        log_is = (
            torch.log(torch.clamp(A_eff_actual * self.A_richardson, min=1e-30))
            + 2.0 * torch.log(t_k)
            - phi_b / v_t
        )
        i_s = torch.exp(torch.clamp(log_is, min=-100.0, max=10.0))  # [batch]

        # 4. Mobility and Drift resistance
        # mu = mu_300 * (T / 300)^(-alpha) / (1 + N_net / N_ref)^beta
        mu = (
            mu_300
            * torch.pow(t_k / 300.0, -alpha)
            / torch.pow(1.0 + n_net_cm3 / self.N_ref, self.beta)
        )
        # rho = 1 / (q * mu * N_net) in ohm*cm
        rho = 1.0 / (self.q * mu * n_net_cm3)
        # t_D in cm: t_d_um * 1e-4
        t_d_cm = t_d_um * 1e-4
        r_drift = (rho * t_d_cm) / A_eff_actual  # [batch]
        r_s = torch.clamp(r_drift + R_c, min=1e-6)  # [batch]

        # Ensure correct shapes for broadcasting with v_grid
        # wf, t_k are [batch, 1], v_grid is [1, num_voltages]
        if v_grid.ndim == 1:
            v_grid = v_grid.unsqueeze(0)  # [1, num_v]

        if wf.ndim == 1:
            wf = wf.unsqueeze(1)
            t_k = t_k.unsqueeze(1)
            v_t = v_t.unsqueeze(1)
            i_s = i_s.unsqueeze(1)
            log_is = log_is.unsqueeze(1)
            r_s = r_s.unsqueeze(1)

        # 5. Lambert-W implicit diode solution:
        # z = (I_s * R_s / (n * V_t)) * exp((V + I_s * R_s) / (n * V_t))
        # ln(z) = ln(I_s * R_s / (n * V_t)) + (V + I_s * R_s) / (n * V_t)
        n_vt = n * v_t
        is_rs = i_s * r_s
        term1 = torch.log(torch.clamp(is_rs / n_vt, min=1e-30))
        term2 = (v_grid + is_rs) / n_vt
        log_z = term1 + term2

        # Solve w = W_0(z)
        w = lambert_w_log(log_z)

        # I = (n * V_t / R_s) * w - I_s
        # When w * exp(w) = z, for large V: w ~ ln(z) ~ V / (n * V_t), so I ~ V / R_s
        current = (n_vt / r_s) * w - i_s

        # Guarantee physical bounds: non-negative, clamp minimum at numerical noise floor
        current = torch.clamp(current, min=1e-14)

        return current
