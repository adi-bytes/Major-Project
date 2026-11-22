"""1-D Parallel-Plane Breakdown Model for Vertical beta-Ga2O3 SBD.

Evidence Level: analytic_unvalidated
Note: This analytical model provides the ideal parallel-plane physical limits
for breakdown voltage and specific on-resistance. Real devices typically achieve
50-80% of ideal 1D breakdown due to 2D/3D contact edge electric field crowding.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


class Breakdown1D:
    """Analytical 1D parallel-plane breakdown calculator for beta-Ga2O3.

    Physics:
    - Relative permittivity: epsilon_r = 10.0
    - Permittivity: epsilon_s = epsilon_r * epsilon_0 = 10.0 * 8.8541878e-14 F/cm
    - Critical electric field: E_c0 = 8.0e6 V/cm (8.0 MV/cm) with optional doping scaling:
        E_c(N_net) = E_c0 * (N_net / 1e16)^0.1
    - Non-punch-through (NPT) critical depletion width:
        W_c = epsilon_s * E_c / (q * N_net)
    - If W_c <= t_D: Non-punch-through regime
        V_br = (epsilon_s * E_c^2) / (2 * q * N_net) - V_bi
        W_dep = W_c
    - If W_c > t_D: Punch-through (PT) regime
        V_br = E_c * t_D - (q * N_net * t_D^2) / (2 * epsilon_s) - V_bi
        W_dep = t_D
    - Baliga Figure of Merit (BFOM):
        BFOM = V_br^2 / R_on_sp
    """

    EPSILON_0: float = 8.854187817e-14  # F/cm
    EPSILON_R: float = 10.0  # beta-Ga2O3 relative permittivity
    Q: float = 1.602176634e-19  # Coulombs
    DEFAULT_EC: float = 8.0e6  # V/cm (8 MV/cm)
    V_BI: float = 1.2  # Built-in potential in Volts
    EVIDENCE_LEVEL: str = "analytic_unvalidated"

    def __init__(
        self,
        epsilon_r: float = 10.0,
        e_c0: float = 8.0e6,
        v_bi: float = 1.2,
        scale_ec_with_doping: bool = True,
    ):
        self.epsilon_r = float(epsilon_r)
        self.epsilon_s = self.epsilon_r * self.EPSILON_0
        self.e_c0 = float(e_c0)
        self.v_bi = float(v_bi)
        self.scale_ec = scale_ec_with_doping

    def compute_ec(self, n_net_cm3: float | np.ndarray) -> float | np.ndarray:
        """Compute critical electric field with optional weak doping dependence."""
        if not self.scale_ec:
            if isinstance(n_net_cm3, np.ndarray):
                return np.full_like(n_net_cm3, self.e_c0)
            return self.e_c0
        # Weak empirical power-law dependence: E_c ~ (N_net / 1e16)^0.1 clamped in [5e6, 9e6]
        if isinstance(n_net_cm3, np.ndarray):
            scale = np.power(np.maximum(n_net_cm3, 1e14) / 1.0e16, 0.1)
            return np.clip(self.e_c0 * scale, 5.0e6, 9.5e6)
        scale = math.pow(max(float(n_net_cm3), 1e14) / 1.0e16, 0.1)
        return min(max(self.e_c0 * scale, 5.0e6), 9.5e6)

    def calculate_numpy(
        self,
        t_d_um: float | np.ndarray,
        n_net_cm3: float | np.ndarray,
        mu_cm2_vs: float | np.ndarray = 200.0,
    ) -> dict[str, Any]:
        """Compute 1D breakdown metrics for NumPy scalars or arrays.

        Args:
            t_d_um: Drift layer thickness in micrometers.
            n_net_cm3: Net donor concentration in cm^-3.
            mu_cm2_vs: Electron mobility in cm^2/(V*s).

        Returns:
            Dictionary with v_br, regime, w_dep_um, r_drift_sp_mohm_cm2, bfom_mw_cm2.
        """
        t_d_cm = np.asarray(t_d_um, dtype=np.float64) * 1.0e-4
        n_net = np.maximum(np.asarray(n_net_cm3, dtype=np.float64), 1.0e14)
        mu = np.maximum(np.asarray(mu_cm2_vs, dtype=np.float64), 1.0)

        e_c = self.compute_ec(n_net)
        q = self.Q
        eps_s = self.epsilon_s

        # Critical depletion width at Ec: W_c = eps_s * E_c / (q * N_net)
        w_c_cm = (eps_s * e_c) / (q * n_net)

        # Regimes: W_c <= t_d_cm is Non-Punch-Through (NPT); W_c > t_d_cm is Punch-Through (PT)
        is_npt = w_c_cm <= t_d_cm

        # NPT breakdown voltage: V_br_npt = 0.5 * eps_s * E_c^2 / (q * N_net) - V_bi
        v_br_npt = 0.5 * (eps_s * np.square(e_c)) / (q * n_net) - self.v_bi

        # PT breakdown voltage: V_br_pt = E_c * t_D - 0.5 * (q * N_net * t_D^2) / eps_s - V_bi
        v_br_pt = e_c * t_d_cm - 0.5 * (q * n_net * np.square(t_d_cm)) / eps_s - self.v_bi

        v_br = np.where(is_npt, v_br_npt, v_br_pt)
        v_br = np.maximum(v_br, 0.0)

        w_dep_cm = np.where(is_npt, w_c_cm, t_d_cm)
        w_dep_um = w_dep_cm * 1.0e4

        # Specific drift resistance in ohm*cm^2: R_drift_sp = t_D / (q * mu * N_net)
        r_drift_sp = t_d_cm / (q * mu * n_net)  # ohm*cm^2
        r_drift_sp_mohm_cm2 = r_drift_sp * 1000.0  # mOhm*cm^2

        # Baliga Figure of Merit in MW/cm^2: BFOM = V_br^2 / R_drift_sp in W/cm^2 -> / 1e6 for MW/cm^2
        bfom_mw_cm2 = (np.square(v_br) / np.maximum(r_drift_sp, 1e-12)) / 1.0e6

        regime = np.where(is_npt, "non_punch_through", "punch_through")

        if np.ndim(v_br) == 0:
            return {
                "v_br": float(v_br),
                "regime": str(regime),
                "w_dep_um": float(w_dep_um),
                "w_c_um": float(w_c_cm * 1.0e4),
                "e_c_v_cm": float(e_c),
                "r_drift_sp_mohm_cm2": float(r_drift_sp_mohm_cm2),
                "bfom_mw_cm2": float(bfom_mw_cm2),
                "evidence_level": self.EVIDENCE_LEVEL,
            }

        return {
            "v_br": v_br,
            "regime": regime,
            "w_dep_um": w_dep_um,
            "w_c_um": w_c_cm * 1.0e4,
            "e_c_v_cm": e_c,
            "r_drift_sp_mohm_cm2": r_drift_sp_mohm_cm2,
            "bfom_mw_cm2": bfom_mw_cm2,
            "evidence_level": self.EVIDENCE_LEVEL,
        }

    def calculate_torch(
        self,
        t_d_um: torch.Tensor,
        n_net_cm3: torch.Tensor,
        mu_cm2_vs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute 1D breakdown metrics differentiably in PyTorch.

        Args:
            t_d_um: Drift layer thickness in micrometers [batch].
            n_net_cm3: Net donor concentration in cm^-3 [batch].
            mu_cm2_vs: Optional electron mobility in cm^2/(V*s) [batch].

        Returns:
            Dictionary with v_br, w_dep_um, r_drift_sp_mohm_cm2, bfom_mw_cm2 tensors.
        """
        dtype, device = t_d_um.dtype, t_d_um.device
        if mu_cm2_vs is None:
            mu_cm2_vs = torch.full_like(t_d_um, 200.0)

        t_d_cm = t_d_um * 1.0e-4
        n_net = torch.clamp(n_net_cm3, min=1.0e14)
        mu = torch.clamp(mu_cm2_vs, min=1.0)

        q = torch.tensor(self.Q, dtype=dtype, device=device)
        eps_s = torch.tensor(self.epsilon_s, dtype=dtype, device=device)
        e_c0 = torch.tensor(self.e_c0, dtype=dtype, device=device)
        v_bi = torch.tensor(self.v_bi, dtype=dtype, device=device)

        if self.scale_ec:
            scale = torch.pow(n_net / 1.0e16, 0.1)
            e_c = torch.clamp(e_c0 * scale, min=5.0e6, max=9.5e6)
        else:
            e_c = e_c0

        # W_c = eps_s * E_c / (q * N_net)
        w_c_cm = (eps_s * e_c) / (q * n_net)

        # Smooth soft selection between NPT and PT:
        # NPT: 0.5 * eps_s * E_c^2 / (q * N_net) - V_bi
        # PT: E_c * t_D - 0.5 * q * N_net * t_D^2 / eps_s - V_bi
        v_br_npt = 0.5 * (eps_s * torch.square(e_c)) / (q * n_net) - v_bi
        v_br_pt = e_c * t_d_cm - 0.5 * (q * n_net * torch.square(t_d_cm)) / eps_s - v_bi

        # Condition: W_c <= t_d_cm => NPT
        # Using exact torch.where for non-differentiable boundary, but branches are differentiable
        is_npt = w_c_cm <= t_d_cm
        v_br = torch.where(is_npt, v_br_npt, v_br_pt)
        v_br = torch.clamp(v_br, min=0.0)

        w_dep_cm = torch.where(is_npt, w_c_cm, t_d_cm)
        w_dep_um = w_dep_cm * 1.0e4

        r_drift_sp = t_d_cm / (q * mu * n_net)  # ohm*cm^2
        r_drift_sp_mohm_cm2 = r_drift_sp * 1000.0  # mOhm*cm^2

        bfom_mw_cm2 = (torch.square(v_br) / torch.clamp(r_drift_sp, min=1e-12)) / 1.0e6

        return {
            "v_br": v_br,
            "w_dep_um": w_dep_um,
            "w_c_um": w_c_cm * 1.0e4,
            "e_c_v_cm": e_c,
            "r_drift_sp_mohm_cm2": r_drift_sp_mohm_cm2,
            "bfom_mw_cm2": bfom_mw_cm2,
        }
