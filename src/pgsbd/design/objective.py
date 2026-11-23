"""Design requirements and objective functions for Tier B Inverse Design."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from pgsbd.physics.breakdown import Breakdown1D
from pgsbd.physics.extract_torch import extract_metrics_torch
from pgsbd.physics.sbd_core import SBDPhysicsCore


@dataclass
class DesignRequirements:
    """Target specifications and tolerance constraints for device synthesis."""

    target_v_br: float | None = None  # Minimum breakdown voltage (V)
    target_r_on_sp: float | None = None  # Maximum specific on-resistance (mOhm*cm^2)
    target_v_o: float | None = None  # Maximum turn-on voltage (V)
    target_ss: float | None = None  # Maximum subthreshold swing (mV/dec)
    target_i_on_4v: float | None = None  # Minimum forward current at 4V (A)
    tolerance: float = 0.10  # Acceptance tolerance (e.g. 10% = 0.10)
    fixed_temperature: float | None = 300.0  # Optional fixed operating temperature (K)
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "v_br": 1.0,
            "r_on_sp": 1.0,
            "v_o": 0.5,
            "ss": 0.2,
            "i_on": 0.5,
        }
    )


# Normalized parameter bounds for Tier B: [WF, T, t_D, log10(N_net), d]
PARAM_BOUNDS = {
    "WF": (4.99, 5.19),
    "T": (300.0, 500.0),
    "t_D_um": (3.0, 15.0),
    "log_N_net": (15.0, 16.7),  # 1.0e15 to 5.0e16
    "d_um": (50.0, 500.0),
}


def unit_to_physical(u: np.ndarray | torch.Tensor) -> dict[str, Any]:
    """Convert normalized unit coordinates [0, 1]^5 to physical device parameters."""
    if isinstance(u, torch.Tensor):
        wf = PARAM_BOUNDS["WF"][0] + u[..., 0] * (PARAM_BOUNDS["WF"][1] - PARAM_BOUNDS["WF"][0])
        t = PARAM_BOUNDS["T"][0] + u[..., 1] * (PARAM_BOUNDS["T"][1] - PARAM_BOUNDS["T"][0])
        t_d = PARAM_BOUNDS["t_D_um"][0] + u[..., 2] * (
            PARAM_BOUNDS["t_D_um"][1] - PARAM_BOUNDS["t_D_um"][0]
        )
        log_n = PARAM_BOUNDS["log_N_net"][0] + u[..., 3] * (
            PARAM_BOUNDS["log_N_net"][1] - PARAM_BOUNDS["log_N_net"][0]
        )
        n_net = torch.pow(10.0, log_n)
        d_um = PARAM_BOUNDS["d_um"][0] + u[..., 4] * (
            PARAM_BOUNDS["d_um"][1] - PARAM_BOUNDS["d_um"][0]
        )
        return {"WF": wf, "T": t, "t_D_um": t_d, "N_net_cm3": n_net, "d_um": d_um}

    u_arr = np.asarray(u, dtype=float)
    wf = PARAM_BOUNDS["WF"][0] + u_arr[..., 0] * (PARAM_BOUNDS["WF"][1] - PARAM_BOUNDS["WF"][0])
    t = PARAM_BOUNDS["T"][0] + u_arr[..., 1] * (PARAM_BOUNDS["T"][1] - PARAM_BOUNDS["T"][0])
    t_d = PARAM_BOUNDS["t_D_um"][0] + u_arr[..., 2] * (
        PARAM_BOUNDS["t_D_um"][1] - PARAM_BOUNDS["t_D_um"][0]
    )
    log_n = PARAM_BOUNDS["log_N_net"][0] + u_arr[..., 3] * (
        PARAM_BOUNDS["log_N_net"][1] - PARAM_BOUNDS["log_N_net"][0]
    )
    n_net = np.power(10.0, log_n)
    d_um = PARAM_BOUNDS["d_um"][0] + u_arr[..., 4] * (
        PARAM_BOUNDS["d_um"][1] - PARAM_BOUNDS["d_um"][0]
    )
    return {"WF": wf, "T": t, "t_D_um": t_d, "N_net_cm3": n_net, "d_um": d_um}


def physical_to_unit(wf: float, t: float, t_d: float, n_net: float, d_um: float) -> np.ndarray:
    """Convert physical parameters to normalized unit coordinates [0, 1]^5."""
    u0 = (wf - PARAM_BOUNDS["WF"][0]) / (PARAM_BOUNDS["WF"][1] - PARAM_BOUNDS["WF"][0])
    u1 = (t - PARAM_BOUNDS["T"][0]) / (PARAM_BOUNDS["T"][1] - PARAM_BOUNDS["T"][0])
    u2 = (t_d - PARAM_BOUNDS["t_D_um"][0]) / (PARAM_BOUNDS["t_D_um"][1] - PARAM_BOUNDS["t_D_um"][0])
    log_n = np.log10(max(n_net, 1.0e14))
    u3 = (log_n - PARAM_BOUNDS["log_N_net"][0]) / (
        PARAM_BOUNDS["log_N_net"][1] - PARAM_BOUNDS["log_N_net"][0]
    )
    u4 = (d_um - PARAM_BOUNDS["d_um"][0]) / (PARAM_BOUNDS["d_um"][1] - PARAM_BOUNDS["d_um"][0])
    return np.clip(np.array([u0, u1, u2, u3, u4], dtype=float), 0.0, 1.0)


class DesignEvaluator:
    """Forward physics evaluator that evaluates candidates against design requirements."""

    def __init__(
        self,
        physics_core: SBDPhysicsCore | None = None,
        breakdown_calc: Breakdown1D | None = None,
        calib_theta: dict[str, float] | None = None,
    ):
        self.core = physics_core or SBDPhysicsCore()
        self.breakdown = breakdown_calc or Breakdown1D()
        self.calib_theta = calib_theta or {
            "chi_eff": 3.9841,
            "ideality_n": 1.0463,
            "A_eff_cm2": 1.711e-4,
            "mu_300": 250.0,
            "alpha": 1.4047,
            "R_c_ohm": 6.5e-11,
        }
        self.v_grid_np = np.linspace(0.0, 4.0, 52)
        self.v_grid_torch = torch.tensor(self.v_grid_np, dtype=torch.float64)

    def evaluate_candidate(
        self,
        wf: float,
        temperature_k: float,
        t_d_um: float,
        n_net_cm3: float,
        d_um: float,
    ) -> dict[str, Any]:
        """Compute all forward and breakdown figures of merit for a candidate design."""
        # 1. 1D breakdown calculation
        bd_res = self.breakdown.calculate_numpy(
            t_d_um=t_d_um,
            n_net_cm3=n_net_cm3,
            mu_cm2_vs=float(self.calib_theta.get("mu_300", 250.0)),
        )

        # 2. Forward I-V curve via SBD physics core
        # Active anode area in cm^2 from diameter d_um: Area = pi * (d/2 * 1e-4)^2
        radius_cm = (d_um / 2.0) * 1.0e-4
        anode_area_cm2 = math.pi * (radius_cm**2)

        # Update theta with actual area
        theta_cand = dict(self.calib_theta)
        theta_cand["A_eff_cm2"] = anode_area_cm2

        with torch.no_grad():
            wf_t = torch.tensor([wf], dtype=torch.float64)
            t_t = torch.tensor([temperature_k], dtype=torch.float64)
            td_t = torch.tensor([t_d_um], dtype=torch.float64)
            nn_t = torch.tensor([n_net_cm3], dtype=torch.float64)

            i_curve = self.core.forward(
                v_grid=self.v_grid_torch,
                wf=wf_t,
                t_k=t_t,
                t_d_um=td_t,
                n_net_cm3=nn_t,
                theta=theta_cand,
            )
            i_curve_np = i_curve.squeeze(0).cpu().numpy()

            fom = extract_metrics_torch(
                v_grid=self.v_grid_torch,
                currents=i_curve,
                area_cm2=anode_area_cm2,
            )

        r_diff = float(fom["r_diff"].item())
        ron_sp = float(fom["ron_sp_mohm_cm2"].item())
        v_o = float(fom["v_o"].item())
        ss = float(fom["ss_mV_dec"].item())
        i_on_4v = float(i_curve_np[-1])
        i_leak_0v = float(i_curve_np[0])

        return {
            "parameters": {
                "WF": float(wf),
                "temperature": float(temperature_k),
                "t_D_um": float(t_d_um),
                "N_net_cm3": float(n_net_cm3),
                "d_um": float(d_um),
                "anode_area_cm2": float(anode_area_cm2),
            },
            "metrics": {
                "v_br": float(bd_res["v_br"]),
                "v_br_ideal": float(bd_res["v_br_ideal"]),
                "regime": str(bd_res["regime"]),
                "w_dep_um": float(bd_res["w_dep_um"]),
                "bfom_mw_cm2": float(bd_res["bfom_mw_cm2"]),
                "r_on_sp": float(ron_sp),
                "r_diff": float(r_diff),
                "v_o": float(v_o),
                "ss_mV_dec": float(ss),
                "i_on_4V": float(i_on_4v),
                "i_leak_0V": float(i_leak_0v),
            },
            "iv_curve": {
                "V": self.v_grid_np.tolist(),
                "I": i_curve_np.tolist(),
            },
            "evidence_tags": {
                "v_br": "analytic_unvalidated",
                "r_on_sp": "physics_extrapolated",
                "v_o": "tcad_validated"
                if (4.99 <= wf <= 5.19 and 300 <= temperature_k <= 500)
                else "physics_extrapolated",
                "ss_mV_dec": "tcad_validated"
                if (4.99 <= wf <= 5.19 and 300 <= temperature_k <= 500)
                else "physics_extrapolated",
                "i_on_4V": "physics_extrapolated",
                "t_D_um": "physics_extrapolated",
                "N_net_cm3": "physics_extrapolated",
                "d_um": "physics_extrapolated",
            },
        }

    def compute_loss_torch(
        self,
        u: torch.Tensor,
        reqs: DesignRequirements,
    ) -> torch.Tensor:
        """Vectorized differentiable loss function for batch u in [B, 5]."""
        p = unit_to_physical(u)
        wf = p["WF"]
        t = torch.full_like(wf, reqs.fixed_temperature) if reqs.fixed_temperature else p["T"]
        t_d = p["t_D_um"]
        n_net = p["N_net_cm3"]
        d_um = p["d_um"]

        # 1. Breakdown
        bd = self.breakdown.calculate_torch(t_d, n_net)
        v_br = bd["v_br"]

        # 2. Forward I-V & metrics
        radius_cm = (d_um / 2.0) * 1.0e-4
        anode_area_cm2 = math.pi * (radius_cm**2)

        theta_cand = dict(self.calib_theta)
        # Note: A_eff is scaled per candidate
        i_curve = self.core.forward(
            v_grid=self.v_grid_torch,
            wf=wf,
            t_k=t,
            t_d_um=t_d,
            n_net_cm3=n_net,
            d_um=d_um,
            theta=theta_cand,
        )
        fom = extract_metrics_torch(
            v_grid=self.v_grid_torch,
            currents=i_curve,
            area_cm2=1.711e-4,  # baseline normalization
        )
        r_diff = fom["r_diff"]
        ron_sp = r_diff * anode_area_cm2 * 1000.0
        v_o = fom["v_o"]
        ss = fom["ss_mV_dec"]
        i_on = i_curve[:, -1]

        loss = torch.zeros_like(wf)

        # Loss terms based on user targets (penalize violation)
        if reqs.target_v_br is not None:
            # We want v_br >= target_v_br
            rel_err = torch.relu((reqs.target_v_br - v_br) / max(reqs.target_v_br, 1.0))
            loss = loss + reqs.weights.get("v_br", 1.0) * (rel_err**2)

        if reqs.target_r_on_sp is not None:
            # We want ron_sp <= target_r_on_sp
            rel_err = torch.relu((ron_sp - reqs.target_r_on_sp) / max(reqs.target_r_on_sp, 0.1))
            loss = loss + reqs.weights.get("r_on_sp", 1.0) * (rel_err**2)

        if reqs.target_v_o is not None:
            # We want v_o <= target_v_o
            rel_err = torch.relu((v_o - reqs.target_v_o) / max(reqs.target_v_o, 0.1))
            loss = loss + reqs.weights.get("v_o", 0.5) * (rel_err**2)

        if reqs.target_ss is not None:
            # We want ss <= target_ss
            rel_err = torch.relu((ss - reqs.target_ss) / max(reqs.target_ss, 10.0))
            loss = loss + reqs.weights.get("ss", 0.2) * (rel_err**2)

        if reqs.target_i_on_4v is not None:
            # We want i_on >= target_i_on_4v
            rel_err = torch.relu((reqs.target_i_on_4v - i_on) / max(reqs.target_i_on_4v, 1e-4))
            loss = loss + reqs.weights.get("i_on", 0.5) * (rel_err**2)

        return loss
