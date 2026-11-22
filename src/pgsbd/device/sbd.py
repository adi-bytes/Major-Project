"""Vertical beta-Ga2O3 Schottky Barrier Diode (SBD) Device Specification.

Device Plug-in #1 in the PG-SBD Framework.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pgsbd.config import load_config
from pgsbd.data.metrics import extract_metrics_numpy
from pgsbd.device.spec import DeviceSpec, VariableBounds


class BetaGa2O3SBD(DeviceSpec):
    """Plug-in #1: Vertical beta-Ga2O3 Schottky Barrier Diode."""

    def __init__(self, config: dict[str, Any] | None = None):
        if config is None:
            config = load_config()
        self.config = config
        self.bounds_cfg = config.get("bounds", {})
        self.base_cfg = config.get("baseline_device", {})

    @property
    def device_id(self) -> str:
        return "beta_ga2o3_sbd"

    @property
    def display_name(self) -> str:
        return "Vertical β-Ga₂O₃ Schottky Barrier Diode"

    @property
    def material(self) -> str:
        return "β-Ga₂O₃"

    def get_variable_bounds(self) -> dict[str, VariableBounds]:
        return {
            "WF": VariableBounds(
                name="Effective Anode Work Function",
                symbol="WF",
                unit="eV",
                lower=float(self.bounds_cfg.get("WF", [4.99, 5.19])[0]),
                upper=float(self.bounds_cfg.get("WF", [4.99, 5.19])[1]),
                default=5.05,
                scale="linear",
                tier="Tier A",
                evidence_level="tcad_validated",
            ),
            "temperature": VariableBounds(
                name="Operating Temperature",
                symbol="T",
                unit="K",
                lower=float(self.bounds_cfg.get("T", [300.0, 500.0])[0]),
                upper=float(self.bounds_cfg.get("T", [300.0, 500.0])[1]),
                default=300.0,
                scale="linear",
                tier="Tier A",
                evidence_level="tcad_validated",
            ),
            "t_D_um": VariableBounds(
                name="Drift Layer Thickness",
                symbol="t_D",
                unit="μm",
                lower=float(self.bounds_cfg.get("t_D_um", [3.0, 15.0])[0]),
                upper=float(self.bounds_cfg.get("t_D_um", [3.0, 15.0])[1]),
                default=float(self.base_cfg.get("t_D_um", 9.0)),
                scale="linear",
                tier="Tier B",
                evidence_level="physics_extrapolated",
            ),
            "N_net_cm3": VariableBounds(
                name="Net Donor Concentration",
                symbol="N_net",
                unit="cm⁻³",
                lower=float(self.bounds_cfg.get("N_net_cm3", [1.0e15, 5.0e16])[0]),
                upper=float(self.bounds_cfg.get("N_net_cm3", [1.0e15, 5.0e16])[1]),
                default=float(self.base_cfg.get("N_net_cm3", 6.0e15)),
                scale="log",
                tier="Tier B",
                evidence_level="physics_extrapolated",
            ),
            "d_um": VariableBounds(
                name="Anode Diameter",
                symbol="d",
                unit="μm",
                lower=float(self.bounds_cfg.get("d_um", [50.0, 500.0])[0]),
                upper=float(self.bounds_cfg.get("d_um", [50.0, 500.0])[1]),
                default=float(self.base_cfg.get("d_um", 200.0)),
                scale="linear",
                tier="Tier B",
                evidence_level="physics_extrapolated",
            ),
        }

    def forward_physics(
        self,
        v_grid: torch.Tensor,
        wf: torch.Tensor,
        t_k: torch.Tensor,
        t_d_um: torch.Tensor | None = None,
        n_net_cm3: torch.Tensor | None = None,
        d_um: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # Import lazily to avoid circular dependencies
        from pgsbd.physics.sbd_core import SBDPhysicsCore

        core = kwargs.get("core")
        if core is None:
            core = SBDPhysicsCore()
        return core.forward(
            v_grid=v_grid,
            wf=wf,
            t_k=t_k,
            t_d_um=t_d_um,
            n_net_cm3=n_net_cm3,
            d_um=d_um,
        )

    def extract_metrics(
        self,
        v_grid: np.ndarray,
        currents: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, float]:
        return extract_metrics_numpy(v_grid, currents, **kwargs)

    def get_evidence_map(self) -> dict[str, str]:
        return {
            "WF": "tcad_validated",
            "temperature": "tcad_validated",
            "t_D_um": "physics_extrapolated",
            "N_net_cm3": "physics_extrapolated",
            "d_um": "physics_extrapolated",
            "von": "tcad_validated",
            "v_o": "tcad_validated",
            "ss_mV_dec": "tcad_validated",
            "r_diff": "tcad_validated",
            "i_at_vop": "tcad_validated",
            "bv_est": "analytic_unvalidated",
            "ron_sp_mohm_cm2": "physics_extrapolated",
        }
