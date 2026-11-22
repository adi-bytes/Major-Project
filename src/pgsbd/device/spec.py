"""Device specification interface (DeviceSpec).

Defines an abstract schema for power semiconductor devices to ensure
extensibility beyond beta-Ga2O3 SBDs without hard-coding device assumptions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class VariableBounds:
    """Numerical range and scale for an input variable."""

    name: str
    symbol: str
    unit: str
    lower: float
    upper: float
    default: float
    scale: str  # "linear" or "log"
    tier: str  # "Tier A" or "Tier B"
    evidence_level: str  # "tcad_validated", "physics_extrapolated", "analytic_unvalidated"


class DeviceSpec(ABC):
    """Abstract device specification interface."""

    @property
    @abstractmethod
    def device_id(self) -> str:
        """Unique machine-readable identifier for the device plugin."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name."""

    @property
    @abstractmethod
    def material(self) -> str:
        """Primary semiconductor material."""

    @abstractmethod
    def get_variable_bounds(self) -> dict[str, VariableBounds]:
        """Return schema and bounds for all design and operating variables."""

    @abstractmethod
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
        """Compute forward I-V curve via differentiable analytic physics."""

    @abstractmethod
    def extract_metrics(
        self,
        v_grid: np.ndarray,
        currents: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Extract figures of merit from an I-V curve."""

    @abstractmethod
    def get_evidence_map(self) -> dict[str, str]:
        """Map each metric/variable to its evidence level."""
