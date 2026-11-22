"""Pydantic v2 schemas for PG-SBD Designer FastAPI API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

HONESTY_NOTICE = (
    "Research prototype trained on TCAD simulation data. "
    "Not validated for fabrication or experimental device design."
)


class BaseSchema(BaseModel):
    """Base schema with strict forbid-extra configuration."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(BaseSchema):
    """Health check response schema."""

    status: str = Field("ok", description="Service status")
    device: str = Field(..., description="Device plug-in name")
    evidence_level: str = Field("tcad_validated", description="Current evidence level tag")
    artifacts_loaded: dict[str, bool] = Field(..., description="Loaded artifacts status")
    honesty_notice: str = Field(HONESTY_NOTICE, description="Mandatory honesty notice")


class VariableBound(BaseSchema):
    """Range and unit for an input variable."""

    name: str
    unit: str
    min_val: float
    max_val: float
    is_sweepable: bool


class DeviceMetaResponse(BaseSchema):
    """Device metadata and operational bounds."""

    device_name: str
    material: str
    description: str
    evidence_level: str = "tcad_validated"
    variables: list[VariableBound]
    fixed_parameters: dict[str, Any]
    voltage_grid: dict[str, float | int]
    honesty_notice: str = HONESTY_NOTICE


class PredictRequest(BaseSchema):
    """Inference request for forward I-V prediction."""

    wf: float = Field(
        ...,
        ge=4.99,
        le=5.19,
        description="Anode work function (eV). Allowed range: [4.99, 5.19].",
    )
    temperature: float = Field(
        ...,
        ge=300.0,
        le=500.0,
        description="Lattice temperature (K). Allowed range: [300.0, 500.0].",
    )
    v_custom: list[float] | None = Field(
        default=None,
        description="Optional custom voltage evaluation grid (V). Must be non-empty and non-negative.",
    )

    @model_validator(mode="after")
    def validate_v_custom(self) -> PredictRequest:
        if self.v_custom is not None:
            if len(self.v_custom) == 0:
                raise ValueError("v_custom must not be empty if provided.")
            if any(v < 0.0 for v in self.v_custom):
                raise ValueError("All voltages in v_custom must be non-negative (>= 0.0 V).")
            if any(v > 10.0 for v in self.v_custom):
                raise ValueError("Voltages in v_custom must not exceed 10.0 V.")
        return self


class ExtractedMetrics(BaseSchema):
    """Key device figures of merit."""

    v_o: float = Field(..., description="Turn-on voltage at 10 uA (V)")
    r_on_sp: float = Field(..., description="Specific on-resistance (mOhm-cm^2)")
    ss_mV_dec: float = Field(..., description="Subthreshold swing (mV/decade)")
    i_on_4V: float = Field(..., description="Forward current at 4 V (A)")
    i_leak_0V: float = Field(..., description="Leakage current at 0 V (A)")


class PredictResponse(BaseSchema):
    """Forward I-V prediction response with uncertainty and physics baseline."""

    V: list[float] = Field(..., description="Voltage points (V)")
    I_mean: list[float] = Field(..., description="Ensemble mean current (A)")
    I_std: list[float] = Field(..., description="Ensemble current standard deviation (A)")
    I_lower: list[float] = Field(..., description="Lower 95% confidence interval current (A)")
    I_upper: list[float] = Field(..., description="Upper 95% confidence interval current (A)")
    I_phys: list[float] = Field(..., description="Analytic physics core current (A)")
    metrics: ExtractedMetrics = Field(..., description="Extracted figures of merit")
    flags: list[str] = Field(
        default_factory=list, description="Verifier and uncertainty warning flags"
    )
    evidence_level: str = Field("tcad_validated", description="Evidence tier")
    honesty_notice: str = Field(HONESTY_NOTICE, description="Mandatory honesty notice")


class ReverseRequest(BaseSchema):
    """Reverse Mode request for parameter recovery from an input I-V curve."""

    v: list[float] = Field(..., min_length=2, description="Voltage array (V). Min length 2.")
    i: list[float] = Field(..., min_length=2, description="Current array (A). Min length 2.")

    @model_validator(mode="after")
    def validate_arrays(self) -> ReverseRequest:
        if len(self.v) != len(self.i):
            raise ValueError(
                f"Array length mismatch: v has {len(self.v)} points while i has {len(self.i)} points."
            )
        if any(v_val < 0.0 for v_val in self.v):
            raise ValueError("All voltage values in 'v' must be non-negative (>= 0.0 V).")
        if any(i_val <= 0.0 for i_val in self.i):
            raise ValueError("All current values in 'i' must be strictly positive (> 0.0 A).")
        return self


class ParameterRecovery(BaseSchema):
    """Recovered device parameters with uncertainty intervals."""

    WF: float = Field(..., description="Estimated anode work function (eV)")
    T: float = Field(..., description="Estimated temperature (K)")
    WF_uncertainty: float = Field(..., description="WF half-width uncertainty interval (eV)")
    T_uncertainty: float = Field(..., description="T half-width uncertainty interval (K)")
    WF_interval: list[float] = Field(..., description="[WF_min, WF_max] uncertainty interval")
    T_interval: list[float] = Field(..., description="[T_min, T_max] uncertainty interval")


class FitResidual(BaseSchema):
    """Fit residual quality metrics."""

    mae_decades: float = Field(..., description="Mean absolute error in log10 current (decades)")
    rmse_decades: float = Field(
        ..., description="Root mean squared error in log10 current (decades)"
    )
    objective_loss: float = Field(..., description="Final objective loss value")


class NearestTCAD(BaseSchema):
    """Closest simulated TCAD dataset entry."""

    index: int = Field(..., description="Dataset row index in training split")
    WF: float = Field(..., description="Simulated work function (eV)")
    T: float = Field(..., description="Simulated temperature (K)")
    distance: float = Field(..., description="Normalized Euclidean distance in (WF, T) space")


class ReverseResponse(BaseSchema):
    """Reverse Mode parameter recovery response."""

    recovered: ParameterRecovery
    fit_residual: FitResidual
    reconstructed: PredictResponse
    nearest_tcad: NearestTCAD
    warnings: list[str] = Field(default_factory=list, description="Boundary and ambiguity warnings")
    status: str = Field("met", description="Recovery status verdict")
    evidence_level: str = Field("tcad_validated", description="Evidence tier")
    honesty_notice: str = Field(HONESTY_NOTICE, description="Mandatory honesty notice")


class ValidationResponse(BaseSchema):
    """Validation benchmark metrics across splits and models."""

    metrics: dict[str, Any] = Field(..., description="Full benchmark metrics dataset")
    evidence_level: str = Field("tcad_validated", description="Evidence tier")
    honesty_notice: str = Field(HONESTY_NOTICE, description="Mandatory honesty notice")


# --- Tier B Design Mode Schemas ---


class DesignCandidate(BaseSchema):
    """Synthesized device candidate with extracted metrics and verdict."""

    archetype: str = Field(..., description="Archetype title, e.g. 'Balanced Compromise'")
    verdict: str = Field(
        ..., description="'met' | 'met_within_tolerance' | 'compromise' | 'infeasible'"
    )
    verdict_reason: str = Field(..., description="Explanation for verdict")
    parameters: dict[str, float] = Field(
        ..., description="Synthesized physical parameters (WF, T, t_D_um, N_net_cm3, d_um)"
    )
    metrics: dict[str, Any] = Field(
        ..., description="Extracted metrics (v_br, r_on_sp, v_o, ss_mV_dec, i_on_4V, etc.)"
    )
    evidence_tags: dict[str, str] = Field(..., description="Evidence level per variable and metric")
    iv_curve: dict[str, list[float]] = Field(
        ..., description="Forward simulated I-V curve {'V': [...], 'I': [...]}"
    )
    checks: dict[str, Any] = Field(
        default_factory=dict, description="Per-requirement verification checks"
    )


class DesignRequest(BaseSchema):
    """Tier B inverse design synthesis request."""

    target_v_br: float | None = Field(
        default=None, ge=100.0, le=10000.0, description="Target breakdown voltage (V)"
    )
    target_r_on_sp: float | None = Field(
        default=None, ge=0.01, le=500.0, description="Target specific on-resistance (mOhm*cm^2)"
    )
    target_v_o: float | None = Field(
        default=None, ge=0.2, le=2.5, description="Target turn-on voltage (V)"
    )
    target_ss: float | None = Field(
        default=None, ge=50.0, le=500.0, description="Target subthreshold swing (mV/dec)"
    )
    target_i_on_4v: float | None = Field(
        default=None, ge=1e-4, le=100.0, description="Target forward current at 4V (A)"
    )
    tolerance: float = Field(
        default=0.10, ge=0.01, le=0.50, description="Acceptance tolerance band (default 0.10 = 10%)"
    )
    method: str = Field(default="nsga2", description="Optimization method: 'nsga2' or 'gradient'")
    fixed_temperature: float | None = Field(
        default=300.0, ge=300.0, le=500.0, description="Operating temperature (K)"
    )
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "v_br": 1.0,
            "r_on_sp": 1.0,
            "v_o": 0.5,
            "ss": 0.2,
            "i_on": 0.5,
        }
    )


class DesignResponse(BaseSchema):
    """Tier B design mode synthesis response."""

    candidates: list[DesignCandidate] = Field(..., description="Top synthesized candidate designs")
    pareto_front: list[dict[str, Any]] = Field(
        default_factory=list, description="Non-dominated Pareto front points"
    )
    requirements: dict[str, Any] = Field(..., description="Input specifications and tolerances")
    method_used: str = Field(..., description="Optimization algorithm executed")
    evidence_level: str = Field("physics_extrapolated", description="Evidence tier")
    honesty_notice: str = Field(HONESTY_NOTICE, description="Mandatory honesty notice")


class DesignPreset(BaseSchema):
    """Pre-verified feasible device design preset."""

    id: str
    name: str
    description: str
    requirements: dict[str, Any]


class DesignPresetsResponse(BaseSchema):
    """List of pre-verified feasible device design presets."""

    presets: list[DesignPreset]
    honesty_notice: str = Field(HONESTY_NOTICE, description="Mandatory honesty notice")


class ExportTCADRequest(BaseSchema):
    """Request to export Sentaurus TCAD deck for a candidate design."""

    parameters: dict[str, float] = Field(
        ..., description="Device parameters (WF, t_D_um, N_net_cm3, d_um, etc.)"
    )
    device_name: str = Field("beta_ga2o3_sbd", description="Device identifier")


class ExportTCADResponse(BaseSchema):
    """Exported Sentaurus TCAD simulation deck files."""

    sdevice_cmd: str = Field(..., description="Sentaurus Device simulation command deck")
    sde_cmd: str = Field(..., description="Sentaurus Structure Editor geometry file")
    readme_md: str = Field(..., description="Mandatory untested deck safety notice")
    device_params: dict[str, float]
    honesty_notice: str = Field(HONESTY_NOTICE, description="Mandatory honesty notice")


class ExportReportRequest(BaseSchema):
    """Request to export design synthesis report."""

    candidates: list[DesignCandidate] = Field(..., description="Synthesized candidate designs")
    requirements: dict[str, Any] = Field(..., description="Design specifications")


class ExportReportResponse(BaseSchema):
    """Exported design synthesis markdown report."""

    markdown: str = Field(..., description="Formatted markdown report")
    title: str = Field(..., description="Report title")
    honesty_notice: str = Field(HONESTY_NOTICE, description="Mandatory honesty notice")
