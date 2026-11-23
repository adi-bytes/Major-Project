"""API route definitions for PG-SBD Designer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request, status

from pgsbd.api.schemas import (
    DesignCandidate,
    DesignPreset,
    DesignPresetsResponse,
    DesignRequest,
    DesignResponse,
    DeviceMetaResponse,
    ExportReportRequest,
    ExportReportResponse,
    ExportTCADRequest,
    ExportTCADResponse,
    ExtractedMetrics,
    HealthResponse,
    NearestTCAD,
    ParameterRecovery,
    PredictRequest,
    PredictResponse,
    ReverseRequest,
    ReverseResponse,
    ValidationResponse,
    VariableBound,
)
from pgsbd.data.metrics import extract_metrics_numpy
from pgsbd.design.cluster import cluster_and_select_candidates
from pgsbd.design.gradient import GradientDesignEngine
from pgsbd.design.nsga import NSGADesignEngine
from pgsbd.design.objective import DesignEvaluator, DesignRequirements
from pgsbd.design.verify import verify_candidate_verdict

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request) -> HealthResponse:
    """Check API health and loaded artifact availability."""
    artifacts_status = {
        "ensemble_models": bool(getattr(request.app.state, "ensemble", None) is not None),
        "reverse_engine": bool(getattr(request.app.state, "reverse_engine", None) is not None),
        "calibration": bool(getattr(request.app.state, "calib", None) is not None),
        "validation_metrics": bool(getattr(request.app.state, "metrics", None) is not None),
    }

    return HealthResponse(
        status="ok",
        device="Vertical Beta-Ga2O3 Schottky Barrier Diode (SBD)",
        evidence_level="tcad_validated",
        artifacts_loaded=artifacts_status,
    )


@router.get("/meta", response_model=DeviceMetaResponse)
@router.get("/devices", response_model=DeviceMetaResponse)
def get_device_meta(_request: Request) -> DeviceMetaResponse:
    """Retrieve device metadata, variable bounds, and operational constraints."""
    return DeviceMetaResponse(
        device_name="Vertical Beta-Ga2O3 Schottky Barrier Diode (SBD)",
        material="Beta-Ga2O3",
        description=(
            "Vertical beta-gallium oxide Schottky barrier diode simulated in Sentaurus TCAD "
            "(Dhillon et al., IEEE TED 68(11), 2021) with cylindrical coordinates, self-heating, "
            "and incomplete ionization."
        ),
        evidence_level="tcad_validated",
        variables=[
            VariableBound(
                name="WF",
                unit="eV",
                min_val=4.99,
                max_val=5.19,
                is_sweepable=True,
            ),
            VariableBound(
                name="temperature",
                unit="K",
                min_val=300.0,
                max_val=500.0,
                is_sweepable=True,
            ),
        ],
        fixed_parameters={
            "t_Drift_um": 9.0,
            "D_N_cm3": 4.0e16,
            "D_C_cm3": 3.4e16,
            "N_net_cm3": 6.0e15,
            "anode_radius_um": 73.8,
            "anode_area_cm2": 1.711e-4,
            "epsilon_r": 10.0,
        },
        voltage_grid={
            "v_min": 0.0,
            "v_max": 4.0,
            "num_points": 52,
            "step_v": 4.0 / 51,
        },
    )


@router.post("/predict", response_model=PredictResponse)
def predict_forward(request: Request, payload: PredictRequest) -> PredictResponse:
    """Run forward physics-guided surrogate prediction for specified (WF, T)."""
    ensemble = getattr(request.app.state, "ensemble", None)
    if ensemble is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Surrogate model ensemble is not loaded.",
        )

    sim = ensemble.predict(
        wf=payload.wf,
        temperature_k=payload.temperature,
    )

    v_arr = np.array(sim["V"], dtype=float)
    i_mean_arr = np.array(sim["I_mean"], dtype=float)
    i_std_arr = np.array(sim["I_std"], dtype=float)  # std in log10 space
    i_phys_arr = np.array(sim["I_phys"], dtype=float)

    log_mean = np.log10(np.maximum(i_mean_arr, 1e-14))
    i_lower_arr = 10.0 ** (log_mean - 1.96 * i_std_arr)
    i_upper_arr = 10.0 ** (log_mean + 1.96 * i_std_arr)

    # Off-grid custom voltage interpolation if requested
    if payload.v_custom is not None:
        v_eval = np.array(payload.v_custom, dtype=float)
        i_mean_arr = 10.0 ** np.interp(v_eval, v_arr, np.log10(np.maximum(i_mean_arr, 1e-14)))
        i_lower_arr = 10.0 ** np.interp(v_eval, v_arr, np.log10(np.maximum(i_lower_arr, 1e-14)))
        i_upper_arr = 10.0 ** np.interp(v_eval, v_arr, np.log10(np.maximum(i_upper_arr, 1e-14)))
        i_phys_arr = 10.0 ** np.interp(v_eval, v_arr, np.log10(np.maximum(i_phys_arr, 1e-14)))
        i_std_arr = np.interp(v_eval, v_arr, i_std_arr)
        v_arr = v_eval

    fom = extract_metrics_numpy(v_arr, i_mean_arr)
    r_diff = float(fom.get("r_diff", 0.0))
    r_on_sp = float(fom.get("r_on_sp", r_diff * 1.711e-4 * 1000.0))

    metrics_obj = ExtractedMetrics(
        v_o=float(fom.get("v_o", fom.get("von", 0.0))),
        r_on_sp=r_on_sp,
        ss_mV_dec=float(fom.get("ss_mV_dec", 0.0)),
        i_on_4V=float(i_mean_arr[-1]),
        i_leak_0V=float(i_mean_arr[0]),
    )

    return PredictResponse(
        V=[float(v) for v in v_arr],
        I_mean=[float(x) for x in i_mean_arr],
        I_std=[float(x) for x in i_std_arr],
        I_lower=[float(x) for x in i_lower_arr],
        I_upper=[float(x) for x in i_upper_arr],
        I_phys=[float(x) for x in i_phys_arr],
        metrics=metrics_obj,
        flags=sim.get("flags", []),
        evidence_level="tcad_validated",
    )


@router.post("/reverse", response_model=ReverseResponse)
def reverse_recover(request: Request, payload: ReverseRequest) -> ReverseResponse:
    """Recover device parameters (WF, T) from supplied experimental or simulated I-V curve."""
    reverse_engine = getattr(request.app.state, "reverse_engine", None)
    if reverse_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reverse mode engine is not loaded.",
        )

    res = reverse_engine.recover_parameters(
        v_input=np.array(payload.v, dtype=float),
        i_input=np.array(payload.i, dtype=float),
    )

    rec = res["recovered"]
    fit = res["fit_residual"]
    recon = res["reconstructed"]
    nearest = res["nearest_tcad"]

    v_arr = np.array(recon["V"], dtype=float)
    i_mean_arr = np.array(recon["I_mean"], dtype=float)
    i_std_arr = np.array(recon["I_std"], dtype=float)
    i_phys_arr = np.array(recon["I_phys"], dtype=float)

    log_mean = np.log10(np.maximum(i_mean_arr, 1e-14))
    i_lower_arr = 10.0 ** (log_mean - 1.96 * i_std_arr)
    i_upper_arr = 10.0 ** (log_mean + 1.96 * i_std_arr)

    recon_fom = extract_metrics_numpy(v_arr, i_mean_arr)
    recon_r_diff = float(recon_fom.get("r_diff", 0.0))
    recon_r_on_sp = float(recon_fom.get("r_on_sp", recon_r_diff * 1.711e-4 * 1000.0))

    reconstructed_pred = PredictResponse(
        V=[float(v) for v in v_arr],
        I_mean=[float(x) for x in i_mean_arr],
        I_std=[float(x) for x in i_std_arr],
        I_lower=[float(x) for x in i_lower_arr],
        I_upper=[float(x) for x in i_upper_arr],
        I_phys=[float(x) for x in i_phys_arr],
        metrics=ExtractedMetrics(
            v_o=float(recon_fom.get("v_o", recon_fom.get("von", 0.0))),
            r_on_sp=recon_r_on_sp,
            ss_mV_dec=float(recon_fom.get("ss_mV_dec", 0.0)),
            i_on_4V=float(i_mean_arr[-1]),
            i_leak_0V=float(i_mean_arr[0]),
        ),
        flags=recon.get("flags", []),
        evidence_level="tcad_validated",
    )

    return ReverseResponse(
        recovered=ParameterRecovery(
            WF=float(rec["WF"]),
            T=float(rec["T"]),
            WF_uncertainty=float(rec["WF_uncertainty"]),
            T_uncertainty=float(rec["T_uncertainty"]),
            WF_interval=[float(x) for x in rec["WF_interval"]],
            T_interval=[float(x) for x in rec["T_interval"]],
        ),
        fit_residual=fit,
        reconstructed=reconstructed_pred,
        nearest_tcad=NearestTCAD(
            index=int(nearest["index"]),
            WF=float(nearest["WF"]),
            T=float(nearest["T"]),
            distance=float(nearest["distance"]),
        ),
        warnings=res.get("warnings", []),
        status=res.get("status", "met"),
        evidence_level="tcad_validated",
    )


@router.get("/validation", response_model=ValidationResponse)
def get_validation(request: Request) -> ValidationResponse:
    """Retrieve precomputed validation metrics and ablation benchmarks."""
    metrics: dict[str, Any] | None = getattr(request.app.state, "metrics", None)
    if metrics is None:
        metrics_p = Path("artifacts/metrics.json")
        if metrics_p.exists():
            with open(metrics_p, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Validation metrics artifact not found.",
            )

    return ValidationResponse(
        metrics=metrics,
        evidence_level="tcad_validated",
    )


# --- Tier B Design Mode Endpoints ---

PREVERIFIED_PRESETS = [
    DesignPreset(
        id="high_voltage_1200v",
        name="1.2 kV Power Rectifier",
        description="Standard 1200 V class power diode balancing moderate specific on-resistance and high blocking voltage.",
        requirements={
            "target_v_br": 1200.0,
            "target_r_on_sp": 10.0,
            "target_v_o": 1.0,
            "target_ss": 100.0,
            "target_i_on_4v": 0.08,
            "tolerance": 0.10,
        },
    ),
    DesignPreset(
        id="low_loss_600v",
        name="600 V Fast-Switching Diode",
        description="Low turn-on voltage and minimal forward on-resistance for high-efficiency power converter applications.",
        requirements={
            "target_v_br": 600.0,
            "target_r_on_sp": 3.5,
            "target_v_o": 0.65,
            "target_ss": 85.0,
            "target_i_on_4v": 0.12,
            "tolerance": 0.10,
        },
    ),
    DesignPreset(
        id="ultra_high_voltage_2500v",
        name="2.5 kV Ultra-High Voltage Diode",
        description="Thick drift layer design for extreme breakdown voltage requirements in transmission or heavy-duty drives.",
        requirements={
            "target_v_br": 2500.0,
            "target_r_on_sp": 25.0,
            "target_v_o": 1.2,
            "target_ss": 120.0,
            "target_i_on_4v": 0.05,
            "tolerance": 0.10,
        },
    ),
]


@router.get("/design/presets", response_model=DesignPresetsResponse)
def get_design_presets() -> DesignPresetsResponse:
    """Retrieve pre-verified feasible device design presets."""
    return DesignPresetsResponse(presets=PREVERIFIED_PRESETS)


@router.post("/design", response_model=DesignResponse)
def run_design_optimization(request: Request, payload: DesignRequest) -> DesignResponse:
    """Synthesize optimal device parameters from electrical requirements using Tier B engine."""
    evaluator = DesignEvaluator(
        physics_core=getattr(request.app.state, "physics_core", None),
        calib_theta=getattr(request.app.state, "calib", None),
    )

    reqs = DesignRequirements(
        target_v_br=payload.target_v_br,
        target_r_on_sp=payload.target_r_on_sp,
        target_v_o=payload.target_v_o,
        target_ss=payload.target_ss,
        target_i_on_4v=payload.target_i_on_4v,
        tolerance=payload.tolerance,
        fixed_temperature=payload.fixed_temperature,
        weights=payload.weights,
    )

    if payload.method == "gradient":
        engine = GradientDesignEngine(evaluator=evaluator)
        pool = engine.run_optimization(reqs)
        method_name = "256-Start Sobol + Adam Local Optimization"
    else:
        nsga_engine = NSGADesignEngine(evaluator=evaluator, pop_size=40, n_gen=25)
        pool = nsga_engine.run_optimization(reqs)
        method_name = "NSGA-II Multi-Objective Evolutionary Algorithm"

    # Fallback to gradient if NSGA produced empty or too few candidates
    if not pool:
        engine = GradientDesignEngine(evaluator=evaluator)
        pool = engine.run_optimization(reqs)
        method_name = "256-Start Sobol + Adam (Fallback)"

    # Cluster to select 4-5 diverse candidates
    selected = cluster_and_select_candidates(pool, n_clusters=4)

    # Format candidates with verified verdicts
    candidate_cards: list[DesignCandidate] = []
    for cand in selected:
        v_res = verify_candidate_verdict(cand, reqs)
        candidate_cards.append(
            DesignCandidate(
                archetype=cand.get("archetype", "Candidate"),
                verdict=v_res["verdict"],
                verdict_reason=v_res["reason"],
                parameters=cand["parameters"],
                metrics=cand["metrics"],
                evidence_tags=cand.get("evidence_tags", {}),
                iv_curve=cand["iv_curve"],
                checks=v_res.get("checks", {}),
            )
        )

    # Compile Pareto front coordinates for visualization
    pareto_data = [
        {
            "v_br": c["metrics"]["v_br"],
            "r_on_sp": c["metrics"]["r_on_sp"],
            "v_o": c["metrics"]["v_o"],
            "i_on_4V": c["metrics"]["i_on_4V"],
            "t_D_um": c["parameters"]["t_D_um"],
            "N_net_cm3": c["parameters"]["N_net_cm3"],
            "archetype": c.get("archetype", "Pareto Point"),
        }
        for c in pool
    ]

    return DesignResponse(
        candidates=candidate_cards,
        pareto_front=pareto_data,
        requirements=payload.model_dump(),
        method_used=method_name,
        evidence_level="physics_extrapolated",
    )


@router.post("/export/tcad", response_model=ExportTCADResponse)
def export_tcad_deck(payload: ExportTCADRequest) -> ExportTCADResponse:
    """Generate Sentaurus TCAD simulation deck files with mandatory untested-deck notice."""
    p = payload.parameters
    wf = p.get("WF", 5.05)
    t_d = p.get("t_D_um", 9.0)
    n_net = p.get("N_net_cm3", 6.0e15)
    d_um = p.get("d_um", 200.0)
    radius_um = d_um / 2.0
    t_k = float(p.get("temperature", 300.0))

    sdevice_content = f"""# Synopsys Sentaurus Device Command Deck (sdevice.cmd)
# Generated by PG-SBD Designer (Tier B Synthesis Engine)
# Target Device: Vertical beta-Ga2O3 Schottky Barrier Diode

File {{
    Grid      = "sde_msh.tdr"
    Parameter = "models.par"
    Plot      = "sdevice.tdr"
    Current   = "sdevice.plt"
    Output    = "sdevice.log"
}}

Electrode {{
    {{ Name = "anode"   Voltage = 0.0 WorkFunction = {wf:.4f} BarrierLowering }}
    {{ Name = "cathode" Voltage = 0.0 }}
}}

Physics {{
    Fermi
    IncompleteIonization
    Temperature = {t_k:.1f}
    Recombination(
        SRH(DopingDep)
    )
    Mobility(
        DopingDep
        HighFieldSaturation
    )
    EffectiveIntrinsicDensity(OldSlotboom)
}}

Math {{
    Extrapolate
    RelErr = 1e-4
    Digits = 5
    Iterations = 50
    Method = ILS
    Number_of_Threads = 4
}}

Solve {{
    Coupled(Iterations=100) {{ Poisson }}
    Coupled {{ Poisson Electron }}

    # Forward I-V Sweep (0 V to 4.0 V)
    Quasistationary(
        InitialStep = 0.01  Increment = 1.3
        MinStep     = 1e-5  MaxStep   = 0.05
        Goal {{ Name = "anode" Voltage = 4.0 }}
    ) {{
        Coupled {{ Poisson Electron }}
    }}
}}
"""

    sde_content = f"""# Synopsys Sentaurus Structure Editor Script (sde.cmd)
# Geometry and Meshing for Vertical beta-Ga2O3 SBD

(sde:clear)

# Dimensions (micrometers)
(define R_ANODE {radius_um:.2f})
(define T_DRIFT {t_d:.2f})
(define T_SUBSTRATE 10.0)
(define R_TOTAL (+ R_ANODE 30.0))

# Substrate Region (n+ heavily doped)
(sdegeo:create-rectangle
    (position 0.0 0.0 0.0)
    (position R_TOTAL T_SUBSTRATE 0.0)
    "GaO_substrate" "substrate_region")

# Drift Layer Region (n- low doped)
(sdegeo:create-rectangle
    (position 0.0 T_SUBSTRATE 0.0)
    (position R_TOTAL (+ T_SUBSTRATE T_DRIFT) 0.0)
    "GaO_drift" "drift_region")

# Doping Definitions
(sdedr:define-constant-profile "SubstrateDoping" "PhosphorusActiveConcentration" 4.0e18)
(sdedr:define-constant-profile-material "SubstrateDopingMat" "SubstrateDoping" "GaO_substrate")

(sdedr:define-constant-profile "DriftDoping" "SiliconActiveConcentration" {n_net:.3e})
(sdedr:define-constant-profile-material "DriftDopingMat" "DriftDoping" "GaO_drift")

# Contact Definitions
(sdegeo:define-contact-set "cathode" 4 (color:rgb 1 0 0) "##")
(sdegeo:define-contact-set "anode"   4 (color:rgb 0 0 1) "##")

(sdegeo:set-current-contact-interface "cathode")
(sdegeo:set-contact-boundary-edges (find-edge-id (position (/ R_TOTAL 2.0) 0.0 0.0)))

(sdegeo:set-current-contact-interface "anode")
(sdegeo:set-contact-boundary-edges (find-edge-id (position (/ R_ANODE 2.0) (+ T_SUBSTRATE T_DRIFT) 0.0)))

# Mesh Generation
(sdedr:define-regrid-settings "DefaultSettings")
(sdemesh:build-mesh "sde_msh")
"""

    readme_content = f"""# UNTESTED TCAD SIMULATION DECK NOTICE

**CRITICAL DISCLAIMER: RESEARCH PROTOTYPE ONLY**
This simulation deck was automatically generated by the PG-SBD Designer framework (Tier B).

- **NOT experimentally validated**: Physical devices have not been fabricated or measured for these exact parameters.
- **NOT pre-executed in Synopsys Sentaurus TCAD**: Convergence, meshing density, and exact model calibration files must be independently verified.
- **2D/3D Edge Effects**: This deck uses 2D planar or cylindrical geometry. Edge termination (guard rings, field plates, bevels) is NOT included and must be added before high-voltage breakdown analysis.

### Synthesized Device Specifications:
- Anode Work Function (WF): {wf:.4f} eV
- Drift Layer Thickness (t_D): {t_d:.2f} μm
- Net Donor Concentration (N_net): {n_net:.3e} cm⁻³
- Anode Diameter (d): {d_um:.1f} μm
- Evidence Classification: `physics_extrapolated` (Tier B)
"""

    return ExportTCADResponse(
        sdevice_cmd=sdevice_content,
        sde_cmd=sde_content,
        readme_md=readme_content,
        device_params=p,
    )


@router.post("/export/report", response_model=ExportReportResponse)
def export_design_report(payload: ExportReportRequest) -> ExportReportResponse:
    """Generate comprehensive synthesis report in Markdown."""
    reqs = payload.requirements
    cands = payload.candidates

    report_lines = [
        "# PG-SBD Designer — Tier B Device Synthesis Report",
        "",
        (
            "> **[HONESTY NOTICE]** Research prototype trained on TCAD simulation data. "
            "Not validated for fabrication or experimental device design."
        ),
        "",
        "## 1. Design Requirements & Specifications",
        "",
        "| Requirement | Target Specification | Status |",
        "|---|---|---|",
    ]

    for k, val in reqs.items():
        if val is not None and k.startswith("target_"):
            report_lines.append(f"| `{k}` | **{val}** | Active |")

    report_lines.extend(
        [
            f"| Acceptance Tolerance | **{reqs.get('tolerance', 0.1) * 100:.0f}%** | Constraint Band |",
            "",
            "## 2. Synthesized Candidate Devices",
            "",
            "| Archetype | Verdict | V_br (V) | Ron·A (mΩ·cm²) | V_o (V) | I_on@4V (A) | Drift t_D (μm) | Doping N_net (cm⁻³) |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )

    for c in cands:
        p = c.parameters
        m = c.metrics
        report_lines.append(
            f"| **{c.archetype}** | `{c.verdict}` | {m.get('v_br', 0.0):.1f} | {m.get('r_on_sp', 0.0):.2f} | "
            f"{m.get('v_o', 0.0):.2f} | {m.get('i_on_4V', 0.0):.3e} | {p.get('t_D_um', 0.0):.2f} | {p.get('N_net_cm3', 0.0):.2e} |"
        )

    report_lines.extend(
        [
            "",
            "## 3. Evidence & Uncertainty Classification",
            "",
            "- **Breakdown Voltage ($V_{br}$)**: `analytic_unvalidated` (1-D parallel-plane model, assumes ideal edge termination).",
            "- **Structural Variables ($t_D, N_{net}, d$)**: `physics_extrapolated` (Extrapolated beyond baseline TCAD 9 μm / 6e15 cm⁻³ design).",
            "- **Turn-on ($V_0$) & Subthreshold Swing ($SS$)**: `tcad_validated` for baseline (WF, T) range; `physics_extrapolated` otherwise.",
            "",
            "---",
            "*Report generated by PG-SBD Designer — Phase 8 Synthesis Engine.*",
        ]
    )

    return ExportReportResponse(
        markdown="\n".join(report_lines),
        title="PG-SBD Tier B Device Synthesis Report",
    )
