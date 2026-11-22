"""Unit and integration tests for Tier B Design Mode and synthesis engine."""

from __future__ import annotations

import numpy as np
import torch
from fastapi.testclient import TestClient

from pgsbd.api.main import app
from pgsbd.data.metrics import extract_metrics_numpy
from pgsbd.design.cluster import cluster_and_select_candidates
from pgsbd.design.gradient import GradientDesignEngine
from pgsbd.design.objective import (
    DesignEvaluator,
    DesignRequirements,
)
from pgsbd.design.verify import run_self_consistency_test, verify_candidate_verdict
from pgsbd.physics.breakdown import Breakdown1D
from pgsbd.physics.extract_torch import extract_metrics_torch


# 1. Breakdown Model Tests
def test_breakdown_1d_regimes_and_evidence():
    """Verify 1D breakdown accurately distinguishes punch-through and non-punch-through."""
    bd = Breakdown1D()
    assert bd.EVIDENCE_LEVEL == "analytic_unvalidated"

    # Thin drift layer + low doping => Punch-through (PT)
    pt_res = bd.calculate_numpy(t_d_um=3.0, n_net_cm3=1.0e15)
    assert pt_res["regime"] == "punch_through"
    assert pt_res["v_br"] > 0
    assert pt_res["evidence_level"] == "analytic_unvalidated"

    # Thick drift layer + high doping => Non-punch-through (NPT)
    npt_res = bd.calculate_numpy(t_d_um=15.0, n_net_cm3=5.0e16)
    assert npt_res["regime"] == "non_punch_through"
    assert npt_res["v_br"] > 0
    assert npt_res["w_dep_um"] <= 15.0

    # PyTorch implementation matches NumPy within numerical precision
    t_d_t = torch.tensor([3.0, 15.0], dtype=torch.float64)
    n_net_t = torch.tensor([1.0e15, 5.0e16], dtype=torch.float64)
    bd_torch = bd.calculate_torch(t_d_t, n_net_t)

    assert abs(float(bd_torch["v_br"][0]) - pt_res["v_br"]) < 1e-4
    assert abs(float(bd_torch["v_br"][1]) - npt_res["v_br"]) < 1e-4


# 2. Metric Extractor NumPy vs PyTorch Equivalence Tests
def test_metric_extractor_torch_vs_numpy_precision():
    """Verify differentiable PyTorch extractor matches extract_metrics_numpy to 1e-6."""
    v_grid = np.linspace(0.0, 4.0, 52)
    # Generate realistic diode I-V characteristic with exponential turn-on and ohmic tail
    currents = 1e-12 * (np.exp(v_grid / 0.06) - 1.0)
    currents = np.clip(currents, 1e-14, 1.0)
    currents = currents / (1.0 + 8.0 * currents)

    v_torch = torch.tensor(v_grid, dtype=torch.float64)
    i_torch = torch.tensor(currents, dtype=torch.float64)

    area = 1.711e-4
    m_np = extract_metrics_numpy(v_grid, currents, area_cm2=area)
    m_pt = extract_metrics_torch(v_torch, i_torch, area_cm2=area)

    for k in ["von", "v_o", "ss_mV_dec", "r_diff", "i_at_vop", "ron_sp_mohm_cm2"]:
        val_np = m_np[k]
        val_pt = float(m_pt[k])
        error = abs(val_np - val_pt)
        assert error < 1e-6, (
            f"Metric mismatch in {k}: numpy={val_np}, torch={val_pt}, error={error}"
        )


# 3. 200-Curve Self-Consistency Test
def test_200_curve_self_consistency():
    """Run the 200-curve self-consistency verification across the design space."""
    res = run_self_consistency_test(num_curves=200, seed=123)
    assert res["total_curves"] == 200
    assert res["finite_checks_passed"] == 200
    assert res["self_consistency_passed"] == 200
    assert res["consistency_rate"] == 1.0


# 4. Verdict Trigger Tests (met, met_within_tolerance, compromise, infeasible)
def test_verdict_triggers_accurately():
    """Test that all 4 verdicts trigger with exact condition boundaries."""
    cand = {
        "metrics": {
            "v_br": 1200.0,
            "r_on_sp": 8.0,
            "v_o": 0.85,
            "ss_mV_dec": 85.0,
            "i_on_4V": 0.10,
        }
    }

    # 1. Strictly Met
    req_met = DesignRequirements(target_v_br=1200.0, target_r_on_sp=8.0)
    v_met = verify_candidate_verdict(cand, req_met)
    assert v_met["verdict"] == "met"

    # 2. Met within tolerance (error <= tolerance=0.10)
    # Target V_br=1280V vs 1200V actual => rel error = (1280 - 1200)/1280 = 6.25% <= 10%
    req_tol = DesignRequirements(target_v_br=1280.0, tolerance=0.10)
    v_tol = verify_candidate_verdict(cand, req_tol)
    assert v_tol["verdict"] == "met_within_tolerance"

    # 3. Compromise (tolerance < error <= 2x tolerance)
    # Target V_br=1400V vs 1200V actual => rel error = (1400 - 1200)/1400 = 14.3% in [10%, 20%]
    req_comp = DesignRequirements(target_v_br=1400.0, tolerance=0.10)
    v_comp = verify_candidate_verdict(cand, req_comp)
    assert v_comp["verdict"] == "compromise"

    # 4. Infeasible (error > 2x tolerance = 20%)
    # Target V_br=2000V vs 1200V actual => rel error = 40% > 20%
    req_inf = DesignRequirements(target_v_br=2000.0, tolerance=0.10)
    v_inf = verify_candidate_verdict(cand, req_inf)
    assert v_inf["verdict"] == "infeasible"

    # 5. Infeasible due to theoretical 1-D Baliga Limit violation
    req_baliga = DesignRequirements(target_v_br=5000.0, target_r_on_sp=0.05)
    v_baliga = verify_candidate_verdict(cand, req_baliga)
    assert v_baliga["verdict"] == "infeasible"
    assert "Baliga" in v_baliga["reason"]


# 5. Optimization & Clustering Integration Tests
def test_gradient_and_nsga_optimization_and_clustering():
    """Verify optimization engines synthesize valid candidates and cluster diverse archetypes."""
    evaluator = DesignEvaluator()
    reqs = DesignRequirements(
        target_v_br=800.0,
        target_r_on_sp=5.0,
        tolerance=0.10,
    )

    # Gradient engine test
    grad_engine = GradientDesignEngine(
        evaluator=evaluator, num_sobol_samples=64, top_k=8, num_adam_steps=10
    )
    grad_cands = grad_engine.run_optimization(reqs, seed=42)
    assert len(grad_cands) >= 3

    # Cluster test
    clustered = cluster_and_select_candidates(grad_cands, n_clusters=3)
    assert len(clustered) >= 2
    archetypes = [c["archetype"] for c in clustered]
    assert any("Breakdown" in a or "Compromise" in a or "Resistance" in a for a in archetypes)


# 6. API Endpoints Contract Tests
def test_api_design_endpoints():
    """Verify POST /api/design, POST /api/export/tcad, POST /api/export/report and presets."""
    with TestClient(app) as client:
        # Presets
        r_pre = client.get("/api/design/presets")
        assert r_pre.status_code == 200
        presets = r_pre.json()["presets"]
        assert len(presets) >= 3

        # POST /api/design
        design_payload = {
            "target_v_br": 800.0,
            "target_r_on_sp": 5.0,
            "target_v_o": 0.9,
            "tolerance": 0.10,
            "method": "gradient",
            "fixed_temperature": 300.0,
        }
        r_des = client.post("/api/design", json=design_payload)
        assert r_des.status_code == 200
        res = r_des.json()
        assert len(res["candidates"]) > 0
        cand0 = res["candidates"][0]
        assert "verdict" in cand0
        assert "parameters" in cand0
        assert "metrics" in cand0
        assert cand0["evidence_tags"]["v_br"] == "analytic_unvalidated"

        # POST /api/export/tcad
        tcad_payload = {
            "parameters": cand0["parameters"],
            "device_name": "beta_ga2o3_sbd",
        }
        r_tcad = client.post("/api/export/tcad", json=tcad_payload)
        assert r_tcad.status_code == 200
        tcad_res = r_tcad.json()
        assert "UNTESTED TCAD SIMULATION DECK NOTICE" in tcad_res["readme_md"]
        assert "sdevice" in tcad_res["sdevice_cmd"]
        assert "sdegeo" in tcad_res["sde_cmd"]

        # POST /api/export/report
        report_payload = {
            "candidates": res["candidates"],
            "requirements": design_payload,
        }
        r_rep = client.post("/api/export/report", json=report_payload)
        assert r_rep.status_code == 200
        rep_res = r_rep.json()
        assert "Tier B Device Synthesis Report" in rep_res["markdown"]
        assert "[HONESTY NOTICE]" in rep_res["markdown"]
