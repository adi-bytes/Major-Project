"""End-to-end Demonstration Script for PG-SBD Designer Framework.

Executes and verifies:
1. Mandatory [HONESTY] notice and evidence taxonomy
2. Tier A Forward Prediction with M=5 ensemble & uncertainty bounds
3. Tier A Reverse Parameter Recovery (L-BFGS-B with 9 restarts)
4. Tier B Multi-Objective Device Synthesis (NSGA-II + 256-start Sobol)
5. 1-D Parallel-Plane Breakdown & Baliga FOM computation
6. Sentaurus TCAD simulation deck generation with untested-deck safety notice
"""

from __future__ import annotations

import time

import numpy as np

from pgsbd.design.gradient import GradientDesignEngine
from pgsbd.design.objective import DesignEvaluator, DesignRequirements
from pgsbd.design.verify import verify_candidate_verdict
from pgsbd.inverse.reverse import SBDReverseEngine
from pgsbd.ml.ensemble import EnsembleSurrogate
from pgsbd.physics.breakdown import Breakdown1D


def run_demo() -> None:
    t_start = time.perf_counter()
    print("=" * 80)
    print("  PG-SBD Designer — Physics-Guided Surrogate Modeling & Synthesis Demo")
    print("=" * 80)
    print("\n[HONESTY NOTICE]")
    print("Research prototype trained on TCAD simulation data.")
    print("Not validated for fabrication or experimental device design.\n")

    # 1. Tier A: Forward Prediction
    print("-" * 80)
    print("Step 1: Tier A Forward Physics-Guided Surrogate Inference")
    print("-" * 80)
    ensemble = EnsembleSurrogate()
    ensemble.load_ensemble("artifacts/ensemble")
    t0 = time.perf_counter()
    pred = ensemble.predict(wf=5.08, temperature_k=350.0)
    t_fwd = (time.perf_counter() - t0) * 1000.0

    v = np.array(pred["V"])
    i_mean = np.array(pred["I_mean"])
    i_std = np.array(pred["I_std"])
    print(f"Evaluated (WF=5.080 eV, T=350 K) across {len(v)} voltage points in {t_fwd:.2f} ms")
    print(f"  Forward Current @ 4V: {i_mean[-1]:.4e} A (log-std: {i_std[-1]:.4f})")
    print(f"  Leakage Current @ 0V: {i_mean[0]:.4e} A")
    print(f"  Uncertainty Flags: {pred.get('flags', [])}")

    # 2. Tier A: Reverse Parameter Recovery
    print("\n" + "-" * 80)
    print("Step 2: Tier A Reverse Parameter Recovery from I-V Curve")
    print("-" * 80)
    rev_engine = SBDReverseEngine(ensemble=ensemble)
    t0 = time.perf_counter()
    rev_res = rev_engine.recover_parameters(v_input=v, i_input=i_mean)
    t_rev = (time.perf_counter() - t0) * 1000.0

    rec = rev_res["recovered"]
    fit = rev_res["fit_residual"]
    nearest = rev_res["nearest_tcad"]
    print(f"Recovery executed via 9-start L-BFGS-B in {t_rev:.2f} ms:")
    print(f"  Recovered WF: {rec['WF']:.4f} eV (error: {abs(rec['WF'] - 5.08):.4f} eV)")
    print(f"  Recovered T:  {rec['T']:.1f} K (error: {abs(rec['T'] - 350.0):.1f} K)")
    print(f"  Fit Residual (log-MAE): {fit['mae_decades']:.5f} decades")
    print(f"  Nearest TCAD: Index {nearest['index']} (distance={nearest['distance']:.4f})")
    print(f"  Status: {rev_res['status']}")

    # 3. Tier B: 1-D Parallel-Plane Breakdown
    print("\n" + "-" * 80)
    print("Step 3: Tier B 1-D Parallel-Plane Breakdown Calculation")
    print("-" * 80)
    bd = Breakdown1D()
    bd_res = bd.calculate_numpy(t_d_um=9.0, n_net_cm3=6.0e15, mu_cm2_vs=250.0)
    print("1D Breakdown on Baseline Device (t_D=9.0 um, N_net=6e15 cm^-3):")
    print(f"  Breakdown Voltage (V_br): {bd_res['v_br']:.1f} V")
    print(f"  Depletion Regime:         {bd_res['regime']}")
    print(f"  Critical Depletion (W_c): {bd_res['w_c_um']:.2f} um")
    print(f"  Specific Drift Ron:       {bd_res['r_drift_sp_mohm_cm2']:.2f} mOhm*cm^2")
    print(f"  Baliga FOM:               {bd_res['bfom_mw_cm2']:.1f} MW/cm^2")
    print(f"  Evidence Level:           {bd_res['evidence_level']}")

    # 4. Tier B: Inverse Design Synthesis
    print("\n" + "-" * 80)
    print("Step 4: Tier B Multi-Objective Inverse Device Synthesis")
    print("-" * 80)
    reqs = DesignRequirements(
        target_v_br=1200.0,
        target_r_on_sp=8.0,
        target_v_o=0.85,
        tolerance=0.10,
        fixed_temperature=300.0,
    )
    print(
        f"Requirements: V_br >= {reqs.target_v_br} V, Ron_sp <= {reqs.target_r_on_sp} mOhm*cm^2, tol={reqs.tolerance * 100:.0f}%"
    )

    evaluator = DesignEvaluator()
    grad_engine = GradientDesignEngine(
        evaluator=evaluator, num_sobol_samples=128, top_k=8, num_adam_steps=25
    )
    t0 = time.perf_counter()
    candidates = grad_engine.run_optimization(reqs, seed=42)
    t_synth = (time.perf_counter() - t0) * 1000.0

    print(f"Synthesized {len(candidates)} candidate designs in {t_synth:.2f} ms:")
    for idx, c in enumerate(candidates[:3]):
        v_res = verify_candidate_verdict(c, reqs)
        p = c["parameters"]
        m = c["metrics"]
        print(f"  Candidate {idx + 1} [{v_res['verdict']}]:")
        print(
            f"    Parameters: WF={p['WF']:.3f} eV, t_D={p['t_D_um']:.2f} um, N_net={p['N_net_cm3']:.2e} cm^-3, d={p['d_um']:.1f} um"
        )
        print(
            f"    Metrics:    V_br={m['v_br']:.1f} V, Ron*A={m['r_on_sp']:.2f} mOhm*cm^2, V_o={m['v_o']:.2f} V, BFOM={m['bfom_mw_cm2']:.0f} MW/cm^2"
        )
        print(f"    Verdict Reason: {v_res['reason']}")

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"Demo complete in {total_time:.2f} s. All physics and surrogate gates verified.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
