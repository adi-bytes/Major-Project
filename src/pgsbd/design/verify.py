"""Forward verification and verdict classification for Tier B device synthesis."""

from __future__ import annotations

from typing import Any

import numpy as np

from pgsbd.design.objective import (
    PARAM_BOUNDS,
    DesignEvaluator,
    DesignRequirements,
)


def verify_candidate_verdict(
    candidate: dict[str, Any],
    reqs: DesignRequirements,
) -> dict[str, Any]:
    """Verify forward candidate metrics against design requirements and assign verdict.

    Verdicts:
    - "met": Every target requirement is strictly satisfied.
    - "met_within_tolerance": All requirements satisfied within acceptance tolerance.
    - "compromise": Some targets met, but one or more targets traded off within 2x tolerance.
    - "infeasible": Severe constraint violations (> 2x tolerance) or physically impossible
      specifications (e.g., exceeding theoretical 1D Baliga figure-of-merit limit).

    Returns:
        Dictionary with verdict, max_relative_error, and per-metric checks.
    """
    metrics = candidate.get("metrics", {})
    tol = reqs.tolerance
    checks: dict[str, dict[str, Any]] = {}

    v_br = float(metrics.get("v_br", 0.0))
    ron_sp = float(metrics.get("r_on_sp", 1e6))
    v_o = float(metrics.get("v_o", 10.0))
    ss = float(metrics.get("ss_mV_dec", 1000.0))
    i_on = float(metrics.get("i_on_4V", 0.0))

    # Physics limit check: Theoretical 1D parallel plane Baliga Limit for beta-Ga2O3
    # BFOM = V_br^2 / R_on_sp (W/cm^2). Ideal theoretical max ~ 34,000 MW/cm^2
    # If a user demands V_br=5000V and Ron_sp=0.1 mOhm*cm^2 => BFOM = 250,000 MW/cm^2 (impossible)
    if reqs.target_v_br is not None and reqs.target_r_on_sp is not None:
        target_bfom = (reqs.target_v_br**2) / (reqs.target_r_on_sp * 1e-3) / 1e6  # MW/cm^2
        if target_bfom > 45_000.0:  # Exceeds ideal Ga2O3 Baliga limit
            return {
                "verdict": "infeasible",
                "reason": (
                    f"Requested targets (V_br={reqs.target_v_br}V, Ron_sp={reqs.target_r_on_sp}mOhm*cm^2) "
                    f"yield a Baliga FOM of {target_bfom:.0f} MW/cm^2, which exceeds the theoretical "
                    "1-D physical limit for beta-Ga2O3 (~34,000 MW/cm^2)."
                ),
                "checks": {},
                "max_relative_error": 9.99,
            }

    relative_errors = []
    strictly_met = True

    # 1. Breakdown voltage: target is minimum
    if reqs.target_v_br is not None:
        passed = v_br >= reqs.target_v_br
        rel_err = max(0.0, (reqs.target_v_br - v_br) / reqs.target_v_br)
        checks["v_br"] = {
            "target": reqs.target_v_br,
            "actual": v_br,
            "unit": "V",
            "passed": passed,
            "rel_error": rel_err,
        }
        if not passed:
            strictly_met = False
        relative_errors.append(rel_err)

    # 2. Specific on-resistance: target is maximum
    if reqs.target_r_on_sp is not None:
        passed = ron_sp <= reqs.target_r_on_sp
        rel_err = max(0.0, (ron_sp - reqs.target_r_on_sp) / reqs.target_r_on_sp)
        checks["r_on_sp"] = {
            "target": reqs.target_r_on_sp,
            "actual": ron_sp,
            "unit": "mΩ·cm²",
            "passed": passed,
            "rel_error": rel_err,
        }
        if not passed:
            strictly_met = False
        relative_errors.append(rel_err)

    # 3. Turn-on voltage: target is maximum
    if reqs.target_v_o is not None:
        passed = v_o <= reqs.target_v_o
        rel_err = max(0.0, (v_o - reqs.target_v_o) / reqs.target_v_o)
        checks["v_o"] = {
            "target": reqs.target_v_o,
            "actual": v_o,
            "unit": "V",
            "passed": passed,
            "rel_error": rel_err,
        }
        if not passed:
            strictly_met = False
        relative_errors.append(rel_err)

    # 4. Subthreshold swing: target is maximum
    if reqs.target_ss is not None:
        passed = ss <= reqs.target_ss
        rel_err = max(0.0, (ss - reqs.target_ss) / reqs.target_ss)
        checks["ss_mV_dec"] = {
            "target": reqs.target_ss,
            "actual": ss,
            "unit": "mV/dec",
            "passed": passed,
            "rel_error": rel_err,
        }
        if not passed:
            strictly_met = False
        relative_errors.append(rel_err)

    # 5. Forward current at 4V: target is minimum
    if reqs.target_i_on_4v is not None:
        passed = i_on >= reqs.target_i_on_4v
        rel_err = max(0.0, (reqs.target_i_on_4v - i_on) / reqs.target_i_on_4v)
        checks["i_on_4V"] = {
            "target": reqs.target_i_on_4v,
            "actual": i_on,
            "unit": "A",
            "passed": passed,
            "rel_error": rel_err,
        }
        if not passed:
            strictly_met = False
        relative_errors.append(rel_err)

    max_rel_err = max(relative_errors) if relative_errors else 0.0

    if strictly_met:
        verdict = "met"
        reason = "All design targets are strictly satisfied."
    elif max_rel_err <= tol:
        verdict = "met_within_tolerance"
        reason = f"All requirements met within {tol * 100:.0f}% specified tolerance (max error {max_rel_err * 100:.1f}%)."
    elif max_rel_err <= 2.0 * tol:
        verdict = "compromise"
        reason = (
            f"Physical Pareto trade-off: some targets met, while others exhibit moderate "
            f"trade-off ({max_rel_err * 100:.1f}% error, exceeding {tol * 100:.0f}% tolerance)."
        )
    else:
        verdict = "infeasible"
        reason = (
            f"Significant target violation ({max_rel_err * 100:.1f}% error) beyond 2x tolerance "
            f"({2 * tol * 100:.0f}%). Targets are contradictory or physically out of reach."
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "max_relative_error": float(max_rel_err),
        "checks": checks,
    }


def run_self_consistency_test(
    evaluator: DesignEvaluator | None = None,
    num_curves: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Run 200-curve self-consistency verification across the parameter space.

    Generates 200 valid random device configurations, extracts metrics, evaluates them,
    and validates self-consistency and accuracy of the verification engine.
    """
    evaluator = evaluator or DesignEvaluator()
    rng = np.random.RandomState(seed)

    # Sample random physical parameters
    wf_samples = rng.uniform(PARAM_BOUNDS["WF"][0], PARAM_BOUNDS["WF"][1], num_curves)
    t_samples = rng.uniform(PARAM_BOUNDS["T"][0], PARAM_BOUNDS["T"][1], num_curves)
    td_samples = rng.uniform(PARAM_BOUNDS["t_D_um"][0], PARAM_BOUNDS["t_D_um"][1], num_curves)
    log_nn_samples = rng.uniform(
        PARAM_BOUNDS["log_N_net"][0], PARAM_BOUNDS["log_N_net"][1], num_curves
    )
    nn_samples = np.power(10.0, log_nn_samples)
    d_samples = rng.uniform(PARAM_BOUNDS["d_um"][0], PARAM_BOUNDS["d_um"][1], num_curves)

    passed_count = 0
    finite_checks = 0

    for i in range(num_curves):
        cand = evaluator.evaluate_candidate(
            wf=float(wf_samples[i]),
            temperature_k=float(t_samples[i]),
            t_d_um=float(td_samples[i]),
            n_net_cm3=float(nn_samples[i]),
            d_um=float(d_samples[i]),
        )
        m = cand["metrics"]

        # Ensure all metrics are finite and positive
        is_finite = (
            np.isfinite(m["v_br"])
            and np.isfinite(m["r_on_sp"])
            and np.isfinite(m["v_o"])
            and np.isfinite(m["ss_mV_dec"])
            and np.isfinite(m["i_on_4V"])
            and m["v_br"] > 0
            and m["r_on_sp"] > 0
        )
        if is_finite:
            finite_checks += 1

        # Check self-consistency: if we set targets equal to candidate metrics, verdict must be "met"
        self_reqs = DesignRequirements(
            target_v_br=float(m["v_br"]),
            target_r_on_sp=float(m["r_on_sp"]),
            target_v_o=float(m["v_o"] + 0.05),  # relaxed slightly
        )
        v_res = verify_candidate_verdict(cand, self_reqs)
        if v_res["verdict"] == "met":
            passed_count += 1

    return {
        "total_curves": num_curves,
        "finite_checks_passed": finite_checks,
        "self_consistency_passed": passed_count,
        "consistency_rate": passed_count / num_curves,
    }
