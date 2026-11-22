"""Evaluation script for PG-SBD models."""

from __future__ import annotations

import json
from pathlib import Path


def main():
    metrics_path = Path("artifacts/metrics.json")
    if not metrics_path.exists():
        print(f"Metrics not found at {metrics_path}. Running train script to generate ...")
        from scripts.train import train_and_eval_baselines

        train_and_eval_baselines()
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print("=" * 60)
    print("PG-SBD HELD-OUT TEST METRICS (Dhillon et al. dataset)")
    print("=" * 60)
    test_metrics = metrics.get("test", {})
    for model_name, m in test_metrics.items():
        print(
            f"[{model_name}] log10-MAE: {m['log10_mae_all']:.4f} dec | "
            f"Vo-MAE: {m['vo_mae_V'] * 1000:.1f} mV | "
            f"SS-MAE: {m['ss_mae_mV_dec']:.1f} mV/dec | "
            f"Rdiff-RelErr: {m['rdiff_rel_err_pct']:.1f}% | "
            f"Von-MAE: {m['von_mae_V'] * 1000:.1f} mV"
        )

    print("\n" + "=" * 60)
    print("EXTRAPOLATION BLOCK METRICS (WF > 5.16 eV, T > 470 K)")
    print("=" * 60)
    extrap_metrics = metrics.get("extrapolation", {})
    for model_name, m in extrap_metrics.items():
        print(
            f"[{model_name}] log10-MAE: {m['log10_mae_all']:.4f} dec | "
            f"Vo-MAE: {m['vo_mae_V'] * 1000:.1f} mV | "
            f"SS-MAE: {m['ss_mae_mV_dec']:.1f} mV/dec | "
            f"Rdiff-RelErr: {m['rdiff_rel_err_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
