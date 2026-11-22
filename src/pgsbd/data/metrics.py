"""Metric extraction module for I-V curves (NumPy implementation).

Extracts key physical figures of merit:
- Von at I_ref (default 1 mA) via log-linear interpolation
- Vo at 10 uA
- Subthreshold slope SS between 10 uA and 1 uA in mV/dec
- Rdiff (ohm) via linear regression over the last 5 points (V vs I slope)
- I at V_op (default 2 V) via interpolation
- Ron_sp (mohm*cm^2) = Rdiff * area * 1000
"""

from __future__ import annotations

from typing import Any

import numpy as np


def extract_metrics_numpy(
    v_grid: np.ndarray,
    currents: np.ndarray,
    i_ref: float = 1.0e-3,
    v_op: float = 2.0,
    vo_current: float = 1.0e-5,
    ss_low: float = 1.0e-6,
    ss_high: float = 1.0e-5,
    rdiff_points: int = 5,
    area_cm2: float | None = None,
) -> dict[str, float]:
    """Extract figures of merit from an I-V curve given 1D arrays of V and I.

    Args:
        v_grid: 1D array of voltages (e.g. 0 to 4 V, 52 points).
        currents: 1D array of currents in Amperes.
        i_ref: Reference current for turn-on voltage (default 1 mA = 1e-3 A).
        v_op: Operating voltage for forward current evaluation (default 2.0 V).
        vo_current: Current for Vo extraction (default 10 uA = 1e-5 A).
        ss_low: Lower current for subthreshold slope (default 1 uA = 1e-6 A).
        ss_high: Upper current for subthreshold slope (default 10 uA = 1e-5 A).
        rdiff_points: Number of points at the high-voltage end for linear regression.
        area_cm2: Optional device active area in cm^2 for Ron_sp computation.

    Returns:
        Dictionary of extracted scalar metrics.
    """
    v_grid = np.asarray(v_grid, dtype=np.float64)
    currents = np.asarray(currents, dtype=np.float64)

    # Floor nonpositive noise to avoid log issues
    safe_currents = np.maximum(currents, 1.0e-14)
    log_i = np.log10(safe_currents)

    # 1. Von at i_ref via log-linear interpolation
    # Since current increases with V, find where log_i crosses log10(i_ref)
    target_log_iref = np.log10(i_ref)
    if target_log_iref < log_i[0]:
        v_on = float(v_grid[0])
    elif target_log_iref > log_i[-1]:
        v_on = float(v_grid[-1])
    else:
        v_on = float(np.interp(target_log_iref, log_i, v_grid))

    # 2. Vo at vo_current (10 uA = 1e-5 A)
    target_log_vo = np.log10(vo_current)
    if target_log_vo < log_i[0]:
        v_o = float(v_grid[0])
    elif target_log_vo > log_i[-1]:
        v_o = float(v_grid[-1])
    else:
        v_o = float(np.interp(target_log_vo, log_i, v_grid))

    # 3. Subthreshold slope SS between ss_low and ss_high (1 uA to 10 uA) in mV/dec
    target_log_low = np.log10(ss_low)
    target_log_high = np.log10(ss_high)
    v_low = float(np.interp(target_log_low, log_i, v_grid))
    v_high = float(np.interp(target_log_high, log_i, v_grid))
    # SS = (V_high - V_low) / (log10(I_high) - log10(I_low)) in V/dec -> * 1000 for mV/dec
    decades = target_log_high - target_log_low
    ss_mv_dec = float((v_high - v_low) / decades * 1000.0)

    # 4. Rdiff (ohms) via linear regression of V vs I over the last rdiff_points
    v_tail = v_grid[-rdiff_points:]
    i_tail = currents[-rdiff_points:]
    # Fit V = m * I + c => m is dV/dI = Rdiff
    if len(i_tail) >= 2 and (np.max(i_tail) - np.min(i_tail)) > 1.0e-12:
        poly = np.polyfit(i_tail, v_tail, deg=1)
        r_diff = float(poly[0])
    else:
        r_diff = float((v_tail[-1] - v_tail[0]) / max(i_tail[-1] - i_tail[0], 1.0e-12))

    # 5. I at v_op (default 2.0 V) via linear interpolation
    i_at_vop = float(np.interp(v_op, v_grid, currents))

    metrics: dict[str, Any] = {
        "von": v_on,
        "v_o": v_o,
        "ss_mV_dec": ss_mv_dec,
        "r_diff": r_diff,
        "i_at_vop": i_at_vop,
    }

    if area_cm2 is not None and area_cm2 > 0:
        # Ron_sp in mohm*cm^2: R (ohm) * area (cm^2) * 1000 mohm/ohm
        metrics["ron_sp_mohm_cm2"] = float(r_diff * area_cm2 * 1000.0)

    return metrics
