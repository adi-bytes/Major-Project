"""DBSCAN and k-means diversity clustering for Pareto front candidates."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans


def cluster_and_select_candidates(
    candidates: list[dict[str, Any]],
    n_clusters: int = 4,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Cluster candidates on the Pareto front and select diverse representative designs.

    Selects distinct archetypes:
    1. Maximum Breakdown Voltage (High-Voltage specialist)
    2. Minimum Specific On-Resistance (Efficiency / Low-Loss specialist)
    3. Balanced Compromise (Knee point closest to normalized utopian point)
    4. Minimum Turn-On Voltage (Low forward drop specialist)
    5. Maximum Forward Current (Power capacity specialist)

    Args:
        candidates: List of evaluated candidate dictionaries.
        n_clusters: Number of representative candidates to return.
        weights: Optional preference weights.

    Returns:
        List of selected representative candidates labeled with an archetype title.
    """
    if not candidates:
        return []

    if len(candidates) <= n_clusters:
        for idx, c in enumerate(candidates):
            c["archetype"] = f"Candidate {idx + 1}"
        return candidates

    # Extract feature matrix for clustering: [V_br, R_on_sp, V_o, I_on_4V]
    v_brs = np.array([c["metrics"]["v_br"] for c in candidates])
    rons = np.array([c["metrics"]["r_on_sp"] for c in candidates])
    vos = np.array([c["metrics"]["v_o"] for c in candidates])
    ions = np.array([c["metrics"]["i_on_4V"] for c in candidates])

    # Normalize features to [0, 1]
    def norm_feat(arr: np.ndarray) -> np.ndarray:
        span = np.max(arr) - np.min(arr)
        if span < 1e-12:
            return np.zeros_like(arr)
        return (arr - np.min(arr)) / span

    v_br_norm = norm_feat(v_brs)
    ron_norm = norm_feat(rons)
    vo_norm = norm_feat(vos)
    ion_norm = norm_feat(ions)

    # 1. Specialist: Max Breakdown
    idx_max_vbr = int(np.argmax(v_brs))
    # 2. Specialist: Min On-Resistance
    idx_min_ron = int(np.argmin(rons))
    # 3. Specialist: Min Turn-on
    idx_min_vo = int(np.argmin(vos))
    # 4. Specialist: Max Current
    idx_max_ion = int(np.argmax(ions))

    # 5. Balanced compromise (distance to utopian: max v_br, min ron, min vo, max ion)
    # Utopian target: [1.0, 0.0, 0.0, 1.0]
    dist_utopian = np.sqrt(
        (1.0 - v_br_norm) ** 2
        + (ron_norm - 0.0) ** 2
        + (vo_norm - 0.0) ** 2
        + (1.0 - ion_norm) ** 2
    )
    idx_balanced = int(np.argmin(dist_utopian))

    selected_indices: list[int] = []
    archetypes: list[str] = []

    def try_add(idx: int, title: str) -> None:
        if idx not in selected_indices:
            selected_indices.append(idx)
            archetypes.append(title)

    try_add(idx_balanced, "Balanced Compromise")
    try_add(idx_max_vbr, "Maximum Breakdown")
    try_add(idx_min_ron, "Lowest On-Resistance")
    try_add(idx_min_vo, "Lowest Turn-On")
    try_add(idx_max_ion, "High Forward Current")

    # If we still need more to satisfy n_clusters, run KMeans on remaining candidates
    if len(selected_indices) < n_clusters and len(candidates) > len(selected_indices):
        X = np.column_stack([v_br_norm, ron_norm, vo_norm, ion_norm])
        remaining = [i for i in range(len(candidates)) if i not in selected_indices]
        n_needed = min(n_clusters - len(selected_indices), len(remaining))
        if n_needed > 0:
            kmeans = KMeans(n_clusters=n_needed, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X[remaining])
            for cluster_k in range(n_needed):
                cluster_members = [remaining[i] for i, lbl in enumerate(labels) if lbl == cluster_k]
                if cluster_members:
                    # Pick member closest to cluster center
                    center = kmeans.cluster_centers_[cluster_k]
                    best_m = min(
                        cluster_members,
                        key=lambda m: float(np.linalg.norm(X[m] - center)),
                    )
                    try_add(best_m, f"Pareto Cluster {len(selected_indices) + 1}")

    result = []
    for idx, title in zip(selected_indices, archetypes):
        cand = dict(candidates[idx])
        cand["archetype"] = title
        result.append(cand)

    return result
