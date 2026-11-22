"""Leakage-safe grouped dataset splitting module.

Ensures that identical or repeated (WF, T) operating points never leak across splits.
Extracts:
1. Grouped 70/15/15 train/val/test split over unique (WF, T) in-domain pairs.
2. Held-out extrapolation corner block (WF > WF_min and T > T_min) kept strictly separate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def create_grouped_splits(
    df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    extrapolation_corner: dict[str, float] | None = None,
    output_json_path: str | Path | None = "artifacts/splits.json",
) -> dict[str, Any]:
    """Create leakage-safe splits grouped by unique (WF, temperature) pairs.

    Args:
        df: Input DataFrame containing 'WF' and 'temperature' columns.
        seed: Random seed for deterministic shuffling.
        train_ratio: Ratio for training set (from in-domain pool).
        val_ratio: Ratio for validation set.
        test_ratio: Ratio for test set.
        extrapolation_corner: Dict with 'WF_min' and 'T_min' defining extrapolation block.
        output_json_path: Optional path to save splits.json artifact.

    Returns:
        Dict with split indices, summary statistics, and metadata.
    """
    if extrapolation_corner is None:
        extrapolation_corner = {"WF_min": 5.16, "T_min": 470.0}

    wf_min_corner = extrapolation_corner.get("WF_min", 5.16)
    t_min_corner = extrapolation_corner.get("T_min", 470.0)

    # Flag extrapolation samples
    is_extrap = (df["WF"] > wf_min_corner) & (df["temperature"] > t_min_corner)
    extrap_indices = df.index[is_extrap].tolist()
    in_domain_indices = df.index[~is_extrap].tolist()

    df_in_domain = df.loc[in_domain_indices].copy()

    # Create group identifier for unique (WF, temperature) pairs
    # Round to 6 decimal places to avoid float precision mismatch
    df_in_domain["group_id"] = (
        df_in_domain["WF"].round(6).astype(str)
        + "_"
        + df_in_domain["temperature"].round(6).astype(str)
    )

    unique_groups = df_in_domain["group_id"].unique()
    rng = np.random.RandomState(seed)
    shuffled_groups = rng.permutation(unique_groups)

    n_groups = len(shuffled_groups)
    n_train = int(np.floor(train_ratio * n_groups))
    n_val = int(np.floor(val_ratio * n_groups))

    train_groups = set(shuffled_groups[:n_train])
    val_groups = set(shuffled_groups[n_train : n_train + n_val])
    test_groups = set(shuffled_groups[n_train + n_val :])

    train_idx = df_in_domain.index[df_in_domain["group_id"].isin(train_groups)].tolist()
    val_idx = df_in_domain.index[df_in_domain["group_id"].isin(val_groups)].tolist()
    test_idx = df_in_domain.index[df_in_domain["group_id"].isin(test_groups)].tolist()

    # Leakage verification assertion: sets of (WF, T) in train, val, test must be disjoint
    train_pairs = set(zip(df.loc[train_idx, "WF"], df.loc[train_idx, "temperature"]))
    val_pairs = set(zip(df.loc[val_idx, "WF"], df.loc[val_idx, "temperature"]))
    test_pairs = set(zip(df.loc[test_idx, "WF"], df.loc[test_idx, "temperature"]))
    extrap_pairs = set(zip(df.loc[extrap_indices, "WF"], df.loc[extrap_indices, "temperature"]))

    assert train_pairs.isdisjoint(val_pairs), "Leakage detected: train and val share (WF, T) pairs!"
    assert train_pairs.isdisjoint(test_pairs), (
        "Leakage detected: train and test share (WF, T) pairs!"
    )
    assert val_pairs.isdisjoint(test_pairs), "Leakage detected: val and test share (WF, T) pairs!"
    assert train_pairs.isdisjoint(extrap_pairs), (
        "Leakage detected: train and extrap share (WF, T) pairs!"
    )

    splits = {
        "seed": seed,
        "total_samples": len(df),
        "extrapolation_corner_definition": {
            "WF_min": wf_min_corner,
            "T_min": t_min_corner,
        },
        "counts": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
            "extrapolation": len(extrap_indices),
            "in_domain_unique_groups": int(n_groups),
            "train_unique_groups": len(train_groups),
            "val_unique_groups": len(val_groups),
            "test_unique_groups": len(test_groups),
        },
        "indices": {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
            "extrapolation": extrap_indices,
        },
    }

    if output_json_path is not None:
        out_p = Path(output_json_path)
        if not out_p.is_absolute():
            root = Path(__file__).resolve().parents[3]
            out_p = root / output_json_path
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)

    return splits
