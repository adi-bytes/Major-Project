"""Unit tests for leakage-safe grouped dataset splitting."""

from __future__ import annotations

import json
from pathlib import Path

from pgsbd.data.load import load_dataset
from pgsbd.data.splits import create_grouped_splits


def test_split_leakage_and_integrity():
    df, _, _, _ = load_dataset()
    splits = create_grouped_splits(df, seed=42)

    indices = splits["indices"]
    train_idx = indices["train"]
    val_idx = indices["val"]
    test_idx = indices["test"]
    extrap_idx = indices["extrapolation"]

    # 1. Check all index sets are mutually disjoint
    s_train = set(train_idx)
    s_val = set(val_idx)
    s_test = set(test_idx)
    s_extrap = set(extrap_idx)

    assert s_train.isdisjoint(s_val), "Train and Val indices overlap!"
    assert s_train.isdisjoint(s_test), "Train and Test indices overlap!"
    assert s_val.isdisjoint(s_test), "Val and Test indices overlap!"
    assert s_train.isdisjoint(s_extrap), "Train and Extrapolation indices overlap!"
    assert s_val.isdisjoint(s_extrap), "Val and Extrapolation indices overlap!"
    assert s_test.isdisjoint(s_extrap), "Test and Extrapolation indices overlap!"

    # 2. Total row count matches
    total_split_rows = len(s_train) + len(s_val) + len(s_test) + len(s_extrap)
    assert total_split_rows == len(df), f"Expected {len(df)} total rows, got {total_split_rows}"

    # 3. Check (WF, temperature) operating pairs are 100% disjoint (NO DATA LEAKAGE)
    train_pairs = set(zip(df.loc[train_idx, "WF"], df.loc[train_idx, "temperature"]))
    val_pairs = set(zip(df.loc[val_idx, "WF"], df.loc[val_idx, "temperature"]))
    test_pairs = set(zip(df.loc[test_idx, "WF"], df.loc[test_idx, "temperature"]))
    extrap_pairs = set(zip(df.loc[extrap_idx, "WF"], df.loc[extrap_idx, "temperature"]))

    assert train_pairs.isdisjoint(val_pairs), (
        "Data leakage: train and val share (WF, T) operating points!"
    )
    assert train_pairs.isdisjoint(test_pairs), (
        "Data leakage: train and test share (WF, T) operating points!"
    )
    assert val_pairs.isdisjoint(test_pairs), (
        "Data leakage: val and test share (WF, T) operating points!"
    )
    assert train_pairs.isdisjoint(extrap_pairs), (
        "Data leakage: train and extrap share (WF, T) operating points!"
    )
    assert val_pairs.isdisjoint(extrap_pairs), (
        "Data leakage: val and extrap share (WF, T) operating points!"
    )

    # 4. Verify extrapolation corner condition
    for idx in extrap_idx:
        assert df.loc[idx, "WF"] > 5.16
        assert df.loc[idx, "temperature"] > 470.0


def test_splits_artifact_file_reproducibility(tmp_path: Path):
    df, _, _, _ = load_dataset()
    artifact_path = tmp_path / "splits_test.json"
    splits1 = create_grouped_splits(df, seed=42, output_json_path=artifact_path)

    with open(artifact_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["counts"] == splits1["counts"]
    assert loaded["indices"]["train"] == splits1["indices"]["train"]
