"""Unit tests for ML surrogate baselines and evaluation metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pgsbd.eval.validate import evaluate_predictions
from pgsbd.ml.baselines import (
    HistGradientBoostingBaseline,
    KNNBaseline,
    PlainMLPBaseline,
    RidgeBaseline,
    save_scaler_artifact,
)


@pytest.fixture
def mock_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(42)
    v_grid = np.linspace(0.0, 4.0, 52)
    # 20 samples, 2 parameters
    X = np.random.uniform(low=[4.99, 300.0], high=[5.19, 500.0], size=(20, 2))
    # Simulated currents: exponentially increasing with V, linear tail
    currents = []
    for i in range(20):
        c = 1e-12 * np.exp(v_grid * 5.0)
        c = np.clip(c, 1e-14, 0.1)
        currents.append(c)
    Y = np.array(currents)
    log_Y = np.log10(Y)
    return v_grid, X, Y, log_Y


def test_ridge_baseline(mock_dataset):
    _v_grid, X, _Y, log_Y = mock_dataset
    model = RidgeBaseline()
    model.fit(X[:15], log_Y[:15])
    preds = model.predict(X[15:])
    assert preds.shape == (5, 52)
    assert np.all(np.isfinite(preds))


def test_hist_gradient_boosting_baseline(mock_dataset):
    _v_grid, X, _Y, log_Y = mock_dataset
    model = HistGradientBoostingBaseline(max_iter=10)
    model.fit(X[:15], log_Y[:15])
    preds = model.predict(X[15:])
    assert preds.shape == (5, 52)
    assert np.all(np.isfinite(preds))


def test_knn_baseline(mock_dataset):
    _v_grid, X, _Y, log_Y = mock_dataset
    model = KNNBaseline(n_neighbors=2)
    model.fit(X[:15], log_Y[:15])
    preds = model.predict(X[15:])
    assert preds.shape == (5, 52)
    assert np.all(np.isfinite(preds))


def test_plain_mlp_baseline(mock_dataset):
    _v_grid, X, _Y, log_Y = mock_dataset
    model = PlainMLPBaseline(hidden_dims=(32, 32), epochs=10, batch_size=8)
    model.fit(X[:15], log_Y[:15], X_val=X[15:], log_y_val=log_Y[15:])
    preds = model.predict(X[15:])
    assert preds.shape == (5, 52)
    assert np.all(np.isfinite(preds))


def test_evaluate_predictions(mock_dataset):
    v_grid, _X, Y, log_Y = mock_dataset
    # Test perfect prediction evaluation
    metrics = evaluate_predictions(v_grid, Y, log_Y)
    assert metrics["log10_mae_all"] < 1e-6
    assert metrics["vo_mae_V"] < 1e-6
    assert metrics["ss_mae_mV_dec"] < 1e-4
    assert metrics["rdiff_rel_err_pct"] < 1e-4


def test_save_scaler_artifact(tmp_path: Path):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    data = np.array([[5.0, 300.0], [5.1, 400.0], [5.2, 500.0]])
    scaler.fit(data)
    out_file = tmp_path / "test_scaler.json"
    res = save_scaler_artifact(scaler, out_file)
    assert out_file.exists()
    assert "mean" in res
    assert len(res["mean"]) == 2
