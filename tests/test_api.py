"""API contract and integration tests for PG-SBD Designer FastAPI backend (Phase 6)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pgsbd.api.main import app
from pgsbd.data.load import load_dataset


@pytest.fixture(scope="module")
def client() -> TestClient:
    """TestClient instance with lifespan startup executed."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_curve():
    """Load a sample held-out curve for reverse mode testing."""
    _, v_grid, X, Y = load_dataset()
    with open("artifacts/splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_idx = splits["indices"]["test"]
    idx = test_idx[0]
    return {
        "wf": float(X[idx, 0]),
        "temperature": float(X[idx, 1]),
        "v": [float(v) for v in v_grid],
        "i": [float(curr) for curr in Y[idx]],
    }


def test_root_endpoint(client: TestClient):
    """Verify root / endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "PG-SBD Designer API"
    assert "/docs" in data["docs_url"]
    assert "Not validated for fabrication" in data["honesty_notice"]


def test_health_endpoint(client: TestClient):
    """Verify /api/health reports operational status and loaded artifacts."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["evidence_level"] == "tcad_validated"
    assert "Beta-Ga2O3" in data["device"]
    assert data["artifacts_loaded"]["ensemble_models"] is True
    assert data["artifacts_loaded"]["reverse_engine"] is True
    assert data["artifacts_loaded"]["calibration"] is True
    assert data["artifacts_loaded"]["validation_metrics"] is True
    assert "Not validated for fabrication" in data["honesty_notice"]


def test_meta_and_devices_endpoints(client: TestClient):
    """Verify /api/meta and its alias /api/devices return identical schema and bounds."""
    resp_meta = client.get("/api/meta")
    resp_devices = client.get("/api/devices")

    assert resp_meta.status_code == 200
    assert resp_devices.status_code == 200
    assert resp_meta.json() == resp_devices.json()

    data = resp_meta.json()
    assert data["device_name"] == "Vertical Beta-Ga2O3 Schottky Barrier Diode (SBD)"
    assert data["material"] == "Beta-Ga2O3"
    assert data["evidence_level"] == "tcad_validated"

    # Verify sweepable variable bounds
    vars_dict = {v["name"]: v for v in data["variables"]}
    assert "WF" in vars_dict
    assert vars_dict["WF"]["min_val"] == 4.99
    assert vars_dict["WF"]["max_val"] == 5.19
    assert vars_dict["WF"]["unit"] == "eV"

    assert "temperature" in vars_dict
    assert vars_dict["temperature"]["min_val"] == 300.0
    assert vars_dict["temperature"]["max_val"] == 500.0
    assert vars_dict["temperature"]["unit"] == "K"

    # Verify fixed parameters
    assert data["fixed_parameters"]["t_Drift_um"] == 9.0
    assert data["fixed_parameters"]["N_net_cm3"] == 6.0e15


def test_predict_valid_payload(client: TestClient):
    """Verify /api/predict generates complete I-V response on valid (WF, T)."""
    payload = {"wf": 5.08, "temperature": 380.0}
    response = client.post("/api/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["V"]) == 52
    assert len(data["I_mean"]) == 52
    assert len(data["I_std"]) == 52
    assert len(data["I_phys"]) == 52
    assert data["evidence_level"] == "tcad_validated"

    # Figures of merit
    assert "v_o" in data["metrics"]
    assert "r_on_sp" in data["metrics"]
    assert "ss_mV_dec" in data["metrics"]
    assert data["metrics"]["v_o"] > 0.0


def test_predict_custom_voltage_grid(client: TestClient):
    """Verify /api/predict handles custom voltage grids."""
    v_custom = [0.0, 0.5, 1.0, 2.0, 3.0, 3.9]
    payload = {"wf": 5.10, "temperature": 400.0, "v_custom": v_custom}
    response = client.post("/api/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["V"]) == len(v_custom)
    assert np.allclose(data["V"], v_custom)
    assert len(data["I_mean"]) == len(v_custom)


@pytest.mark.parametrize(
    "invalid_payload,expected_field",
    [
        ({"wf": 4.90, "temperature": 350.0}, "wf"),  # WF below 4.99
        ({"wf": 5.25, "temperature": 350.0}, "wf"),  # WF above 5.19
        ({"wf": 5.05, "temperature": 250.0}, "temperature"),  # T below 300 K
        ({"wf": 5.05, "temperature": 550.0}, "temperature"),  # T above 500 K
        (
            {"wf": 5.05, "temperature": 350.0, "client_path": "/var/data"},
            "client_path",
        ),  # Extra field
        ({}, "wf"),  # Missing fields
    ],
)
def test_predict_invalid_payloads_422(client: TestClient, invalid_payload, expected_field):
    """Verify strict 422 bound validation with descriptive messages on invalid inputs."""
    response = client.post("/api/predict", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"
    detail_str = str(data["detail"]).lower()
    assert expected_field.lower() in detail_str


def test_reverse_valid_curve(client: TestClient, sample_curve):
    """Verify /api/reverse parameter recovery on a held-out TCAD curve."""
    payload = {"v": sample_curve["v"], "i": sample_curve["i"]}
    response = client.post("/api/reverse", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "met"
    assert data["evidence_level"] == "tcad_validated"

    rec = data["recovered"]
    assert 4.99 <= rec["WF"] <= 5.19
    assert 300.0 <= rec["T"] <= 500.0
    assert rec["WF_uncertainty"] > 0
    assert rec["T_uncertainty"] > 0

    assert "mae_decades" in data["fit_residual"]
    assert "index" in data["nearest_tcad"]
    assert len(data["reconstructed"]["V"]) == 52


@pytest.mark.parametrize(
    "invalid_reverse_payload,error_fragment",
    [
        ({"v": [0.0, 1.0, 2.0], "i": [1e-12, 1e-6]}, "length mismatch"),
        ({"v": [-0.5, 1.0], "i": [1e-12, 1e-6]}, "non-negative"),
        ({"v": [0.0, 1.0], "i": [-1e-12, 1e-6]}, "strictly positive"),
        ({"v": [], "i": []}, "min_length"),
        ({"v": [0.0, 1.0], "i": [1e-12, 1e-6], "dataset_file": "secret.csv"}, "extra"),
    ],
)
def test_reverse_invalid_payloads_422(client: TestClient, invalid_reverse_payload, error_fragment):
    """Verify 422 rejection on mismatched or non-physical reverse inputs."""
    response = client.post("/api/reverse", json=invalid_reverse_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"


def test_validation_endpoint(client: TestClient):
    """Verify /api/validation returns the full metrics artifact."""
    response = client.get("/api/validation")
    assert response.status_code == 200
    data = response.json()
    assert data["evidence_level"] == "tcad_validated"
    metrics = data["metrics"]
    assert "test" in metrics
    assert "extrapolation" in metrics
    assert "ablation" in metrics
    assert "reverse_mode" in metrics
