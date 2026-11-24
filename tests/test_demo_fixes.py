"""Regression tests for design synthesis, breakdown derating, and reverse recovery."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pgsbd.api.main import app
from pgsbd.physics.breakdown import Breakdown1D

BASE_REQ = {
    "target_v_br": 600,
    "target_r_on_sp": 3.5,
    "target_v_o": 0.65,
    "target_ss": 85,
    "target_i_on_4v": 0.12,
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("method", ["nsga2", "gradient"])
@pytest.mark.parametrize("temp", [300.0, 350.0, 450.0])
def test_fixed_temperature_honoured(client, method, temp):
    """D1: every candidate must be at the requested operating temperature."""
    r = client.post("/api/design", json={**BASE_REQ, "method": method, "fixed_temperature": temp})
    assert r.status_code == 200
    for cand in r.json()["candidates"]:
        assert cand["parameters"]["temperature"] == pytest.approx(temp, abs=1e-6)


def test_breakdown_is_derated():
    """D2: reported breakdown is eta_term x the ideal 1-D limit."""
    bd = Breakdown1D()
    res = bd.calculate_numpy(t_d_um=9.0, n_net_cm3=6e15)
    assert res["v_br"] == pytest.approx(0.5 * res["v_br_ideal"])
    assert res["v_br"] < res["v_br_ideal"]


def test_design_reports_derated_breakdown(client):
    r = client.post("/api/design", json={**BASE_REQ, "method": "nsga2"})
    for cand in r.json()["candidates"]:
        m = cand["metrics"]
        assert m["v_br"] == pytest.approx(0.5 * m["v_br_ideal"], rel=1e-6)
        assert m["v_br"] >= BASE_REQ["target_v_br"] * 0.9 or cand["verdict"] != "met"


def test_reverse_flags_poor_fit(client):
    """D5: a curve that does not come from this device must not return status 'met'."""
    v = np.linspace(0, 4, 52).tolist()
    r = client.post("/api/reverse", json={"v": v, "i": np.linspace(1e-9, 1e-1, 52).tolist()})
    assert r.status_code == 200
    assert r.json()["status"] == "poor_fit"


def test_reverse_accepts_tiny_negative_current(client):
    """D5: raw TCAD curves contain ~-1e-18 A at V=0 and must be accepted."""
    v = np.linspace(0, 4, 52)
    i = np.logspace(-12, -1, 52)
    i[0] = -1.4e-18
    r = client.post("/api/reverse", json={"v": v.tolist(), "i": i.tolist()})
    assert r.status_code == 200


def test_export_uses_requested_temperature(client):
    """D8: exported Sdevice deck uses the requested temperature."""
    r = client.post(
        "/api/export/tcad",
        json={
            "parameters": {
                "WF": 5.05,
                "temperature": 350.0,
                "t_D_um": 9.0,
                "N_net_cm3": 6e15,
                "d_um": 200.0,
            }
        },
    )
    assert r.status_code == 200
    assert "Temperature = 350.0" in r.json()["sdevice_cmd"]
