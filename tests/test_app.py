import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402
from train_model import MODEL_PATH, main as train  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def ensure_model():
    if not MODEL_PATH.exists():
        train()


@pytest.fixture()
def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


VALID = {
    "distance_km": 5,
    "prep_time_min": 15,
    "courier_experience_years": 3,
    "traffic": "medium",
    "weather": "clear",
}


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


def test_index_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Delivery Time Prediction" in res.data


def test_api_predict_valid(client):
    res = client.post("/api/predict", json=VALID)
    assert res.status_code == 200
    minutes = res.get_json()["delivery_time_min"]
    assert isinstance(minutes, float)
    assert 5 <= minutes <= 120


def test_longer_distance_takes_longer(client):
    near = client.post("/api/predict", json={**VALID, "distance_km": 1}).get_json()
    far = client.post("/api/predict", json={**VALID, "distance_km": 18}).get_json()
    assert far["delivery_time_min"] > near["delivery_time_min"]


def test_high_traffic_slower_than_low(client):
    low = client.post("/api/predict", json={**VALID, "traffic": "low"}).get_json()
    high = client.post("/api/predict", json={**VALID, "traffic": "high"}).get_json()
    assert high["delivery_time_min"] > low["delivery_time_min"]


@pytest.mark.parametrize(
    "bad",
    [
        {**VALID, "distance_km": -1},
        {**VALID, "distance_km": "abc"},
        {**VALID, "traffic": "gridlock"},
        {k: v for k, v in VALID.items() if k != "weather"},
    ],
)
def test_api_predict_invalid(client, bad):
    res = client.post("/api/predict", json=bad)
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_api_rejects_non_json(client):
    res = client.post("/api/predict", data="nope", content_type="text/plain")
    assert res.status_code == 400


def test_form_post_shows_prediction(client):
    res = client.post("/", data={k: str(v) for k, v in VALID.items()})
    assert res.status_code == 200
    assert b"Estimated delivery time" in res.data


def test_form_post_shows_error(client):
    res = client.post("/", data={**{k: str(v) for k, v in VALID.items()}, "distance_km": "0"})
    assert res.status_code == 400
    assert b"distance_km" in res.data
