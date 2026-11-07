import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(tmp_path):
    report = json.loads((ROOT / "reports/lab7_pipeline_report.json").read_text())
    model_path = ROOT / report["candidate_artifact"]
    metadata_path = ROOT / report["candidate_metadata"]
    app = create_app(model_path, metadata_path, tmp_path / "inference.jsonl")
    with TestClient(app) as test_client:
        yield test_client


def valid_car() -> dict:
    return {
        "name": "Maruti 800 AC",
        "year": 2007,
        "km_driven": 70000,
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
    }


def test_health_and_metadata(client):
    assert client.get("/health").json() == {
        "status": "healthy",
        "model_loaded": True,
    }
    assert client.get("/metadata").json()["model_family"] == "Random_Forest"


def test_predict_price_and_log_request(client, tmp_path):
    response = client.post("/predict", json=valid_car())

    assert response.status_code == 200
    result = response.json()
    assert result["currency"] == "INR"
    assert result["model_name"] == "Random_Forest"
    assert result["predicted_selling_price"] > 0

    log_path = tmp_path / "inference.jsonl"
    event = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert event["status"] == "success"
    assert event["predicted_selling_price"] == result["predicted_selling_price"]
    assert event["latency_ms"] >= 0


def test_unseen_category_uses_training_pipeline(client):
    payload = valid_car()
    payload["fuel"] = "Hydrogen"
    assert client.post("/predict", json=payload).status_code == 200


def test_exported_deployment_bundle_serves_prediction(tmp_path):
    app = create_app(log_path=tmp_path / "deployment-inference.jsonl")
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json=valid_car())
    assert response.status_code == 200
    assert response.json()["predicted_selling_price"] > 0


@pytest.mark.parametrize(
    ("changes", "expected_status"),
    [
        ({"year": 2021}, 422),
        ({"km_driven": -1}, 422),
        ({"unexpected": "value"}, 422),
    ],
)
def test_invalid_prediction_inputs(client, changes, expected_status):
    payload = valid_car()
    payload.update(changes)
    assert client.post("/predict", json=payload).status_code == expected_status