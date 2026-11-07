import pandas as pd

from src.monitor_regression import build_report


def baseline_frame():
    return pd.DataFrame(
        {
            "year": [2018, 2019, 2020, 2018],
            "km_driven": [10000, 20000, 30000, 40000],
            "fuel": ["Petrol", "Diesel", "Petrol", "Diesel"],
            "seller_type": ["Individual"] * 4,
            "transmission": ["Manual"] * 4,
            "owner": ["First Owner"] * 4,
        }
    )


def test_monitor_reports_volume_drift_and_labeled_regression_metrics():
    events = [
        {
            "request_id": "one",
            "status": "success",
            "input": {
                "year": 2019,
                "km_driven": 20000,
                "fuel": "Petrol",
                "seller_type": "Individual",
                "transmission": "Manual",
                "owner": "First Owner",
            },
            "predicted_selling_price": 100000,
            "latency_ms": 4.0,
        },
        {
            "request_id": "two",
            "status": "failure",
            "input": {},
            "predicted_selling_price": None,
            "latency_ms": 2.0,
        },
    ]
    labels = pd.DataFrame(
        {"request_id": ["one"], "actual_selling_price": [110000]}
    )

    report = build_report(
        events,
        baseline_frame(),
        baseline_predictions=pd.Series([90000, 100000, 110000, 120000]).to_numpy(),
        labels=labels,
        psi_threshold=0.2,
        maximum_mae=200000,
        minimum_r2=0.0,
    )

    assert report["prediction_volume"] == 2
    assert report["success_count"] == 1
    assert report["failure_count"] == 1
    assert report["mean_latency_ms"] == 3.0
    assert report["drift_status"] == "insufficient_traffic"
    assert report["actual_performance"]["labeled_rows"] == 1
    assert report["actual_performance"]["r2"] is None
    assert report["actual_performance"]["status"] == "DEGRADED_OR_INSUFFICIENT"


def test_monitor_labels_no_traffic_as_insufficient():
    report = build_report(
        [],
        baseline_frame(),
        baseline_predictions=pd.Series([1, 2, 3, 4]).to_numpy(),
        labels=None,
        psi_threshold=0.2,
        maximum_mae=200000,
        minimum_r2=0.0,
    )
    assert report["status"] == "insufficient_traffic"
    assert report["drift_status"] == "insufficient_traffic"