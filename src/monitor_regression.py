"""Summarize regression API traffic, drift, and optional labeled performance."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]


def population_stability_index(reference: np.ndarray, current: np.ndarray) -> float | None:
    reference = np.asarray(reference)
    current = np.asarray(current)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if not len(reference) or not len(current):
        return None
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_counts = np.histogram(reference, bins=edges)[0].astype(float)
    current_counts = np.histogram(current, bins=edges)[0].astype(float)
    reference_share = (reference_counts + 1e-6) / (reference_counts.sum() + 1e-6 * len(edges - 1))
    current_share = (current_counts + 1e-6) / (current_counts.sum() + 1e-6 * len(edges - 1))
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float | None:
    if reference.empty or current.empty:
        return None
    categories = set(reference.astype(str)) | set(current.astype(str))
    if not categories:
        return None
    reference_share = reference.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0)
    current_share = current.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0)
    reference_share = (reference_share + 1e-6) / (1 + 1e-6 * len(categories))
    current_share = (current_share + 1e-6) / (1 + 1e-6 * len(categories))
    return float(((current_share - reference_share) * np.log(current_share / reference_share)).sum())


def read_events(log_path: Path) -> list[dict]:
    if not log_path.is_file():
        return []
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_report(
    events: list[dict],
    baseline: pd.DataFrame,
    baseline_predictions: np.ndarray,
    labels: pd.DataFrame | None,
    psi_threshold: float,
    maximum_mae: float,
    minimum_r2: float,
    minimum_samples: int = 30,
) -> dict:
    successes = [event for event in events if event.get("status") == "success"]
    failures = [event for event in events if event.get("status") != "success"]
    latencies = [float(event["latency_ms"]) for event in events if event.get("latency_ms") is not None]
    predictions = np.asarray(
        [event["predicted_selling_price"] for event in successes], dtype=float
    )
    report = {
        "status": "insufficient_traffic" if not events else "completed",
        "prediction_volume": len(events),
        "success_count": len(successes),
        "failure_count": len(failures),
        "success_rate": len(successes) / len(events) if events else None,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else None,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "input_drift_psi": {},
        "prediction_drift_psi": population_stability_index(baseline_predictions, predictions),
        "minimum_samples_for_drift": minimum_samples,
        "actual_performance": {"status": "insufficient_labels", "labeled_rows": 0},
        "limitations": (
            "Traffic-derived drift reflects only the logged sample; demonstration traffic may be simulated. "
            "Actual-price metrics require labels joined by request_id and do not establish population-level performance."
        ),
    }

    if successes:
        current = pd.DataFrame([event["input"] for event in successes])
        reference_year = int(baseline["year"].max())
        current["vehicle_age"] = reference_year - pd.to_numeric(current["year"])
        baseline_features = baseline.assign(vehicle_age=reference_year - baseline["year"])
        for column in ("km_driven", "vehicle_age"):
            report["input_drift_psi"][column] = population_stability_index(
                baseline_features[column].to_numpy(dtype=float),
                current[column].to_numpy(dtype=float),
            )
        for column in ("fuel", "seller_type", "transmission", "owner"):
            report["input_drift_psi"][column] = categorical_psi(
                baseline[column], current[column]
            )
    all_psi = [
        value
        for value in [*report["input_drift_psi"].values(), report["prediction_drift_psi"]]
        if value is not None
    ]
    report["drift_status"] = (
        "insufficient_traffic"
        if not all_psi or len(successes) < minimum_samples
        else "DRIFT" if any(value > psi_threshold for value in all_psi) else "stable"
    )

    if labels is not None and not labels.empty:
        required = {"request_id", "actual_selling_price"}
        if not required.issubset(labels.columns):
            raise ValueError(f"Label file must contain columns: {sorted(required)}")
        joined = pd.DataFrame(
            [
                {
                    "request_id": event.get("request_id"),
                    "predicted": event.get("predicted_selling_price"),
                }
                for event in successes
            ]
        ).merge(labels[list(required)], on="request_id", how="inner")
        if not joined.empty:
            actual = pd.to_numeric(joined["actual_selling_price"], errors="raise")
            predicted = pd.to_numeric(joined["predicted"], errors="raise")
            mae = float(mean_absolute_error(actual, predicted))
            rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
            r2 = float(r2_score(actual, predicted)) if len(joined) > 1 else None
            passed = len(joined) > 1 and mae <= maximum_mae and r2 >= minimum_r2
            report["actual_performance"] = {
                "status": "PASSED" if passed else "DEGRADED_OR_INSUFFICIENT",
                "labeled_rows": len(joined),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
    return report


def run_monitoring(log_path: Path | None = None, labels_path: Path | None = None) -> dict:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    dataset_path = ROOT / config["data"]["raw_path"]
    baseline = pd.read_csv(dataset_path).drop_duplicates()
    baseline["year"] = pd.to_numeric(baseline["year"], errors="raise")
    report_data = json.loads((ROOT / config["paths"]["report"]).read_text(encoding="utf-8"))
    model_path = ROOT / report_data["candidate_artifact"]
    model = joblib.load(model_path)
    reference_year = int(report_data["reference_year"])
    features = baseline.assign(vehicle_age=reference_year - baseline["year"])[
        report_data["feature_columns"]
    ]
    baseline_predictions = model.predict(features)
    log_path = log_path or ROOT / config["paths"]["inference_log"]
    labels_path = labels_path or ROOT / config["paths"]["labeled_predictions"]
    labels = pd.read_csv(labels_path) if labels_path.is_file() else None
    report = build_report(
        read_events(log_path),
        baseline,
        baseline_predictions,
        labels,
        float(config["quality_gate"]["maximum_distribution_psi"]),
        float(config["quality_gate"]["maximum_mae"]),
        float(config["quality_gate"]["minimum_r2"]),
        int(config["monitoring"]["minimum_samples_for_drift"]),
    )
    output = ROOT / config["paths"]["monitoring_report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--labels-path", type=Path)
    arguments = parser.parse_args()
    run_monitoring(arguments.log_path, arguments.labels_path)