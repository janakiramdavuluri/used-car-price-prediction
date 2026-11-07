"""Train and compare a candidate from an explicitly supplied new-data batch."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.run_lab7_pipeline import validate_data
from src.preprocess import preprocessor
from src.train import random_forest


def retrain_candidate(batch_path: Path, report_path: Path | None = None) -> dict:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    if not batch_path.is_file():
        raise FileNotFoundError(f"Incoming training batch not found: {batch_path}")
    base_path = ROOT / config["data"]["raw_path"]
    base = validate_data(pd.read_csv(base_path), config["data"]["required_columns"])
    batch = validate_data(pd.read_csv(batch_path), config["data"]["required_columns"])
    if len(pd.concat([base, batch], ignore_index=True).drop_duplicates()) != len(base) + len(batch):
        raise ValueError("Incoming batch contains records already present in the training dataset")

    base_reference_year = int(base["year"].max())
    base["vehicle_age"] = base_reference_year - base["year"]
    feature_columns = [
        "name", "fuel", "seller_type", "transmission", "owner", "km_driven", "vehicle_age"
    ]
    x_base = base[feature_columns]
    y_base = base[config["data"]["target_column"]]
    x_train, x_test, y_train, y_test = train_test_split(
        x_base,
        y_base,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    base_report = json.loads((ROOT / config["paths"]["report"]).read_text(encoding="utf-8"))
    incumbent = joblib.load(ROOT / base_report["candidate_artifact"])
    incumbent_predictions = incumbent.predict(x_test)
    incumbent_metrics = _metrics(y_test, incumbent_predictions)

    candidate_reference_year = max(base_reference_year, int(batch["year"].max()))
    candidate_train_features = x_train.copy()
    candidate_train_features["vehicle_age"] = candidate_reference_year - (
        base_reference_year - x_train["vehicle_age"]
    )
    batch_features = batch.assign(vehicle_age=candidate_reference_year - batch["year"])[feature_columns]
    training_features = pd.concat([candidate_train_features, batch_features], ignore_index=True)
    training_target = pd.concat(
        [y_train.reset_index(drop=True), batch[config["data"]["target_column"]].reset_index(drop=True)],
        ignore_index=True,
    )
    candidate = Pipeline(
        [("preprocessor", clone(preprocessor)), ("model", clone(random_forest))]
    )
    candidate.fit(training_features, training_target)
    candidate_test = x_test.copy()
    candidate_test["vehicle_age"] = candidate_reference_year - (base_reference_year - x_test["vehicle_age"])
    candidate_metrics = _metrics(y_test, candidate.predict(candidate_test[feature_columns]))

    maximum_regression = float(config["retraining"]["maximum_mae_regression_fraction"])
    gate_passed = (
        candidate_metrics["mae"] <= float(config["quality_gate"]["maximum_mae"])
        and candidate_metrics["r2"] >= float(config["quality_gate"]["minimum_r2"])
        and candidate_metrics["mae"] <= incumbent_metrics["mae"] * (1 + maximum_regression)
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_dir = ROOT / config["paths"]["candidates_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"used_car_retrained_{timestamp}.joblib"
    metadata_path = artifact_path.with_suffix(".json")
    batch_hash = hashlib.sha256(batch_path.read_bytes()).hexdigest()
    metadata = {
        "model_family": "Random_Forest",
        "reference_year": candidate_reference_year,
        "feature_columns": feature_columns,
        "batch_sha256": batch_hash,
        "metrics": candidate_metrics,
    }
    if gate_passed:
        joblib.dump(candidate, artifact_path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="Used_Car_Retraining_Candidate") as run:
        mlflow.log_params(
            {
                "model_family": "Random_Forest",
                "batch_sha256": batch_hash,
                "new_training_rows": len(batch),
                "holdout_rows": len(y_test),
                "reference_year": candidate_reference_year,
                "random_state": config["split"]["random_state"],
                "quality_gate_passed": gate_passed,
            }
        )
        mlflow.log_metrics(
            {
                **{f"candidate_{key}": value for key, value in candidate_metrics.items()},
                **{f"incumbent_{key}": value for key, value in incumbent_metrics.items()},
            }
        )
        if gate_passed:
            mlflow.log_artifact(str(artifact_path), artifact_path="candidate")
            mlflow.log_artifact(str(metadata_path), artifact_path="candidate")

    report = {
        "status": "PASSED" if gate_passed else "QUALITY_GATE_FAILED",
        "incoming_batch": str(batch_path),
        "batch_sha256": batch_hash,
        "new_training_rows": len(batch),
        "holdout_rows": len(y_test),
        "incumbent_metrics": incumbent_metrics,
        "candidate_metrics": candidate_metrics,
        "mlflow_run_id": run.info.run_id,
        "maximum_mae_regression_fraction": maximum_regression,
        "candidate_artifact": str(artifact_path.relative_to(ROOT)) if gate_passed else None,
        "production_promotion": "not performed; requires explicit approval",
        "registry_action": "none",
    }
    report_path = report_path or ROOT / "outputs/retraining/retraining_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not gate_passed:
        raise RuntimeError("Retrained candidate failed the configured comparison gate")
    return report


def _metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mean_squared_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    retrain_candidate(args.batch, args.report)