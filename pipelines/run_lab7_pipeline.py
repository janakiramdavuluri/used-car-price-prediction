"""Run the Used Cars regression pipeline without changing registry aliases."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess import preprocessor
from src.train import extra_tree, linear_reg, random_forest, ridge_reg


def load_config() -> dict:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_data(data: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    missing = sorted(set(required_columns) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data[required_columns].isna().any().any():
        raise ValueError("Required columns contain missing values")

    for column in ("year", "selling_price", "km_driven"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any():
            raise ValueError(f"Column {column} contains non-numeric values")
    if (data["selling_price"] <= 0).any():
        raise ValueError("selling_price must be positive")
    if (data["km_driven"] < 0).any():
        raise ValueError("km_driven cannot be negative")
    if (data["year"] % 1 != 0).any():
        raise ValueError("year must contain whole numbers")
    return data.drop_duplicates().copy()


def compare_registry_reference(candidate: dict) -> dict:
    client = MlflowClient()
    reference = None
    for alias in ("production", "candidate"):
        try:
            version = client.get_model_version_by_alias("UsedCarPricePredictor", alias)
            reference = {
                "alias": alias,
                "version": version.version,
                "metrics": client.get_run(version.run_id).data.metrics,
            }
            break
        except MlflowException:
            continue
    if reference is None:
        return {"status": "no_registry_reference", "promotion_requires_approval": True}

    metrics = reference["metrics"]
    improves_reference = (
        "mae" in metrics
        and "r2" in metrics
        and candidate["mae"] < metrics["mae"]
        and candidate["r2"] >= metrics["r2"]
    )
    return {
        "reference_alias": reference["alias"],
        "reference_version": reference["version"],
        "reference_metrics": metrics,
        "candidate_run_id": candidate["run_id"],
        "candidate_improves_reference": improves_reference,
        "decision": "eligible_for_explicit_review" if improves_reference else "retain_reference",
        "promotion_requires_approval": True,
        "registry_action": "none",
    }


def run_pipeline() -> dict:
    config = load_config()
    data_path = ROOT / config["data"]["raw_path"]
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    versions_dir = ROOT / config["paths"]["dataset_versions_dir"]
    versions_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = versions_dir / f"{data_hash}.csv"
    if not snapshot_path.exists():
        shutil.copy2(data_path, snapshot_path)
    data = validate_data(
        pd.read_csv(data_path), config["data"]["required_columns"]
    )
    reference_year = int(data["year"].max())
    data["vehicle_age"] = reference_year - data["year"]

    feature_columns = [
        "name",
        "fuel",
        "seller_type",
        "transmission",
        "owner",
        "km_driven",
        "vehicle_age",
    ]
    features = data[feature_columns]
    target = data[config["data"]["target_column"]]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    estimators = {
        "Linear_Regression": linear_reg,
        "Ridge_Regression": ridge_reg,
        "Random_Forest": random_forest,
        "Extra_Trees": extra_tree,
    }
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    results = []
    fitted_models = {}
    for name, estimator in estimators.items():
        model = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("model", clone(estimator)),
            ]
        )
        with mlflow.start_run(run_name=name) as active_run:
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            metrics = {
                "mae": float(mean_absolute_error(y_test, predictions)),
                "mse": float(mean_squared_error(y_test, predictions)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "r2": float(r2_score(y_test, predictions)),
            }
            mlflow.log_params(
                {
                    "model_family": name,
                    "reference_year": reference_year,
                    "test_size": config["split"]["test_size"],
                    "random_state": config["split"]["random_state"],
                    "dataset_sha256": data_hash,
                    "training_rows": len(x_train),
                    "test_rows": len(x_test),
                }
            )
            mlflow.log_metrics(metrics)
            results.append(
                {
                    "model": name,
                    **metrics,
                    "run_id": active_run.info.run_id,
                }
            )
            fitted_models[name] = model

    results.sort(key=lambda result: result["mae"])
    selected = results[0]
    gate = {
        "maximum_mae": config["quality_gate"]["maximum_mae"],
        "minimum_r2": config["quality_gate"]["minimum_r2"],
        "passed": (
            selected["mae"] <= config["quality_gate"]["maximum_mae"]
            and selected["r2"] >= config["quality_gate"]["minimum_r2"]
        ),
    }

    report = {
        "status": "PASSED" if gate["passed"] else "QUALITY_GATE_FAILED",
        "dataset": str(data_path.relative_to(ROOT)),
        "dataset_sha256": data_hash,
        "dataset_snapshot": str(snapshot_path.relative_to(ROOT)),
        "rows_after_validation": len(data),
        "reference_year": reference_year,
        "feature_columns": feature_columns,
        "metrics": results,
        "selected_model": selected["model"],
        "quality_gate": gate,
        "lifecycle_comparison": compare_registry_reference(selected),
        "registry_action": "none; registration and alias changes require approval",
    }

    report_path = ROOT / config["paths"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if gate["passed"]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        artifact_dir = ROOT / config["paths"]["candidates_dir"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"used_car_candidate_{timestamp}.joblib"
        joblib.dump(fitted_models[selected["model"]], artifact_path)
        metadata_path = artifact_path.with_suffix(".json")
        metadata = {
            "model_family": selected["model"],
            "reference_year": reference_year,
            "feature_columns": feature_columns,
            "dataset_sha256": data_hash,
            "run_id": selected["run_id"],
            "metrics": {
                key: selected[key] for key in ("mae", "mse", "rmse", "r2")
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        report["candidate_artifact"] = str(artifact_path.relative_to(ROOT))
        report["candidate_metadata"] = str(metadata_path.relative_to(ROOT))
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not gate["passed"]:
        raise RuntimeError("Selected model failed the configured regression quality gate")
    return report


if __name__ == "__main__":
    run_pipeline()