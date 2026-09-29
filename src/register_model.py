"""Register the latest completed Random Forest MLflow run without retraining.

Run from the project root:
    python src/register_model.py
"""

import mlflow
from mlflow import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "Used_Car_Price_Prediction"
MODEL_NAME = "UsedCarPricePredictor"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {EXPERIMENT}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "tags.mlflow.runName = 'Random_Forest' "
            "AND attributes.status = 'FINISHED'"
        ),
        order_by=["start_time DESC"],
    )
    if runs.empty:
        raise RuntimeError("No completed Random Forest run found.")

    run_id = runs.iloc[0]["run_id"]
    print(f"Selected run: {run_id}")

    # Recent MLflow versions may log models as separate entities rather than
    # traditional run artifacts. Prefer a logged-model URI when available.
    logged_models = client.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"source_run_id = '{run_id}'",
    )
    model_uri = None
    for model in logged_models:
        if model.name == "model":
            model_uri = f"models:/{model.model_id}"
            break
    if model_uri is None:
        model_uri = f"runs:/{run_id}/model"

    print(f"Registering: {model_uri}")
    registered = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    client.set_registered_model_alias(
        name=MODEL_NAME, alias="candidate", version=registered.version
    )
    print(f"Registered {MODEL_NAME} version {registered.version} (alias: candidate)")


if __name__ == "__main__":
    main()
