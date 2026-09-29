"""Predict a used-car selling price using the MLflow registered candidate model.

Run from the project root:
    python src/predict.py --name "Maruti 800 AC" --year 2007 --km-driven 70000 \
        --fuel Petrol --seller-type Individual --transmission Manual \
        --owner "First Owner"
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "UsedCarPricePredictor"
MODEL_ALIAS = "candidate"
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"


def predict_price(name, year, km_driven, fuel, seller_type, transmission, owner):
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    if not version.run_id:
        raise RuntimeError("Registered model has no source run ID; supply a model version linked to a training run.")
    run = client.get_run(version.run_id)
    if "reference_year" not in run.data.params:
        raise RuntimeError("Training run does not record reference_year; cannot reproduce vehicle_age.")
    reference_year = int(run.data.params["reference_year"])
    if year > reference_year:
        raise ValueError(f"Car year {year} is later than the training reference year {reference_year}.")
    if km_driven < 0:
        raise ValueError("km_driven cannot be negative")
    features = pd.DataFrame([{
        "name": name,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "km_driven": km_driven,
        "vehicle_age": reference_year - year,
    }])
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    return float(model.predict(features)[0])


def main():
    parser = argparse.ArgumentParser(description="Predict used-car selling price")
    parser.add_argument("--name", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--km-driven", type=int, required=True)
    parser.add_argument("--fuel", required=True)
    parser.add_argument("--seller-type", required=True)
    parser.add_argument("--transmission", required=True)
    parser.add_argument("--owner", required=True)
    args = parser.parse_args()
    price = predict_price(args.name, args.year, args.km_driven, args.fuel,
                          args.seller_type, args.transmission, args.owner)
    print(f"Predicted selling price: ₹{price:,.2f}")


if __name__ == "__main__":
    main()
