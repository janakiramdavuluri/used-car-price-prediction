import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.model_loader import ModelLoader, ROOT
from api.monitoring_logger import log_inference
from api.schemas import HealthResponse, PredictionResponse, UsedCarInput


def create_app(
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    log_path: str | Path | None = None,
) -> FastAPI:
    loader = ModelLoader(model_path, metadata_path)
    inference_log = Path(
        log_path or os.getenv("USED_CAR_INFERENCE_LOG", ROOT / "logs/inference/inference.jsonl")
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        loader.load()
        yield

    app = FastAPI(
        title="Used Cars Price Prediction API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.model_loader = loader

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        if not loader.is_ready():
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return HealthResponse(status="healthy", model_loaded=True)

    @app.get("/metadata")
    def metadata() -> dict:
        return {
            "service": "Used Cars Price Prediction API",
            "model_family": loader.metadata.get("model_family"),
            "reference_year": loader.metadata.get("reference_year"),
            "metrics": loader.metadata.get("metrics"),
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(car: UsedCarInput) -> PredictionResponse:
        started = time.perf_counter()
        payload = car.model_dump()
        try:
            if not loader.is_ready():
                raise HTTPException(status_code=503, detail="Model is not loaded")
            reference_year = int(loader.metadata["reference_year"])
            if car.year > reference_year:
                raise HTTPException(
                    status_code=422,
                    detail=f"year must not exceed model reference year {reference_year}",
                )
            features = pd.DataFrame(
                [
                    {
                        "name": car.name,
                        "fuel": car.fuel,
                        "seller_type": car.seller_type,
                        "transmission": car.transmission,
                        "owner": car.owner,
                        "km_driven": car.km_driven,
                        "vehicle_age": reference_year - car.year,
                    }
                ],
                columns=loader.metadata["feature_columns"],
            )
            prediction = float(loader.model.predict(features)[0])
            if not math.isfinite(prediction):
                raise ValueError("Model returned a non-finite price")
            log_inference(
                inference_log,
                payload,
                prediction=prediction,
                latency_ms=(time.perf_counter() - started) * 1000,
                status="success",
            )
            return PredictionResponse(
                predicted_selling_price=prediction,
                model_name=str(loader.metadata["model_family"]),
            )
        except HTTPException as error:
            log_inference(
                inference_log,
                payload,
                prediction=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                status="failure",
                error=str(error.detail),
            )
            raise
        except Exception as error:
            log_inference(
                inference_log,
                payload,
                prediction=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                status="failure",
                error=type(error).__name__,
            )
            raise HTTPException(status_code=500, detail="Prediction failed") from error

    return app


app = create_app()