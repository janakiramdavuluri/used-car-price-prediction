# Used Cars Regression MLOps Labs 7–13

This document describes the regression adaptation of the remaining MLOps labs. The project keeps the existing CarDekho target and feature engineering; all model metrics are regression metrics.

## Lab 7: Pipeline and Lifecycle

Run from the repository root:

```powershell
python pipelines/run_lab7_pipeline.py
```

The pipeline validates required columns and numeric constraints, drops exact duplicate rows, stores an immutable SHA-256-named dataset snapshot under `data/versions/`, creates a deterministic train/test split, and compares Linear Regression, Ridge, Random Forest, and Extra Trees using MAE, MSE, RMSE, and R². MLflow records model parameters and metrics. The selected candidate is serialized together with metadata containing its reference year, feature order, dataset hash, and run ID.

The configured quality gate is MAE <= 200,000 and R² >= 0.0. It is a project demonstration threshold, not a business-approved price-error limit. The pipeline reads the current registry reference and writes a candidate comparison; it does not register a model, move an alias, or promote to Production.

Primary outputs:

- `reports/lab7_pipeline_report.json`
- `data/versions/<sha256>.csv`
- `models/candidates/used_car_candidate_<timestamp>.joblib` and matching JSON metadata
- MLflow experiment `Used_Car_Price_Prediction`

## Labs 8–9: CI and Failure Handling

`.github/workflows/lab8_ci.yml` runs the regression pipeline, prepares the gated model bundle, executes the full tests, and builds the API image on relevant pushes, pull requests, or manual dispatch. `.github/workflows/lab9_failure_tests.yml` runs malformed-data and unavailable-artifact tests when relevant files change or on manual dispatch. Validation is fail-fast: missing columns, invalid prices/distances, or missing model bundles stop the affected operation.

Local checks:

```powershell
python -m pytest tests/test_pipeline_validation.py tests/test_failure_handling.py -q
python -m pytest tests -q
```

## Lab 10: FastAPI Serving

The API uses the serialized sklearn `Pipeline`, so request processing shares the training encoder and scaler. The service derives `vehicle_age` using the reference year in the matching metadata file.

Request fields are `name`, `year`, `km_driven`, `fuel`, `seller_type`, `transmission`, and `owner`. A successful response contains `predicted_selling_price`, `currency` (`INR`), and `model_name`.

```powershell
python src/prepare_deployment.py
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Endpoints: `GET /health`, `GET /metadata`, and `POST /predict`. Inference events include request ID, input, prediction, status, and latency in `logs/inference/inference.jsonl`. Requests for years later than the model reference year are rejected.

## Lab 11: Docker and Kubernetes

The bundle exporter refuses to overwrite an existing deployment bundle unless `--overwrite` is explicitly supplied. It packages the quality-gated candidate, not an MLflow Production model.

```powershell
python src/prepare_deployment.py
docker build -t used-car-price-api:lab11 .
docker run --rm -p 8000:8000 used-car-price-api:lab11
```

The container runs as a non-root user and has writable inference-log storage. Kubernetes manifests are in `k8s/`; apply them only when the intended cluster is available and selected:

```powershell
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl port-forward service/used-car-price-api 8000:8000
```

Kubernetes logs currently use container-local storage. Production monitoring should forward logs to a durable external sink or attach persistent storage.

## Labs 12–13: Monitoring and Candidate Retraining

Run monitoring after inference events are available:

```powershell
python src/monitor_regression.py
```

The report includes traffic volume, success/failure rate, latency, PSI for numeric/categorical inputs and predicted prices, plus MAE/RMSE/R² when `data/monitoring/labeled_predictions.csv` provides `request_id,actual_selling_price`. Drift is reported only after 30 successful requests. Fewer events or missing labels are explicitly reported as insufficient evidence.

Retraining requires a separate, validated incoming batch; it never runs automatically:

```powershell
python src/retrain_candidate.py --batch path\to\validated_new_batch.csv
```

The retraining comparison uses the original deterministic holdout, checks for overlap with existing rows, writes a timestamped candidate only if quality and incumbent-comparison gates pass, and does not change registry aliases or Production. The report is written to `outputs/retraining/retraining_report.json`.

## Verification Snapshot

The inspected dataset contains 4,340 source rows and eight expected columns. The Lab 7 run validated 3,577 unique rows and selected Random Forest with MAE 157,505, RMSE 399,513, and R² 0.5045. The regression gate passed. The run tied registry candidate version 2 on these metrics, so the existing registry candidate was retained.

API tests passed against both the candidate and exported deployment bundle. The Docker image built and returned a healthy response and a predicted price of 95,600 INR for a smoke request. Monitoring recorded that request, but correctly classified the sample as insufficient for drift and actual-price evaluation because it has fewer than 30 successful requests and no actual labels.

The local Kubernetes context was `docker-desktop`, but its API server was unreachable during verification. Kubernetes resources were therefore not applied. The retraining CLI used a temporary simulated five-row batch; it did not add data to the repository or change registry state. Candidate MAE/RMSE/R² were 157,675 / 399,254 / 0.5052 versus incumbent values 157,505 / 399,513 / 0.5045. The candidate passed the configured 2% MAE comparison allowance and was logged as an MLflow run, but was not promoted. Such demonstration traffic/data is not evidence of production drift or model performance.