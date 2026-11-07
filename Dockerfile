FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    USED_CAR_MODEL_PATH=/app/deployment/model.joblib \
    USED_CAR_METADATA_PATH=/app/deployment/model_metadata.json \
    USED_CAR_INFERENCE_LOG=/app/logs/inference/inference.jsonl

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs/inference \
    && chown -R appuser:appuser /app/logs

COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser deployment/model.joblib deployment/model_metadata.json ./deployment/

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]