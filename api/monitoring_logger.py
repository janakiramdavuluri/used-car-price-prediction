import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def log_inference(
    path: Path,
    input_data: dict[str, Any],
    *,
    prediction: float | None,
    latency_ms: float,
    status: str,
    error: str | None = None,
) -> str:
    request_id = uuid.uuid4().hex
    event = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input_data,
        "predicted_selling_price": prediction,
        "latency_ms": latency_ms,
        "status": status,
        "error": error,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, allow_nan=False) + "\n")
    return request_id