import json
import os
from pathlib import Path
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parents[1]


class ModelLoader:
    def __init__(
        self,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(
            model_path or os.getenv("USED_CAR_MODEL_PATH", ROOT / "deployment/model.joblib")
        )
        self.metadata_path = Path(
            metadata_path
            or os.getenv("USED_CAR_METADATA_PATH", ROOT / "deployment/model_metadata.json")
        )
        self.model: Any = None
        self.metadata: dict[str, Any] = {}

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Model metadata not found: {self.metadata_path}")
        with self.metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        required = {"model_family", "reference_year", "feature_columns"}
        if not required.issubset(metadata):
            raise ValueError(f"Model metadata is missing keys: {sorted(required - metadata.keys())}")
        model = joblib.load(self.model_path)
        if not callable(getattr(model, "predict", None)):
            raise TypeError("Loaded artifact does not provide a predict method")
        self.model = model
        self.metadata = metadata

    def is_ready(self) -> bool:
        return self.model is not None