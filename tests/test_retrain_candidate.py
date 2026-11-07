import json
from pathlib import Path

import pandas as pd

from src.retrain_candidate import retrain_candidate

ROOT = Path(__file__).resolve().parents[1]


def test_retraining_validates_batch_and_never_promotes(tmp_path):
    source = pd.read_csv(ROOT / "data/raw/CAR DETAILS FROM CAR DEKHO.csv").drop_duplicates()
    batch = source.head(5).copy()
    batch["name"] = batch["name"].astype(str) + " monitored-batch"
    batch["km_driven"] = batch["km_driven"] + 1000
    batch_path = tmp_path / "simulated_new_batch.csv"
    batch.to_csv(batch_path, index=False)
    report_path = tmp_path / "retraining_report.json"

    report = retrain_candidate(batch_path, report_path)

    assert report["status"] == "PASSED"
    assert report["new_training_rows"] == 5
    assert report["production_promotion"] == "not performed; requires explicit approval"
    assert report["registry_action"] == "none"
    assert Path(report["candidate_artifact"]).is_file()
    assert json.loads(report_path.read_text())["status"] == "PASSED"