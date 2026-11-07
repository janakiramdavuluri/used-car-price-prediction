"""Export a quality-gated candidate as a local serving bundle."""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare_deployment(
    candidate_path: Path | None = None,
    metadata_path: Path | None = None,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    report_path = ROOT / "reports/lab7_pipeline_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASSED" or not report.get("quality_gate", {}).get("passed"):
        raise RuntimeError("Refusing to package a candidate that did not pass its quality gate")

    source_model = candidate_path or ROOT / report["candidate_artifact"]
    source_metadata = metadata_path or (
        candidate_path.with_suffix(".json") if candidate_path else ROOT / report["candidate_metadata"]
    )
    output_dir = output_dir or ROOT / "deployment"
    output_model = output_dir / "model.joblib"
    output_metadata = output_dir / "model_metadata.json"

    if not source_model.is_file() or not source_metadata.is_file():
        raise FileNotFoundError("Candidate model and matching metadata must both exist")
    if not overwrite and (output_model.exists() or output_metadata.exists()):
        raise FileExistsError(
            f"Deployment bundle already exists in {output_dir}; pass --overwrite to replace it"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    staged_model = output_dir / "model.joblib.tmp"
    staged_metadata = output_dir / "model_metadata.json.tmp"
    shutil.copy2(source_model, staged_model)
    shutil.copy2(source_metadata, staged_metadata)
    staged_model.replace(output_model)
    staged_metadata.replace(output_metadata)
    return output_model, output_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    model_path, metadata_path = prepare_deployment(
        args.candidate,
        args.metadata,
        args.output_dir,
        args.overwrite,
    )
    print(f"Model bundle: {model_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()