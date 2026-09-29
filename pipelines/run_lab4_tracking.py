"""Lab 4: MLflow tracking and reproducibility check. Run from project root."""
from pathlib import Path
import subprocess, sys
import mlflow

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'mlflow.db'
mlflow.set_tracking_uri(f'sqlite:///{DB.as_posix()}')

# Existing MLflow training is reused rather than duplicated.
subprocess.run([sys.executable, str(ROOT / 'src/train_mlflow.py')], cwd=ROOT, check=True)
experiment = mlflow.get_experiment_by_name('Used_Car_Price_Prediction')
if experiment is None:
    raise RuntimeError('MLflow experiment missing')
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],filter_string="attributes.status = 'FINISHED'")
expected = {'Linear_Regression','Ridge_Regression','Random_Forest','Extra_Trees'}
found = set(runs['tags.mlflow.runName'].dropna()) if not runs.empty else set()
missing = expected - found
if missing:
    raise RuntimeError(f'Missing completed runs: {sorted(missing)}')
print('Lab 4 tracking passed; completed runs:', sorted(found & expected))
print('Reproducibility note: compare repeated runs with identical data, split, and random seeds before claiming numerical reproducibility.')
