"""Lab 6: register selected model and create registry report; no automatic production promotion."""
from pathlib import Path
import subprocess,sys,json
import mlflow
from mlflow import MlflowClient

ROOT=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(ROOT/'src/register_model.py')],cwd=ROOT,check=True)
mlflow.set_tracking_uri(f'sqlite:///{(ROOT/'mlflow.db').as_posix()}')
client=MlflowClient()
name='UsedCarPricePredictor'
version=client.get_model_version_by_alias(name,'candidate')
report={'model_name':name,'candidate_version':version.version,'run_id':version.run_id,'source':version.source,'status':version.status,'production_promotion':'not performed; requires separate approval'}
path=ROOT/'reports/lab6_registry_report.json'
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Registry report:',path)
