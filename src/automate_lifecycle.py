from pathlib import Path
import mlflow
from mlflow import MlflowClient
ROOT=Path(__file__).resolve().parents[1]
def inspect_candidate():
    mlflow.set_tracking_uri(f'sqlite:///{(ROOT / "mlflow.db").as_posix()}')
    client=MlflowClient()
    v=client.get_model_version_by_alias('UsedCarPricePredictor','candidate')
    return {'name':v.name,'version':v.version,'run_id':v.run_id,'status':v.status,'alias':'candidate'}
if __name__=='__main__':print(inspect_candidate())
