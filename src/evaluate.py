import json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
ROOT=Path(__file__).resolve().parents[1]
def evaluate_model(model,X_test,y_test,report_path=None):
    pred=model.predict(X_test)
    result={'mae':float(mean_absolute_error(y_test,pred)),'rmse':float(np.sqrt(mean_squared_error(y_test,pred))),'r2':float(r2_score(y_test,pred))}
    if report_path:
        p=Path(report_path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2))
    return result
