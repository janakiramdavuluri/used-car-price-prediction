import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
import numpy as np
from src.preprocess_pipeline import prepare_data
ROOT=Path(__file__).resolve().parents[1]
def validate_outputs():
    X_train,X_test,y_train,y_test,_=prepare_data()
    def finite(x):return bool(np.isfinite(x.data if hasattr(x,'data') else x).all())
    ok=X_train.shape[0]==len(y_train) and X_test.shape[0]==len(y_test) and X_train.shape[1]==X_test.shape[1] and finite(X_train) and finite(X_test)
    report={'status':'PASSED' if ok else 'FAILED','train_shape':X_train.shape,'test_shape':X_test.shape}
    p=ROOT/'reports/output_validation.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(report,indent=2))
    if not ok:raise AssertionError(report)
    return report
if __name__=='__main__':print(validate_outputs())
