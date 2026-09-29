import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.preprocess_pipeline import prepare_data
ROOT=Path(__file__).resolve().parents[1]
def validate_reproducibility():
    X_train,X_test,y_train,y_test,_=prepare_data()
    predictions=[]
    for _ in range(2):
        model=RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=1)
        model.fit(X_train,y_train)
        predictions.append(model.predict(X_test))
    passed_=bool(np.array_equal(*predictions))
    report={'status':'PASSED' if passed_ else 'FAILED','predictions_identical':passed_,'test_rows':len(y_test)}
    p=ROOT/'reports/reproducibility_report.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(report,indent=2))
    if not passed_:raise AssertionError(report)
    return report
if __name__=='__main__':print(validate_reproducibility())
