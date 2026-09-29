"""Lab 3: reproducible baseline training and evaluation. Run from project root."""
from pathlib import Path
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocess import preprocessor
from src.train import linear_reg, ridge_reg, random_forest, extra_tree

DATA = ROOT / 'data/raw/CAR DETAILS FROM CAR DEKHO.csv'
REPORT = ROOT / 'reports/lab3_baseline_metrics.csv'

def main():
    df = pd.read_csv(DATA).drop_duplicates()
    required = ['name','year','selling_price','km_driven','fuel','seller_type','transmission','owner']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    df = df.dropna(subset=required).copy()
    # Match the training reference year in your existing project if different.
    df['vehicle_age'] = 2026 - df['year']
    X = df[['name','fuel','seller_type','transmission','owner','km_driven','vehicle_age']]
    y = df['selling_price']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    results = []
    for name, estimator in [('Linear Regression',linear_reg),('Ridge Regression',ridge_reg),('Random Forest',random_forest),('Extra Trees',extra_tree)]:
        from sklearn.base import clone
        model = Pipeline([('preprocessor',clone(preprocessor)),('model',clone(estimator))])
        model.fit(X_train,y_train)
        pred = model.predict(X_test)
        results.append({'model':name,'mae':mean_absolute_error(y_test,pred),'rmse':mean_squared_error(y_test,pred)**0.5,'r2':r2_score(y_test,pred)})
        print(f'{name}: MAE={results[-1]["mae"]:.2f}, RMSE={results[-1]["rmse"]:.2f}, R2={results[-1]["r2"]:.4f}')
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(REPORT,index=False)
    print('Saved:', REPORT)

if __name__ == '__main__':
    main()
