"""Train and track used-car regressors with MLflow. Run from project root."""
from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from preprocess import preprocessor
from train import train_models  # Existing model definitions are reused below
from train import linear_reg, ridge_reg, random_forest, extra_tree

DATA = ROOT / 'data/raw/CAR DETAILS FROM CAR DEKHO.csv'
EXPERIMENT = 'Used_Car_Price_Prediction'


def run():
    df = pd.read_csv(DATA).drop_duplicates().copy()
    df = df.dropna(subset=['selling_price', 'year', 'km_driven'])
    df = df[(df.selling_price > 0) & (df.km_driven >= 0)]
    reference_year = int(df.year.max())
    df['vehicle_age'] = reference_year - df.year
    X = df[['name', 'fuel', 'seller_type', 'transmission', 'owner', 'km_driven', 'vehicle_age']]
    y = df['selling_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)
    models = {'Linear Regression': linear_reg, 'Ridge Regression': ridge_reg,
              'Random Forest': random_forest, 'Extra Trees': extra_tree}
    results = []
    for name, estimator in models.items():
        with mlflow.start_run(run_name=name.replace(' ', '_')):
            pipeline = Pipeline([('preprocessor', clone(preprocessor)), ('model', clone(estimator))])
            mlflow.log_param('model_family', name)
            mlflow.log_param('reference_year', reference_year)
            mlflow.log_param('test_size', 0.2)
            mlflow.log_param('split_seed', 42)
            mlflow.log_params({f'model_{k}': v for k, v in estimator.get_params().items() if isinstance(v, (str, int, float, bool))})
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            metrics = {'mae': float(mean_absolute_error(y_test, pred)),
                       'rmse': float(np.sqrt(mean_squared_error(y_test, pred))),
                       'r2': float(r2_score(y_test, pred))}
            mlflow.log_metrics(metrics)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_test, pred, alpha=0.4, s=10)
            bounds = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
            ax.plot(bounds, bounds, linestyle='--')
            ax.set(xlabel='Actual selling price', ylabel='Predicted selling price', title=name)
            fig.tight_layout()
            plot_path = ROOT / 'actual_vs_predicted.png'
            fig.savefig(plot_path)
            plt.close(fig)
            mlflow.log_artifact(str(plot_path), artifact_path='plots')
            plot_path.unlink()
            mlflow.sklearn.log_model(sk_model=pipeline,
    name="model",
    skops_trusted_types=[
        "sklearn.compose._column_transformer._RemainderColsList",
        "sklearn.tree._tree.Tree"
    ])
            results.append({'model': name, **metrics, 'run_id': mlflow.active_run().info.run_id})
            print(f'{name}: MAE={metrics["mae"]:.2f} RMSE={metrics["rmse"]:.2f} R2={metrics["r2"]:.4f}')
    out = ROOT / 'reports'
    out.mkdir(exist_ok=True)
    pd.DataFrame(results).sort_values('mae').to_csv(out / 'mlflow_model_comparison.csv', index=False)
    print('Saved reports/mlflow_model_comparison.csv; launch: mlflow ui --backend-store-uri ./mlruns')

if __name__ == '__main__':
    run()
