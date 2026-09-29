"""Lab 5: production data checks and preprocessor output validation."""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.base import clone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.preprocess import preprocessor

DATA=ROOT/'data/raw/CAR DETAILS FROM CAR DEKHO.csv'
REPORT=ROOT/'reports/lab5_validation.txt'
df=pd.read_csv(DATA)
required=['name','year','selling_price','km_driven','fuel','seller_type','transmission','owner']
missing=set(required)-set(df.columns)
if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
if df[required].isna().any().any(): raise ValueError('Missing values in required columns')
if (df['selling_price']<=0).any(): raise ValueError('Nonpositive selling prices')
if (df['km_driven']<0).any(): raise ValueError('Negative km_driven')
df=df.drop_duplicates().copy()
df['vehicle_age']=2026-df['year']
if (df['vehicle_age']<0).any(): raise ValueError('Vehicle year after reference year 2026')
X=df[['name','fuel','seller_type','transmission','owner','km_driven','vehicle_age']]
processed=clone(preprocessor).fit_transform(X)
if processed.shape[0]!=len(X): raise RuntimeError('Row count changed in preprocessing')
if not np.isfinite(processed.data if hasattr(processed,'data') else processed).all():
    raise RuntimeError('Non-finite values in processed features')
REPORT.parent.mkdir(parents=True,exist_ok=True)
REPORT.write_text(f'PASS: {len(X)} unique rows, {processed.shape[1]} processed features\n',encoding='utf-8')
print(REPORT.read_text())
