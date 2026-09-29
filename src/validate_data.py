"""Validate raw CarDekho used-car data before preprocessing.
Run from the project root: python src/validate_data.py
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data/raw/CAR DETAILS FROM CAR DEKHO.csv'
REPORT_DIR = ROOT / 'reports'
REQUIRED = ['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
CATEGORIES = {
    'fuel': {'Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'},
    'seller_type': {'Individual', 'Dealer', 'Trustmark Dealer'},
    'transmission': {'Manual', 'Automatic'},
    'owner': {'First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'},
}

def validate_data(path=DATA_PATH):
    df = pd.read_csv(path)
    issues = []
    missing_cols = sorted(set(REQUIRED) - set(df.columns))
    extra_cols = sorted(set(df.columns) - set(REQUIRED))
    if missing_cols:
        issues.append({'row': None, 'column': '*', 'issue': f'Missing columns: {missing_cols}'})
    if extra_cols:
        issues.append({'row': None, 'column': '*', 'issue': f'Unexpected columns: {extra_cols}'})
    for col in REQUIRED:
        if col not in df:
            continue
        for idx in df.index[df[col].isna()]:
            issues.append({'row': int(idx) + 2, 'column': col, 'issue': 'Missing value'})
    for col, lower, upper in [('year', 1886, 2026), ('selling_price', 1, None), ('km_driven', 0, None)]:
        if col not in df:
            continue
        numeric = pd.to_numeric(df[col], errors='coerce')
        invalid = numeric.isna() | numeric.lt(lower)
        if upper is not None:
            invalid |= numeric.gt(upper)
        if col == 'year':
            invalid |= numeric.notna() & numeric.mod(1).ne(0)
        for idx in df.index[invalid & df[col].notna()]:
            issues.append({'row': int(idx) + 2, 'column': col, 'issue': f'Invalid {col}: {df.at[idx, col]}'})
    for col, allowed in CATEGORIES.items():
        if col not in df:
            continue
        for idx in df.index[df[col].notna() & ~df[col].isin(allowed)]:
            issues.append({'row': int(idx) + 2, 'column': col, 'issue': f'Unexpected category: {df.at[idx, col]}'})
    if 'name' in df:
        for idx in df.index[df['name'].notna() & df['name'].astype(str).str.strip().eq('')]:
            issues.append({'row': int(idx) + 2, 'column': 'name', 'issue': 'Blank car name'})
    duplicate_count = int(df.duplicated().sum())
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(issues, columns=['row', 'column', 'issue']).to_csv(REPORT_DIR / 'validation_errors.csv', index=False)
    summary = {'rows': len(df), 'columns': len(df.columns), 'duplicates': duplicate_count,
               'validation_errors': len(issues), 'schema_valid': not issues,
               'note': 'Exact duplicate rows are reported, not treated as schema failures.'}
    (REPORT_DIR / 'validation_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return not issues

if __name__ == '__main__':
    raise SystemExit(0 if validate_data() else 1)
