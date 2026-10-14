"""Run in the SAME notebook session as training: %run export_from_notebook.py

Requires df/df_pred/df_shares, saved .keras models or stock_models, and
ORIGINAL fitted scalers in stock_scalers (or equivalent mapping).
"""
from pathlib import Path
import re
import shutil
import pandas as pd
import joblib

ROOT = Path('dse_mobile_app')
for folder in ('models', 'scalers', 'data'):
    (ROOT / folder).mkdir(parents=True, exist_ok=True)

def safe(code):
    return re.sub(r'[^A-Za-z0-9_-]', '_', str(code).strip().upper())

def export_dse_app_data(historical_df, models=None, scalers=None, saved_models_dir='DSE_LSTM_Models'):
    """Export dataset, models, and original fitted scalers for the mobile app."""
    frame = historical_df.copy()
    frame.columns = frame.columns.astype(str).str.strip().str.lower().str.replace(' ', '_', regex=False)
    if 'close' not in frame.columns:
        for old in ('closep*', 'closep'):
            if old in frame.columns:
                frame = frame.rename(columns={old: 'close'})
                break
    required = {'trading_code', 'date', 'close'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {required - set(frame.columns)}. A dated historical dataset is required.')
    frame = frame[['trading_code', 'date', 'close']].copy()
    frame['trading_code'] = frame['trading_code'].astype('string').str.strip().str.upper()
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce')
    frame['close'] = pd.to_numeric(frame['close'], errors='coerce')
    frame = frame.dropna().query('close > 0').sort_values(['trading_code', 'date'])
    frame.to_csv(ROOT / 'data' / 'historical_prices.csv', index=False)
    n_models = 0
    if models:
        for code, model in models.items():
            model.save(ROOT / 'models' / (safe(code) + '.keras'))
            n_models += 1
    elif Path(saved_models_dir).is_dir():
        for p in Path(saved_models_dir).glob('*.keras'):
            shutil.copy2(p, ROOT / 'models' / p.name)
            n_models += 1
    if not scalers:
        print('WARNING: No original fitted scalers provided. Predictions remain unavailable until scalers are exported.')
    else:
        for code, scaler in scalers.items():
            joblib.dump(scaler, ROOT / 'scalers' / (safe(code) + '.joblib'))
    print(f'Exported {len(frame):,} historical rows and {n_models} model files to {ROOT.resolve()}')
    print('Next: copy these folders into the app project, then run streamlit run app.py')

if __name__ == '__main__':
    print('Import export_dse_app_data and call it with your training dataframe and original fitted scalers.')
    print("Example: export_dse_app_data(df_pred, models=stock_models, scalers=stock_scalers)")
