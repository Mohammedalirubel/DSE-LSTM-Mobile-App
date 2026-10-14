from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parent
models = {p.stem for p in (base / 'models').glob('*.keras')}
scalers = {p.stem for p in (base / 'scalers').glob('*.joblib')}
path = base / 'data' / 'historical_prices.csv'
if path.exists():
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)
    codes = set(data['trading_code'].dropna().astype(str).str.strip().str.upper()) if 'trading_code' in data else set()
else:
    codes = set()
ready = codes & models & scalers
print(f'Historical codes: {len(codes)} | Models: {len(models)} | Scalers: {len(scalers)} | Ready matches: {len(ready)}')
print('Model codes missing scalers (first 20):', sorted(models - scalers)[:20])
print('Model codes missing historical data (first 20):', sorted(models - codes)[:20])
print('Ready codes (first 20):', sorted(ready)[:20])
