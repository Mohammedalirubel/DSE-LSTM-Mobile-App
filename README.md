# DSE LSTM Mobile Web App

A phone-friendly Streamlit interface for existing per-ticker TensorFlow LSTM models.

## Required project files
requirements.txt
streamlit>=1.40,<2.0
tensorflow-cpu==2.20.0
numpy>=1.26,<3.0
pandas>=2.0,<3.0
scikit-learn>=1.4,<2.0
matplotlib>=3.8,<4.0
joblib>=1.3,<2.0
plotly>=5.20,<7.0

dse-lstm-mobile-app/
├── app.py
├── requirements.txt
├── README.md
├── models/
├── scalers/
└── data/
```

`data/historical_prices.csv` must contain `trading_code,date,close` columns, with at least 20 historical closing prices for each share. Alternative `closep*`/`closep` columns are accepted. The app only lists share codes for which historical prices, model and fitted scaler all exist. It does not provide live DSE data.

## Export from the SAME training session

You must save each scaler **as it was fitted during training**, not fit a new scaler later. For example, if your training loop stores scalers in `stock_scalers` and models in `stock_models`:

```python
import os, re, joblib
os.makedirs('scalers', exist_ok=True)
os.makedirs('models', exist_ok=True)
for code, scaler in stock_scalers.items():
    safe = re.sub(r'[^A-Za-z0-9_-]', '_', code.upper())
    joblib.dump(scaler, f'scalers/{safe}.joblib')
for code, model in stock_models.items():
    safe = re.sub(r'[^A-Za-z0-9_-]', '_', code.upper())
    model.save(f'models/{safe}.keras')
```

If scalers were never saved, you should rerun training and save them. Re-fitting from the entire historical dataset may produce a different scaler and invalidate inference. Model architecture assumes a single close-price input with a lookback window of 20; set `LSTM_WINDOW` if your trained models used a different window. Use the same data transformations and feature order as training.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal. For access on your phone on the same Wi-Fi, run `streamlit run app.py --server.address 0.0.0.0` and open your computer's LAN IP on port 8501 (allow through firewall).

## Publish online

1. Upload the project to a GitHub repository, including your own CSV, `.keras` models, and `.joblib` scalers. Use private hosting for proprietary data. GitHub has per-file size limits; large models may need separate artifact storage.
2. Deploy the repository on Streamlit Community Cloud with `app.py` as the entrypoint, or deploy on another compatible Python host.
3. Open the public app URL in your phone browser. Use your browser's **Add to Home Screen** action to create an app-like icon.

## Caveats

- This is a mobile-friendly web app (PWA-like browser shortcut), not an Android APK or iOS App Store package.
- Stock-market calendar uses Sunday–Thursday and omits DSE holidays.
- Ticker exclusions are heuristic and are not an official current equity classification.
- Recursive LSTM forecasts may drift or become non-positive; invalid outputs are not presented as valid price targets.
- Long-horizon predictions are highly uncertain and not financial advice.

## One-step notebook export and asset diagnostics

Copy `export_from_notebook.py` into the training notebook's working directory, then run:

```python
from export_from_notebook import export_dse_app_data
export_dse_app_data(df_pred, models=stock_models, scalers=stock_scalers)
```

The `stock_scalers` mapping must contain the **original fitted training scalers**. If models are already saved on disk and `stock_models` is unavailable, omit `models=` to copy `.keras` files from `DSE_LSTM_Models`. If the notebook never retained its original scalers, retrain with scaler persistence; do not silently fit new ones for deployment.

After copying exported assets into this app's `data/`, `models/`, and `scalers/` directories, run `python diagnose_assets.py` to count ready trading codes. The package intentionally contains no invented model weights, stock history, or forecasts.
