import os
import re
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from pandas.tseries.offsets import CustomBusinessDay

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / 'models'
DATA_PATH = BASE / 'data' / 'historical_prices.csv'
SCALER_DIR = BASE / 'scalers'
WINDOW = int(os.getenv('LSTM_WINDOW', '20'))

EXCLUDED = set('''1JANATAMF 1STPRIMFMF AIBL1STIMF ATCSLGF CAPMBDBLMF CAPMIBBLMF DBH1STMF EBL1STMF EBLNRBMF EXIM1STMF FBFIF IC3RDNRB ICBAGRANI1 ICBAMCL2ND ICBEPMF1S1 ICBSONALI1 IFILISLMF1 LANKABA1MF LRGLOBMF1 MBL1STMF NCCBLMF1 NLI1STMF PHPMF1 POPULAR1MF PRIME1ICBA RELIANCE1 SEMLIBBLSF SEMLLECMF TRUSTB1MF VAMLBDG1 VAMLRBBF VAMLBDMF1'''.split())

def equity_candidate(code):
    c = str(code).strip().upper()
    return bool(c) and c not in EXCLUDED and not re.fullmatch(r'TB\d+Y\d+', c) and not any(x in c for x in ('BOND', 'SUKUK', 'DEB'))

def safe_name(code):
    return re.sub(r'[^A-Za-z0-9_-]', '_', code)

@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)
    for src in ('closep*', 'closep'):
        if src in df.columns and 'close' not in df.columns:
            df = df.rename(columns={src: 'close'})
    required = {'trading_code', 'date', 'close'}
    if not required.issubset(df.columns):
        raise ValueError(f'CSV requires columns {sorted(required)}. Missing: {sorted(required - set(df.columns))}')
    df['trading_code'] = df['trading_code'].astype('string').str.strip().str.upper()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['trading_code', 'date', 'close'])
    df = df[df['close'] > 0]
    return df.sort_values(['trading_code', 'date']).drop_duplicates(['trading_code', 'date'], keep='last')

@st.cache_resource(show_spinner=False)
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)

@st.cache_resource(show_spinner=False)
def load_scaler(path):
    return joblib.load(path)

def available_codes(df):
    historical = set(df['trading_code'].unique())
    return sorted(c for c in historical if equity_candidate(c) and (MODEL_DIR / (safe_name(c) + '.keras')).is_file() and (SCALER_DIR / (safe_name(c) + '.joblib')).is_file())

def predict(code, target, df):
    rows = df[df['trading_code'] == code].sort_values('date')
    if len(rows) < WINDOW:
        raise ValueError(f'{code}: at least {WINDOW} historical prices are required.')
    last_date = rows['date'].iloc[-1].date()
    if target <= last_date:
        raise ValueError(f'Choose a date after the last recorded trading date ({last_date}).')
    # DSE generally operates Sunday through Thursday; public holidays not included.
    dse_weekday = CustomBusinessDay(weekmask='Sun Mon Tue Wed Thu')
    future_dates = pd.date_range(start=pd.Timestamp(last_date) + dse_weekday, end=pd.Timestamp(target), freq=dse_weekday)
    if len(future_dates) == 0:
        raise ValueError('Selected date has no forecast trading sessions.')
    if len(future_dates) > 60:
        st.warning('Long-horizon recursive forecasts are especially unreliable.')
    scaler = load_scaler(str(SCALER_DIR / (safe_name(code) + '.joblib')))
    model = load_model(str(MODEL_DIR / (safe_name(code) + '.keras')))
    scaled = scaler.transform(rows[['close']].to_numpy(dtype=float))
    sequence = scaled[-WINDOW:].reshape(1, WINDOW, 1)
    predicted_scaled = []
    for _ in future_dates:
        nxt = float(model(sequence, training=False).numpy().reshape(-1)[0])
        if not np.isfinite(nxt):
            raise ValueError('Model returned a non-finite prediction.')
        predicted_scaled.append(nxt)
        sequence = np.concatenate([sequence[:, 1:, :], np.array([[[nxt]]])], axis=1)
    prices = scaler.inverse_transform(np.array(predicted_scaled).reshape(-1, 1)).reshape(-1)
    forecast = pd.DataFrame({'Date': future_dates, 'Predicted Close (BDT)': prices})
    return rows, forecast

st.set_page_config(page_title='DSE Stock Predictor', page_icon='📈', layout='centered')
st.markdown('''<style>
.block-container{max-width:850px;padding-top:1.2rem;padding-left:1rem;padding-right:1rem}
h1{font-size:1.9rem!important} div[data-testid="stMetricValue"]{font-size:1.5rem}
.stButton button{width:100%;min-height:3rem;border-radius:10px}
@media(max-width:600px){h1{font-size:1.55rem!important}.block-container{padding-top:.8rem}}
</style>''', unsafe_allow_html=True)
st.title('📈 DSE LSTM Stock Predictor')
st.caption('Mobile-friendly research tool • Forecasts are estimates, not investment advice')

try:
    df = load_data()
except Exception as exc:
    st.error(f'Could not load historical prices: {exc}')
    st.stop()
if df is None:
    st.warning('Historical data missing. Add data/historical_prices.csv and redeploy.')
    st.code('trading_code,date,close\nBRACBANK,2026-10-12,66.10', language='csv')
    st.stop()

codes = available_codes(df)
if not codes:
    st.warning('No ready shares found. Each share needs historical prices, a .keras model and its original fitted .joblib scaler.')
    st.info('See README.md for model/scaler export and folder setup.')
    st.stop()

st.caption(f'🟢 {len(codes)} trained share codes available')
code = st.selectbox('Trading code (type to search)', codes, index=codes.index('BRACBANK') if 'BRACBANK' in codes else 0)
rows = df[df['trading_code'] == code]
last_date = rows['date'].max().date()
selected_date = st.date_input('Future prediction date', value=max(date.today() + timedelta(days=30), last_date + timedelta(days=1)), min_value=last_date + timedelta(days=1))
st.caption(f'Latest historical close: BDT {rows["close"].iloc[-1]:,.2f} on {last_date:%d %b %Y}')

if st.button('Predict closing price', type='primary'):
    try:
        with st.spinner('Loading LSTM model and forecasting…'):
            history, forecast = predict(code, selected_date, df)
        actual = float(history['close'].iloc[-1]); predicted = float(forecast['Predicted Close (BDT)'].iloc[-1]); change = predicted - actual
        if predicted <= 0:
            st.error('Model produced a non-positive closing price. This forecast is invalid and should not be used.')
        else:
            st.success(f'Forecast calculated for {code}')
            a, b = st.columns(2)
            a.metric('Predicted close', f'BDT {predicted:,.2f}', f'{change/actual*100:+.2f}%')
            b.metric('Last actual close', f'BDT {actual:,.2f}')
            st.caption(f'Forecast sessions: {len(forecast)} • Requested date: {selected_date} • Last modeled session: {forecast["Date"].iloc[-1].date()}')
            fig, ax = plt.subplots(figsize=(9, 4))
            recent = history.tail(90)
            ax.plot(recent['date'], recent['close'], label='Historical close', linewidth=2)
            ax.plot(forecast['Date'], forecast['Predicted Close (BDT)'], label='LSTM forecast', linewidth=2)
            ax.axvline(history['date'].iloc[-1], linestyle='--', alpha=.5)
            ax.set_ylabel('BDT'); ax.legend(); ax.grid(alpha=.2); fig.autofmt_xdate(); fig.tight_layout()
            st.pyplot(fig); plt.close(fig)
            st.subheader('Forecast data')
            st.dataframe(forecast.tail(15), hide_index=True, use_container_width=True)
            summary = pd.DataFrame([{'Trading Code': code, 'Last Actual Date': last_date, 'Last Actual Close (BDT)': actual, 'Requested Date': selected_date, 'Last Forecast Trading Date': forecast['Date'].iloc[-1].date(), 'Forecast Steps': len(forecast), 'Predicted Close (BDT)': predicted, 'Estimated Return (%)': (predicted/actual - 1)*100}])
            st.download_button('Download summary CSV', summary.to_csv(index=False).encode(), file_name=f'{code}_summary.csv', mime='text/csv')
            st.download_button('Download forecast CSV', forecast.to_csv(index=False).encode(), file_name=f'{code}_forecast.csv', mime='text/csv')
    except Exception as exc:
        st.error(f'Prediction failed: {exc}')

st.divider()
st.caption('Trading calendar assumes Sunday–Thursday but does not exclude DSE market holidays. Original training scalers are mandatory. Predictions can be inaccurate, especially over long periods. Equity filtering uses a known exclusion list and ticker-name heuristics; verify against the official DSE equity list for full coverage.')
