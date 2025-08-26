# eth_scalper.py
# Çalıştır:  streamlit run eth_scalper.py

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import io
import zipfile
import requests
from ta.volatility import AverageTrueRange
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="ETH 5m Scalper Backtest", layout="wide")

# =========================
# Yardımcı Fonksiyonlar
# =========================

def ema(series: pd.Series, length: int):
    return series.ewm(span=length, adjust=False).mean()

def tsi(close: pd.Series, r: int = 25, s: int = 13):
    m = close.diff()
    ema1 = m.ewm(span=r, adjust=False).mean()
    ema2 = ema1.ewm(span=s, adjust=False).mean()
    a = m.abs()
    a1 = a.ewm(span=r, adjust=False).mean()
    a2 = a1.ewm(span=s, adjust=False).mean()
    val = 100 * (ema2 / a2)
    return val

def bullish_engulf(prev_open, prev_close, curr_open, curr_close) -> bool:
    return (prev_close < prev_open) and (curr_close > curr_open) and \
           (curr_open <= prev_close) and (curr_close >= prev_open)

def bearish_engulf(prev_open, prev_close, curr_open, curr_close) -> bool:
    return (prev_close > prev_open) and (curr_close < curr_open) and \
           (curr_open >= prev_close) and (curr_close <= prev_open)

# Binance arşiv yardımcıları
BINANCE_BASE = "https://data.binance.vision"

def list_last_n_months(n=12):
    today = pd.Timestamp.utcnow().normalize().to_pydatetime().date().replace(day=1)
    months = []
    for i in range(n):
        d = (pd.Timestamp(today) - pd.DateOffset(months=i)).to_pydatetime().date()
        months.append((d.year, d.month))
    months.sort()
    return months

def monthly_url_futures_eth_5m(year, month):
    return f"{BINANCE_BASE}/data/futures/um/monthly/klines/ETHUSDT/5m/ETHUSDT-5m-{year:04d}-{month:02d}.zip"

def fetch_zip(url, timeout=30, retries=3, backoff=1.5):
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.content
            elif r.status_code == 404:
                return None
        except Exception:
            pass
        if i < retries - 1:
            time.sleep(backoff * (i+1))
    return None

def parse_month_zip(zbytes):
    import pandas as pd
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(name) as f:
            df = pd.read_csv(f, header=None)
    df = df.iloc[:, :6].copy()
    df.columns = ["timestamp","open","high","low","close","volume"]
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    idx = pd.to_datetime(ts, unit="ms" if ts.max() > 1e12 else "s", utc=True)
    df.index = idx
    df = df[["open","high","low","close","volume"]].astype(float)
    return df.sort_index()

# =========================
# CSV okuyucu
# =========================
def read_and_prepare_csv(file, start_date, end_date):
    df = pd.read_csv(file)
    lower_cols = [c.lower() for c in df.columns]
    required = ["open","high","low","close","volume"]
    for req in required:
        if req not in lower_cols:
            raise ValueError("CSV sütunları 'open, high, low, close, volume' içermeli.")
    if "timestamp" in lower_cols:
        ts_col = df.columns[lower_cols.index("timestamp")]
        ts_vals = pd.to_numeric(df[ts_col], errors="coerce")
        if ts_vals.max() > 1e12:
            idx = pd.to_datetime(ts_vals, unit="ms", utc=True)
        else:
            idx = pd.to_datetime(ts_vals, unit="s", utc=True)
    elif "date" in lower_cols:
        ts_col = df.columns[lower_cols.index("date")]
        idx = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    else:
        raise ValueError("CSV'de 'timestamp' veya 'date' sütunu yok.")
    ren = {}
    for name in ["open","high","low","close","volume"]:
        ren[df.columns[lower_cols.index(name)]] = name
    df = df.rename(columns=ren)
    df.index = idx
    df = df[["open","high","low","close","volume"]].dropna()
    if len(df) > 2:
        median_diff = (df.index.to_series().diff().median())
        if pd.notna(median_diff) and median_diff <= pd.Timedelta(minutes=1, seconds=5):
            df = df.resample("5T").agg({
                "open":"first","high":"max","low":"min","close":"last","volume":"sum"
            }).dropna()
    df = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]
    return df

# =========================
# UI
# =========================
st.title("ETH 5m Scalper – Backtest (TSI D+W, EMA Pullback, Engulf, ATR×2, Risk)")

with st.sidebar:
    st.header("Ayarlar")
    data_source = st.selectbox(
        "Veri kaynağı",
        ["CSV yükle", "Binance Arşiv (otomatik)", "API (ccxt)"],  # API sona alındı
        index=1
    )
    start_date = st.date_input("Başlangıç", value=(dt.date.today() - dt.timedelta(days=365)))
    end_date   = st.date_input("Bitiş", value=dt.date.today())
    if data_source == "CSV yükle":
        csv_file = st.file_uploader("CSV seç", type=["csv"])
    else:
        csv_file = None
    run_btn = st.button("Backtest Çalıştır")

# =========================
# Veri kaynağı seçimi
# =========================
if run_btn:
    if data_source == "CSV yükle":
        if csv_file is None:
            st.error("CSV dosyası seçilmedi.")
            st.stop()
        df_5m = read_and_prepare_csv(csv_file, start_date, end_date)

    elif data_source == "Binance Arşiv (otomatik)":
        months = list_last_n_months(12)
        frames = []
        for (y, m) in months:
            url = monthly_url_futures_eth_5m(y, m)
            zb = fetch_zip(url)
            if zb is None:
                continue
            try:
                dfm = parse_month_zip(zb)
                frames.append(dfm)
                time.sleep(0.4)
            except Exception as e:
                st.write(f"{y}-{m} okunamadı: {e}")
        if not frames:
            st.error("Arşivden veri alınamadı.")
            st.stop()
        df_5m = pd.concat(frames)
        df_5m = df_5m.loc[(df_5m.index.date >= start_date) & (df_5m.index.date <= end_date)]

    elif data_source == "API (ccxt)":
        try:
            import ccxt  # gecikmeli import
        except Exception as e:
            import sys, pkgutil
            py = sys.version.replace("\n"," ")
            has_ccxt = pkgutil.find_loader("ccxt") is not None
            st.error(
                f"ccxt import edilemedi: {e}\n\n"
                f"Python: {py}\n"
                f"ccxt bulundu mu? {has_ccxt}\n"
                "Muhtemel neden: runtime.txt uygulanmadı veya ccxt kurulmadı."
            )
            st.stop()
        st.success("ccxt başarıyla import edildi. Burada API çağrılarını çalıştırabilirsin.")
        st.stop()

    st.success(f"5m bar sayısı: {len(df_5m):,}")
    st.dataframe(df_5m.head())
