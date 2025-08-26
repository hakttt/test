# eth_scalper.py
# Çalıştır: streamlit run eth_scalper.py

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import io
import zipfile
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="ETH 5m Scalper Backtest", layout="wide")

# ============ Yardımcılar ============

def ema(series: pd.Series, length: int):
    return series.ewm(span=length, adjust=False).mean()

def tsi(close: pd.Series, r: int = 25, s: int = 13):
    m = close.diff()
    ema1 = m.ewm(span=r, adjust=False).mean()
    ema2 = ema1.ewm(span=s, adjust=False).mean()
    a1 = m.abs().ewm(span=r, adjust=False).mean()
    a2 = a1.ewm(span=s, adjust=False).mean()
    return 100 * (ema2 / a2)

def atr_sma(df: pd.DataFrame, length: int = 14):
    """Basit ATR (SMA). Wilder EMA istersen .ewm(alpha=1/length) ile değiştirebilirsin."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift()
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()

def bullish_engulf(prev_open, prev_close, curr_open, curr_close) -> bool:
    return (prev_close < prev_open) and (curr_close > curr_open) and \
           (curr_open <= prev_close) and (curr_close >= prev_open)

def bearish_engulf(prev_open, prev_close, curr_open, curr_close) -> bool:
    return (prev_close > prev_open) and (curr_close < curr_open) and \
           (curr_open >= prev_close) and (curr_close <= prev_open)

# ---- Binance arşiv yardımcıları (USDT-M futures, ETHUSDT, 5m)
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

# ---- Hızlı IO + Cache

@st.cache_data(show_spinner=False)
def load_any(file_or_path):
    """CSV / Parquet / Feather otomatik okur (hız için Parquet önerilir)."""
    name = getattr(file_or_path, "name", str(file_or_path))
    if name.endswith(".parquet"):
        return pd.read_parquet(file_or_path)
    if name.endswith(".feather"):
        return pd.read_feather(file_or_path)
    return pd.read_csv(file_or_path)

@st.cache_data(show_spinner=False)
def normalize_to_5m(df_in: pd.DataFrame):
    """timestamp & kolon isimleri normalize + gerekirse 5m resample."""
    df = df_in.copy()
    lower = [c.lower() for c in df.columns]
    # timestamp/datetime bul
    if "timestamp" in lower:
        col = df.columns[lower.index("timestamp")]
        ts_vals = pd.to_numeric(df[col], errors="coerce")
        idx = pd.to_datetime(ts_vals, unit="ms" if ts_vals.max() > 1e12 else "s", utc=True)
    elif "date" in lower:
        col = df.columns[lower.index("date")]
        idx = pd.to_datetime(df[col], utc=True, errors="coerce")
    else:
        # Eğer zaten index datetime ise
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        else:
            raise ValueError("CSV/Veri 'timestamp' veya 'date' sütunu içermeli ya da datetime index olmalı.")
    # sütunları standartla
    need = ["open","high","low","close","volume"]
    rename = {}
    for n in need:
        if n in lower:
            rename[df.columns[lower.index(n)]] = n
        else:
            raise ValueError(f"'{n}' sütunu eksik.")
    df = df.rename(columns=rename)
    df.index = idx
    df = df[need].dropna().sort_index()
    # 1m ise 5m'ye resample
    if len(df) > 2:
        median_diff = (df.index.to_series().diff().median())
        if pd.notna(median_diff) and median_diff <= pd.Timedelta(minutes=1, seconds=5):
            df = df.resample("5T").agg({
                "open":"first","high":"max","low":"min","close":"last","volume":"sum"
            }).dropna()
    return df

@st.cache_data(show_spinner=False)
def filter_range(df: pd.DataFrame, start_date, end_date):
    return df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]

@st.cache_data(show_spinner=False)
def compute_indicators(df_5m: pd.DataFrame, ema_fast=7, ema_mid=13, ema_slow=26, atr_len=14, tsi_r=25, tsi_s=13):
    out = df_5m.copy()
    out[f"ema_{ema_fast}"] = ema(out["close"], ema_fast)
    out[f"ema_{ema_mid}"]  = ema(out["close"], ema_mid)
    out[f"ema_{ema_slow}"] = ema(out["close"], ema_slow)
    out["atr"] = atr_sma(out, atr_len)
    out["vol_ma"] = out["volume"].rolling(20).mean()

    # TSI günlük/haftalık
    D = out["close"].resample("1D").last().to_frame("close").dropna()
    D["TSI"] = tsi(D["close"], tsi_r, tsi_s); D["tsi_color"] = np.sign(D["TSI"]).astype(int)

    W = out["close"].resample("1W").last().to_frame("close").dropna()
    W["TSI"] = tsi(W["close"], tsi_r, tsi_s); W["tsi_color"] = np.sign(W["TSI"]).astype(int)
    return out, D[["TSI","tsi_color"]], W[["TSI","tsi_color"]]

# ============ Backtest ============

def backtest(
    df_5m: pd.DataFrame,
    tsi_day: pd.DataFrame,
    tsi_week: pd.DataFrame,
    ema_fast=7, ema_mid=13, ema_slow=26,
    vol_mult=1.3,
    atr_mult=2.0,
    swing_lookback=10,
    entry_offset_bps=2.0,   # 2 bps = %0.02
    rr=2.0,
    # Sermaye & risk
    initial_capital=1000.0,
    risk_mode="dynamic",    # 'dynamic' veya 'fixed'
    risk_fixed_usdt=20.0,   # fixed ise sabit risk
    risk_pct=0.02,          # dynamic ise % risk
    leverage=10.0,
    fee_perc=0.0005         # her bacak için komisyon (örn. %0.05)
):
    df = df_5m.copy()
    ef, em, es = f"ema_{ema_fast}", f"ema_{ema_mid}", f"ema_{ema_slow}"

    # TSI'yi 5m bara bindir
    day = tsi_day.copy()
    week = tsi_week.copy()
    df = df.join(day.add_suffix("_D"), how="left")
    df = df.join(week.add_suffix("_W"), how="left")
    if "tsi_color_D" in df.columns: df["tsi_color_D"] = df["tsi_color_D"].ffill()
    if "tsi_color_W" in df.columns: df["tsi_color_W"] = df["tsi_color_W"].ffill()

    in_position = False
    pos_side = None
    entry_price = np.nan
    stop_price  = np.nan
    tp_price    = np.nan
    entry_qty   = np.nan
    entry_time  = None
    entry_risk_usdt = np.nan

    equity = initial_capital
    equity_curve = []
    if len(df):
        equity_curve.append((df.index[0], equity))
    trades = []
    prev = None

    for ts, row in df.iterrows():
        if prev is None:
            prev = row
            continue

        ema_long_ok = (row[ef] > row[em] > row[es])
        ema_short_ok = (row[ef] < row[em] < row[es])

        tsi_long_gate = (row.get("tsi_color_D", 0) == 1) and (row.get("tsi_color_W", 0) == 1)
        tsi_short_gate = (row.get("tsi_color_D", 0) == -1) and (row.get("tsi_color_W", 0) == -1)

        if not in_position:
            touched_long  = ema_long_ok  and ((prev["low"] <= prev[em]) or (prev["low"] <= prev[ef]) or (prev["low"] <= prev[es]))
            touched_short = ema_short_ok and ((prev["high"] >= prev[em]) or (prev["high"] >= prev[ef]) or (prev["high"] >= prev[es]))

            bull_eng = bullish_engulf(prev["open"], prev["close"], row["open"], row["close"])
            bear_eng = bearish_engulf(prev["open"], prev["close"], row["open"], row["close"])

            vol_ma = 0.0 if np.isnan(row["vol_ma"]) else row["vol_ma"]
            vol_ok = (row["volume"] >= vol_mult * vol_ma)

            risk_usdt = risk_fixed_usdt if risk_mode == "fixed" else equity * risk_pct
            if risk_usdt <= 0:
                prev = row; continue

            if touched_long and bull_eng and vol_ok and tsi_long_gate:
                entry = row["close"] * (1 + entry_offset_bps / 10000.0)
                swing_low = df.loc[:ts, "low"].tail(swing_lookback).min()
                atr_stop  = row["close"] - row["atr"] * atr_mult
                stop      = min(swing_low, atr_stop)
                stop_dist = entry - stop
                if stop_dist <= 0 or np.isnan(stop_dist):
                    prev = row; continue
                qty = risk_usdt / stop_dist
                max_notional = equity * leverage
                notional = qty * entry
                if max_notional > 0 and notional > max_notional:
                    qty = max_notional / entry
                if qty <= 0:
                    prev = row; continue
                tp = entry + rr * stop_dist
                in_position = True; pos_side="long"
                entry_price=entry; stop_price=stop; tp_price=tp
                entry_qty=qty; entry_time=ts; entry_risk_usdt=risk_usdt

            elif touched_short and bear_eng and vol_ok and tsi_short_gate:
                entry = row["close"] * (1 - entry_offset_bps / 10000.0)
                swing_high = df.loc[:ts, "high"].tail(swing_lookback).max()
                atr_stop   = row["close"] + row["atr"] * atr_mult
                stop       = max(swing_high, atr_stop)
                stop_dist  = stop - entry
                if stop_dist <= 0 or np.isnan(stop_dist):
                    prev = row; continue
                qty = risk_usdt / stop_dist
                max_notional = equity * leverage
                notional = qty * entry
                if max_notional > 0 and notional > max_notional:
                    qty = max_notional / entry
                if qty <= 0:
                    prev = row; continue
                tp = entry - rr * stop_dist
                in_position = True; pos_side="short"
                entry_price=entry; stop_price=stop; tp_price=tp
                entry_qty=qty; entry_time=ts; entry_risk_usdt=risk_usdt

        else:
            exit_price = None; reason=None
            if pos_side == "long":
                if row["low"] <= stop_price: exit_price = stop_price; reason="SL"
                elif row["high"] >= tp_price: exit_price = tp_price; reason="TP"
                if exit_price is not None:
                    fees = fee_perc*entry_price*entry_qty + fee_perc*exit_price*entry_qty
                    pnl  = (exit_price - entry_price)*entry_qty - fees
                    equity += pnl
                    trades.append({
                        "side":"long","entry_time":entry_time,"exit_time":ts,
                        "entry":entry_price,"exit":exit_price,"qty":entry_qty,
                        "risk_usdt":entry_risk_usdt,"stop_dist":(entry_price-stop_price),
                        "pnl":pnl,"reason":reason
                    })
                    in_position=False; equity_curve.append((ts,equity))

            elif pos_side == "short":
                if row["high"] >= stop_price: exit_price = stop_price; reason="SL"
                elif row["low"] <= tp_price:  exit_price = tp_price;  reason="TP"
                if exit_price is not None:
                    fees = fee_perc*entry_price*entry_qty + fee_perc*exit_price*entry_qty
                    pnl  = (entry_price - exit_price)*entry_qty - fees
                    equity += pnl
                    trades.append({
                        "side":"short","entry_time":entry_time,"exit_time":ts,
                        "entry":entry_price,"exit":exit_price,"qty":entry_qty,
                        "risk_usdt":entry_risk_usdt,"stop_dist":(stop_price-entry_price),
                        "pnl":pnl,"reason":reason
                    })
                    in_position=False; equity_curve.append((ts,equity))
        prev = row

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["return_pct_on_entry"] = trades_df["pnl"] / (trades_df["entry"] * trades_df["qty"]) * 100.0
    eq_df = pd.DataFrame(equity_curve, columns=["time","equity"]).set_index("time")
    return trades_df, eq_df

# ============ UI ============

st.title("ETH 5m Scalper – Backtest (TSI D+W, EMA Pullback, Engulf, ATR×2, Risk)")

with st.sidebar:
    st.header("Ayarlar")
    data_source = st.selectbox("Veri kaynağı",
        ["CSV yükle", "Binance Arşiv (otomatik)", "API (ccxt)"], index=1)
    start_date = st.date_input("Başlangıç", value=(dt.date.today() - dt.timedelta(days=365)))
    end_date   = st.date_input("Bitiş", value=dt.date.today())

    if data_source == "CSV yükle":
        csv_file = st.file_uploader("CSV / Parquet / Feather", type=["csv","parquet","feather"])
    else:
        csv_file = None

    st.markdown("---")
    st.subheader("Göstergeler")
    ema_fast = st.number_input("EMA Hızlı (5m)", 3, 50, 7)
    ema_mid  = st.number_input("EMA Orta (5m)", 5, 100, 13)
    ema_slow = st.number_input("EMA Yavaş (5m)", 5, 200, 26)
    tsi_r    = st.number_input("TSI r", 1, 100, 25)
    tsi_s    = st.number_input("TSI s", 1, 100, 13)
    vol_mult = st.slider("Hacim filtresi (× ort. 20)", 1.0, 3.0, 1.3, 0.1)
    atr_len  = st.number_input("ATR Periyodu (5m)", 5, 100, 14)
    atr_mult = st.slider("ATR Çarpanı (Stop)", 1.0, 5.0, 2.0, 0.1)
    swing_look = st.number_input("Swing lookback (bar)", 3, 50, 10)
    entry_offset_bps = st.number_input("Giriş offset (bps)", 0.0, 20.0, 2.0, 0.5)
    rr = st.slider("Risk/Ödül (TP/SL)", 1.0, 5.0, 2.0, 0.1)

    st.markdown("---")
    st.subheader("Sermaye & Risk")
    initial_capital = st.number_input("Başlangıç Sermaye (USDT)", 100.0, 1_000_000.0, 1000.0, 50.0)
    risk_mode = st.selectbox("Risk modu", ["dynamic","fixed"], index=0,
                             help="dynamic: her işlemde güncel bakiyenin %'i; fixed: sabit USDT")
    risk_pct = st.slider("Risk % (dynamic)", 0.1, 10.0, 2.0, 0.1) / 100.0
    risk_fixed_usdt = st.number_input("Sabit risk (USDT)", 1.0, 10_000.0, 20.0, 1.0)
    leverage = st.slider("Kaldıraç (x)", 1.0, 50.0, 10.0, 0.5)
    fee_perc = st.number_input("Komisyon (her bacak, %)", 0.0, 1.0, 0.05, 0.01) / 100.0

    st.markdown("---")
    dev_fast = st.toggle("Geliştirme modu (grafikleri atla)", value=False)

    run_btn = st.button("Backtest Çalıştır")

# API alanları (opsiyonel)
if data_source == "API (ccxt)":
    st.info("API modu için ccxt gerekir. Sembol: ETH/USDT. Futures için binanceusdm önerilir.")
    colA, colB = st.columns(2)
    with colA:
        exchange_name = st.selectbox("Borsa", ["binance", "binanceusdm", "bybit", "okx"], index=0)
    with colB:
        symbol = st.text_input("Sembol", value="ETH/USDT")

# ============ Veri Yükleme ============

if run_btn:
    if data_source == "CSV yükle":
        if csv_file is None:
            st.error("Dosya seçilmedi."); st.stop()
        raw = load_any(csv_file)
        df_5m_all = normalize_to_5m(raw)
        df_5m = filter_range(df_5m_all, start_date, end_date)

    elif data_source == "Binance Arşiv (otomatik)":
        st.info("Binance arşivinden son 12 ay 5m ETHUSDT (USDT-M futures) indiriliyor…")
        frames = []
        for (y, m) in list_last_n_months(12):
            url = monthly_url_futures_eth_5m(y, m)
            st.write(f"- {y}-{m:02d} indiriliyor…")
            zb = fetch_zip(url)
            if zb is None:
                st.write("  (yayınlanmamış/404, geçiliyor)")
                continue
            try:
                dfm = parse_month_zip(zb)
                frames.append(dfm)
                time.sleep(0.3)
            except Exception as e:
                st.write(f"  (okunamadı: {e})")
        if not frames:
            st.error("Arşivden veri alınamadı."); st.stop()
        df_5m_all = pd.concat(frames).sort_index()
        df_5m = filter_range(df_5m_all, start_date, end_date)
        st.caption("İstersen aşağıdan Parquet olarak kaydedip bir sonraki denemede çok daha hızlı açabilirsin.")
        if st.button("Birleştirilmiş veriyi Parquet olarak kaydet"):
            try:
                df_5m_all.to_parquet("ETHUSDT-5m-last12m.parquet")
                st.success("Kaydedildi: ETHUSDT-5m-last12m.parquet")
            except Exception as e:
                st.error(f"Kaydedilemedi: {e}")

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
        st.info("API'den OHLCV çekiliyor… (rate-limit dostu)")
        ex = getattr(ccxt, exchange_name)({"enableRateLimit": True, "timeout": 30000})
        if exchange_name == "binanceusdm":
            ex.options = {**getattr(ex, "options", {}), "defaultType": "future"}
        # markets yükleme
        for i in range(3):
            try:
                ex.load_markets(); break
            except Exception:
                if i == 2:
                    st.error("load_markets başarısız. CSV veya Arşiv modunu deneyin."); st.stop()
                time.sleep(1.5*(i+1))
        # 5m veri sayfalama
        since_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
        df_list = []; fetch_since = since_ms
        limit_per_call = 1000; calls = 0; max_calls = 60
        while True:
            try:
                data = ex.fetch_ohlcv(symbol, timeframe="5m", since=fetch_since, limit=limit_per_call)
            except Exception:
                time.sleep(1.0)
                data = ex.fetch_ohlcv(symbol, timeframe="5m", since=fetch_since, limit=limit_per_call)
            if not data: break
            chunk = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], unit="ms", utc=True)
            chunk.set_index("timestamp", inplace=True)
            df_list.append(chunk)
            last_ts = int(chunk.index[-1].timestamp() * 1000)
            if chunk.index[-1].date() >= end_date: break
            fetch_since = last_ts + 5*60*1000
            calls += 1
            if calls >= max_calls: break
            time.sleep(0.4)
        if not df_list:
            st.error("Veri alınamadı."); st.stop()
        df_5m_all = pd.concat(df_list).sort_index()
        df_5m = filter_range(df_5m_all, start_date, end_date)

    else:
        st.error("Bilinmeyen veri kaynağı."); st.stop()

    if df_5m.empty:
        st.warning("Seçilen aralıkta veri yok."); st.stop()

    st.success(f"5m bar sayısı: {len(df_5m):,}")

    # ============ İndikatörler ============
    df_5m, tsi_D, tsi_W = compute_indicators(
        df_5m, ema_fast=ema_fast, ema_mid=ema_mid, ema_slow=ema_slow,
        atr_len=atr_len, tsi_r=tsi_r, tsi_s=tsi_s
    )

    # ============ Backtest ============
    st.info("Backtest çalışıyor…")
    trades, eq = backtest(
        df_5m, tsi_D, tsi_W,
        ema_fast=ema_fast, ema_mid=ema_mid, ema_slow=ema_slow,
        vol_mult=vol_mult, atr_mult=atr_mult, swing_lookback=swing_look,
        entry_offset_bps=entry_offset_bps, rr=rr,
        initial_capital=initial_capital, risk_mode=risk_mode,
        risk_fixed_usdt=risk_fixed_usdt, risk_pct=risk_pct,
        leverage=leverage, fee_perc=fee_perc
    )

    # ============ Çıktılar ============
    col1, col2 = st.columns([3,2])

    with col2:
        st.subheader("Sonuçlar")
        if trades.empty:
            st.warning("Trade oluşmadı – filtreler çok sıkı olabilir ya da veri kısa.")
        else:
            total_pnl = trades["pnl"].sum()
            wins = (trades["pnl"] > 0).sum()
            winrate = 100 * wins / len(trades)
            ending_cap = eq["equity"].iloc[-1] if not eq.empty else initial_capital
            max_dd = None
            if not eq.empty:
                roll_max = eq["equity"].cummax()
                dd_series = (eq["equity"] / roll_max - 1.0) * 100
                max_dd = dd_series.min()
            st.metric("Toplam Trade", len(trades))
            st.metric("Kazanma Oranı %", f"{winrate:.1f}")
            st.metric("Toplam PnL (USDT)", f"{total_pnl:.2f}")
            st.metric("Bitiş Sermaye (USDT)", f"{ending_cap:.2f}")
            if max_dd is not None:
                st.metric("Maks. Drawdown %", f"{max_dd:.2f}")
            if not dev_fast:
                st.dataframe(trades.sort_values("entry_time").reset_index(drop=True))

    with col1:
        if not dev_fast:
            plot_tsi = True
            if plot_tsi:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.72, 0.28], vertical_spacing=0.05,
                                    subplot_titles=("Fiyat (5m)", "TSI (Günlük & Haftalık)"))
            else:
                fig = make_subplots(rows=1, cols=1)

            fig.add_trace(go.Candlestick(
                x=df_5m.index, open=df_5m["open"], high=df_5m["high"],
                low=df_5m["low"], close=df_5m["close"], name="ETH 5m"
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m[f"ema_{ema_fast}"], name=f"EMA {ema_fast}", mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m[f"ema_{ema_mid}"],  name=f"EMA {ema_mid}",  mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m[f"ema_{ema_slow}"], name=f"EMA {ema_slow}", mode="lines"), row=1, col=1)

            if not trades.empty:
                longs  = trades[trades["side"]=="long"]
                shorts = trades[trades["side"]=="short"]
                fig.add_trace(go.Scatter(x=longs["entry_time"], y=longs["entry"], mode="markers",
                                         name="Long Entry", marker=dict(symbol="triangle-up", size=9)), row=1, col=1)
                fig.add_trace(go.Scatter(x=longs["exit_time"], y=longs["exit"], mode="markers",
                                         name="Long Exit", marker=dict(symbol="x", size=8)), row=1, col=1)
                fig.add_trace(go.Scatter(x=shorts["entry_time"], y=shorts["entry"], mode="markers",
                                         name="Short Entry", marker=dict(symbol="triangle-down", size=9)), row=1, col=1)
                fig.add_trace(go.Scatter(x=shorts["exit_time"], y=shorts["exit"], mode="markers",
                                         name="Short Exit", marker=dict(symbol="x", size=8)), row=1, col=1)

            if plot_tsi:
                tsiD = tsi_D["TSI"].reindex(df_5m.index, method="ffill")
                tsiW = tsi_W["TSI"].reindex(df_5m.index, method="ffill")
                fig.add_trace(go.Scatter(x=df_5m.index, y=tsiD, name="TSI Daily", mode="lines"), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_5m.index, y=tsiW, name="TSI Weekly", mode="lines"), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_5m.index, y=[0]*len(df_5m), name="TSI 0", mode="lines",
                                         line=dict(dash="dot", width=1)), row=2, col=1)

            fig.update_layout(height=750, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("TSI Son Değerler"):
        st.write("Günlük TSI:", None if tsi_D.empty else float(tsi_D["TSI"].dropna().iloc[-1]))
        st.write("Haftalık TSI:", None if tsi_W.empty else float(tsi_W["TSI"].dropna().iloc[-1]))
