# streamlit run eth_scalper.py

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt

# Optional: install via pip if running locally the first time
# pip install ccxt plotly ta
import plotly.graph_objects as go
import ccxt
from ta.volatility import AverageTrueRange

st.set_page_config(page_title="ETH 5m Scalper Backtest", layout="wide")

# -----------------------------
# Helpers
# -----------------------------

def ema(series: pd.Series, length: int):
    return series.ewm(span=length, adjust=False).mean()

def tsi(close: pd.Series, r: int = 25, s: int = 13):
    # True Strength Index (classic)
    m = close.diff()
    ema1 = m.ewm(span=r, adjust=False).mean()
    ema2 = ema1.ewm(span=s, adjust=False).mean()

    a = m.abs()
    a1 = a.ewm(span=r, adjust=False).mean()
    a2 = a1.ewm(span=s, adjust=False).mean()

    tsi_val = 100 * (ema2 / a2)
    return tsi_val

def bullish_engulf(prev_open, prev_close, curr_open, curr_close):
    # Classic bullish engulfing: prev red, curr green and body engulfs
    return (prev_close < prev_open) and (curr_close > curr_open) and (curr_open <= prev_close) and (curr_close >= prev_open)

def bearish_engulf(prev_open, prev_close, curr_open, curr_close):
    return (prev_close > prev_open) and (curr_close < curr_open) and (curr_open >= prev_close) and (curr_close <= prev_open)


def fetch_ohlcv(exchange, symbol: str, timeframe: str, since_ms=None, limit=1500):
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("UTC")
    df.set_index("timestamp", inplace=True)
    return df


def resample_tsi(df_5m: pd.DataFrame, rule: str, r: int, s: int):
    o = df_5m["open"].resample(rule).first()
    h = df_5m["high"].resample(rule).max()
    l = df_5m["low"].resample(rule).min()
    c = df_5m["close"].resample(rule).last()
    v = df_5m["volume"].resample(rule).sum()
    ohlc = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna()
    ohlc["TSI"] = tsi(ohlc["close"], r=r, s=s)
    # Green if TSI > 0, Red if TSI < 0
    ohlc["tsi_color"] = np.where(ohlc["TSI"] > 0, 1, np.where(ohlc["TSI"] < 0, -1, 0))
    return ohlc[["TSI", "tsi_color"]]


def compute_indicators(df: pd.DataFrame, ema_fast=7, ema_mid=13, ema_slow=26, atr_len=14):
    df = df.copy()
    df[f"ema_{ema_fast}"] = ema(df["close"], ema_fast)
    df[f"ema_{ema_mid}"] = ema(df["close"], ema_mid)
    df[f"ema_{ema_slow}"] = ema(df["close"], ema_slow)

    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=atr_len)
    df["atr"] = atr.average_true_range()

    # Rolling vol mean for volume filter
    df["vol_ma"] = df["volume"].rolling(20).mean()
    return df


def backtest(df_5m: pd.DataFrame,
             tsi_day: pd.DataFrame,
             tsi_week: pd.DataFrame,
             ema_fast=7, ema_mid=13, ema_slow=26,
             vol_mult=1.3,
             atr_mult=2.0,
             swing_lookback=10,
             entry_offset_bps=2,  # 2 bps = 0.02%
             rr=2.0,
             # NEW: capital & risk params
             initial_capital=1000.0,
             risk_mode="dynamic",  # 'dynamic' or 'fixed'
             risk_fixed_usdt=20.0,  # used if fixed
             risk_pct=0.02,         # used if dynamic
             leverage=10.0,
             fee_perc=0.0005  # per side, e.g. 0.05%
             ):

    df = df_5m.copy()
    ef, em, es = f"ema_{ema_fast}", f"ema_{ema_mid}", f"ema_{ema_slow}"

    # Join TSI filters to 5m
    day = tsi_day.copy()
    week = tsi_week.copy()
    df = df.join(day.add_suffix("_D"), how="left")
    df = df.join(week.add_suffix("_W"), how="left")
    df[["tsi_color_D", "tsi_color_W"]] = df[["tsi_color_D", "tsi_color_W"]].ffill()

    # Position state
    in_position = False
    pos_side = None  # 'long' or 'short'
    entry_price = np.nan
    stop_price = np.nan
    tp_price = np.nan

    # Capital state
    equity = initial_capital
    equity_curve = [(df.index[0] if len(df) else pd.NaT, equity)]

    trades = []

    # We iterate row by row
    prev = None
    for ts, row in df.iterrows():
        if prev is None:
            prev = row
            continue

        # Determine EMA order
        ema_long_ok = (row[ef] > row[em] > row[es])
        ema_short_ok = (row[ef] < row[em] < row[es])

        # Daily+Weekly TSI gate
        tsi_long_gate = (row.get("tsi_color_D", 0) == 1) and (row.get("tsi_color_W", 0) == 1)
        tsi_short_gate = (row.get("tsi_color_D", 0) == -1) and (row.get("tsi_color_W", 0) == -1)

        # Entry logic only if flat
        if not in_position:
            # Check EMA touch on previous bar
            touched_long = ema_long_ok and ((prev["low"] <= prev[em]) or (prev["low"] <= prev[ef]) or (prev["low"] <= prev[es]))
            touched_short = ema_short_ok and ((prev["high"] >= prev[em]) or (prev["high"] >= prev[ef]) or (prev["high"] >= prev[es]))

            # Engulfing on current bar relative to previous
            bull_eng = bullish_engulf(prev["open"], prev["close"], row["open"], row["close"]) if not np.isnan(prev["open"]) else False
            bear_eng = bearish_engulf(prev["open"], prev["close"], row["open"], row["close"]) if not np.isnan(prev["open"]) else False

            # Volume filter on engulf bar
            vol_ok = (row["volume"] >= vol_mult * (row["vol_ma"] if not np.isnan(row["vol_ma"]) else 0))

            # Risk amount for this trade
            risk_usdt = risk_fixed_usdt if risk_mode == "fixed" else equity * risk_pct
            risk_usdt = max(0.0, float(risk_usdt))
            if risk_usdt <= 0:
                prev = row
                continue

            # LONG entry
            if touched_long and bull_eng and vol_ok and tsi_long_gate:
                entry = row["close"] * (1 + entry_offset_bps / 10000)
                swing_low = df.loc[:ts, "low"].tail(swing_lookback).min()
                atr_stop = row["close"] - row["atr"] * atr_mult
                stop = min(swing_low, atr_stop)
                stop_dist = max(0.0, entry - stop)
                if stop_dist <= 0:
                    prev = row
                    continue
                # Position sizing
                qty = risk_usdt / stop_dist
                # Leverage cap
                max_notional = equity * leverage
                notional = qty * entry
                if notional > max_notional and max_notional > 0:
                    qty = max_notional / entry
                    notional = qty * entry
                if qty <= 0:
                    prev = row
                    continue
                tp = entry + rr * stop_dist

                in_position = True
                pos_side = 'long'
                entry_price = entry
                stop_price = stop
                tp_price = tp
                entry_time = ts
                entry_qty = qty

            # SHORT entry
            elif touched_short and bear_eng and vol_ok and tsi_short_gate:
                entry = row["close"] * (1 - entry_offset_bps / 10000)
                swing_high = df.loc[:ts, "high"].tail(swing_lookback).max()
                atr_stop = row["close"] + row["atr"] * atr_mult
                stop = max(swing_high, atr_stop)
                stop_dist = max(0.0, stop - entry)
                if stop_dist <= 0:
                    prev = row
                    continue
                qty = risk_usdt / stop_dist
                max_notional = equity * leverage
                notional = qty * entry
                if notional > max_notional and max_notional > 0:
                    qty = max_notional / entry
                    notional = qty * entry
                if qty <= 0:
                    prev = row
                    continue
                tp = entry - rr * stop_dist

                in_position = True
                pos_side = 'short'
                entry_price = entry
                stop_price = stop
                tp_price = tp
                entry_time = ts
                entry_qty = qty

        else:
            # Manage open position
            if pos_side == 'long':
                exit_price = None
                reason = None
                if row["low"] <= stop_price:
                    exit_price = stop_price
                    reason = "SL"
                elif row["high"] >= tp_price:
                    exit_price = tp_price
                    reason = "TP"
                if exit_price is not None:
                    # Fees (entry + exit)
                    fees = fee_perc * entry_price * entry_qty + fee_perc * exit_price * entry_qty
                    pnl = (exit_price - entry_price) * entry_qty - fees
                    equity += pnl
                    trades.append({
                        "side": "long", "entry_time": entry_time, "exit_time": ts,
                        "entry": entry_price, "exit": exit_price, "qty": entry_qty,
                        "risk_usdt": risk_usdt, "stop_dist": (entry_price - stop_price),
                        "pnl": pnl, "pnl_pct": (pnl / max(1e-9, equity)) * 100, "reason": reason
                    })
                    in_position = False
                    equity_curve.append((ts, equity))
            else:
                exit_price = None
                reason = None
                if row["high"] >= stop_price:
                    exit_price = stop_price
                    reason = "SL"
                elif row["low"] <= tp_price:
                    exit_price = tp_price
                    reason = "TP"
                if exit_price is not None:
                    fees = fee_perc * entry_price * entry_qty + fee_perc * exit_price * entry_qty
                    pnl = (entry_price - exit_price) * entry_qty - fees
                    equity += pnl
                    trades.append({
                        "side": "short", "entry_time": entry_time, "exit_time": ts,
                        "entry": entry_price, "exit": exit_price, "qty": entry_qty,
                        "risk_usdt": risk_usdt, "stop_dist": (stop_price - entry_price),
                        "pnl": pnl, "pnl_pct": (pnl / max(1e-9, equity)) * 100, "reason": reason
                    })
                    in_position = False
                    equity_curve.append((ts, equity))

        prev = row

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["return_pct_on_entry"] = trades_df["pnl"] / (trades_df["entry"] * trades_df["qty"]) * 100.0

    eq_df = pd.DataFrame(equity_curve, columns=["time", "equity"]).set_index("time")
    return trades_df, eq_df


# -----------------------------
# UI
# -----------------------------
st.title("ETH 5m Scalper – Backtest (TSI D+W, EMA Pullback, Engulf, ATR×2, 1:2 RR)")

with st.sidebar:
    st.header("Ayarlar")
    data_source = st.selectbox("Veri kaynağı", ["API (ccxt)", "CSV yükle"], index=0)
    exchange_name = st.selectbox("Borsa", ["binance"], index=0, disabled=(data_source=="CSV yükle"))
    symbol = st.text_input("Sembol (Perp)", value="ETH/USDT", disabled=(data_source=="CSV yükle"))
    start_date = st.date_input("Başlangıç", value=(dt.date.today() - dt.timedelta(days=180)))
    end_date = st.date_input("Bitiş", value=dt.date_today := dt.date.today())

    if data_source == "CSV yükle":
        st.info("CSV sütunları: timestamp, open, high, low, close, volume. Zaman damgası ms veya s olabilir.")
        csv_file = st.file_uploader("CSV seç", type=["csv"])
    else:
        csv_file = None

    ema_fast = st.number_input("EMA Hızlı (5m)", 3, 50, 7)
    ema_mid = st.number_input("EMA Orta (5m)", 5, 100, 13)
    ema_slow = st.number_input("EMA Yavaş (5m)", 5, 200, 26)

    tsi_r = st.number_input("TSI r", 1, 100, 25)
    tsi_s = st.number_input("TSI s", 1, 100, 13)

    vol_mult = st.slider("Hacim filtresi (× ort. 20)", 1.0, 3.0, 1.3, 0.1)
    atr_len = st.number_input("ATR Periyodu (5m)", 5, 100, 14)
    atr_mult = st.slider("ATR Çarpanı (Stop)", 1.0, 5.0, 2.0, 0.1)

    swing_look = st.number_input("Swing lookback (bar)", 3, 50, 10)
    entry_offset_bps = st.number_input("Giriş offset (bps)", 0.0, 20.0, 2.0, 0.5)
    rr = st.slider("Risk/Ödül (TP/SL)", 1.0, 5.0, 2.0, 0.1)

    st.markdown("---")
    st.subheader("Sermaye & Risk")
    initial_capital = st.number_input("Başlangıç Sermaye (USDT)", 100.0, 1_000_000.0, 1000.0, 50.0)
    risk_mode = st.selectbox("Risk modu", ["dynamic", "fixed"], index=0, help="dynamic: her işlem güncel bakiyenin %'i; fixed: sabit USDT")
    risk_pct = st.slider("Risk % (dynamic)", 0.1, 10.0, 2.0, 0.1) / 100.0
    risk_fixed_usdt = st.number_input("Sabit risk (USDT)", 1.0, 10_000.0, 20.0, 1.0)
    leverage = st.slider("Kaldıraç (x)", 1.0, 50.0, 10.0, 0.5)
    fee_perc = st.number_input("Komisyon (her bacak, %)", 0.0, 1.0, 0.05, 0.01) / 100.0

    plot_tsi = st.checkbox("TSI çizgilerini göster (alt panel)", value=False)

    run_btn = st.button("Backtest Çalıştır")


if run_btn:
    if data_source == "CSV yükle":
        if csv_file is None:
            st.error("Lütfen bir CSV dosyası seçin.")
            st.stop()
        st.info("CSV okunuyor…")
        try:
            df_5m = pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"CSV okunamadı: {e}")
            st.stop()
        # Column normalization
        cols = {c.lower(): c for c in df_5m.columns}
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        # Try to map lower-case
        lower_cols = [c.lower() for c in df_5m.columns]
        if not all(r in lower_cols for r in ["open","high","low","close","volume"]):
            st.error("CSV sütunları 'open, high, low, close, volume' içermeli.")
            st.stop()
        # Timestamp handling
        if "timestamp" in lower_cols:
            ts_col = df_5m.columns[lower_cols.index("timestamp")]
            ts_vals = pd.to_numeric(df_5m[ts_col], errors='coerce')
            if ts_vals.max() > 1e12:
                # likely ms
                idx = pd.to_datetime(ts_vals, unit='ms', utc=True)
            else:
                idx = pd.to_datetime(ts_vals, unit='s', utc=True)
        elif "date" in lower_cols:
            ts_col = df_5m.columns[lower_cols.index("date")]
            idx = pd.to_datetime(df_5m[ts_col], utc=True, errors='coerce')
        else:
            st.error("CSV'de 'timestamp' veya 'date' kolonu bulunamadı.")
            st.stop()
        df_5m = df_5m.rename(columns={df_5m.columns[lower_cols.index("open")]:"open",
                                      df_5m.columns[lower_cols.index("high")]:"high",
                                      df_5m.columns[lower_cols.index("low")]:"low",
                                      df_5m.columns[lower_cols.index("close")]:"close",
                                      df_5m.columns[lower_cols.index("volume")]:"volume"})
        df_5m.index = idx
        df_5m = df_5m[["open","high","low","close","volume"]].dropna()
        # Resample to 5m if needed
        if df_5m.index.to_series().diff().median() < pd.Timedelta(minutes=5):
            df_5m = df_5m.resample("5T").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
        # Date filter
        df_5m = df_5m.loc[(df_5m.index.date >= start_date) & (df_5m.index.date <= end_date)]
        if df_5m.empty:
            st.error("Seçilen tarih aralığında veri bulunamadı.")
            st.stop()
    else:
        st.info("Veri indiriliyor… (ccxt)")
        ex = getattr(ccxt, exchange_name)({"enableRateLimit": True})
        ex.load_markets()
        since_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
        df_5m_list = []
        fetch_since = since_ms
        while True:
            df_chunk = fetch_ohlcv(ex, symbol, "5m", since_ms=fetch_since, limit=1500)
            if df_chunk.empty:
                break
            df_5m_list.append(df_chunk)
            last_ts = int(df_chunk.index[-1].timestamp() * 1000)
            if df_chunk.index[-1].date() >= end_date:
                break
            fetch_since = last_ts + 5 * 60 * 1000
            # Increased cap for longer history
            if len(df_5m_list) > 200:
                break
        if not df_5m_list:
            st.error("Veri alınamadı. Sembol veya tarih aralığını kontrol edin.")
            st.stop()
        df_5m = pd.concat(df_5m_list)
        df_5m = df_5m.loc[(df_5m.index.date >= start_date) & (df_5m.index.date <= end_date)]
    df_5m = df_5m.loc[(df_5m.index.date >= start_date) & (df_5m.index.date <= end_date)]

    st.success(f"5m bar sayısı: {len(df_5m)}")

    # İndikatörler (5m)
    df_5m = compute_indicators(df_5m, ema_fast=ema_fast, ema_mid=ema_mid, ema_slow=ema_slow, atr_len=atr_len)

    # Günlük + Haftalık TSI
    st.info("TSI (Günlük & Haftalık) hesaplanıyor…")
    tsi_D = resample_tsi(df_5m, "1D", r=tsi_r, s=tsi_s)
    tsi_W = resample_tsi(df_5m, "1W", r=tsi_r, s=tsi_s)

    # Backtest
    st.info("Backtest çalışıyor…")
    trades, eq = backtest(df_5m, tsi_D, tsi_W, ema_fast=ema_fast, ema_mid=ema_mid, ema_slow=ema_slow,
                      vol_mult=vol_mult, atr_mult=atr_mult, swing_lookback=swing_look,
                      entry_offset_bps=entry_offset_bps, rr=rr,
                      initial_capital=initial_capital, risk_mode=risk_mode,
                      risk_fixed_usdt=risk_fixed_usdt, risk_pct=risk_pct,
                      leverage=leverage, fee_perc=fee_perc)

    col1, col2 = st.columns([3, 2])

    with col1:
        # Plotly chart with EMAs and marks
        fig = go.Figure(data=[go.Candlestick(x=df_5m.index, open=df_5m['open'], high=df_5m['high'], low=df_5m['low'], close=df_5m['close'], name='ETH 5m')])
        fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m[f"ema_{ema_fast}"], name=f"EMA {ema_fast}", mode='lines'))
        fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m[f"ema_{ema_mid}"], name=f"EMA {ema_mid}", mode='lines'))
        fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m[f"ema_{ema_slow}"], name=f"EMA {ema_slow}", mode='lines'))

        if not trades.empty:
            longs = trades[trades['side']=="long"]
            shorts = trades[trades['side']=="short"]

            fig.add_trace(go.Scatter(x=longs['entry_time'], y=longs['entry'], mode='markers', name='Long Entry', marker=dict(symbol='triangle-up', size=9)))
            fig.add_trace(go.Scatter(x=longs['exit_time'], y=longs['exit'], mode='markers', name='Long Exit', marker=dict(symbol='x', size=8)))

            fig.add_trace(go.Scatter(x=shorts['entry_time'], y=shorts['entry'], mode='markers', name='Short Entry', marker=dict(symbol='triangle-down', size=9)))
            fig.add_trace(go.Scatter(x=shorts['exit_time'], y=shorts['exit'], mode='markers', name='Short Exit', marker=dict(symbol='x', size=8)))

        if plot_tsi:
            # Add TSI daily/weekly as separate subplot-like traces (scaled)
            # We'll plot them as secondary y by normalizing
            tsiD = tsi_D['TSI'].reindex(df_5m.index, method='ffill')
            tsiW = tsi_W['TSI'].reindex(df_5m.index, method='ffill')
            # Shift them to bottom using additive offset relative to price min
            base = df_5m['close'].min()
            scale = (df_5m['close'].max() - base) / 8.0
            fig.add_trace(go.Scatter(x=df_5m.index, y=base + tsiD/100*scale, name='TSI Daily (scaled)', mode='lines'))
            fig.add_trace(go.Scatter(x=df_5m.index, y=base + tsiW/100*scale, name='TSI Weekly (scaled)', mode='lines'))

        fig.update_layout(height=700, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sonuçlar")
        if trades.empty:
            st.warning("Trade oluşmadı – filtreler çok sıkı olabilir ya da veri aralığı kısa.")
        else:
            total_pnl = trades['pnl'].sum()
            wins = (trades['pnl'] > 0).sum()
            losses = (trades['pnl'] <= 0).sum()
            winrate = 100 * wins / len(trades)
            ending_cap = eq['equity'].iloc[-1] if not eq.empty else initial_capital
            ret_pct = 100 * (ending_cap / initial_capital - 1)
            max_dd = None
            if not eq.empty:
                roll_max = eq['equity'].cummax()
                dd_series = (eq['equity'] / roll_max - 1.0) * 100
                max_dd = dd_series.min()

            st.metric("Toplam Trade", len(trades))
            st.metric("Kazanma Oranı %", f"{winrate:.1f}")
            st.metric("Toplam PnL (USDT)", f"{total_pnl:.2f}")
            st.metric("Bitiş Sermaye (USDT)", f"{ending_cap:.2f}")
            if max_dd is not None:
                st.metric("Maks. Drawdown %", f"{max_dd:.2f}")

            st.dataframe(trades.sort_values('entry_time').reset_index(drop=True))

    # Equity curve
    with st.expander("Sermaye Eğrisi"):
        if not eq.empty:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=eq.index, y=eq['equity'], mode='lines', name='Equity'))
            fig_eq.update_layout(height=300)
            st.plotly_chart(fig_eq, use_container_width=True)

    with st.expander("TSI (Günlük & Haftalık) – son değerler"):
        st.write("Günlük TSI son değer:", tsi_D["TSI"].dropna().iloc[-1] if not tsi_D.empty else None)
        st.write("Haftalık TSI son değer:", tsi_W["TSI"].dropna().iloc[-1] if not tsi_W.empty else None)("TSI (Günlük & Haftalık) – son değerler"):
        st.write("Günlük TSI son değer:", tsi_D["TSI"].dropna().iloc[-1] if not tsi_D.empty else None)
        st.write("Haftalık TSI son değer:", tsi_W["TSI"].dropna().iloc[-1] if not tsi_W.empty else None)

else:
    st.info("Soldaki menüden tarih aralığı ve parametreleri seçip 'Backtest Çalıştır' butonuna bas.")
