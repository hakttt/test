# eth_scalper.py
# Çalıştır:  streamlit run eth_scalper.py
# Gerekli paketler:
#   pip install streamlit ccxt plotly ta pandas numpy

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import ccxt
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
    """True Strength Index (basit)"""
    m = close.diff()
    ema1 = m.ewm(span=r, adjust=False).mean()
    ema2 = ema1.ewm(span=s, adjust=False).mean()
    a = m.abs()
    a1 = a.ewm(span=r, adjust=False).mean()
    a2 = a1.ewm(span=s, adjust=False).mean()
    val = 100 * (ema2 / a2)
    return val

def bullish_engulf(prev_open, prev_close, curr_open, curr_close) -> bool:
    # Önceki kırmızı + şimdiki yeşil, gövde tamamen sarıyor
    return (prev_close < prev_open) and (curr_close > curr_open) and \
           (curr_open <= prev_close) and (curr_close >= prev_open)

def bearish_engulf(prev_open, prev_close, curr_open, curr_close) -> bool:
    # Önceki yeşil + şimdiki kırmızı, gövde tamamen sarıyor
    return (prev_close > prev_open) and (curr_close < curr_open) and \
           (curr_open >= prev_close) and (curr_close <= prev_open)

def fetch_ohlcv(exchange, symbol: str, timeframe: str, since_ms=None, limit=1500):
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
    if not data:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"]).set_index("timestamp")
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
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
    ohlc["tsi_color"] = np.where(ohlc["TSI"] > 0, 1, np.where(ohlc["TSI"] < 0, -1, 0))
    return ohlc[["TSI","tsi_color"]]

def compute_indicators(df: pd.DataFrame, ema_fast=7, ema_mid=13, ema_slow=26, atr_len=14):
    df = df.copy()
    df[f"ema_{ema_fast}"] = ema(df["close"], ema_fast)
    df[f"ema_{ema_mid}"] = ema(df["close"], ema_mid)
    df[f"ema_{ema_slow}"] = ema(df["close"], ema_slow)
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=atr_len)
    df["atr"] = atr.average_true_range()
    df["vol_ma"] = df["volume"].rolling(20).mean()
    return df

# =========================
# Backtest Motoru
# =========================

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
    pos_side = None     # 'long' / 'short'
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
            # EMA bandına dokunuş (bir önceki bar)
            touched_long  = ema_long_ok  and ((prev["low"] <= prev[em]) or (prev["low"] <= prev[ef]) or (prev["low"] <= prev[es]))
            touched_short = ema_short_ok and ((prev["high"] >= prev[em]) or (prev["high"] >= prev[ef]) or (prev["high"] >= prev[es]))

            # Engulf (mevcut bar, önceki barı sarıyor)
            bull_eng = bullish_engulf(prev["open"], prev["close"], row["open"], row["close"])
            bear_eng = bearish_engulf(prev["open"], prev["close"], row["open"], row["close"])

            # Hacim filtresi (engulf bar)
            vol_ma = 0.0 if np.isnan(row["vol_ma"]) else row["vol_ma"]
            vol_ok = (row["volume"] >= vol_mult * vol_ma)

            # Bu işleme ayrılacak risk (USDT)
            risk_usdt = risk_fixed_usdt if risk_mode == "fixed" else equity * risk_pct
            if risk_usdt <= 0:
                prev = row
                continue

            # LONG ENTRY
            if touched_long and bull_eng and vol_ok and tsi_long_gate:
                entry = row["close"] * (1 + entry_offset_bps / 10000.0)

                swing_low = df.loc[:ts, "low"].tail(swing_lookback).min()
                atr_stop  = row["close"] - row["atr"] * atr_mult
                stop      = min(swing_low, atr_stop)  # daha aşağıdaki (uzak) stop
                stop_dist = entry - stop
                if stop_dist <= 0 or np.isnan(stop_dist):
                    prev = row
                    continue

                qty = risk_usdt / stop_dist
                # kaldıraç tavanı
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
                pos_side    = "long"
                entry_price = entry
                stop_price  = stop
                tp_price    = tp
                entry_qty   = qty
                entry_time  = ts
                entry_risk_usdt = risk_usdt

            # SHORT ENTRY
            elif touched_short and bear_eng and vol_ok and tsi_short_gate:
                entry = row["close"] * (1 - entry_offset_bps / 10000.0)

                swing_high = df.loc[:ts, "high"].tail(swing_lookback).max()
                atr_stop   = row["close"] + row["atr"] * atr_mult
                stop       = max(swing_high, atr_stop)  # daha yukarıdaki (uzak) stop
                stop_dist  = stop - entry
                if stop_dist <= 0 or np.isnan(stop_dist):
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
                pos_side    = "short"
                entry_price = entry
                stop_price  = stop
                tp_price    = tp
                entry_qty   = qty
                entry_time  = ts
                entry_risk_usdt = risk_usdt

        else:
            # Pozisyon yönetimi
            exit_price = None
            reason = None
            if pos_side == "long":
                if row["low"] <= stop_price:
                    exit_price = stop_price; reason = "SL"
                elif row["high"] >= tp_price:
                    exit_price = tp_price;  reason = "TP"

                if exit_price is not None:
                    fees = fee_perc * entry_price * entry_qty + fee_perc * exit_price * entry_qty
                    pnl  = (exit_price - entry_price) * entry_qty - fees
                    equity += pnl
                    trades.append({
                        "side": "long",
                        "entry_time": entry_time, "exit_time": ts,
                        "entry": entry_price, "exit": exit_price,
                        "qty": entry_qty, "risk_usdt": entry_risk_usdt,
                        "stop_dist": (entry_price - stop_price),
                        "pnl": pnl, "reason": reason
                    })
                    in_position = False
                    equity_curve.append((ts, equity))

            elif pos_side == "short":
                if row["high"] >= stop_price:
                    exit_price = stop_price; reason = "SL"
                elif row["low"] <= tp_price:
                    exit_price = tp_price;  reason = "TP"

                if exit_price is not None:
                    fees = fee_perc * entry_price * entry_qty + fee_perc * exit_price * entry_qty
                    pnl  = (entry_price - exit_price) * entry_qty - fees
                    equity += pnl
                    trades.append({
                        "side": "short",
                        "entry_time": entry_time, "exit_time": ts,
                        "entry": entry_price, "exit": exit_price,
                        "qty": entry_qty, "risk_usdt": entry_risk_usdt,
                        "stop_dist": (stop_price - entry_price),
                        "pnl": pnl, "reason": reason
                    })
                    in_position = False
                    equity_curve.append((ts, equity))

        prev = row

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["return_pct_on_entry"] = trades_df["pnl"] / (trades_df["entry"] * trades_df["qty"]) * 100.0

    eq_df = pd.DataFrame(equity_curve, columns=["time","equity"]).set_index("time")
    return trades_df, eq_df

# =========================
# UI
# =========================

st.title("ETH 5m Scalper – Backtest (TSI D+W, EMA Pullback, Engulf, ATR×2, Risk)")

with st.sidebar:
    st.header("Ayarlar")
    data_source = st.selectbox("Veri kaynağı", ["API (ccxt)", "CSV yükle"], index=0)
    exchange_name = st.selectbox("Borsa", ["binance"], index=0, disabled=(data_source=="CSV yükle"))
    symbol = st.text_input("Sembol (Perp)", value="ETH/USDT", disabled=(data_source=="CSV yükle"))
    start_date = st.date_input("Başlangıç", value=(dt.date.today() - dt.timedelta(days=365)))
    end_date   = st.date_input("Bitiş", value=dt.date.today())

    if data_source == "CSV yükle":
        st.info("CSV sütunları: timestamp, open, high, low, close, volume (timestamp s veya ms).")
        csv_file = st.file_uploader("CSV seç", type=["csv"])
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
    plot_tsi = st.checkbox("TSI çizgilerini alt panelde göster", value=True)

    st.markdown("---")
    st.subheader("Sermaye & Risk")
    initial_capital = st.number_input("Başlangıç Sermaye (USDT)", 100.0, 1_000_000.0, 1000.0, 50.0)
    risk_mode = st.selectbox("Risk modu", ["dynamic","fixed"], index=0, help="dynamic: her işlemde güncel bakiyenin %'i; fixed: sabit USDT")
    risk_pct = st.slider("Risk % (dynamic)", 0.1, 10.0, 2.0, 0.1) / 100.0
    risk_fixed_usdt = st.number_input("Sabit risk (USDT)", 1.0, 10_000.0, 20.0, 1.0)
    leverage = st.slider("Kaldıraç (x)", 1.0, 50.0, 10.0, 0.5)
    fee_perc = st.number_input("Komisyon (her bacak, %)", 0.0, 1.0, 0.05, 0.01) / 100.0

    run_btn = st.button("Backtest Çalıştır")

# =========================
# CSV Okuyucu
# =========================

def read_and_prepare_csv(file, start_date, end_date):
    df = pd.read_csv(file)
    lower_cols = [c.lower() for c in df.columns]
    required = ["open","high","low","close","volume"]
    for req in required:
        if req not in lower_cols:
            raise ValueError("CSV sütunları 'open, high, low, close, volume' içermeli.")
    # timestamp
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
        raise ValueError("CSV'de 'timestamp' veya 'date' sütunu bulunamadı.")

    ren = {}
    for name in ["open","high","low","close","volume"]:
        ren[df.columns[lower_cols.index(name)]] = name
    df = df.rename(columns=ren)
    df.index = idx
    df = df[["open","high","low","close","volume"]].dropna()

    # Eğer 1m ise 5m'ye resample
    if len(df) > 2:
        median_diff = (df.index.to_series().diff().median())
        if pd.notna(median_diff) and median_diff <= pd.Timedelta(minutes=1, seconds=5):
            df = df.resample("5T").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

    # Tarih filtresi
    df = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]
    if df.empty:
        raise ValueError("Seçilen tarih aralığında veri bulunamadı.")
    return df

# =========================
# Veri Yükleme + Backtest + Görselleştirme
# =========================

if run_btn:
    if data_source == "CSV yükle":
        if csv_file is None:
            st.error("CSV dosyası seçilmedi.")
            st.stop()
        st.info("CSV okunuyor…")
        try:
            df_5m = read_and_prepare_csv(csv_file, start_date, end_date)
        except Exception as e:
            st.error(f"CSV okunamadı: {e}")
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
            if len(df_5m_list) > 200:  # ~3 yıl civarı
                break
        if not df_5m_list:
            st.error("Veri alınamadı. Sembol veya tarih aralığını kontrol edin.")
            st.stop()
        df_5m = pd.concat(df_5m_list)
        df_5m = df_5m.loc[(df_5m.index.date >= start_date) & (df_5m.index.date <= end_date)]

    st.success(f"5m bar sayısı: {len(df_5m)}")

    # Göstergeler
    df_5m = compute_indicators(df_5m, ema_fast=ema_fast, ema_mid=ema_mid, ema_slow=ema_slow, atr_len=atr_len)

    # TSI (Günlük & Haftalık)
    tsi_D = resample_tsi(df_5m, "1D", r=tsi_r, s=tsi_s)
    tsi_W = resample_tsi(df_5m, "1W", r=tsi_r, s=tsi_s)

    # Backtest
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

    # Görselleştirme
    col1, col2 = st.columns([3, 2])

    with col1:
        if plot_tsi:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.72, 0.28], vertical_spacing=0.05,
                                subplot_titles=("Fiyat (5m)", "TSI (Günlük & Haftalık)"))
        else:
            fig = make_subplots(rows=1, cols=1)

        # Fiyat + EMA
        fig.add_trace(
            go.Candlestick(
                x=df_5m.index, open=df_5m["open"], high=df_5m["high"],
                low=df_5m["low"], close=df_5m["close"], name="ETH 5m"
            ),
            row=1, col=1
        )
        fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m[f"ema_{ema_fast}"], name=f"EMA {ema_fast}", mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m[f"ema_{ema_mid}"],  name=f"EMA {ema_mid}",  mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m[f"ema_{ema_slow}"], name=f"EMA {ema_slow}", mode="lines"), row=1, col=1)

        # Entry/Exit işaretleri
        if not trades.empty:
            longs  = trades[trades["side"]=="long"]
            shorts = trades[trades["side"]=="short"]

            fig.add_trace(
                go.Scatter(
                    x=longs["entry_time"], y=longs["entry"],
                    mode="markers", name="Long Entry",
                    marker=dict(symbol="triangle-up", size=9)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=longs["exit_time"], y=longs["exit"],
                    mode="markers", name="Long Exit",
                    marker=dict(symbol="x", size=8)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=shorts["entry_time"], y=shorts["entry"],
                    mode="markers", name="Short Entry",
                    marker=dict(symbol="triangle-down", size=9)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=shorts["exit_time"], y=shorts["exit"],
                    mode="markers", name="Short Exit",
                    marker=dict(symbol="x", size=8)
                ),
                row=1, col=1
            )

        # TSI alt panel
        if plot_tsi:
            tsiD = tsi_D["TSI"].reindex(df_5m.index, method="ffill")
            tsiW = tsi_W["TSI"].reindex(df_5m.index, method="ffill")
            fig.add_trace(go.Scatter(x=df_5m.index, y=tsiD, name="TSI Daily", mode="lines"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_5m.index, y=tsiW, name="TSI Weekly", mode="lines"), row=2, col=1)
            # 0 hattı
            fig.update_yaxes(title_text="", row=2, col=1)
            fig.add_hline(y=0, line=dict(width=1, dash="dot"), row=2, col=1)

        fig.update_layout(height=750, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sonuçlar")
        if trades.empty:
            st.warning("Trade oluşmadı – filtreler çok sıkı olabilir ya da veri aralığı kısa.")
        else:
            total_pnl = trades["pnl"].sum()
            wins = (trades["pnl"] > 0).sum()
            losses = (trades["pnl"] <= 0).sum()
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

            st.dataframe(trades.sort_values("entry_time").reset_index(drop=True))

    with st.expander("Sermaye Eğrisi"):
        if not eq.empty:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=eq.index, y=eq["equity"], mode="lines", name="Equity"))
            fig_eq.update_layout(height=300)
            st.plotly_chart(fig_eq, use_container_width=True)

    with st.expander("TSI (Günlük & Haftalık) – Son Değerler"):
        st.write("Günlük TSI son değer:", None if tsi_D.empty else float(tsi_D["TSI"].dropna().iloc[-1]))
        st.write("Haftalık TSI son değer:", None if tsi_W.empty else float(tsi_W["TSI"].dropna().iloc[-1]))
