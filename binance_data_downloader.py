# binance_data_downloader.py
# Çalıştır: streamlit run binance_data_downloader.py
# Amaç: BTCUSDT ve ETHUSDT için 5m, 1h, 1w, 1M OHLCV'yi Binance arşivinden indirip
#       tek .parquet (veya isteğe göre parçalı .parquet'leri .zip) halinde indirmeye sunmak.

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import io
import zipfile
import requests

st.set_page_config(page_title="Binance Arşiv İndirici (BTC & ETH)", layout="wide")

BINANCE_BASE = "https://data.binance.vision"
INTERVALS_ALL = ["5m", "1h", "1d", "1w", "1M"]  # hedef TF'ler (1d eklendi)

MARKETS = {
    "USDT-M Futures": "data/futures/um/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{yyyy}-{mm}.zip",
    "Spot":           "data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{yyyy}-{mm}.zip",
}

# ------------------ Yardımcılar ------------------

def months_range(start_yyyymm: str, end_yyyymm: str):
    """YYYY-MM → YYYY-MM (dahil) aralığını aylar bazında döndürür."""
    s_year, s_month = map(int, start_yyyymm.split("-"))
    e_year, e_month = map(int, end_yyyymm.split("-"))
    start = dt.date(s_year, s_month, 1)
    end = dt.date(e_year, e_month, 1)
    cur = start
    out = []
    while cur <= end:
        out.append((cur.year, cur.month))
        if cur.month == 12:
            cur = dt.date(cur.year + 1, 1, 1)
        else:
            cur = dt.date(cur.year, cur.month + 1, 1)
    return out

def fetch_zip(url, timeout=30, retries=3, backoff=1.5):
    """ZIP indir; 404 ise None döner; transient hatalarda retry."""
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
            time.sleep(backoff * (i + 1))
    return None

def parse_month_zip(zbytes):
    """Binance monthly/klines CSV → DataFrame(index=UTC Datetime, cols: ohlcv)."""
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(name) as f:
            df = pd.read_csv(f, header=None)
    # Kolonlar: open_time(ms), open, high, low, close, volume, close_time, ...
    df = df.iloc[:, :6].copy()
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    idx = pd.to_datetime(ts, unit="ms" if ts.max() > 1e12 else "s", utc=True)
    df.index = idx
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df.sort_index()

def build_dataset(symbols, month_pairs, market_key, intervals, progress_bar=None, status=None):
    """
    Her sembol & interval için aylık zipleri indirir, tek uzun tablo oluşturur.
    Dönüş: DataFrame(index=timestamp UTC, cols: open, high, low, close, volume, symbol, timeframe)
    """
    path_tmpl = MARKETS[market_key]
    records = []
    total_steps = len(symbols) * len(intervals) * max(1, len(month_pairs))
    step = 0

    for sym in symbols:
        for interval in intervals:
            frames = []
            for (y, m) in month_pairs:
                step += 1
                url = f"{BINANCE_BASE}/" + path_tmpl.format(
                    symbol=sym, interval=interval, yyyy=f"{y:04d}", mm=f"{m:02d}"
                )
                if status is not None:
                    status.write(f"İndiriliyor: **{sym} {interval} {y}-{m:02d}**")
                if progress_bar is not None:
                    progress_bar.progress(min(step / total_steps, 1.0))
                zb = fetch_zip(url)
                if zb is None:
                    continue
                try:
                    dfm = parse_month_zip(zb)
                    frames.append(dfm)
                    time.sleep(0.12)  # nazik rate-limit
                except Exception as e:
                    if status is not None:
                        status.write(f"Uyarı: {sym} {interval} {y}-{m:02d} okunamadı: {e}")

            if frames:
                df_all = pd.concat(frames).sort_index()
                df_all["symbol"] = sym
                df_all["timeframe"] = interval
                records.append(df_all)

    if not records:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "symbol", "timeframe"])
    big = pd.concat(records).sort_index()
    big = big[["open", "high", "low", "close", "volume", "symbol", "timeframe"]]
    # Sıkıştırma dostu dtypes
    big["symbol"] = big["symbol"].astype("category")
    big["timeframe"] = big["timeframe"].astype("category")
    return big

def save_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf)  # pyarrow ile
    buf.seek(0)
    return buf.read()

def build_split_zip_bytes(df: pd.DataFrame) -> bytes:
    """
    df’yi (symbol, timeframe) bölerek her biri için ayrı .parquet yaz;
    hepsini tek .zip içinde döndür.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for sym in sorted(df["symbol"].astype(str).unique()):
            for tf in sorted(df["timeframe"].astype(str).unique()):
                dfx = df[(df["symbol"].astype(str) == sym) & (df["timeframe"].astype(str) == tf)]
                if dfx.empty:
                    continue
                inner = io.BytesIO()
                dfx.to_parquet(inner)
                inner.seek(0)
                z.writestr(f"{sym}_{tf}.parquet", inner.read())
    buf.seek(0)
    return buf.read()

# ------------------ UI ------------------

st.title("Binance Arşiv İndirici → BTC & ETH • 5m, 1h, 1w, 1M")

with st.sidebar:
    st.header("Ayarlar")
    market_key = st.radio("Piyasa", list(MARKETS.keys()), index=0, horizontal=True)

    intervals = st.multiselect(
        "Zaman dilimleri",
        INTERVALS_ALL,
        default=["5m", "1h", "1w", "1M"],
        help="Sadece ham OHLCV indirilecek. İndikatör yok."
    )

    # Bu projede sadece BTC & ETH
    symbols_default = ["BTCUSDT", "ETHUSDT"]
    st.caption("Semboller sabit: BTCUSDT, ETHUSDT")

    today = dt.date.today()
    last_month = (today.replace(day=1) - dt.timedelta(days=1)).strftime("%Y-%m")
    three_years_ago = (today.replace(day=1) - pd.DateOffset(months=36)).strftime("%Y-%m")

    c1, c2 = st.columns(2)
    with c1:
        start_yyyymm = st.text_input("Başlangıç (YYYY-MM)", value=str(three_years_ago))
    with c2:
        end_yyyymm = st.text_input("Bitiş (YYYY-MM)", value=str(last_month))

    split_files = st.checkbox("Parçalı çıktı (symbol×TF ayrı .parquet → .zip)", value=False)
    run_btn = st.button("İndir ve Çıktı Oluştur", type="primary")

if run_btn:
    # Girdi kontrolleri
    try:
        months = months_range(start_yyyymm, end_yyyymm)
    except Exception:
        st.error("Tarih formatı hatalı. Örn: 2022-01")
        st.stop()

    if not intervals:
        st.error("En az bir zaman dilimi seçin.")
        st.stop()

    symbols = ["BTCUSDT", "ETHUSDT"]
    st.info(f"{market_key} | TF: {', '.join(intervals)} | Semboller: {', '.join(symbols)}")
    prog = st.progress(0.0)
    status = st.empty()
    big = build_dataset(symbols, months, market_key, intervals, progress_bar=prog, status=status)

    if big.empty:
        st.error("Hiç veri indirilemedi (seçimler/aylar mevcut olmayabilir).")
        st.stop()

    st.success(
        f"Toplam bar: {len(big):,} | Semboller: {big['symbol'].nunique()} | TF: {big['timeframe'].unique().tolist()}"
    )

    # Hızlı özet
    counts = big.groupby(["symbol", "timeframe"]).size().rename("bars").reset_index()
    st.dataframe(counts, use_container_width=True)

    # Çıktı
    if split_files:
        zip_bytes = build_split_zip_bytes(big)
        file_name = f"binance_{'futures' if 'futures' in MARKETS[market_key] else 'spot'}_{'-'.join(intervals)}_{start_yyyymm}_{end_yyyymm}_BTC_ETH_parts.zip"
        st.download_button(
            "Parçalı .parquet’ler (ZIP) indir",
            data=zip_bytes,
            file_name=file_name,
            mime="application/zip"
        )
        st.caption("Dropbox’a yüklerken sembol/timeframe bazında daha küçük dosyalar işine yarayabilir.")
    else:
        pq_bytes = save_parquet_bytes(big)
        file_name = f"binance_{'futures' if 'futures' in MARKETS[market_key] else 'spot'}_{'-'.join(intervals)}_{start_yyyymm}_{end_yyyymm}_BTC_ETH.parquet"
        st.download_button(
            "Tek .parquet indir",
            data=pq_bytes,
            file_name=file_name,
            mime="application/octet-stream"
        )
        st.caption("Tek dosya istiyorsan bu seçeneği kullan. Sonra Dropbox’a bu tek .parquet’i yükleyebilirsin.")
