# Ailey-Bailey's Master Data Provider v1.5 (Primary + Backup Providers)

import os
import pandas as pd
import yfinance as yf
from datetime import datetime
import requests
import time

TZ_NY = "America/New_York"


def _std_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = ["Open", "High", "Low", "Close", "Adj Close", "AdjClose", "Volume"]
    out = pd.DataFrame(index=df.index.copy())
    if "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "AdjClose"})
    for c in ["Open", "High", "Low", "Close", "AdjClose", "Volume"]:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    if "AdjClose" not in out.columns and "Close" in out.columns:
        out["AdjClose"] = out["Close"]
    return out.dropna(how="all")


def _get(url, params, timeout=20, max_retries=2):
    last_exception = None
    for _ in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as exc:
            last_exception = exc
            time.sleep(0.6)
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Request failed after retries without a specific exception.")


def daily_eodhd(sym: str):
    token = os.getenv("EODHD_TOKEN")
    if not token:
        raise RuntimeError("EODHD_TOKEN is missing")
    url = f"https://eodhd.com/api/eod/{sym}.US"
    params = {
        "api_token": token,
        "from": "2000-01-01",
        "to": datetime.utcnow().strftime("%Y-%m-%d"),
        "fmt": "json"
    }
    r = _get(url, params)
    j = r.json()
    if not isinstance(j, list) or not j:
        raise RuntimeError(f"EODHD returned no daily data for {sym}")
    df = (
        pd.DataFrame(j)
        .rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adjusted_close": "AdjClose",
            "volume": "Volume",
            "date": "Date"
        })
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    df.index = df.index.tz_localize(TZ_NY, nonexistent="shift_forward", ambiguous="NaT")
    return _std_ohlcv(df), "eodhd"


def fetch_yf(sym: str, interval="1d"):
    period = "60d" if interval != "1d" else "max"
    ticker = yf.Ticker(sym)
    df = ticker.history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"yfinance found no data for {sym} with interval {interval}")
    df.index = df.index.tz_convert(TZ_NY)
    return _std_ohlcv(df), "yfinance"


def _bars_per_day(interval: str) -> int:
    mapping = {"1m": 390, "2m": 195, "5m": 78, "10m": 39, "15m": 26, "30m": 13, "1h": 7}
    return mapping.get(interval, 78)


def intra_twelvedata(sym: str, interval="5m", days=3):
    token = os.getenv("TWELVE_TOKEN")
    if not token:
        raise RuntimeError("TWELVE_TOKEN missing")
    interval_map = {
        "1m": "1min",
        "2m": "2min",
        "5m": "5min",
        "10m": "10min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h"
    }
    itv = interval_map.get(interval, "5min")
    bars_per_interval = _bars_per_day(interval) * max(1, int(days)) + 5
    outputsize = min(5000, bars_per_interval)
    params = {
        "symbol": sym,
        "interval": itv,
        "outputsize": outputsize,
        "timezone": TZ_NY,
        "format": "JSON",
        "apikey": token
    }
    url = "https://api.twelvedata.com/time_series"
    r = _get(url, params, timeout=20)
    j = r.json()
    if j.get("status") == "error" or "values" not in j:
        raise RuntimeError(j.get("message", "twelvedata intraday error"))
    df = (
        pd.DataFrame(j["values"])
        .rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "datetime": "Date"
        })
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    if df.index.tz is None:
        df.index = df.index.tz_localize(TZ_NY, nonexistent="shift_forward", ambiguous="NaT")
    else:
        df.index = df.index.tz_convert(TZ_NY)
    df["AdjClose"] = df["Close"]
    return _std_ohlcv(df), "twelvedata"


def fetch_all_data(symbol: str):
    print(f"Fetching all data for {symbol}...")
    data_frames = {}
    source = "N/A"

    daily_providers = [
        (fetch_yf, {"interval": "1d"}),
        (daily_eodhd, {})
    ]
    for func, params in daily_providers:
        try:
            df_d, source = func(symbol, **params)
            if df_d is not None and not df_d.empty:
                print(f" -> Successfully fetched daily data for {symbol} from '{source}'")
                data_frames["D"] = df_d
                break
        except Exception as exc:
            print(f" -> Provider '{func.__name__}' failed for daily data: {exc}")

    if "D" not in data_frames:
        print(f" -> [ERROR] All daily providers failed for {symbol}. Aborting.")
        return None, "Failed"

    try:
        agg_rules = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "AdjClose": "last", "Volume": "sum"}
        data_frames["W"] = data_frames["D"].resample("W-FRI").agg(agg_rules).dropna(how="all")
        data_frames["M"] = data_frames["D"].resample("ME").agg(agg_rules).dropna(how="all")
        print(f" -> Resampling to W/M successful for {symbol}")
    except Exception as exc:
        print(f" -> [Warning] Failed to resample D/W/M data for {symbol}: {exc}")

    intraday_providers = [
        (fetch_yf, {}),
        (intra_twelvedata, {})
    ]
    interval_days = {"5m": 3, "15m": 5, "30m": 7}
    for interval in ["5m", "15m", "30m"]:
        for func, params in intraday_providers:
            try:
                current = params.copy()
                if func is fetch_yf:
                    current["interval"] = interval
                else:
                    current.setdefault("interval", interval)
                    current.setdefault("days", interval_days.get(interval, 3))
                df_intra, intra_source = func(symbol, **current)
                if df_intra is not None and not df_intra.empty:
                    print(f" -> Successfully fetched {interval} data for {symbol} from '{intra_source}'")
                    data_frames[interval] = df_intra
                    break
            except Exception as exc:
                print(f" -> Provider '{func.__name__}' failed for {interval} data: {exc}")
        if interval not in data_frames:
            print(f" -> [Warning] All providers failed for {interval} data of {symbol}.")
            data_frames[interval] = pd.DataFrame()

    return data_frames, source
