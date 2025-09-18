# Ailey-Bailey's Alpha-Core OS Data Builder v6.0 (Strategy Enhanced)
import os
import json
import statistics
import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from datetime import datetime
import providers_master as providers
import providers_sentiment as sentiment_provider
import providers_meta as meta_provider
ROOT = Path(os.getenv("ALPHA_CORE_ROOT", str(Path(__file__).resolve().parent))).expanduser().resolve()
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON_PATH = DATA_DIR / "alpha_core_data.json"
TRADE_HISTORY_DEFAULT = DATA_DIR / "trades_history.json"
ASSET_SYMBOLS = ["QQQ", "QLD", "TQQQ"]
MACRO_SYMBOLS = {"UST10Y": "^TNX", "DXY": "DX-Y.NYB"}
VOL_SYMBOL = {"VIX": "^VIX"}
def get_last_value(series, precision=4):
    if series is None:
        return None
    clean = series.dropna()
    if clean.empty:
        return None
    val = clean.iloc[-1]
    return round(float(val), precision) if pd.notna(val) else None
def calculate_indicators(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    df.ta.sma(length=9, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=100, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.sma(length=300, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.rsi(length=7, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.obv(append=True)
    df.ta.bbands(length=20, std=2.0, append=True)
    if 'Volume' in df.columns:
        df['VMA_20'] = df['Volume'].rolling(20, min_periods=1).mean()
        df['VMA_50'] = df['Volume'].rolling(50, min_periods=1).mean()
    if 'OBV' in df.columns:
        df['OBV_MA20'] = df['OBV'].rolling(20, min_periods=1).mean()
    if all(x in df.columns for x in ['EMA_20', 'ATRr_14']):
        df['KC_upper'] = df['EMA_20'] + 2.5 * df['ATRr_14']
        df['KC_lower'] = df['EMA_20'] - 2.5 * df['ATRr_14']
    if 'High' in df.columns:
        df['prev20_high'] = df['High'].shift(1).rolling(20, min_periods=1).max()
    if 'Low' in df.columns:
        df['prev20_low'] = df['Low'].shift(1).rolling(20, min_periods=1).min()
    return df
def create_daily_rt(df_d: pd.DataFrame, df_5m: pd.DataFrame):
    if df_d is None or df_d.empty:
        return pd.DataFrame()
    d_rt = df_d.copy()
    if df_5m is None or df_5m.empty:
        return d_rt
    last_day = d_rt.index[-1]
    today_intraday = df_5m[df_5m.index.date == last_day.date()]
    if not today_intraday.empty:
        d_rt.at[last_day, "High"] = max(d_rt.at[last_day, "High"], today_intraday["High"].max())
        d_rt.at[last_day, "Low"] = min(d_rt.at[last_day, "Low"], today_intraday["Low"].min())
        d_rt.at[last_day, "Close"] = today_intraday["Close"].iloc[-1]
        if "AdjClose" in d_rt.columns:
            d_rt.at[last_day, "AdjClose"] = today_intraday["Close"].iloc[-1]
    return d_rt
def calc_rate_of_change(series: pd.Series, periods: int):
    if series is None or series.dropna().empty:
        return None
    clean = series.dropna()
    if len(clean) <= periods:
        return None
    start = clean.iloc[-periods - 1]
    end = clean.iloc[-1]
    if start == 0:
        return None
    return round((float(end) / float(start) - 1) * 100, 2)
def calc_regression_slope(series: pd.Series, lookback: int = 20):
    if series is None:
        return None
    clean = series.dropna()
    if len(clean) < lookback:
        return None
    window = clean.iloc[-lookback:]
    x = np.arange(len(window))
    slope = np.polyfit(x, window, 1)[0]
    return round(float(slope), 6)
def classify_macro_trend(close, ma20, roc5, slope):
    if close is None or ma20 is None:
        return "unknown"
    roc = roc5 if roc5 is not None else 0.0
    slope_val = slope if slope is not None else 0.0
    if close > ma20 and roc >= 0 and slope_val >= 0:
        return "bullish"
    if close < ma20 and roc <= 0 and slope_val <= 0:
        return "bearish"
    return "mixed"
def compute_trend_alert(distance_pct, roc5):
    if distance_pct is None or roc5 is None:
        return "unknown"
    if distance_pct > 0.75 and roc5 > 0.5:
        return "strong_up"
    if distance_pct < -0.75 and roc5 < -0.5:
        return "strong_down"
    return "stable"
def safe_value(series: pd.Series, key: str):
    if series is None or key not in series:
        return None
    val = series[key]
    if pd.isna(val):
        return None
    return float(val)
def derive_asset_states(latest: pd.Series):
    out = {}
    close = safe_value(latest, "Close")
    ma20 = safe_value(latest, "SMA_20")
    ma50 = safe_value(latest, "SMA_50")
    ma200 = safe_value(latest, "SMA_200")
    adx = safe_value(latest, "ADX_14")
    rsi14 = safe_value(latest, "RSI_14")
    rsi7 = safe_value(latest, "RSI_7")
    macd_line = safe_value(latest, "MACD_12_26_9")
    macd_signal = safe_value(latest, "MACDs_12_26_9")
    macd_hist = safe_value(latest, "MACDh_12_26_9")
    volume = safe_value(latest, "Volume")
    vma20 = safe_value(latest, "VMA_20")
    obv = safe_value(latest, "OBV")
    obv_ma20 = safe_value(latest, "OBV_MA20")
    kc_upper = safe_value(latest, "KC_upper")
    kc_lower = safe_value(latest, "KC_lower")
    atr = safe_value(latest, "ATRr_14")
    high = safe_value(latest, "High")
    prev20_high = safe_value(latest, "prev20_high")
    prev20_low = safe_value(latest, "prev20_low")
    if ma20 and close:
        out["price_vs_ma20_pct"] = round((close / ma20 - 1) * 100, 2)
    if ma50 and close:
        out["price_vs_ma50_pct"] = round((close / ma50 - 1) * 100, 2)
    if ma200 and close:
        out["price_vs_ma200_pct"] = round((close / ma200 - 1) * 100, 2)
    if adx is not None:
        if adx >= 22:
            out["regime_tag"] = "trend"
        elif adx < 18:
            out["regime_tag"] = "range"
        else:
            out["regime_tag"] = "transition"
    if volume and vma20 and vma20 != 0:
        ratio = volume / vma20
        out["volume_vs_vma20"] = round(ratio, 3)
        out["volume_confirmation"] = ratio >= 1.5
    if obv is not None and obv_ma20:
        diff = obv - obv_ma20
        out["obv_vs_ma20_diff"] = round(diff, 1)
        out["obv_confirmation"] = diff >= 0
    rsibase = rsi14 if rsi14 is not None else rsi7
    if rsibase is not None:
        if rsibase >= 70:
            out["rsi_state"] = "overbought"
        elif rsibase <= 30:
            out["rsi_state"] = "oversold"
        else:
            out["rsi_state"] = "neutral"
    if macd_line is not None and macd_signal is not None:
        if macd_line > macd_signal:
            out["macd_state"] = "bullish"
        elif macd_line < macd_signal:
            out["macd_state"] = "bearish"
        else:
            out["macd_state"] = "flat"
    if macd_hist is not None:
        out["macd_momentum"] = round(macd_hist, 4)
    if close is not None and kc_upper is not None and kc_lower is not None:
        if close >= kc_upper:
            out["band_position"] = "upper_break"
        elif close <= kc_lower:
            out["band_position"] = "lower_break"
        else:
            out["band_position"] = "inside_band"
    if close is not None and prev20_high is not None and prev20_low is not None:
        if close > prev20_high:
            out["breakout_state"] = "new_high"
        elif close < prev20_low:
            out["breakout_state"] = "new_low"
        else:
            out["breakout_state"] = "range"
    if high is not None and atr is not None:
        out["atr_trailing_stop"] = round(high - 2.5 * atr, 4)
    trend_score = 0
    if close and ma20 and close > ma20:
        trend_score += 1
    else:
        trend_score -= 1
    if close and ma50 and close > ma50:
        trend_score += 1
    else:
        trend_score -= 1
    if close and ma200 and close > ma200:
        trend_score += 1
    else:
        trend_score -= 1
    if macd_line is not None and macd_signal is not None:
        trend_score += 1 if macd_line >= macd_signal else -1
    if out.get("rsi_state") == "overbought":
        trend_score -= 1
    elif out.get("rsi_state") == "oversold":
        trend_score += 1
    out["trend_bias_score"] = trend_score
    if trend_score >= 2:
        out["trend_bias_label"] = "bullish"
    elif trend_score <= -2:
        out["trend_bias_label"] = "bearish"
    else:
        out["trend_bias_label"] = "neutral"
    if ma20 and ma50 and ma200:
        out["ma_stack_alignment"] = ma20 >= ma50 >= ma200
    return out
def evaluate_filter_summary(asset_daily: dict, macro_ctx: dict, vol_ctx: dict) -> dict:
    summary = {
        "macro": {},
        "volatility": {},
        "trend": {},
        "volume": {},
        "ensemble": {},
    }
    ust_distance = macro_ctx.get("UST10Y_ma_distance_pct")
    dxy_alert = macro_ctx.get("DXY_trend_alert")
    ust_condition = ust_distance is not None and ust_distance <= 0
    dxy_condition = dxy_alert != "strong_up"
    macro_pass = bool(ust_condition and dxy_condition)
    summary["macro"] = {
        "yield_below_ma": ust_condition,
        "dxy_not_breakout": dxy_condition,
        "pass": macro_pass,
    }
    vix_alert = vol_ctx.get("VIX_trend_alert")
    vix_distance = vol_ctx.get("VIX_ma_distance_pct")
    vix_change = vol_ctx.get("VIX_roc_5d_pct")
    vol_pass = vix_alert != "strong_up" and (vix_distance is None or vix_distance <= 15) and (vix_change is None or vix_change <= 15)
    summary["volatility"] = {
        "vix_alert": vix_alert,
        "vix_distance_pct": vix_distance,
        "vix_roc_5d_pct": vix_change,
        "pass": vol_pass,
    }
    trend_bias = asset_daily.get("trend_bias_label")
    ma_alignment = asset_daily.get("ma_stack_alignment", False)
    regime = asset_daily.get("regime_tag")
    trend_pass = trend_bias == "bullish" and ma_alignment and regime != "range"
    summary["trend"] = {
        "trend_bias": trend_bias,
        "ma_alignment": ma_alignment,
        "regime": regime,
        "pass": trend_pass,
    }
    volume_pass = bool(asset_daily.get("volume_confirmation")) or bool(asset_daily.get("obv_confirmation"))
    summary["volume"] = {
        "volume_confirmation": bool(asset_daily.get("volume_confirmation")),
        "obv_confirmation": bool(asset_daily.get("obv_confirmation")),
        "pass": volume_pass,
    }
    macd_state = asset_daily.get("macd_state")
    rsi_state = asset_daily.get("rsi_state")
    ensemble_pass = macd_state == "bullish" and rsi_state != "overbought"
    summary["ensemble"] = {
        "macd_state": macd_state,
        "rsi_state": rsi_state,
        "pass": ensemble_pass,
    }
    blockers = []
    if not macro_pass:
        blockers.append("macro")
    if not vol_pass:
        blockers.append("volatility")
    if not trend_pass:
        blockers.append("trend")
    if not volume_pass:
        blockers.append("volume")
    if not ensemble_pass:
        blockers.append("ensemble")
    summary["blocking_reasons"] = blockers
    summary["final_decision"] = "pass" if not blockers else "block"
    summary["score"] = sum(1 for flag in (macro_pass, vol_pass, trend_pass, volume_pass, ensemble_pass) if flag)
    return summary
def main():
    print("===== Alpha-Core OS Data Build Start (v6.0 Enhanced) =====")
    result = json.loads('{"as_of": "", "data_source": {}, "macro": {}, "vol": {}, "assets": {}, "risk": {}, "filters": {}, "sentiment": {}, "market_sentiment": {}, "meta": {}}')
    result["as_of"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # --- Macro & Volatility Data ---
    print("\n--- Processing Macro & Volatility Data ---")
    for name, sym in {**MACRO_SYMBOLS, **VOL_SYMBOL}.items():
        data_frames, src = providers.fetch_all_data(sym)
        if data_frames is None:
            continue
        d = data_frames.get('D')
        if d is not None and not d.empty:
            d = calculate_indicators(d)
            result["data_source"][name] = src
            target_obj = result["macro"] if name in MACRO_SYMBOLS else result["vol"]
            close_val = get_last_value(d["Close"])
            ma20_val = get_last_value(d.get("SMA_20"))
            roc5 = calc_rate_of_change(d["Close"], 5)
            roc10 = calc_rate_of_change(d["Close"], 10)
            slope = calc_regression_slope(d.get("SMA_20"))
            distance_pct = None
            if ma20_val not in (None, 0) and close_val is not None:
                distance_pct = round((close_val / ma20_val - 1) * 100, 2)
            target_obj[f"{name}_close"] = close_val
            target_obj[f"{name}_MA20"] = ma20_val
            target_obj[f"{name}_BB20_2U"] = get_last_value(d.get("BBU_20_2.0"))
            if len(d["Close"].dropna()) > 5:
                target_obj[f"{name}_5d_chg_pct"] = round((d["Close"].iloc[-1] / d["Close"].iloc[-6] - 1) * 100, 2)
            target_obj[f"{name}_roc_5d_pct"] = roc5
            target_obj[f"{name}_roc_10d_pct"] = roc10
            target_obj[f"{name}_ma20_slope"] = slope
            target_obj[f"{name}_ma_distance_pct"] = distance_pct
            target_obj[f"{name}_trend_state"] = classify_macro_trend(close_val, ma20_val, roc5, slope)
            target_obj[f"{name}_trend_alert"] = compute_trend_alert(distance_pct, roc5)
    # --- Asset Data ---
    print("\n--- Processing Asset Data ---")
    for sym in ASSET_SYMBOLS:
        data_frames, src = providers.fetch_all_data(sym)
        if data_frames is None:
            continue
        result["data_source"][sym] = src
        result["assets"][sym] = {}
        df_d_rt_base = create_daily_rt(data_frames.get('D'), data_frames.get('5m'))
        if not df_d_rt_base.empty:
            data_frames['D_RT'] = calculate_indicators(df_d_rt_base.copy())
        for period, df in data_frames.items():
            if df is None or df.empty:
                continue
            df_processed = calculate_indicators(df.copy()) if period != 'D_RT' else df
            latest = df_processed.iloc[-1]
            data_dict = {}
            field_map = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "MA9": "SMA_9",
                "MA20": "SMA_20",
                "MA50": "SMA_50",
                "MA100": "SMA_100",
                "MA200": "SMA_200",
                "MA300": "SMA_300",
                "EMA20": "EMA_20",
                "ATR14": "ATRr_14",
                "RSI7": "RSI_7",
                "RSI14": "RSI_14",
                "STOCH_K": "STOCHk_14_3_3",
                "STOCH_D": "STOCHd_14_3_3",
                "MACD_line": "MACD_12_26_9",
                "MACD_signal": "MACDs_12_26_9",
                "MACD_hist": "MACDh_12_26_9",
                "ADX14": "ADX_14",
                "VMA20": "VMA_20",
                "VMA50": "VMA_50",
                "OBV": "OBV",
                "OBV_MA20": "OBV_MA20",
                "KC_upper": "KC_upper",
                "KC_lower": "KC_lower",
                "prev20_high": "prev20_high",
                "prev20_low": "prev20_low"
            }
            for key, col in field_map.items():
                if col in latest and pd.notna(latest[col]):
                    data_dict[key] = round(float(latest[col]), 4)
            data_dict.update(derive_asset_states(latest))
            if period == 'D':
                filter_summary = evaluate_filter_summary(data_dict, result["macro"], result["vol"])
                data_dict["filter_summary"] = filter_summary
                result["filters"][sym] = filter_summary
            result["assets"][sym][period] = data_dict
    # --- Finalization ---
    print("\n--- Populating Risk, Sentiment, Meta Data ---")

    # --- Market Sentiment Analysis ---
    market_analysis = sentiment_provider.analyze_market_data({
        "macro": result["macro"],
        "vol": result["vol"],
    })
    result["market_sentiment"] = market_analysis
    result["risk"] = {
        "account_krw": 11000000,
        "krw_usd_fx": 1350,
        "risk_per_trade_pct": 0.8,
        "risk_budget_daily_pct": 2.0,
        "slippage_bps": 2,
        "fee_bps": 2
    }
    sentiment_snapshot = sentiment_provider.collect_sentiment(ASSET_SYMBOLS)
    if sentiment_snapshot.get("status") != "ok":
        print(" -> Sentiment providers unavailable or partial; defaults applied.")
    result["sentiment"] = sentiment_snapshot
    raw_trades = meta_provider.load_trade_history(TRADE_HISTORY_DEFAULT)
    trade_list = []
    long_avg = None
    long_std = None
    if isinstance(raw_trades, dict):
        trade_list = raw_trades.get("trades") or raw_trades.get("last20_trades_pnl") or []
        long_avg = raw_trades.get("long_term_avg")
        long_std = raw_trades.get("long_term_std")
    meta_stats = meta_provider.compute_trade_stats(trade_list, long_avg, long_std)
    if isinstance(raw_trades, dict):
        if raw_trades.get("updated_at"):
            meta_stats.setdefault("updated_at", raw_trades["updated_at"])
        if raw_trades.get("source"):
            meta_stats.setdefault("source", raw_trades["source"])
    if not trade_list and "performance_flag" not in meta_stats:
        meta_stats["performance_flag"] = "no_trade_history"
        print(" -> Trade history not provided; performance diagnostics limited.")
    result["meta"] = meta_stats
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=lambda x: None if isinstance(x, (float, np.floating)) and np.isnan(x) else x)
    print(f"===== Data Build Finished! File saved to: {OUTPUT_JSON_PATH} =====")

if __name__ == "__main__":
    main()
