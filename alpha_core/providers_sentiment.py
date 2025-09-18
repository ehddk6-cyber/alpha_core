"""Sentiment data providers for Alpha-Core."""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import requests


def _fetch_finnhub_sentiment(symbol: str) -> Optional[dict]:
    token = os.getenv("FINNHUB_TOKEN")
    if not token:
        return None
    url = "https://finnhub.io/api/v1/news-sentiment"
    params = {"symbol": symbol, "token": token}
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not data:
            return None
        data["_source"] = "finnhub"
        data["_fetched_at"] = datetime.utcnow().isoformat()
        return data
    except Exception as exc:  # pragma: no cover - graceful fallback
        print(f"Finnhub sentiment fetch failed for {symbol}: {exc}")
        return None


def collect_sentiment(symbols: List[str]) -> Dict[str, Any]:
    """Collect aggregated sentiment information for the provided symbols."""
    results: Dict[str, Any] = {
        "sentiment_score": 0.0,
        "headline_count": 0,
        "confidence": "low",
        "status": "not_available",
        "by_symbol": {},
        "sources": [],
    }

    scores: List[float] = []
    headlines = 0

    for sym in symbols:
        payload = _fetch_finnhub_sentiment(sym)
        if payload is None:
            results["by_symbol"][sym] = {"status": "missing"}
            continue

        results["status"] = "partial"
        results["sources"].append(payload.get("_source"))
        sym_entry = {
            "status": "ok",
            "fetched_at": payload.get("_fetched_at"),
            "news_score": payload.get("newsScore"),
            "bullish_percent": payload.get("bullishPercent"),
            "bearish_percent": payload.get("bearishPercent"),
            "company_news_score": payload.get("companyNewsScore"),
            "sector_avg_news_score": payload.get("sectorAverageNewsScore"),
            "total_news": payload.get("totalNews"),
        }
        results["by_symbol"][sym] = sym_entry

        if isinstance(payload.get("newsScore"), (int, float)):
            scores.append(float(payload["newsScore"]))
        if isinstance(payload.get("totalNews"), (int, float)):
            headlines += int(payload["totalNews"])

    if scores:
        results["sentiment_score"] = round(sum(scores) / len(scores), 4)
        results["headline_count"] = headlines
        results["confidence"] = "high" if len(scores) == len(symbols) else "medium"
        results["status"] = "ok"
    else:
        results.setdefault("detail", "Sentiment providers unavailable; defaults applied.")

    results.setdefault("updated_at", datetime.utcnow().isoformat())
    return results


def analyze_macro_sentiment(macro_data: Dict[str, Any]) -> Dict[str, Any]:
    sentiment = {
        "UST10Y": {"score": 0, "signals": []},
        "DXY": {"score": 0, "signals": []},
    }

    ust = sentiment["UST10Y"]
    if macro_data.get("UST10Y_trend_state") == "bearish":
        ust["score"] -= 1
        ust["signals"].append("금리 하락 추세")
    if (macro_data.get("UST10Y_ma_distance_pct") or 0) < -3:
        ust["score"] -= 1
        ust["signals"].append("MA20 대비 이탈 확대")
    if (macro_data.get("UST10Y_roc_10d_pct") or 0) < -3:
        ust["score"] -= 1
        ust["signals"].append("10일 변화율 급락")

    dxy = sentiment["DXY"]
    if macro_data.get("DXY_trend_state") == "bearish":
        dxy["score"] += 1
        dxy["signals"].append("달러 약세")
    if macro_data.get("DXY_trend_alert") == "strong_down":
        dxy["score"] += 2
        dxy["signals"].append("달러 강한 하락 신호")

    return sentiment


def analyze_vol_sentiment(vol_data: Dict[str, Any]) -> Dict[str, Any]:
    sentiment = {"VIX": {"score": 0, "signals": []}}

    vix = sentiment["VIX"]
    if vol_data.get("VIX_trend_alert") == "strong_up":
        vix["score"] -= 2
        vix["signals"].append("VIX 급등 경고")
    if (vol_data.get("VIX_ma_distance_pct") or 0) > 5:
        vix["score"] -= 1
        vix["signals"].append("VIX MA20 대비 과열")

    return sentiment


def analyze_market_data(data: Dict[str, Any]) -> Dict[str, Any]:
    macro_sentiment = analyze_macro_sentiment(data.get("macro", {}))
    vol_sentiment = analyze_vol_sentiment(data.get("vol", {}))

    total_score = sum(item["score"] for section in (macro_sentiment, vol_sentiment) for item in section.values())

    analysis = {
        "overall_sentiment": "neutral",
        "score": total_score,
        "macro_sentiment": macro_sentiment,
        "vol_sentiment": vol_sentiment,
        "key_points": [],
        "signals": {
            "macro_alerts": [],
            "volatility_alerts": [],
            "trend_changes": [],
        },
    }

    if total_score >= 3:
        analysis["overall_sentiment"] = "very_bullish"
    elif total_score >= 1:
        analysis["overall_sentiment"] = "bullish"
    elif total_score <= -3:
        analysis["overall_sentiment"] = "very_bearish"
    elif total_score <= -1:
        analysis["overall_sentiment"] = "bearish"

    for section in (macro_sentiment, vol_sentiment):
        for indicator in section.values():
            analysis["key_points"].extend(indicator["signals"])

    macro = data.get("macro", {})
    for indicator in ["UST10Y", "DXY"]:
        alert = macro.get(f"{indicator}_trend_alert")
        if alert == "strong_down":
            analysis["signals"]["macro_alerts"].append(f"{indicator} 강한 하락")
        elif alert == "strong_up":
            analysis["signals"]["macro_alerts"].append(f"{indicator} 강한 상승")

    vol = data.get("vol", {})
    if vol.get("VIX_trend_alert") == "strong_up":
        analysis["signals"]["volatility_alerts"].append("VIX 급등")
    if (vol.get("VIX_ma_distance_pct") or 0) > 10:
        analysis["signals"]["volatility_alerts"].append("VIX MA20 대비 10% 이상 이탈")

    for section_name in ["macro", "vol"]:
        for key, value in data.get(section_name, {}).items():
            if key.endswith("_trend_state") and value == "mixed":
                indicator = key.split("_")[0]
                analysis["signals"]["trend_changes"].append(f"{indicator} 추세 전환 위험")

    analysis["updated_at"] = datetime.utcnow().isoformat()
    return analysis
