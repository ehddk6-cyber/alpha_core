"""Market analysis and backtesting utilities for Alpha-Core."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_JSON = DATA_DIR / "alpha_core_data.json"


@dataclass
class FilterDecision:
    symbol: str
    final_decision: str
    score: int
    blocking_reasons: List[str]
    macro_pass: bool
    vol_pass: bool
    trend_pass: bool
    volume_pass: bool
    ensemble_pass: bool


@dataclass
class BacktestReport:
    symbol: str
    strategy: str
    total_return_pct: float
    cagr_pct: float
    volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    trades: int
    equity_curve: pd.DataFrame


class MarketAnalyzer:
    """Helper for reading Alpha-Core JSON outputs and transforming to tabular data."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path or DEFAULT_JSON)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Alpha-Core data not found: {self.data_path}")
        self.data = self._load_json(self.data_path)

    @staticmethod
    def _load_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def get_asset_symbols(self) -> List[str]:
        return list(self.data.get("assets", {}).keys())

    def build_asset_summary(self, period: str = "D") -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for sym, payload in self.data.get("assets", {}).items():
            frame = payload.get(period) or {}
            filter_summary = frame.get("filter_summary") or self.data.get("filters", {}).get(sym, {})
            rows.append(
                {
                    "symbol": sym,
                    "close": frame.get("close"),
                    "ma20": frame.get("MA20"),
                    "ma50": frame.get("MA50"),
                    "trend_bias": frame.get("trend_bias_label"),
                    "regime": frame.get("regime_tag"),
                    "volume_conf": frame.get("volume_confirmation"),
                    "obv_conf": frame.get("obv_confirmation"),
                    "macd_state": frame.get("macd_state"),
                    "rsi_state": frame.get("rsi_state"),
                    "filter_score": (filter_summary or {}).get("score"),
                    "filter_decision": (filter_summary or {}).get("final_decision"),
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            numeric_cols = ["close", "ma20", "ma50", "filter_score"]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        return df.set_index("symbol") if not df.empty else df

    def build_macro_summary(self) -> pd.DataFrame:
        macro = self.data.get("macro", {})
        keys = sorted([k for k in macro.keys() if k.endswith("trend_state")])
        rows = []
        for key in keys:
            prefix = key.replace("_trend_state", "")
            rows.append(
                {
                    "indicator": prefix,
                    "trend_state": macro.get(f"{prefix}_trend_state"),
                    "trend_alert": macro.get(f"{prefix}_trend_alert"),
                    "distance_pct": macro.get(f"{prefix}_ma_distance_pct"),
                    "roc_5d_pct": macro.get(f"{prefix}_roc_5d_pct"),
                    "roc_10d_pct": macro.get(f"{prefix}_roc_10d_pct"),
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            df.set_index("indicator", inplace=True)
        return df

    def latest_market_sentiment(self) -> dict:
        return self.data.get("market_sentiment", {})

    def get_filter_decisions(self) -> List[FilterDecision]:
        filters = self.data.get("filters", {})
        decisions: List[FilterDecision] = []
        for sym, summary in filters.items():
            decisions.append(
                FilterDecision(
                    symbol=sym,
                    final_decision=summary.get("final_decision", "unknown"),
                    score=int(summary.get("score", 0)),
                    blocking_reasons=list(summary.get("blocking_reasons", [])),
                    macro_pass=bool(summary.get("macro", {}).get("pass")),
                    vol_pass=bool(summary.get("volatility", {}).get("pass")),
                    trend_pass=bool(summary.get("trend", {}).get("pass")),
                    volume_pass=bool(summary.get("volume", {}).get("pass")),
                    ensemble_pass=bool(summary.get("ensemble", {}).get("pass")),
                )
            )
        return decisions

    def _csv_path(self, symbol: str, interval: str = "daily") -> Path:
        suffix = "daily" if interval == "daily" else f"intraday_{interval}"
        return DATA_DIR / f"{symbol.lower()}_{suffix}_latest.csv"

    def load_price_history(self, symbol: str, interval: str = "daily") -> pd.DataFrame:
        path = self._csv_path(symbol, interval)
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        df.sort_index(inplace=True)
        df.index.name = "date"
        return df


class BacktestEngine:
    """Simple backtesting logic based on Alpha-Core historical CSVs."""

    def __init__(self, analyzer: MarketAnalyzer):
        self.analyzer = analyzer

    def simulate_sma_trend_strategy(
        self,
        symbol: str,
        short_window: int = 20,
        long_window: int = 50,
        capital: float = 10000.0,
        fee_bps: float = 2.0,
    ) -> BacktestReport:
        price_df = self.analyzer.load_price_history(symbol, interval="daily")
        if price_df.empty:
            raise ValueError(f"Historical data not available for {symbol}")

        df = price_df.copy()
        df["close"] = df["Close"].astype(float)
        df["return"] = df["close"].pct_change().fillna(0.0)
        df["sma_short"] = df["close"].rolling(short_window).mean()
        df["sma_long"] = df["close"].rolling(long_window).mean()
        df.dropna(inplace=True)

        if df.empty:
            raise ValueError(
                "Insufficient data after applying moving averages: "
                f"requires at least {long_window} rows but got {len(df)} for {symbol}"
            )

        df["position"] = np.where(df["sma_short"] > df["sma_long"], 1.0, 0.0)
        df["position_shifted"] = df["position"].shift(1).fillna(0.0)
        df["turnover"] = (df["position_shifted"] - df["position_shifted"].shift(1).fillna(0.0)).abs()
        cost = fee_bps / 10000.0

        df["strategy_return"] = df["position_shifted"] * df["return"] - df["turnover"] * cost
        df["buy_hold"] = df["return"]

        equity = (1 + df["strategy_return"]).cumprod()
        buy_hold = (1 + df["buy_hold"]).cumprod()
        df["equity_curve"] = equity
        df["buy_hold_curve"] = buy_hold

        total_return = equity.iloc[-1] - 1
        holding_period_years = max(len(df) / 252, 1e-9)
        cagr = equity.iloc[-1] ** (1 / holding_period_years) - 1 if equity.iloc[-1] > 0 else -1
        volatility = df["strategy_return"].std() * np.sqrt(252)
        sharpe = (df["strategy_return"].mean() * 252) / volatility if volatility else 0.0
        running_max = equity.cummax()
        drawdown = equity / running_max - 1
        max_dd = drawdown.min()
        df["is_entry"] = (
            (df["turnover"] > 0)
            & (
                df["position_shifted"]
                > df["position_shifted"].shift(1, fill_value=0)
            )
        )
        df["trade_id"] = df["is_entry"].cumsum()

        in_position = df["position_shifted"] > 0
        trade_returns = (
            df.loc[in_position]
            .groupby("trade_id")["strategy_return"]
            .apply(lambda returns: (1 + returns).prod() - 1)
        )
        trade_returns = trade_returns[trade_returns.index > 0]

        wins = (trade_returns > 0).sum()
        total_trades = int(trade_returns.count())
        win_rate = wins / total_trades if total_trades else 0.0

        equity_curve = df[["equity_curve", "buy_hold_curve"]].copy()
        equity_curve.columns = ["strategy", "buy_hold"]
        equity_curve *= capital

        return BacktestReport(
            symbol=symbol,
            strategy=f"SMA{short_window}/{long_window}",
            total_return_pct=total_return * 100,
            cagr_pct=cagr * 100,
            volatility_pct=volatility * 100,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd * 100,
            win_rate_pct=win_rate * 100,
            trades=total_trades,
            equity_curve=equity_curve,
        )


def load_dashboard_payload(analyzer: MarketAnalyzer) -> Dict[str, pd.DataFrame]:
    payload = {
        "asset_summary": analyzer.build_asset_summary(),
        "macro_summary": analyzer.build_macro_summary(),
    }
    sentiment = analyzer.latest_market_sentiment()
    if sentiment:
        payload["market_sentiment"] = pd.json_normalize(sentiment, sep=".")
    return payload


if __name__ == "__main__":
    analyzer = MarketAnalyzer()
    summary = analyzer.build_asset_summary()
    print("Asset Summary:\n", summary)

    engine = BacktestEngine(analyzer)
    for symbol in analyzer.get_asset_symbols():
        try:
            report = engine.simulate_sma_trend_strategy(symbol)
        except ValueError as exc:
            print(f"[WARN] {exc}")
            continue
        print(
            f"\n{symbol} Backtest ({report.strategy})\n"
            f"Total Return: {report.total_return_pct:.2f}%\n"
            f"CAGR: {report.cagr_pct:.2f}% | Sharpe: {report.sharpe_ratio:.2f}\n"
            f"Max Drawdown: {report.max_drawdown_pct:.2f}% | "
            f"Trades: {report.trades} | Win Rate: {report.win_rate_pct:.2f}%"
        )
