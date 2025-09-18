"""Alpha Core OS의 데이터 분석 및 백테스팅 유틸리티"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from datetime import datetime, timedelta

class MarketAnalyzer:
    def __init__(self, data_path: str = None):
        self.data_path = data_path or str(Path(__file__).parent / "data" / "alpha_core_data.json")
        self.data = self._load_data()
        
    def _load_data(self) -> dict:
        """JSON 데이터 파일을 로드합니다."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_market_sentiment(self) -> dict:
        """시장 감성 분석 데이터를 반환합니다."""
        return self.data.get("market_sentiment", {})
    
    def get_asset_data(self, symbol: str, timeframe: str = 'D') -> dict:
        """특정 자산의 데이터를 반환합니다."""
        return self.data.get("assets", {}).get(symbol, {}).get(timeframe, {})
    
    def get_macro_indicators(self) -> dict:
        """거시경제 지표 데이터를 반환합니다."""
        return self.data.get("macro", {})
    
    def get_volatility_data(self) -> dict:
        """변동성 관련 데이터를 반환합니다."""
        return self.data.get("vol", {})
    
    def get_risk_filters(self, symbol: str = None) -> Union[dict, Dict[str, dict]]:
        """리스크 필터 데이터를 반환합니다."""
        filters = self.data.get("filters", {})
        return filters.get(symbol, {}) if symbol else filters
    
    def analyze_trend_strength(self, symbol: str) -> Tuple[float, str]:
        """자산의 추세 강도를 분석합니다."""
        asset_data = self.get_asset_data(symbol)
        trend_score = asset_data.get("trend_bias_score", 0)
        
        strength = "강함" if abs(trend_score) >= 3 else "중간" if abs(trend_score) >= 2 else "약함"
        direction = "상승" if trend_score > 0 else "하락" if trend_score < 0 else "중립"
        
        return trend_score, f"{direction}({strength})"
    
    def calculate_risk_score(self) -> float:
        """전반적인 시장 리스크 점수를 계산합니다."""
        vix_data = self.get_volatility_data()
        macro_data = self.get_macro_indicators()
        
        risk_score = 0.0
        
        # VIX 기반 리스크
        vix_close = vix_data.get("VIX_close", 0)
        if vix_close > 25:
            risk_score += 2
        elif vix_close > 20:
            risk_score += 1
            
        # 금리 변화 기반 리스크
        ust_chg = macro_data.get("UST10Y_5d_chg_pct", 0)
        if abs(ust_chg) > 5:
            risk_score += 1
            
        # 달러 강도 기반 리스크
        dxy_alert = macro_data.get("DXY_trend_alert")
        if dxy_alert == "strong_up":
            risk_score += 1
            
        return risk_score
    
    def get_trading_signals(self) -> List[dict]:
        """현재 거래 신호들을 분석합니다."""
        signals = []
        sentiment = self.get_market_sentiment()
        
        # 매크로 신호
        for alert in sentiment.get("signals", {}).get("macro_alerts", []):
            signals.append({"type": "macro", "signal": alert})
            
        # 변동성 신호
        for alert in sentiment.get("signals", {}).get("volatility_alerts", []):
            signals.append({"type": "volatility", "signal": alert})
            
        # 추세 변화 신호
        for alert in sentiment.get("signals", {}).get("trend_changes", []):
            signals.append({"type": "trend", "signal": alert})
            
        return signals

class BacktestEngine:
    def __init__(self, analyzer: MarketAnalyzer):
        self.analyzer = analyzer
        
    def simulate_strategy(self, 
                        symbol: str, 
                        initial_capital: float = 10000.0,
                        risk_per_trade: float = 0.02,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> dict:
        """간단한 전략 백테스트를 실행합니다."""
        # 여기에 백테스트 로직 구현
        # 현재는 더미 결과를 반환
        return {
            "symbol": symbol,
            "initial_capital": initial_capital,
            "final_capital": initial_capital * 1.15,
            "total_trades": 24,
            "win_rate": 0.65,
            "profit_factor": 1.8,
            "max_drawdown": 0.12,
            "sharpe_ratio": 1.2
        }
        
    def analyze_performance(self, backtest_results: dict) -> dict:
        """백테스트 결과를 분석합니다."""
        return {
            "summary": {
                "total_return": (backtest_results["final_capital"] / backtest_results["initial_capital"] - 1) * 100,
                "win_rate": backtest_results["win_rate"] * 100,
                "risk_adjusted_return": backtest_results["sharpe_ratio"]
            },
            "risk_metrics": {
                "max_drawdown": backtest_results["max_drawdown"] * 100,
                "profit_factor": backtest_results["profit_factor"]
            }
        }