"""Meta-data helpers for Alpha-Core (trade history and performance)."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_trade_history(default_path: Path) -> Dict[str, any]:
    """Load trade history from environment override or default path."""
    env_path = os.getenv("ALPHA_CORE_TRADES_PATH")
    candidates: List[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(default_path)

    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as exc:  # pragma: no cover - graceful fallback
            print(f"Trade history load failed for {path}: {exc}")
    return {}


def compute_trade_stats(trades: List[float], long_term_avg: Optional[float] = None,
                         long_term_std: Optional[float] = None) -> Dict[str, any]:
    from statistics import mean, pstdev

    stats: Dict[str, any] = {}
    clean = [float(x) for x in trades if isinstance(x, (int, float))]
    if clean:
        stats["last20_trades_pnl"] = clean[-20:]
        stats["recent_avg"] = round(mean(clean[-20:]), 4)
        stats["recent_std"] = round(pstdev(clean[-20:]), 4) if len(clean[-20:]) > 1 else 0.0
    else:
        stats["last20_trades_pnl"] = []

    if long_term_avg is not None:
        stats["long_term_avg"] = _safe_float(long_term_avg)
    if long_term_std is not None:
        stats["long_term_std"] = _safe_float(long_term_std)

    if stats.get("last20_trades_pnl") and stats.get("long_term_avg") is not None and stats.get("long_term_std") not in (None, 0):
        z = (stats["recent_avg"] - stats["long_term_avg"]) / stats["long_term_std"]
        stats["performance_deviation_z"] = round(z, 3)
        stats["performance_flag"] = "underperforming" if z < -2 else ("outperforming" if z > 2 else "within_expected")
    else:
        stats.setdefault("performance_flag", "insufficient_data")

    return stats
