from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.fetch import get_stock_data


MARKET_BENCHMARKS = {
    "US": {"ticker": "SPY", "name": "S&P 500"},
    "KR": {"ticker": "069500.KS", "name": "KODEX 200"},
}


@dataclass(slots=True)
class MarketRegime:
    market: str
    benchmark: str
    regime: str
    adjustment: int
    note: str
    trend_strength: float


def classify_market_regime(market: str) -> MarketRegime:
    benchmark_info = MARKET_BENCHMARKS.get(market, MARKET_BENCHMARKS["US"])
    benchmark = benchmark_info["ticker"]
    data = get_stock_data(benchmark)

    if data.empty:
        return MarketRegime(
            market=market,
            benchmark=benchmark,
            regime="중립",
            adjustment=0,
            note="장세 데이터를 못 불러와 중립으로 봅니다.",
            trend_strength=0.0,
        )

    latest = data.iloc[-1]
    close = float(latest["Close"])
    ma20 = float(latest["ma20"])
    ma60 = float(latest["ma60"])
    return_20d = float(latest["return_20d"])
    macd_diff = float(latest["macd_diff"])

    trend_strength = 0.0
    if close > ma20:
        trend_strength += 1.0
    if ma20 > ma60:
        trend_strength += 1.0
    if return_20d > 3:
        trend_strength += 1.0
    if macd_diff > 0:
        trend_strength += 0.5

    if close < ma20:
        trend_strength -= 1.0
    if ma20 < ma60:
        trend_strength -= 1.0
    if return_20d < -3:
        trend_strength -= 1.0
    if macd_diff < 0:
        trend_strength -= 0.5

    if trend_strength >= 2.0:
        return MarketRegime(
            market=market,
            benchmark=benchmark,
            regime="상승장",
            adjustment=5,
            note="시장 전체 흐름이 좋아 추세 매매에 유리합니다.",
            trend_strength=round(trend_strength, 2),
        )
    if trend_strength <= -2.0:
        return MarketRegime(
            market=market,
            benchmark=benchmark,
            regime="약세장",
            adjustment=-6,
            note="시장 전체 흐름이 약해 추격 매매는 보수적으로 봐야 합니다.",
            trend_strength=round(trend_strength, 2),
        )

    return MarketRegime(
        market=market,
        benchmark=benchmark,
        regime="횡보장",
        adjustment=-1,
        note="시장 방향이 애매해 짧고 가볍게 보는 편이 낫습니다.",
        trend_strength=round(trend_strength, 2),
    )
