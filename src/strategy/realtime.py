from __future__ import annotations

import pandas as pd

from src.data.fetch import get_intraday_stock_data
from src.strategy.learning import apply_learning_adjustment, LearningAdjustment
from src.strategy.regime import classify_market_regime
from src.strategy.universe import get_universe


def scan_intraday_market(
    market: str,
    universe: list[dict[str, str]] | None = None,
    interval: str = "5m",
    min_score: int = 55,
    learning_adjustments: dict[tuple[str, str, str], LearningAdjustment] | None = None,
) -> pd.DataFrame:
    candidates = universe if universe is not None else get_universe(market)
    rows: list[dict[str, object]] = []
    regime = classify_market_regime(market)

    for item in candidates:
        data = get_intraday_stock_data(item["ticker"], period="5d", interval=interval)
        if data.empty or len(data) < 25:
            continue

        latest = data.iloc[-1]
        prev = data.iloc[-2]
        score = 40
        reasons: list[str] = []

        if latest["volume_ratio"] >= 2.0:
            score += 20
            reasons.append("지금 거래량이 평소보다 확실히 많습니다.")
        elif latest["volume_ratio"] >= 1.3:
            score += 10
            reasons.append("분봉 거래량이 평소보다 강합니다.")

        if latest["short_return_pct"] >= 1.2:
            score += 18
            reasons.append("방금 올라가는 힘이 강합니다.")
        elif latest["short_return_pct"] >= 0.5:
            score += 8
            reasons.append("짧은 흐름이 위로 살아 있습니다.")

        recent_high = float(data["session_high_20"].iloc[-2])
        if float(latest["Close"]) >= recent_high:
            score += 18
            reasons.append("방금 직전 고점을 넘었습니다.")

        if float(latest["Close"]) > float(latest["vwap_proxy"]):
            score += 12
            reasons.append("장중 평균가 위라 흐름이 강한 편입니다.")

        if float(latest["Close"]) > float(prev["Close"]):
            score += 6

        score = max(0, min(100, int(score + regime.adjustment)))
        if score < min_score:
            continue

        if score >= 82:
            setup = "장중돌파"
        elif score >= 70:
            setup = "급등감시"
        else:
            setup = "초기강세"

        score, learning_delta, learning_note = apply_learning_adjustment(
            base_score=score,
            scan_type="realtime_scan",
            market=market,
            setup=setup,
            adjustments=learning_adjustments,
        )

        if score < min_score:
            continue

        rows.append(
            {
                "ticker": item["ticker"],
                "name": item["name"],
                "setup": setup,
                "score": score,
                "current_price": round(float(latest["Close"]), 2),
                "volume_ratio": round(float(latest["volume_ratio"]), 2),
                "short_return_pct": round(float(latest["short_return_pct"]), 2),
                "above_vwap": bool(float(latest["Close"]) > float(latest["vwap_proxy"])),
                "regime": regime.regime,
                "regime_delta": regime.adjustment,
                "learning_delta": learning_delta,
                "reason": " / ".join(([regime.note] if regime.note else []) + ([learning_note] if learning_note else []) + reasons[:4]),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "name",
                "setup",
                "score",
                "current_price",
                "volume_ratio",
                "short_return_pct",
                "above_vwap",
                "regime",
                "regime_delta",
                "learning_delta",
                "reason",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        by=["score", "volume_ratio", "short_return_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
