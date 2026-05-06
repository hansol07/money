from __future__ import annotations

import pandas as pd

from src.data.fetch import get_intraday_stock_data, get_stock_event_summary, get_stock_news_summary, is_recent_price_data
from src.strategy.learning import apply_context_adjustment, apply_learning_adjustment, ContextAdjustment, LearningAdjustment
from src.strategy.regime import classify_market_regime
from src.strategy.universe import get_universe


def scan_intraday_market(
    market: str,
    universe: list[dict[str, str]] | None = None,
    interval: str = "5m",
    min_score: int = 55,
    force_refresh: bool = False,
    learning_adjustments: dict[tuple[str, str, str], LearningAdjustment] | None = None,
    event_adjustments: dict[str, ContextAdjustment] | None = None,
    news_adjustments: dict[str, ContextAdjustment] | None = None,
) -> pd.DataFrame:
    candidates = universe if universe is not None else get_universe(market)
    rows: list[dict[str, object]] = []
    regime = classify_market_regime(market)

    for item in candidates:
        try:
            data = get_intraday_stock_data(item["ticker"], period="5d", interval=interval, force_refresh=force_refresh)
            if data.empty or len(data) < 25 or not is_recent_price_data(data, max_age_days=1):
                continue

            latest = data.iloc[-1]
            prev = data.iloc[-2]
            event_summary = get_stock_event_summary(item["ticker"])
            news_summary = get_stock_news_summary(item["ticker"])
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

            event_risk = str(event_summary.get("event_risk", "") or "")
            news_bias = str(news_summary.get("news_bias", "중립") or "중립")
            news_score = int(news_summary.get("news_score", 0) or 0)

            if event_risk == "높음":
                score -= 5
                reasons.append("가까운 일정이 있어 장중 흔들림이 커질 수 있습니다.")
            elif event_risk == "중간":
                score -= 2

            if news_bias == "긍정":
                score += min(5, max(1, news_score * 2))
                reasons.append("최근 뉴스 흐름이 우호적입니다.")
            elif news_bias == "부정":
                score -= min(5, max(1, abs(news_score) * 2))
                reasons.append("최근 뉴스 흐름이 부담입니다.")

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
            score, context_delta, context_note = apply_context_adjustment(
                base_score=score,
                event_risk=event_risk,
                news_bias=news_bias,
                event_adjustments=event_adjustments,
                news_adjustments=news_adjustments,
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
                    "context_delta": context_delta,
                    "event_risk": event_risk,
                    "event_note": str(event_summary.get("event_note", "")),
                    "earnings_date": str(event_summary.get("earnings_date", "")),
                    "ex_dividend_date": str(event_summary.get("ex_dividend_date", "")),
                    "news_bias": news_bias,
                    "news_score": news_score,
                    "news_count": int(news_summary.get("news_count", 0) or 0),
                    "learning_delta": learning_delta,
                    "reason": " / ".join(
                        ([regime.note] if regime.note else [])
                        + ([context_note] if context_note else [])
                        + ([learning_note] if learning_note else [])
                        + reasons[:4]
                    ),
                }
            )
        except Exception:
            continue

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
                "context_delta",
                "event_risk",
                "event_note",
                "earnings_date",
                "ex_dividend_date",
                "news_bias",
                "news_score",
                "news_count",
                "learning_delta",
                "reason",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        by=["score", "volume_ratio", "short_return_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
