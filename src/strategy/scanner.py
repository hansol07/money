from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.fetch import get_stock_data
from src.portfolio.models import PositionInput
from src.strategy.learning import apply_learning_adjustment, LearningAdjustment
from src.strategy.regime import classify_market_regime
from src.strategy.recommendation import RecommendationResult, analyze_position
from src.strategy.universe import get_universe


@dataclass(slots=True)
class CandidateProfile:
    trend_score: int
    momentum_score: int
    volume_score: int
    breakout_score: int
    setup: str
    reasons: list[str]


def _build_candidate_profile(data: pd.DataFrame) -> CandidateProfile:
    latest = data.iloc[-1]
    trend_score = 0
    momentum_score = 0
    volume_score = 0
    breakout_score = 0
    reasons: list[str] = []

    if latest["Close"] > latest["ma20"] > latest["ma60"]:
        trend_score += 10
        reasons.append("추세가 위쪽이라 흐름이 좋습니다.")
    elif latest["Close"] > latest["ma20"]:
        trend_score += 5
        reasons.append("단기 흐름은 아직 괜찮습니다.")
    if 45 <= latest["rsi"] <= 68:
        momentum_score += 8
        reasons.append("과열 전이라 한 번 더 갈 힘이 남아 있습니다.")
    if latest["macd_diff"] > 0:
        momentum_score += 7
        reasons.append("상승 탄력이 아직 남아 있습니다.")
    if latest["volume_ratio"] >= 1.3:
        volume_score += 10
        reasons.append("거래량이 붙어서 시장 관심이 높습니다.")
    elif latest["volume_ratio"] >= 1.05:
        volume_score += 5
        reasons.append("거래량이 조금씩 늘고 있습니다.")
    if latest["return_20d"] >= 8:
        momentum_score += 8
        reasons.append("최근 흐름이 다른 종목보다 강합니다.")
    if latest["Close"] >= data["Close"].tail(60).max() * 0.97:
        breakout_score += 12
        reasons.append("전고점 근처라 돌파를 볼 수 있습니다.")
    elif latest["Close"] >= data["Close"].tail(20).max() * 0.98:
        breakout_score += 6
        reasons.append("단기 고점 근처라 한 번 더 뛸 수 있습니다.")

    setup = "눌림목"
    if breakout_score >= 10 and volume_score >= 8:
        setup = "돌파임박"
    elif momentum_score >= 12 and trend_score >= 8:
        setup = "추세지속"
    elif volume_score >= 10 and latest["return_20d"] > 0:
        setup = "수급집중"

    return CandidateProfile(
        trend_score=trend_score,
        momentum_score=momentum_score,
        volume_score=volume_score,
        breakout_score=breakout_score,
        setup=setup,
        reasons=reasons,
    )


def _build_trade_plan(data: pd.DataFrame) -> tuple[float, float, float]:
    latest = data.iloc[-1]
    recent_high = float(data["High"].tail(20).max())
    recent_low = float(data["Low"].tail(20).min())
    entry_price = max(float(latest["Close"]), recent_high * 0.995)
    stop_loss = min(float(latest["ma20"]), recent_low * 1.01)

    if stop_loss >= entry_price:
        stop_loss = entry_price * 0.97

    risk_per_share = max(entry_price - stop_loss, entry_price * 0.02)
    target_1 = entry_price + risk_per_share * 1.8
    return round(entry_price, 2), round(stop_loss, 2), round(target_1, 2)


def analyze_candidate(ticker: str) -> RecommendationResult | None:
    data = get_stock_data(ticker)
    if data.empty:
        return None

    result = analyze_position(
        data=data,
        position=PositionInput(
            ticker=ticker,
            quantity=0,
            avg_price=0,
            cash_budget=0,
            target_weight=0,
        ),
    )

    profile = _build_candidate_profile(data)
    bonus = profile.trend_score + profile.momentum_score + profile.volume_score + profile.breakout_score
    result.score = max(0, min(100, result.score + bonus))

    if result.score >= 80:
        result.action = "오늘매수후보"
    elif result.score >= 68:
        result.action = "관심후보"
    else:
        result.action = "보류"

    result.reasons = profile.reasons + result.reasons
    return result


def scan_market(
    market: str,
    universe: list[dict[str, str]] | None = None,
    min_score: int = 60,
    learning_adjustments: dict[tuple[str, str, str], LearningAdjustment] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    regime = classify_market_regime(market)

    candidates = universe if universe is not None else get_universe(market)
    for item in candidates:
        data = get_stock_data(item["ticker"])
        if data.empty:
            continue
        analyzed = analyze_position(
            data=data,
            position=PositionInput(
                ticker=item["ticker"],
                quantity=0,
                avg_price=0,
                cash_budget=0,
                target_weight=0,
            ),
        )
        profile = _build_candidate_profile(data)
        bonus = profile.trend_score + profile.momentum_score + profile.volume_score + profile.breakout_score
        base_score = max(0, min(100, analyzed.score + bonus + regime.adjustment))
        adjusted_score, learning_delta, learning_note = apply_learning_adjustment(
            base_score=base_score,
            scan_type="today_scan",
            market=market,
            setup=profile.setup,
            adjustments=learning_adjustments,
        )
        analyzed.score = adjusted_score

        if analyzed.score < min_score:
            continue

        if analyzed.score >= 80:
            analyzed.action = "오늘매수후보"
        elif analyzed.score >= 68:
            analyzed.action = "관심후보"
        else:
            analyzed.action = "보류"

        latest = data.iloc[-1]
        entry_price, stop_loss, target_1 = _build_trade_plan(data)
        rows.append(
            {
                "ticker": item["ticker"],
                "name": item["name"],
                "action": analyzed.action,
                "setup": profile.setup,
                "score": analyzed.score,
                "current_price": round(analyzed.current_price, 2),
                "trend_score": profile.trend_score,
                "momentum_score": profile.momentum_score,
                "volume_score": profile.volume_score,
                "breakout_score": profile.breakout_score,
                "volume_ratio": round(float(latest["volume_ratio"]), 2),
                "return_20d": round(float(latest["return_20d"]), 2),
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target_1": target_1,
                "regime": regime.regime,
                "regime_delta": regime.adjustment,
                "learning_delta": learning_delta,
                "reason": " / ".join(
                    ([regime.note] if regime.note else [])
                    + ([learning_note] if learning_note else [])
                    + (profile.reasons + analyzed.reasons)[:4]
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "name",
                "action",
                "setup",
                "score",
                "current_price",
                "trend_score",
                "momentum_score",
                "volume_score",
                "breakout_score",
                "volume_ratio",
                "return_20d",
                "entry_price",
                "stop_loss",
                "target_1",
                "regime",
                "regime_delta",
                "learning_delta",
                "reason",
            ]
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(by=["score", "breakout_score", "ticker"], ascending=[False, False, True]).reset_index(drop=True)
