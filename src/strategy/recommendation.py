from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.portfolio.models import PositionInput


@dataclass(slots=True)
class RecommendationResult:
    ticker: str
    action: str
    score: int
    current_price: float
    return_pct: float
    suggested_buy_pct: int
    suggested_sell_pct: int
    reasons: list[str]


def analyze_position(data: pd.DataFrame, position: PositionInput) -> RecommendationResult:
    latest = data.iloc[-1]
    current_price = float(latest["Close"])
    avg_price = float(position.avg_price or 0)
    quantity = float(position.quantity or 0)

    score = 50
    reasons: list[str] = []

    if latest["Close"] > latest["ma20"] > latest["ma60"]:
        score += 20
        reasons.append("추세가 위쪽이라 흐름이 괜찮습니다.")
    elif latest["Close"] < latest["ma20"] < latest["ma60"]:
        score -= 20
        reasons.append("추세가 아래쪽이라 조금 약합니다.")

    if latest["rsi"] < 35:
        score += 10
        reasons.append("많이 눌린 자리라 반등을 볼 수 있습니다.")
    elif latest["rsi"] > 70:
        score -= 10
        reasons.append("단기 과열 구간이라 추격은 조심해야 합니다.")

    if latest["macd"] > latest["macd_signal"]:
        score += 10
        reasons.append("상승 힘이 아직 살아 있습니다.")
    else:
        score -= 5
        reasons.append("상승 힘이 조금 약해졌습니다.")

    if latest["volume_ratio"] > 1.8:
        score += 10
        reasons.append("거래량이 붙어서 관심이 몰리고 있습니다.")

    if latest["return_20d"] < -12:
        score -= 10
        reasons.append("최근 하락폭이 커서 추세가 꺾였는지 봐야 합니다.")
    elif latest["return_20d"] > 15:
        score += 5
        reasons.append("최근 흐름이 다른 종목보다 강한 편입니다.")

    score = max(0, min(100, int(score)))

    if quantity <= 0:
        action = "신규관심"
        reasons.insert(0, "지금은 새로 들어갈 후보로 봤습니다.")
    elif score >= 75:
        action = "홀드/일부매수"
        reasons.insert(0, "흐름이 좋아서 보유 유지나 소액 추가가 괜찮아 보입니다.")
    elif score >= 55:
        action = "홀드"
        reasons.insert(0, "확신이 아주 강하진 않지만 일단 보유 쪽이 낫습니다.")
    elif score >= 40:
        action = "관망"
        reasons.insert(0, "애매한 자리라 지금은 지켜보는 편이 좋습니다.")
    else:
        action = "일부매도/리스크축소"
        reasons.insert(0, "흐름이 약해서 비중을 줄일지 점검할 필요가 있습니다.")

    return_pct = 0.0
    suggested_buy_pct = 0
    suggested_sell_pct = 0

    if quantity <= 0:
        if score >= 80:
            suggested_buy_pct = 40
        elif score >= 68:
            suggested_buy_pct = 20
    elif action == "홀드/일부매수":
        suggested_buy_pct = 15 if latest["rsi"] >= 65 else 25
    elif action == "홀드":
        suggested_buy_pct = 5 if latest["rsi"] < 40 else 0
    elif action == "관망":
        suggested_sell_pct = 10 if latest["rsi"] > 70 else 0
    else:
        suggested_sell_pct = 25 if avg_price <= 0 or current_price < latest["ma60"] else 15

    if avg_price > 0:
        return_pct = ((current_price - avg_price) / avg_price) * 100
        if return_pct >= 20:
            reasons.append("수익이 많이 난 구간이라 일부 익절을 볼 만합니다.")
            suggested_sell_pct = max(suggested_sell_pct, 20)
        elif return_pct <= -10:
            reasons.append("손실이 커서 물타기보다 기준을 다시 잡는 게 좋습니다.")
            if action in {"홀드/일부매수", "홀드"}:
                suggested_buy_pct = max(suggested_buy_pct, 10)

    return RecommendationResult(
        ticker=position.ticker,
        action=action,
        score=score,
        current_price=current_price,
        return_pct=return_pct,
        suggested_buy_pct=suggested_buy_pct,
        suggested_sell_pct=suggested_sell_pct,
        reasons=reasons,
    )
