from __future__ import annotations

import pandas as pd

from src.data.fetch import get_intraday_stock_data, get_stock_data, get_stock_dividend_yield
from src.strategy.learning import apply_learning_adjustment, LearningAdjustment
from src.strategy.regime import classify_market_regime
from src.strategy.universe import get_high_risk_universe, get_universe


def _volatility_pct(data: pd.DataFrame) -> float:
    returns = data["Close"].pct_change().dropna()
    if returns.empty:
        return 0.0
    return float(returns.tail(252).std() * (252**0.5) * 100)


def _drawdown_pct(data: pd.DataFrame) -> float:
    trailing = data["Close"].tail(252)
    if trailing.empty:
        return 0.0
    return float((trailing / trailing.cummax() - 1).min() * 100)


def _stable_score(data: pd.DataFrame, dividend_yield: float) -> tuple[int, list[str], float, float]:
    latest = data.iloc[-1]
    volatility = _volatility_pct(data)
    drawdown = _drawdown_pct(data)
    annual_return = float(data["Close"].pct_change(252).iloc[-1] * 100) if len(data) > 252 else 0.0

    score = 0
    reasons: list[str] = []
    if latest["Close"] > latest["ma120"]:
        score += 24
        reasons.append("길게 보면 우상향 흐름입니다.")
    if annual_return > 8:
        score += 18
        reasons.append("1년 기준으로도 천천히 올라왔습니다.")
    if volatility <= 24:
        score += 22
        reasons.append("흔들림이 크지 않아 모아가기 좋습니다.")
    if drawdown >= -18:
        score += 18
        reasons.append("큰 하락이 상대적으로 적었습니다.")
    if dividend_yield >= 1.2:
        score += 12
        reasons.append("배당도 있어 버티는 힘이 있습니다.")
    return score, reasons, volatility, drawdown


def _dividend_score(data: pd.DataFrame, dividend_yield: float) -> tuple[int, list[str]]:
    latest = data.iloc[-1]
    score = 0
    reasons: list[str] = []
    if dividend_yield >= 2.0:
        score += 30
        reasons.append("배당 매력이 꽤 있는 편입니다.")
    if latest["Close"] > latest["ma120"]:
        score += 20
        reasons.append("배당주인데 차트 흐름도 무난합니다.")
    if latest["rsi"] >= 45:
        score += 10
    if latest["macd_diff"] > 0:
        score += 10
    if float(data["Close"].pct_change(252).iloc[-1] * 100) > 5 if len(data) > 252 else False:
        score += 15
        reasons.append("배당만 있는 게 아니라 가격 흐름도 괜찮습니다.")
    return score, reasons


def _growth_score(data: pd.DataFrame) -> tuple[int, list[str], float]:
    latest = data.iloc[-1]
    annual_return = float(data["Close"].pct_change(252).iloc[-1] * 100) if len(data) > 252 else 0.0
    volatility = _volatility_pct(data)
    score = 0
    reasons: list[str] = []
    if latest["Close"] > latest["ma20"] > latest["ma60"] > latest["ma120"]:
        score += 28
        reasons.append("위로 정렬된 강한 성장 흐름입니다.")
    if latest["return_60d"] > 18:
        score += 24
        reasons.append("최근 두 달 상승 속도가 빠릅니다.")
    if annual_return > 35:
        score += 20
        reasons.append("1년 동안 강하게 올라온 주도주 후보입니다.")
    if latest["volume_ratio"] > 1.2:
        score += 12
        reasons.append("거래량이 붙어 힘이 실리고 있습니다.")
    if volatility >= 30:
        score += 8
        reasons.append("변동성은 크지만 그만큼 성장 성격도 강합니다.")
    return score, reasons, annual_return


def build_strategy_profiles(market: str, top_n: int = 8) -> dict[str, pd.DataFrame]:
    stable_rows: list[dict[str, object]] = []
    dividend_rows: list[dict[str, object]] = []
    growth_rows: list[dict[str, object]] = []

    for item in get_universe(market):
        data = get_stock_data(item["ticker"])
        if data.empty or len(data) < 180:
            continue

        latest = data.iloc[-1]
        dividend_yield = float(get_stock_dividend_yield(item["ticker"]))
        stable_score, stable_reasons, volatility, drawdown = _stable_score(data, dividend_yield)
        dividend_score, dividend_reasons = _dividend_score(data, dividend_yield)
        growth_score, growth_reasons, annual_return = _growth_score(data)

        base = {
            "ticker": item["ticker"],
            "name": item["name"],
            "current_price": round(float(latest["Close"]), 2),
            "dividend_yield_pct": round(dividend_yield, 2),
            "volatility_pct": round(volatility, 2),
            "drawdown_pct": round(drawdown, 2),
            "return_60d_pct": round(float(latest["return_60d"]), 2),
            "return_1y_pct": round(annual_return, 2),
        }

        stable_rows.append({**base, "score": stable_score, "bucket": "안정형 적립", "reason": " / ".join(stable_reasons[:3])})
        dividend_rows.append({**base, "score": dividend_score, "bucket": "우량 배당", "reason": " / ".join(dividend_reasons[:3])})
        growth_rows.append({**base, "score": growth_score, "bucket": "고위험 성장", "reason": " / ".join(growth_reasons[:3])})

    def _sort_frame(rows: list[dict[str, object]], sort_cols: list[str]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(by=sort_cols, ascending=[False] * len(sort_cols)).head(top_n).reset_index(drop=True)

    return {
        "stable": _sort_frame(stable_rows, ["score", "dividend_yield_pct", "return_1y_pct"]),
        "dividend": _sort_frame(dividend_rows, ["score", "dividend_yield_pct", "return_1y_pct"]),
        "growth": _sort_frame(growth_rows, ["score", "return_1y_pct", "return_60d_pct"]),
    }


def build_short_term_trade_candidates(
    market: str,
    top_n: int = 8,
    interval: str = "5m",
    min_score: int = 65,
    learning_adjustments: dict[tuple[str, str, str], LearningAdjustment] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    regime = classify_market_regime(market)

    for item in get_universe(market):
        try:
            daily = get_stock_data(item["ticker"])
            intraday = get_intraday_stock_data(item["ticker"], period="5d", interval=interval)
            if daily.empty or intraday.empty or len(intraday) < 25:
                continue

            daily_latest = daily.iloc[-1]
            intra_latest = intraday.iloc[-1]
            recent_high = float(intraday["session_high_20"].iloc[-2])
            recent_low = float(intraday["session_low_20"].iloc[-2])
            entry_price = max(float(intra_latest["Close"]), recent_high)
            stop_loss = min(float(intra_latest["vwap_proxy"]), recent_low)
            risk_per_share = max(entry_price - stop_loss, entry_price * 0.008)
            target_1 = entry_price + risk_per_share * 1.5
            target_2 = entry_price + risk_per_share * 2.5

            score = 35
            reasons: list[str] = []
            exit_rule = "VWAP 이탈 또는 손절가 하향 이탈 시 정리"

            if daily_latest["Close"] > daily_latest["ma20"] > daily_latest["ma60"]:
                score += 16
                reasons.append("일봉 흐름이 좋아 단타 받침이 있습니다.")
            if float(intra_latest["volume_ratio"]) >= 1.8:
                score += 18
                reasons.append("분봉 거래량이 강하게 들어왔습니다.")
            if float(intra_latest["short_return_pct"]) >= 0.8:
                score += 16
                reasons.append("방금 탄력이 붙었습니다.")
            if float(intra_latest["Close"]) >= recent_high:
                score += 18
                reasons.append("직전 고점을 넘었습니다.")
            if float(intra_latest["Close"]) > float(intra_latest["vwap_proxy"]):
                score += 12
                reasons.append("평균가 위에서 잘 버티고 있습니다.")
            if 48 <= float(daily_latest["rsi"]) <= 72:
                score += 8

            score = min(100, max(0, int(score + regime.adjustment)))
            if score < min_score:
                continue

            setup = "단타돌파"
            if float(intra_latest["volume_ratio"]) >= 2.2 and float(intra_latest["short_return_pct"]) >= 1.2:
                setup = "초강세추격"
                exit_rule = "1차 목표가 도달 시 절반 매도, VWAP 이탈 시 잔량 정리"
            elif float(intra_latest["Close"]) >= recent_high:
                setup = "장중돌파"
            elif float(intra_latest["Close"]) > float(intra_latest["vwap_proxy"]):
                setup = "VWAP지지"

            score, learning_delta, learning_note = apply_learning_adjustment(
                base_score=score,
                scan_type="short_term_trade",
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
                    "entry_price": round(entry_price, 2),
                    "stop_loss": round(stop_loss, 2),
                    "target_1": round(target_1, 2),
                    "target_2": round(target_2, 2),
                    "short_return_pct": round(float(intra_latest["short_return_pct"]), 2),
                    "volume_ratio": round(float(intra_latest["volume_ratio"]), 2),
                    "risk_reward_1": round((target_1 - entry_price) / max(entry_price - stop_loss, 0.01), 2),
                    "exit_rule": exit_rule,
                    "regime": regime.regime,
                    "regime_delta": regime.adjustment,
                    "learning_delta": learning_delta,
                    "reason": " / ".join(([regime.note] if regime.note else []) + ([learning_note] if learning_note else []) + reasons[:4]),
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
                "entry_price",
                "stop_loss",
                "target_1",
                "target_2",
                "short_return_pct",
                "volume_ratio",
                "risk_reward_1",
                "regime",
                "regime_delta",
                "learning_delta",
                "exit_rule",
                "reason",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        by=["score", "volume_ratio", "short_return_pct"], ascending=[False, False, False]
    ).head(top_n).reset_index(drop=True)


def build_high_risk_trade_candidates(
    market: str,
    top_n: int = 8,
    interval: str = "5m",
    min_score: int = 60,
    learning_adjustments: dict[tuple[str, str, str], LearningAdjustment] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    regime = classify_market_regime(market)

    for item in get_high_risk_universe(market):
        try:
            daily = get_stock_data(item["ticker"])
            intraday = get_intraday_stock_data(item["ticker"], period="5d", interval=interval)
            if daily.empty or intraday.empty or len(intraday) < 25:
                continue

            daily_latest = daily.iloc[-1]
            intra_latest = intraday.iloc[-1]
            recent_high = float(intraday["session_high_20"].iloc[-2])
            recent_low = float(intraday["session_low_20"].iloc[-2])

            entry_price = max(float(intra_latest["Close"]), recent_high)
            stop_loss = min(float(intra_latest["vwap_proxy"]), recent_low)
            if stop_loss >= entry_price:
                stop_loss = entry_price * 0.95

            risk_per_share = max(entry_price - stop_loss, entry_price * 0.015)
            target_1 = entry_price + risk_per_share * 2.0
            target_2 = entry_price + risk_per_share * 3.5

            score = 30
            reasons: list[str] = []
            exit_rule = "손절가 이탈 시 바로 정리, 1차 목표가 도달 시 절반 익절"

            if float(intra_latest["volume_ratio"]) >= 2.2:
                score += 22
                reasons.append("고위험주인데 거래량이 강하게 터졌습니다.")
            if float(intra_latest["short_return_pct"]) >= 1.4:
                score += 22
                reasons.append("짧은 시간에 탄력이 강하게 붙었습니다.")
            if float(intra_latest["Close"]) >= recent_high:
                score += 18
                reasons.append("직전 고점을 넘겨 추격 매매 구간입니다.")
            if float(intra_latest["Close"]) > float(intra_latest["vwap_proxy"]):
                score += 10
                reasons.append("평균가 위에서 버티고 있습니다.")
            if float(daily_latest["volume_ratio"]) >= 1.5:
                score += 10
                reasons.append("일봉 기준으로도 거래량이 살아 있습니다.")

            setup = "고위험추격"
            if float(intra_latest["Close"]) >= recent_high and float(intra_latest["volume_ratio"]) >= 2.8:
                setup = "급등추격"
                exit_rule = "초강세 구간이라 짧게 보고 1차 목표가 근처에서 빠르게 익절"
            elif float(intra_latest["Close"]) > float(intra_latest["vwap_proxy"]):
                setup = "고위험VWAP지지"

            score = min(100, max(0, int(score + regime.adjustment)))
            score, learning_delta, learning_note = apply_learning_adjustment(
                base_score=score,
                scan_type="high_risk_trade",
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
                    "entry_price": round(entry_price, 2),
                    "stop_loss": round(stop_loss, 2),
                    "target_1": round(target_1, 2),
                    "target_2": round(target_2, 2),
                    "short_return_pct": round(float(intra_latest["short_return_pct"]), 2),
                    "volume_ratio": round(float(intra_latest["volume_ratio"]), 2),
                    "risk_reward_1": round((target_1 - entry_price) / max(entry_price - stop_loss, 0.01), 2),
                    "regime": regime.regime,
                    "regime_delta": regime.adjustment,
                    "learning_delta": learning_delta,
                    "exit_rule": exit_rule,
                    "risk_level": "매우높음",
                    "reason": " / ".join(([regime.note] if regime.note else []) + ([learning_note] if learning_note else []) + reasons[:4]),
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
                "entry_price",
                "stop_loss",
                "target_1",
                "target_2",
                "short_return_pct",
                "volume_ratio",
                "risk_reward_1",
                "regime",
                "regime_delta",
                "learning_delta",
                "exit_rule",
                "risk_level",
                "reason",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        by=["score", "volume_ratio", "short_return_pct"], ascending=[False, False, False]
    ).head(top_n).reset_index(drop=True)
