from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data.fetch import (
    get_stock_data,
    is_recent_price_data,
    latest_price_timestamp,
    price_data_freshness_label,
    price_source_label,
)
from src.strategy.universe import get_universe


def _score_monthly(data: pd.DataFrame) -> tuple[int, list[str]]:
    latest = data.iloc[-1]
    score = 0
    reasons: list[str] = []
    if latest["Close"] > latest["ma120"]:
        score += 25
        reasons.append("장기 흐름이 위쪽입니다.")
    if latest["return_60d"] > 12:
        score += 20
        reasons.append("최근 두 달 흐름이 강합니다.")
    if latest["return_20d"] > 5:
        score += 10
        reasons.append("최근 한 달도 괜찮습니다.")
    if latest["macd_diff"] > 0:
        score += 10
    return score, reasons


def _score_weekly(data: pd.DataFrame) -> tuple[int, list[str]]:
    latest = data.iloc[-1]
    score = 0
    reasons: list[str] = []
    if latest["Close"] > latest["ma20"] > latest["ma60"]:
        score += 22
        reasons.append("중기 추세가 잘 정리돼 있습니다.")
    if latest["volume_ratio"] >= 1.2:
        score += 18
        reasons.append("거래량이 조금 더 붙고 있습니다.")
    if latest["Close"] >= data["Close"].tail(40).max() * 0.97:
        score += 16
        reasons.append("중기 고점 근처라 돌파를 볼 수 있습니다.")
    if 45 <= latest["rsi"] <= 68:
        score += 10
    return score, reasons


def _score_daily(data: pd.DataFrame) -> tuple[int, list[str]]:
    latest = data.iloc[-1]
    score = 0
    reasons: list[str] = []
    if latest["volume_ratio"] >= 1.3:
        score += 20
        reasons.append("오늘 거래량이 강한 편입니다.")
    if latest["return_20d"] > 8:
        score += 15
        reasons.append("최근 한 달 흐름이 강합니다.")
    if latest["Close"] >= data["Close"].tail(20).max() * 0.98:
        score += 20
        reasons.append("단기 고점 근처입니다.")
    if latest["macd_diff"] > 0:
        score += 10
    return score, reasons


def _score_next_day(data: pd.DataFrame) -> tuple[int, list[str]]:
    latest = data.iloc[-1]
    score = 0
    reasons: list[str] = []
    if latest["Close"] > latest["Open"]:
        score += 12
        reasons.append("오늘 종가가 시가보다 높게 끝났습니다.")
    if latest["volume_ratio"] >= 1.5:
        score += 18
        reasons.append("거래량이 많이 붙은 채로 끝났습니다.")
    if latest["Close"] >= data["High"].tail(10).max() * 0.985:
        score += 20
        reasons.append("고점 근처에서 끝나서 내일 한 번 더 볼 만합니다.")
    if 50 <= latest["rsi"] <= 70:
        score += 10
    if latest["macd_diff"] > 0:
        score += 10
    return score, reasons


def build_auto_candidate_sets(market: str, top_n: int = 12) -> dict[str, pd.DataFrame]:
    rows = {"monthly": [], "weekly": [], "daily": [], "next_day": []}
    universe = get_universe(market)

    for item in universe:
        data = get_stock_data(item["ticker"])
        if data.empty or not is_recent_price_data(data, max_age_days=3):
            continue

        latest = data.iloc[-1]
        monthly_score, monthly_reasons = _score_monthly(data)
        weekly_score, weekly_reasons = _score_weekly(data)
        daily_score, daily_reasons = _score_daily(data)
        next_day_score, next_day_reasons = _score_next_day(data)
        close_price = float(latest["Close"])
        atr = float(latest.get("atr14", 0) or 0)
        recent_low = float(data["Low"].tail(20).min()) if "Low" in data.columns else close_price * 0.92
        stop_loss = min(close_price * 0.92, recent_low * 0.99)
        if stop_loss <= 0 or stop_loss >= close_price:
            stop_loss = close_price * 0.92
        risk = max(close_price - stop_loss, atr, close_price * 0.03)
        latest_ts = latest_price_timestamp(data)

        base = {
            "market": market,
            "ticker": item["ticker"],
            "name": item["name"],
            "current_price": round(close_price, 2),
            "entry_price": round(close_price, 2),
            "stop_loss": round(stop_loss, 2),
            "target_1": round(close_price + risk * 1.5, 2),
            "target_2": round(close_price + risk * 2.5, 2),
            "target_3": round(close_price + risk * 3.5, 2),
            "quote_as_of": latest_ts.strftime("%Y-%m-%d") if latest_ts is not None else "",
            "data_freshness": price_data_freshness_label(data, intraday=False),
            "price_source": price_source_label(data, intraday=False),
            "price_basis": "현재가 기준 진입, 최근 20일 저점/ATR 기준 손절, 1~3차 목표 자동 산정",
            "volume_ratio": round(float(latest["volume_ratio"]), 2),
            "return_20d": round(float(latest["return_20d"]), 2),
            "captured_at": datetime.now().isoformat(timespec="seconds"),
        }

        rows["monthly"].append({**base, "score": monthly_score, "reason": " / ".join(monthly_reasons[:3])})
        rows["weekly"].append({**base, "score": weekly_score, "reason": " / ".join(weekly_reasons[:3])})
        rows["daily"].append({**base, "score": daily_score, "reason": " / ".join(daily_reasons[:3])})
        rows["next_day"].append({**base, "score": next_day_score, "reason": " / ".join(next_day_reasons[:3])})

    result: dict[str, pd.DataFrame] = {}
    for key, records in rows.items():
        frame = pd.DataFrame(records)
        if frame.empty:
            result[key] = frame
            continue
        result[key] = frame.sort_values(by=["score", "volume_ratio", "ticker"], ascending=[False, False, True]).head(top_n).reset_index(drop=True)
    return result


def build_compounder_candidates(market: str, top_n: int = 12) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    universe = get_universe(market)

    for item in universe:
        data = get_stock_data(item["ticker"])
        if data.empty or len(data) < 252 or not is_recent_price_data(data, max_age_days=3):
            continue

        latest = data.iloc[-1]
        score = 0
        reasons: list[str] = []

        if latest["Close"] > latest["ma120"]:
            score += 20
            reasons.append("장기 우상향 흐름입니다.")
        if latest["return_60d"] > 15:
            score += 15
            reasons.append("최근 흐름이 꽤 강합니다.")
        annual_return = float(data["Close"].pct_change(252).iloc[-1] * 100) if len(data) > 252 else 0.0
        if annual_return > 40:
            score += 20
            reasons.append("1년 성과가 강하게 나왔습니다.")
        max_close = float(data["Close"].tail(252).max())
        if float(latest["Close"]) >= max_close * 0.92:
            score += 15
            reasons.append("고점권을 잘 지키고 있습니다.")
        if latest["macd_diff"] > 0:
            score += 10

        close_price = float(latest["Close"])
        ma120 = float(latest.get("ma120", close_price) or close_price)
        atr = float(latest.get("atr14", 0) or 0)
        risk_buffer = max(atr * 2.2, close_price * 0.12)
        stop_loss = max(ma120 * 0.96, close_price - risk_buffer)
        if stop_loss >= close_price:
            stop_loss = close_price * 0.88
        latest_ts = latest_price_timestamp(data)

        rows.append(
            {
                "market": market,
                "ticker": item["ticker"],
                "name": item["name"],
                "score": score,
                "return_1y_pct": round(annual_return, 2),
                "return_6m_pct": round(float(latest["return_60d"]), 2),
                "current_price": round(close_price, 2),
                "entry_price": round(close_price, 2),
                "stop_loss": round(max(0.01, stop_loss), 2),
                "target_1": round(close_price * 1.15, 2),
                "target_2": round(close_price * 1.28, 2),
                "target_3": round(close_price * 1.45, 2),
                "quote_as_of": latest_ts.strftime("%Y-%m-%d") if latest_ts is not None else "",
                "data_freshness": price_data_freshness_label(data, intraday=False),
                "price_source": price_source_label(data, intraday=False),
                "price_basis": "장기 후보는 현재가 기준 분할 진입, 120일선/ATR 기준 이탈가, 15/28/45% 목표를 기본값으로 산정",
                "reason": " / ".join(reasons[:3]),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "market",
                "ticker",
                "name",
                "score",
                "return_1y_pct",
                "return_6m_pct",
                "current_price",
                "entry_price",
                "stop_loss",
                "target_1",
                "target_2",
                "target_3",
                "quote_as_of",
                "data_freshness",
                "price_source",
                "price_basis",
                "reason",
            ]
        )

    return pd.DataFrame(rows).sort_values(by=["score", "return_1y_pct"], ascending=[False, False]).head(top_n).reset_index(drop=True)
