from __future__ import annotations

import pandas as pd

from src.data.fetch import get_stock_data, get_stock_dividend_details
from src.strategy.universe import get_dividend_universe


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


def _build_accumulate_zone(latest: pd.Series) -> tuple[float, float]:
    anchors = [float(latest["ma20"]), float(latest["ma60"]), float(latest["ma120"])]
    valid = [value for value in anchors if value > 0]
    if not valid:
        close_price = float(latest["Close"])
        return close_price * 0.95, close_price * 1.02

    low = min(valid) * 0.98
    high = sorted(valid)[1] * 1.02 if len(valid) >= 2 else max(valid) * 1.02
    return low, max(low, high)


def _build_action(close_price: float, zone_low: float, zone_high: float, ex_dividend_date: str) -> str:
    if zone_low <= close_price <= zone_high:
        return "지금 모아가기"
    if close_price < zone_low:
        return "분할로 줍기"
    if ex_dividend_date:
        try:
            ex_date = pd.to_datetime(ex_dividend_date, errors="coerce")
            if not pd.isna(ex_date):
                days_gap = (pd.Timestamp(datetime.now().date()) - ex_date.normalize()).days
                if 0 <= days_gap <= 21:
                    return "배당락 후 눌림 체크"
        except Exception:
            pass
    return "추격보다 눌림 대기"


def _stable_dividend_score(data: pd.DataFrame, details: dict[str, object]) -> tuple[int, list[str]]:
    latest = data.iloc[-1]
    volatility = _volatility_pct(data)
    drawdown = _drawdown_pct(data)
    annual_return = float(data["Close"].pct_change(252).iloc[-1] * 100) if len(data) > 252 else 0.0

    dividend_yield = float(details["dividend_yield_pct"])
    dividend_growth_1y = float(details["dividend_growth_1y_pct"])
    dividend_events = int(details["dividend_events_1y"])

    score = 0
    reasons: list[str] = []

    if dividend_yield >= 2.0:
        score += 28
        reasons.append("배당이 꽤 받쳐줘서 버티기 편합니다.")
    if dividend_events >= 2:
        score += 10
        reasons.append("배당 지급 흐름이 이어지고 있습니다.")
    if latest["Close"] > latest["ma120"]:
        score += 18
        reasons.append("긴 흐름도 아직 무너지지 않았습니다.")
    if latest["from_52w_high_pct"] >= -15:
        score += 10
        reasons.append("고점 대비 크게 무너지지 않아 체력이 남아 있습니다.")
    if volatility <= 28:
        score += 16
        reasons.append("흔들림이 심하지 않아 모으기 좋습니다.")
    if drawdown >= -22:
        score += 14
        reasons.append("최근 1년 낙폭이 과하지 않았습니다.")
    if annual_return > 6:
        score += 10
        reasons.append("배당만이 아니라 가격 흐름도 무난합니다.")
    if dividend_growth_1y > 0:
        score += 8

    return score, reasons


def _growth_dividend_score(data: pd.DataFrame, details: dict[str, object]) -> tuple[int, list[str]]:
    latest = data.iloc[-1]
    annual_return = float(data["Close"].pct_change(252).iloc[-1] * 100) if len(data) > 252 else 0.0
    dividend_yield = float(details["dividend_yield_pct"])
    dividend_growth_1y = float(details["dividend_growth_1y_pct"])
    dividend_growth_3y = float(details["dividend_growth_3y_pct"])

    score = 0
    reasons: list[str] = []

    if dividend_growth_1y >= 5:
        score += 26
        reasons.append("최근 배당이 커지는 흐름이 보입니다.")
    if dividend_growth_3y >= 4:
        score += 22
        reasons.append("몇 년 기준으로도 배당이 성장했습니다.")
    if latest["Close"] > latest["ma20"] > latest["ma60"]:
        score += 18
        reasons.append("가격 흐름도 위쪽으로 정리돼 있습니다.")
    if latest["rs_score"] >= 8:
        score += 8
        reasons.append("상대강도도 괜찮아 배당 성장주 성격이 있습니다.")
    if annual_return > 10:
        score += 14
        reasons.append("배당 성장에 가격 상승도 같이 붙었습니다.")
    if dividend_yield >= 1.0:
        score += 8
        reasons.append("수익률도 너무 낮지는 않습니다.")
    if latest["rsi"] <= 68:
        score += 6

    return score, reasons


def build_dividend_profiles(market: str, top_n: int = 8) -> dict[str, pd.DataFrame]:
    stable_rows: list[dict[str, object]] = []
    growth_rows: list[dict[str, object]] = []

    for item in get_dividend_universe(market):
        try:
            data = get_stock_data(item["ticker"])
            if data.empty or len(data) < 180:
                continue

            details = get_stock_dividend_details(item["ticker"])
            dividend_yield = float(details["dividend_yield_pct"])
            if dividend_yield <= 0:
                continue

            latest = data.iloc[-1]
            zone_low, zone_high = _build_accumulate_zone(latest)
            close_price = float(latest["Close"])
            action = _build_action(close_price, zone_low, zone_high, str(details["ex_dividend_date"]))
            pullback_pct = (close_price / float(data["Close"].tail(252).max()) - 1) * 100 if len(data) >= 252 else 0.0

            stable_score, stable_reasons = _stable_dividend_score(data, details)
            growth_score, growth_reasons = _growth_dividend_score(data, details)

            base = {
                "ticker": item["ticker"],
                "name": item["name"],
                "current_price": round(close_price, 2),
                "annual_dividend": round(float(details["annual_dividend"]), 4),
                "dividend_yield_pct": round(dividend_yield, 2),
                "dividend_growth_1y_pct": round(float(details["dividend_growth_1y_pct"]), 2),
                "dividend_growth_3y_pct": round(float(details["dividend_growth_3y_pct"]), 2),
                "dividend_events_1y": int(details["dividend_events_1y"]),
                "ex_dividend_date": str(details["ex_dividend_date"]),
                "accumulate_low": round(zone_low, 2),
                "accumulate_high": round(zone_high, 2),
                "pullback_pct": round(pullback_pct, 2),
                "action": action,
            }

            stable_rows.append(
                {
                    **base,
                    "score": stable_score,
                    "style": "안정 배당",
                    "reason": " / ".join(stable_reasons[:3]),
                }
            )
            growth_rows.append(
                {
                    **base,
                    "score": growth_score,
                    "style": "배당 성장",
                    "reason": " / ".join(growth_reasons[:3]),
                }
            )
        except Exception:
            continue

    def _sort_rows(rows: list[dict[str, object]], sort_columns: list[str]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "name",
                    "score",
                    "style",
                    "current_price",
                    "annual_dividend",
                    "dividend_yield_pct",
                    "dividend_growth_1y_pct",
                    "dividend_growth_3y_pct",
                    "dividend_events_1y",
                    "ex_dividend_date",
                    "accumulate_low",
                    "accumulate_high",
                    "pullback_pct",
                    "action",
                    "reason",
                ]
            )
        return (
            pd.DataFrame(rows)
            .sort_values(by=sort_columns, ascending=[False] * len(sort_columns))
            .head(top_n)
            .reset_index(drop=True)
        )

    return {
        "stable": _sort_rows(stable_rows, ["score", "dividend_yield_pct", "dividend_growth_1y_pct"]),
        "growth": _sort_rows(growth_rows, ["score", "dividend_growth_3y_pct", "dividend_growth_1y_pct"]),
    }
