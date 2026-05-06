from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.fetch import (
    get_latest_quote,
    get_stock_dividend_yield,
    get_stock_event_summary,
    get_stock_data,
    get_stock_news_summary,
    get_stock_profile,
)


@dataclass(slots=True)
class HoldingAnalytics:
    ticker: str
    current_price: float
    market_value: float
    pnl_pct: float
    volatility_pct: float
    drawdown_pct: float
    return_6m_pct: float
    return_1y_pct: float
    dividend_yield_pct: float
    risk_level: str
    style: str
    sector: str
    industry: str


def _classify_outlook(score: int) -> str:
    if score >= 75:
        return "상승 기대"
    if score >= 58:
        return "완만한 우상향"
    if score >= 43:
        return "중립/관찰"
    return "하방 주의"


def _classify_event_risk(earnings_days_left: object, ex_dividend_days_left: object) -> str:
    for value in [earnings_days_left, ex_dividend_days_left]:
        numeric = pd.to_numeric(value, errors="coerce")
        if not pd.isna(numeric) and 0 <= float(numeric) <= 7:
            return "높음"
    for value in [earnings_days_left, ex_dividend_days_left]:
        numeric = pd.to_numeric(value, errors="coerce")
        if not pd.isna(numeric) and 8 <= float(numeric) <= 21:
            return "중간"
    return "낮음"


def _classify_risk(volatility_pct: float, drawdown_pct: float) -> str:
    if volatility_pct >= 45 or drawdown_pct <= -30:
        return "높음"
    if volatility_pct >= 28 or drawdown_pct <= -18:
        return "중간"
    return "낮음"


def _classify_style(return_6m_pct: float, dividend_yield_pct: float, volatility_pct: float) -> str:
    if dividend_yield_pct >= 2.5:
        return "배당"
    if return_6m_pct >= 20:
        return "성장"
    if volatility_pct <= 22:
        return "안정"
    return "혼합"


def _safe_float(value: object, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return float(numeric)


def _build_outlook_action(
    *,
    score: int,
    event_risk: str,
    return_pct: float,
    current_weight_pct: float,
    target_weight_pct: float,
) -> tuple[str, int]:
    weight_gap = current_weight_pct - target_weight_pct

    if score < 43 and return_pct <= -8:
        return "손실관리 우선", 5
    if score < 50 and weight_gap >= 3:
        return "비중 축소 검토", 4
    if event_risk == "높음" and score < 58:
        return "이벤트 전 관찰", 4
    if score >= 75 and weight_gap <= -2:
        return "눌림 추가 가능", 1
    if score >= 72 and return_pct >= 18 and weight_gap >= 2:
        return "일부 차익 실현", 2
    if score >= 58:
        return "보유 유지", 3
    return "관찰 유지", 3


def _plain_score_label(score: int) -> str:
    if score >= 80:
        return "좋음"
    if score >= 65:
        return "괜찮음"
    if score >= 50:
        return "애매함"
    return "주의"


def _classify_momentum_state(latest: pd.Series) -> str:
    close = _safe_float(latest.get("Close", 0))
    ma20 = _safe_float(latest.get("ma20", close))
    ma60 = _safe_float(latest.get("ma60", close))
    rsi = _safe_float(latest.get("rsi", 50), 50)
    macd_diff = _safe_float(latest.get("macd_diff", 0))

    if close > ma20 > ma60 and macd_diff > 0 and 45 <= rsi <= 72:
        return "상승 흐름"
    if close > ma60 and rsi < 45:
        return "눌림 구간"
    if close < ma20 < ma60:
        return "하락 흐름"
    if rsi >= 72:
        return "단기 과열"
    return "중립"


def _build_weight_status(current_weight_pct: float, target_weight_pct: float) -> str:
    if target_weight_pct <= 0:
        return "목표비중 없음"
    gap = current_weight_pct - target_weight_pct
    if gap >= 5:
        return "비중 과다"
    if gap >= 2:
        return "약간 많음"
    if gap <= -5:
        return "비중 부족"
    if gap <= -2:
        return "조금 부족"
    return "비중 적정"


def _build_next_step(action_hint: str, event_risk: str, momentum_state: str, weight_status: str) -> str:
    if action_hint == "손실관리 우선":
        return "손절선/회복 조건 먼저 확인"
    if action_hint == "비중 축소 검토":
        return "반등 시 일부 줄이기"
    if action_hint == "일부 차익 실현":
        return "목표가 근처 분할 익절"
    if event_risk == "높음":
        return "이벤트 지나고 재판단"
    if action_hint == "눌림 추가 가능":
        return "눌림가 근처 분할매수"
    if momentum_state == "상승 흐름" and weight_status in {"비중 부족", "조금 부족"}:
        return "작게 추가 검토"
    return "보유하며 추적"


def _build_reason_summary(
    *,
    momentum_state: str,
    risk_level: str,
    style: str,
    weight_status: str,
    event_risk: str,
    news_bias: str,
    return_pct: float,
) -> str:
    parts = [momentum_state]
    parts.append(f"{style} 성격")
    if risk_level == "높음":
        parts.append("변동성 큼")
    elif risk_level == "낮음":
        parts.append("변동성 낮음")
    if weight_status not in {"목표비중 없음", "비중 적정"}:
        parts.append(weight_status)
    if event_risk == "높음":
        parts.append("가까운 이벤트 주의")
    if news_bias in {"긍정", "부정"}:
        parts.append(f"뉴스 {news_bias}")
    if return_pct >= 20:
        parts.append("수익권")
    elif return_pct <= -10:
        parts.append("손실권")
    return " / ".join(parts[:5])


def analyze_holding(row: dict[str, object]) -> HoldingAnalytics | None:
    ticker = str(row.get("ticker", "")).strip().upper()
    if not ticker:
        return None

    data = get_stock_data(ticker)
    if data.empty:
        return None

    latest = data.iloc[-1]
    current_price = float(latest["Close"])
    quantity = float(row.get("quantity", 0) or 0)
    avg_price = float(row.get("avg_price", 0) or 0)
    market_value = current_price * quantity
    pnl_pct = ((current_price - avg_price) / avg_price) * 100 if avg_price > 0 else 0.0

    returns = data["Close"].pct_change().dropna()
    volatility_pct = float(returns.tail(252).std() * (252 ** 0.5) * 100) if not returns.empty else 0.0
    trailing = data["Close"].tail(252)
    drawdown_pct = float((trailing / trailing.cummax() - 1).min() * 100) if not trailing.empty else 0.0
    return_6m_pct = float(data["Close"].pct_change(126).iloc[-1] * 100) if len(data) > 126 else 0.0
    return_1y_pct = float(data["Close"].pct_change(252).iloc[-1] * 100) if len(data) > 252 else 0.0
    dividend_yield_pct = float(get_stock_dividend_yield(ticker))
    profile = get_stock_profile(ticker)
    risk_level = _classify_risk(volatility_pct, drawdown_pct)
    style = _classify_style(return_6m_pct, dividend_yield_pct, volatility_pct)

    return HoldingAnalytics(
        ticker=ticker,
        current_price=current_price,
        market_value=market_value,
        pnl_pct=pnl_pct,
        volatility_pct=volatility_pct,
        drawdown_pct=drawdown_pct,
        return_6m_pct=return_6m_pct,
        return_1y_pct=return_1y_pct,
        dividend_yield_pct=dividend_yield_pct,
        risk_level=risk_level,
        style=style,
        sector=profile.get("sector", ""),
        industry=profile.get("industry", ""),
    )


def _build_correlation_table(portfolio: pd.DataFrame) -> pd.DataFrame:
    series_map: dict[str, pd.Series] = {}
    for row in portfolio.to_dict("records"):
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        data = get_stock_data(ticker)
        if data.empty:
            continue
        series_map[ticker] = data.set_index("Date")["Close"].pct_change()

    if len(series_map) < 2:
        return pd.DataFrame()

    corr = pd.DataFrame(series_map).dropna(how="all").corr().round(2)
    return corr


def analyze_portfolio(portfolio: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []

    for row in portfolio.to_dict("records"):
        analytics = analyze_holding(row)
        if analytics is None:
            continue

        rows.append(
            {
                "market": row.get("market", ""),
                "ticker": row.get("ticker", ""),
                "name": row.get("name", ""),
                "market_value": round(analytics.market_value, 2),
                "pnl_pct": round(analytics.pnl_pct, 2),
                "volatility_pct": round(analytics.volatility_pct, 2),
                "drawdown_pct": round(analytics.drawdown_pct, 2),
                "return_6m_pct": round(analytics.return_6m_pct, 2),
                "return_1y_pct": round(analytics.return_1y_pct, 2),
                "dividend_yield_pct": round(analytics.dividend_yield_pct, 2),
                "risk_level": analytics.risk_level,
                "style": analytics.style,
                "sector": analytics.sector or "미분류",
                "industry": analytics.industry or "미분류",
            }
        )

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "market",
                "ticker",
                "name",
                "market_value",
                "pnl_pct",
                "volatility_pct",
                "drawdown_pct",
                "return_6m_pct",
                "return_1y_pct",
                "dividend_yield_pct",
                "risk_level",
                "style",
            ]
        )
        return empty, {"total_value": 0.0, "top_weight": 0.0, "us_weight": 0.0, "kr_weight": 0.0}, pd.DataFrame(), pd.DataFrame()

    frame = pd.DataFrame(rows)
    total_value = float(frame["market_value"].sum())
    frame["weight_pct"] = frame["market_value"] / total_value * 100 if total_value > 0 else 0.0

    us_weight = float(frame.loc[frame["market"] == "US", "market_value"].sum() / total_value * 100) if total_value > 0 else 0.0
    kr_weight = float(frame.loc[frame["market"] == "KR", "market_value"].sum() / total_value * 100) if total_value > 0 else 0.0
    top_weight = float(frame["weight_pct"].max()) if not frame.empty else 0.0

    summary = {
        "total_value": total_value,
        "top_weight": top_weight,
        "us_weight": us_weight,
        "kr_weight": kr_weight,
        "avg_dividend_yield": float((frame["dividend_yield_pct"] * frame["weight_pct"] / 100).sum()),
        "avg_volatility": float((frame["volatility_pct"] * frame["weight_pct"] / 100).sum()),
        "growth_weight": float(frame.loc[frame["style"] == "성장", "weight_pct"].sum()),
        "dividend_weight": float(frame.loc[frame["style"] == "배당", "weight_pct"].sum()),
        "defensive_weight": float(frame.loc[frame["style"] == "안정", "weight_pct"].sum()),
        "high_risk_weight": float(frame.loc[frame["risk_level"] == "높음", "weight_pct"].sum()),
    }

    sector_summary = (
        frame.groupby("sector", dropna=False)["weight_pct"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"weight_pct": "sector_weight_pct"})
    )
    correlation = _build_correlation_table(portfolio)
    return frame.sort_values(by="weight_pct", ascending=False).reset_index(drop=True), summary, sector_summary, correlation


def build_rebalance_suggestions(
    analysis: pd.DataFrame,
    summary: dict[str, object],
    sector_summary: pd.DataFrame,
) -> pd.DataFrame:
    suggestions: list[dict[str, object]] = []

    if analysis.empty:
        return pd.DataFrame(columns=["type", "target", "action", "detail"])

    top_row = analysis.iloc[0]
    if float(summary.get("top_weight", 0)) >= 35:
        suggestions.append(
            {
                "type": "집중도",
                "target": str(top_row["ticker"]),
                "action": "비중축소",
                "detail": f"{top_row['ticker']} 비중이 {float(top_row['weight_pct']):.1f}%라 일부 축소를 검토해 보세요.",
            }
        )

    if not sector_summary.empty and float(sector_summary.iloc[0]["sector_weight_pct"]) >= 50:
        top_sector = str(sector_summary.iloc[0]["sector"])
        sector_weight = float(sector_summary.iloc[0]["sector_weight_pct"])
        suggestions.append(
            {
                "type": "섹터",
                "target": top_sector,
                "action": "분산추가",
                "detail": f"{top_sector} 섹터 비중이 {sector_weight:.1f}%라 다른 섹터로 분산하는 편이 좋습니다.",
            }
        )

    if float(summary.get("high_risk_weight", 0)) >= 45:
        risky = analysis[analysis["risk_level"] == "높음"].sort_values(by="weight_pct", ascending=False)
        if not risky.empty:
            target = str(risky.iloc[0]["ticker"])
            suggestions.append(
                {
                    "type": "위험",
                    "target": target,
                    "action": "리스크축소",
                    "detail": f"고위험 종목 비중이 높습니다. 우선 {target} 비중을 점검해 보세요.",
                }
            )

    if float(summary.get("dividend_weight", 0)) < 10:
        suggestions.append(
            {
                "type": "스타일",
                "target": "배당",
                "action": "보강",
                "detail": "배당 스타일 비중이 낮아 현금흐름형 종목을 일부 보강할 수 있습니다.",
            }
        )

    if float(summary.get("growth_weight", 0)) < 20:
        suggestions.append(
            {
                "type": "스타일",
                "target": "성장",
                "action": "보강",
                "detail": "성장 스타일 비중이 낮아 추세형 종목을 일부 보강하는 선택지도 있습니다.",
            }
        )

    if not suggestions:
        suggestions.append(
            {
                "type": "균형",
                "target": "포트폴리오",
                "action": "유지",
                "detail": "현재 기준으로 큰 리밸런싱 필요성은 크지 않아 보입니다.",
            }
        )

    return pd.DataFrame(suggestions)


def build_portfolio_outlook(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current_values: list[float] = []
    prepared_rows: list[dict[str, object]] = []

    for row in portfolio.to_dict("records"):
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        data = get_stock_data(ticker)
        if data.empty:
            continue
        latest = data.iloc[-1]
        quote = get_latest_quote(ticker)
        current_price = pd.to_numeric(quote.get("current_price", latest.get("Close")), errors="coerce")
        if pd.isna(current_price):
            current_price = pd.to_numeric(latest.get("Close", None), errors="coerce")
        quantity = _safe_float(row.get("quantity", 0))
        current_values.append(float(current_price) * quantity if not pd.isna(current_price) else 0.0)
        prepared_rows.append(row)

    total_current_value = float(sum(current_values)) if current_values else 0.0

    for row in prepared_rows:
        ticker = str(row.get("ticker", "")).strip().upper()
        market = str(row.get("market", "")).strip().upper()
        name = str(row.get("name", "") or "")
        if not ticker:
            continue

        data = get_stock_data(ticker)
        if data.empty:
            continue

        latest = data.iloc[-1]
        quote = get_latest_quote(ticker)
        event_summary = get_stock_event_summary(ticker)
        news_summary = get_stock_news_summary(ticker)

        current_price = pd.to_numeric(quote.get("current_price", latest.get("Close")), errors="coerce")
        change_pct = pd.to_numeric(quote.get("change_pct", None), errors="coerce")
        quantity = _safe_float(row.get("quantity", 0))
        avg_price = _safe_float(row.get("avg_price", 0))
        target_weight_pct = _safe_float(row.get("target_weight", 0)) * 100
        current_value = (float(current_price) if not pd.isna(current_price) else float(latest["Close"])) * quantity
        current_weight_pct = current_value / total_current_value * 100 if total_current_value > 0 else 0.0
        return_pct = ((float(current_price) - avg_price) / avg_price * 100) if avg_price > 0 and not pd.isna(current_price) else 0.0
        returns = data["Close"].pct_change().dropna()
        volatility_pct = float(returns.tail(252).std() * (252 ** 0.5) * 100) if not returns.empty else 0.0
        trailing = data["Close"].tail(252)
        drawdown_pct = float((trailing / trailing.cummax() - 1).min() * 100) if not trailing.empty else 0.0
        return_6m_pct = float(data["Close"].pct_change(126).iloc[-1] * 100) if len(data) > 126 else 0.0
        return_1y_pct = float(data["Close"].pct_change(252).iloc[-1] * 100) if len(data) > 252 else 0.0
        dividend_yield_pct = float(get_stock_dividend_yield(ticker))
        risk_level = _classify_risk(volatility_pct, drawdown_pct)
        style = _classify_style(return_6m_pct, dividend_yield_pct, volatility_pct)
        momentum_state = _classify_momentum_state(latest)
        score = 50
        reasons: list[str] = []

        if float(latest["Close"]) > float(latest["ma20"]) > float(latest["ma60"]):
            score += 18
            reasons.append("추세가 위쪽입니다.")
        elif float(latest["Close"]) < float(latest["ma20"]) < float(latest["ma60"]):
            score -= 18
            reasons.append("추세가 아래쪽입니다.")

        rsi = float(latest.get("rsi", 50) or 50)
        if 45 <= rsi <= 68:
            score += 8
            reasons.append("과열 전 구간이라 무난합니다.")
        elif rsi > 72:
            score -= 8
            reasons.append("단기 과열이 있습니다.")
        elif rsi < 35:
            score += 4
            reasons.append("많이 눌린 자리입니다.")

        if float(latest.get("macd_diff", 0) or 0) > 0:
            score += 8
            reasons.append("상승 힘이 남아 있습니다.")
        else:
            score -= 5
            reasons.append("상승 힘이 약해졌습니다.")

        rs_score = pd.to_numeric(latest.get("rs_score", None), errors="coerce")
        if not pd.isna(rs_score):
            if float(rs_score) >= 12:
                score += 10
                reasons.append("시장 대비 강한 종목입니다.")
            elif float(rs_score) <= -8:
                score -= 8
                reasons.append("시장 대비 약한 편입니다.")

        ret_20d = pd.to_numeric(latest.get("return_20d", None), errors="coerce")
        if not pd.isna(ret_20d):
            if float(ret_20d) >= 10:
                score += 6
                reasons.append("최근 한 달 흐름이 강합니다.")
            elif float(ret_20d) <= -10:
                score -= 6
                reasons.append("최근 한 달 흐름이 약합니다.")

        if risk_level == "높음":
            score -= 6
            reasons.append("변동성이 큰 편입니다.")
        elif risk_level == "낮음":
            score += 3

        if style == "배당" and dividend_yield_pct >= 3:
            score += 4
            reasons.append("배당 매력이 있습니다.")

        news_score = int(news_summary.get("news_score", 0) or 0)
        score += max(-6, min(6, news_score * 2))
        if news_summary.get("news_bias") == "긍정":
            reasons.append("최근 뉴스 흐름이 우호적입니다.")
        elif news_summary.get("news_bias") == "부정":
            reasons.append("최근 뉴스 흐름이 부담입니다.")

        event_risk = _classify_event_risk(
            event_summary.get("earnings_days_left"),
            event_summary.get("ex_dividend_days_left"),
        )
        if event_risk == "높음":
            score -= 6
            reasons.append("가까운 이벤트로 변동성이 커질 수 있습니다.")
        elif event_risk == "중간":
            score -= 2

        score = max(0, min(100, int(round(score))))
        outlook = _classify_outlook(score)
        action_hint, priority_rank = _build_outlook_action(
            score=score,
            event_risk=event_risk,
            return_pct=return_pct,
            current_weight_pct=current_weight_pct,
            target_weight_pct=target_weight_pct,
        )

        atr14 = _safe_float(latest.get("atr14", 0))
        ma20 = _safe_float(latest.get("ma20", latest.get("Close", 0)))
        ma60 = _safe_float(latest.get("ma60", latest.get("Close", 0)))
        close_price = _safe_float(latest.get("Close", 0))
        accumulate_price = min(ma20, close_price - atr14 * 0.6) if atr14 > 0 else ma20
        caution_price = min(ma60, close_price - atr14 * 1.2) if atr14 > 0 else ma60
        target_price = close_price + max(atr14 * 2.2, close_price * 0.06)
        weight_gap_pct = current_weight_pct - target_weight_pct
        weight_status = _build_weight_status(current_weight_pct, target_weight_pct)
        next_step = _build_next_step(action_hint, event_risk, momentum_state, weight_status)
        reason_summary = _build_reason_summary(
            momentum_state=momentum_state,
            risk_level=risk_level,
            style=style,
            weight_status=weight_status,
            event_risk=event_risk,
            news_bias=str(news_summary.get("news_bias", "중립")),
            return_pct=return_pct,
        )

        rows.append(
            {
                "market": market,
                "ticker": ticker,
                "name": name,
                "outlook": outlook,
                "outlook_score": score,
                "outlook_label": _plain_score_label(score),
                "action_hint": action_hint,
                "next_step": next_step,
                "priority_rank": priority_rank,
                "current_price": None if pd.isna(current_price) else float(current_price),
                "change_pct": None if pd.isna(change_pct) else float(change_pct),
                "avg_price": avg_price if avg_price > 0 else None,
                "return_pct": round(return_pct, 2) if avg_price > 0 else None,
                "quantity": quantity,
                "current_weight_pct": round(current_weight_pct, 2),
                "target_weight_pct": round(target_weight_pct, 2) if target_weight_pct > 0 else None,
                "weight_gap_pct": round(weight_gap_pct, 2) if target_weight_pct > 0 else None,
                "weight_status": weight_status,
                "style": style,
                "risk_level": risk_level,
                "momentum_state": momentum_state,
                "volatility_pct": round(volatility_pct, 2),
                "drawdown_pct": round(drawdown_pct, 2),
                "return_6m_pct": round(return_6m_pct, 2),
                "return_1y_pct": round(return_1y_pct, 2),
                "dividend_yield_pct": round(dividend_yield_pct, 2),
                "accumulate_price": round(accumulate_price, 2) if accumulate_price > 0 else None,
                "caution_price": round(caution_price, 2) if caution_price > 0 else None,
                "target_price": round(target_price, 2) if target_price > 0 else None,
                "quote_as_of": str(quote.get("as_of", "")),
                "event_risk": event_risk,
                "earnings_date": str(event_summary.get("earnings_date", "")),
                "ex_dividend_date": str(event_summary.get("ex_dividend_date", "")),
                "news_bias": str(news_summary.get("news_bias", "중립")),
                "news_count": int(news_summary.get("news_count", 0) or 0),
                "headline": str(news_summary.get("headline", "")),
                "reason_summary": reason_summary,
                "reason": " / ".join(reasons[:4]) if reasons else reason_summary or "아직 뚜렷한 강약은 크지 않습니다.",
                "event_note": str(event_summary.get("event_note", "")),
                "news_note": str(news_summary.get("news_note", "")),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "market",
                "ticker",
                "name",
                "outlook",
                "outlook_score",
                "outlook_label",
                "action_hint",
                "next_step",
                "priority_rank",
                "current_price",
                "change_pct",
                "avg_price",
                "return_pct",
                "quantity",
                "current_weight_pct",
                "target_weight_pct",
                "weight_gap_pct",
                "weight_status",
                "style",
                "risk_level",
                "momentum_state",
                "volatility_pct",
                "drawdown_pct",
                "return_6m_pct",
                "return_1y_pct",
                "dividend_yield_pct",
                "accumulate_price",
                "caution_price",
                "target_price",
                "quote_as_of",
                "event_risk",
                "earnings_date",
                "ex_dividend_date",
                "news_bias",
                "news_count",
                "headline",
                "reason_summary",
                "reason",
                "event_note",
                "news_note",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        by=["priority_rank", "outlook_score", "ticker"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
