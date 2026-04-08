from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.fetch import get_stock_dividend_yield, get_stock_data, get_stock_profile


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
