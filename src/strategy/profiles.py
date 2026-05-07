from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.data.fetch import (
    get_intraday_stock_data,
    is_recent_price_data,
    latest_price_timestamp,
    get_stock_data,
    get_stock_dividend_yield,
    get_stock_event_summary,
    get_stock_news_summary,
    price_data_freshness_label,
    price_source_label,
)
from src.strategy.learning import apply_context_adjustment, apply_learning_adjustment, ContextAdjustment, LearningAdjustment
from src.strategy.regime import classify_market_regime
from src.strategy.universe import get_high_risk_universe, get_market_sweep_universe, get_universe


ScanProgressCallback = Callable[[dict[str, object]], None]


def _emit_progress(
    callback: ScanProgressCallback | None,
    *,
    market: str,
    scan_type: str,
    done: int,
    total: int,
    ticker: str,
    stage: str,
) -> None:
    if callback is None:
        return
    try:
        callback(
            {
                "market": market,
                "scan_type": scan_type,
                "done": done,
                "total": total,
                "ticker": ticker,
                "stage": stage,
            }
        )
    except Exception:
        return


def _with_diagnostics(frame: pd.DataFrame, diagnostics: dict[str, int]) -> pd.DataFrame:
    frame.attrs["diagnostics"] = diagnostics
    return frame


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
    if latest["rs_score"] >= 15:
        score += 14
        reasons.append("상대강도가 높아 시장 안에서 더 강한 편입니다.")
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


def _latest_float(data: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if data.empty or column not in data.columns:
        return default
    value = pd.to_numeric(data[column].iloc[-1], errors="coerce")
    return default if pd.isna(value) else float(value)


def _passes_short_term_daily_prefilter(data: pd.DataFrame) -> bool:
    if data.empty or len(data) < 80:
        return False
    close = _latest_float(data, "Close")
    ma20 = _latest_float(data, "ma20")
    ma60 = _latest_float(data, "ma60")
    rsi = _latest_float(data, "rsi", 50.0)
    return_20d = _latest_float(data, "return_20d")
    rs_score = _latest_float(data, "rs_score")
    volume_ratio = _latest_float(data, "volume_ratio", 1.0)
    high_60 = float(pd.to_numeric(data["High"].tail(60), errors="coerce").max()) if "High" in data.columns else close

    trend_ok = close > ma20 or (close > ma60 and return_20d > 0)
    momentum_ok = return_20d >= 3 or rs_score >= 8 or close >= high_60 * 0.96
    attention_ok = volume_ratio >= 1.05 or return_20d >= 6
    rsi_ok = 35 <= rsi <= 82
    return bool(trend_ok and momentum_ok and attention_ok and rsi_ok)


def _passes_high_risk_daily_prefilter(data: pd.DataFrame) -> bool:
    if data.empty or len(data) < 60:
        return False
    close = _latest_float(data, "Close")
    ma20 = _latest_float(data, "ma20")
    return_20d = _latest_float(data, "return_20d")
    return_60d = _latest_float(data, "return_60d")
    volume_ratio = _latest_float(data, "volume_ratio", 1.0)
    atr_pct = _latest_float(data, "atr_pct")
    rs_score = _latest_float(data, "rs_score")
    return bool(
        (close > ma20 and (return_20d >= 5 or rs_score >= 8))
        or volume_ratio >= 1.25
        or (atr_pct >= 4 and return_60d >= 8)
    )


def _intraday_chase_risk(intraday: pd.DataFrame) -> dict[str, object]:
    def _num(value: object, default: float = 0.0) -> float:
        numeric = pd.to_numeric(value, errors="coerce")
        return default if pd.isna(numeric) else float(numeric)

    if intraday.empty:
        return {
            "day_return_pct": 0.0,
            "from_day_high_pct": 0.0,
            "chase_penalty": 0,
            "chase_label": "보통",
            "wait_price": 0.0,
            "note": "",
        }

    latest_close = _num(intraday["Close"].iloc[-1])
    if latest_close <= 0:
        return {
            "day_return_pct": 0.0,
            "from_day_high_pct": 0.0,
            "chase_penalty": 0,
            "chase_label": "보통",
            "wait_price": 0.0,
            "note": "",
        }

    if "Datetime" in intraday.columns:
        timestamps = pd.to_datetime(intraday["Datetime"], errors="coerce")
        latest_date = timestamps.iloc[-1].date() if not pd.isna(timestamps.iloc[-1]) else None
        session = intraday[timestamps.dt.date == latest_date] if latest_date else intraday.tail(80)
    else:
        session = intraday.tail(80)
    if session.empty:
        session = intraday.tail(80)

    day_open = _num(session["Open"].iloc[0], latest_close)
    day_high = _num(pd.to_numeric(session["High"], errors="coerce").max(), latest_close)
    vwap = _num(intraday.get("vwap_proxy", pd.Series([latest_close])).iloc[-1], latest_close)
    day_return_pct = ((latest_close / day_open) - 1) * 100 if day_open > 0 else 0.0
    from_day_high_pct = ((latest_close / day_high) - 1) * 100 if day_high > 0 else 0.0

    penalty = 0
    label = "보통"
    note = ""
    if day_return_pct >= 8 and from_day_high_pct >= -1.5:
        penalty = 24
        label = "추격금지"
        note = "오늘 이미 크게 올라 고점 근처입니다. 눌림 확인 전 추격 금지."
    elif day_return_pct >= 5 and from_day_high_pct >= -2.0:
        penalty = 16
        label = "눌림대기"
        note = "당일 상승폭이 커서 바로 추격보다 눌림 진입이 낫습니다."
    elif day_return_pct >= 3:
        penalty = 8
        label = "소액만"
        note = "당일 상승폭이 있어 진입 금액을 줄이는 편이 낫습니다."

    wait_price = min(latest_close, max(vwap, day_high * 0.985))
    if label == "추격금지":
        wait_price = min(vwap * 1.005, latest_close * 0.985)
    elif label == "눌림대기":
        wait_price = min(max(vwap, day_high * 0.975), latest_close * 0.995)

    return {
        "day_return_pct": round(day_return_pct, 2),
        "from_day_high_pct": round(from_day_high_pct, 2),
        "chase_penalty": penalty,
        "chase_label": label,
        "wait_price": round(max(0.01, wait_price), 2),
        "note": note,
    }


def build_strategy_profiles(market: str, top_n: int = 8) -> dict[str, pd.DataFrame]:
    stable_rows: list[dict[str, object]] = []
    dividend_rows: list[dict[str, object]] = []
    growth_rows: list[dict[str, object]] = []

    for item in get_universe(market):
        try:
            data = get_stock_data(item["ticker"])
            if data.empty or len(data) < 180 or not is_recent_price_data(data, max_age_days=3):
                continue

            latest = data.iloc[-1]
            dividend_yield = float(get_stock_dividend_yield(item["ticker"]))
            stable_score, stable_reasons, volatility, drawdown = _stable_score(data, dividend_yield)
            dividend_score, dividend_reasons = _dividend_score(data, dividend_yield)
            growth_score, growth_reasons, annual_return = _growth_score(data)
            latest_ts = latest_price_timestamp(data)

            base = {
                "ticker": item["ticker"],
                "name": item["name"],
                "current_price": round(float(latest["Close"]), 2),
                "quote_as_of": latest_ts.strftime("%Y-%m-%d") if latest_ts is not None else "",
                "data_freshness": price_data_freshness_label(data, intraday=False),
                "price_source": price_source_label(data, intraday=False),
                "dividend_yield_pct": round(dividend_yield, 2),
                "volatility_pct": round(volatility, 2),
                "drawdown_pct": round(drawdown, 2),
                "return_60d_pct": round(float(latest["return_60d"]), 2),
                "return_1y_pct": round(annual_return, 2),
            }

            stable_rows.append({**base, "score": stable_score, "bucket": "안정형 적립", "reason": " / ".join(stable_reasons[:3])})
            dividend_rows.append({**base, "score": dividend_score, "bucket": "우량 배당", "reason": " / ".join(dividend_reasons[:3])})
            growth_rows.append({**base, "score": growth_score, "bucket": "고위험 성장", "reason": " / ".join(growth_reasons[:3])})
        except Exception:
            continue

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
    universe: list[dict[str, str]] | None = None,
    scan_limit: int | None = None,
    progress_callback: ScanProgressCallback | None = None,
    learning_adjustments: dict[tuple[str, str, str], LearningAdjustment] | None = None,
    event_adjustments: dict[str, ContextAdjustment] | None = None,
    news_adjustments: dict[str, ContextAdjustment] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fallback_rows: list[dict[str, object]] = []
    regime = classify_market_regime(market)
    candidates = universe if universe is not None else get_market_sweep_universe(market)
    if scan_limit is not None:
        candidates = candidates[: max(1, int(scan_limit))]
    diagnostics = {
        "scanned": 0,
        "daily_pass": 0,
        "intraday_pass": 0,
        "selected": 0,
        "fallback_selected": 0,
        "errors": 0,
    }

    total = len(candidates)
    for index, item in enumerate(candidates, start=1):
        try:
            diagnostics["scanned"] += 1
            _emit_progress(
                progress_callback,
                market=market,
                scan_type="short_term_trade",
                done=index,
                total=total,
                ticker=item["ticker"],
                stage="일봉 선별",
            )
            daily = get_stock_data(item["ticker"])
            if not is_recent_price_data(daily, max_age_days=3) or not _passes_short_term_daily_prefilter(daily):
                continue
            diagnostics["daily_pass"] += 1
            daily_latest = daily.iloc[-1]
            daily_ts = latest_price_timestamp(daily)
            close = float(daily_latest.get("Close", 0) or 0)
            atr = float(daily_latest.get("atr14", 0) or 0)
            atr_floor = max(atr, close * 0.025)
            daily_score = 45
            daily_reasons: list[str] = ["분봉 확인 전 일봉 기준 예비 단타 후보입니다."]
            if close > float(daily_latest.get("ma20", close) or close):
                daily_score += 12
                daily_reasons.append("20일선 위에서 버티고 있습니다.")
            if float(daily_latest.get("return_20d", 0) or 0) >= 3:
                daily_score += 10
                daily_reasons.append("최근 20일 흐름이 플러스입니다.")
            if float(daily_latest.get("rs_score", 0) or 0) >= 8:
                daily_score += 8
                daily_reasons.append("상대강도가 시장보다 나은 편입니다.")
            if float(daily_latest.get("volume_ratio", 1) or 1) >= 1.05:
                daily_score += 8
                daily_reasons.append("일봉 거래량 관심이 붙었습니다.")
            daily_score = min(100, max(0, int(daily_score + regime.adjustment)))
            if close > 0 and daily_score >= max(45, min_score - 15):
                fallback_rows.append(
                    {
                        "ticker": item["ticker"],
                        "name": item["name"],
                        "setup": "일봉예비",
                        "score": daily_score,
                        "entry_price": round(close, 2),
                        "current_price": round(close, 2),
                        "quote_as_of": daily_ts.strftime("%Y-%m-%d") if daily_ts is not None else "",
                        "data_freshness": price_data_freshness_label(daily, intraday=False),
                        "price_source": price_source_label(daily, intraday=False),
                        "stop_loss": round(max(0.01, close - atr_floor * 1.2), 2),
                        "target_1": round(close + atr_floor * 1.8, 2),
                        "target_2": round(close + atr_floor * 2.8, 2),
                        "short_return_pct": round(float(daily_latest.get("return_5d", 0) or 0), 2),
                        "volume_ratio": round(float(daily_latest.get("volume_ratio", 1) or 1), 2),
                        "atr_pct": round(float(daily_latest.get("atr_pct", 0) or 0), 2),
                        "rs_score": round(float(daily_latest.get("rs_score", 0) or 0), 2),
                        "risk_reward_1": 1.5,
                        "exit_rule": "분봉 거래량 확인 전에는 소액 또는 관찰만",
                        "regime": regime.regime,
                        "regime_delta": regime.adjustment,
                        "context_delta": 0,
                        "event_risk": "",
                        "event_note": "",
                        "earnings_date": "",
                        "ex_dividend_date": "",
                        "news_bias": "중립",
                        "news_score": 0,
                        "news_count": 0,
                        "learning_delta": 0,
                        "data_basis": "일봉 예비",
                        "reason": " / ".join(([regime.note] if regime.note else []) + daily_reasons[:4]),
                    }
                )
            intraday = get_intraday_stock_data(item["ticker"], period="5d", interval=interval)
            if daily.empty or intraday.empty or len(intraday) < 25 or not is_recent_price_data(intraday, max_age_days=1):
                continue
            diagnostics["intraday_pass"] += 1

            daily_latest = daily.iloc[-1]
            intra_latest = intraday.iloc[-1]
            intra_ts = latest_price_timestamp(intraday)
            event_summary = get_stock_event_summary(item["ticker"])
            news_summary = get_stock_news_summary(item["ticker"])
            recent_high = float(intraday["session_high_20"].iloc[-2])
            recent_low = float(intraday["session_low_20"].iloc[-2])
            chase = _intraday_chase_risk(intraday)
            current_price = float(intra_latest["Close"])
            if str(chase["chase_label"]) in {"추격금지", "눌림대기"}:
                entry_price = min(max(float(chase["wait_price"]), recent_high * 0.985), current_price)
            else:
                entry_price = max(current_price, recent_high)
            stop_loss = min(float(intra_latest["vwap_proxy"]), recent_low)
            if stop_loss >= entry_price:
                stop_loss = entry_price * 0.97
            atr = float(daily_latest.get("atr14", 0) or 0)
            risk_per_share = max(entry_price - stop_loss, atr, entry_price * 0.008)
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

            if int(chase["chase_penalty"]) > 0:
                score -= int(chase["chase_penalty"])
                reasons.append(str(chase["note"]))

            event_risk = str(event_summary.get("event_risk", "") or "")
            news_bias = str(news_summary.get("news_bias", "중립") or "중립")
            news_score = int(news_summary.get("news_score", 0) or 0)
            if event_risk == "높음":
                score -= 8
                reasons.append("가까운 일정이 있어 단타 변동성이 커질 수 있습니다.")
            elif event_risk == "중간":
                score -= 3

            if news_bias == "긍정":
                score += min(6, max(2, news_score * 2))
                reasons.append("최근 뉴스 흐름이 우호적입니다.")
            elif news_bias == "부정":
                score -= min(6, max(2, abs(news_score) * 2))
                reasons.append("최근 뉴스 흐름이 부담이라 짧게 보는 편이 좋습니다.")

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
                    "current_price": round(current_price, 2),
                    "quote_as_of": intra_ts.strftime("%Y-%m-%d %H:%M") if intra_ts is not None else "",
                    "data_freshness": price_data_freshness_label(intraday, intraday=True),
                    "price_source": price_source_label(intraday, intraday=True),
                    "entry_price": round(entry_price, 2),
                    "stop_loss": round(stop_loss, 2),
                    "target_1": round(target_1, 2),
                    "target_2": round(target_2, 2),
                    "target_3": round(entry_price + risk_per_share * 3.5, 2),
                    "entry_range": f"{round(entry_price * 0.992, 2)} ~ {round(entry_price * 1.003, 2)}",
                    "short_return_pct": round(float(intra_latest["short_return_pct"]), 2),
                    "day_return_pct": chase["day_return_pct"],
                    "from_day_high_pct": chase["from_day_high_pct"],
                    "chase_risk": chase["chase_label"],
                    "chase_penalty": -int(chase["chase_penalty"]),
                    "volume_ratio": round(float(intra_latest["volume_ratio"]), 2),
                    "atr_pct": round(float(daily_latest.get("atr_pct", 0) or 0), 2),
                    "rs_score": round(float(daily_latest.get("rs_score", 0) or 0), 2),
                    "risk_reward_1": round((target_1 - entry_price) / max(entry_price - stop_loss, 0.01), 2),
                    "exit_rule": exit_rule,
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
                    "price_basis": (
                        f"현재가 {current_price:.2f}, 진입 {entry_price:.2f}, 손절 {stop_loss:.2f}. "
                        f"당일 {float(chase['day_return_pct']):+.2f}%, 고점대비 {float(chase['from_day_high_pct']):+.2f}%, "
                        f"추격위험 {chase['chase_label']}."
                    ),
                    "reason": " / ".join(
                        ([regime.note] if regime.note else [])
                        + ([context_note] if context_note else [])
                        + ([learning_note] if learning_note else [])
                        + reasons[:4]
                    ),
                }
            )
            diagnostics["selected"] += 1
        except Exception:
            diagnostics["errors"] += 1
            continue

    if not rows:
        if fallback_rows:
            fallback = pd.DataFrame(fallback_rows).sort_values(
                by=["score", "volume_ratio", "short_return_pct"], ascending=[False, False, False]
            ).head(top_n).reset_index(drop=True)
            diagnostics["fallback_selected"] = len(fallback)
            return _with_diagnostics(fallback, diagnostics)
        return _with_diagnostics(
            pd.DataFrame(
                columns=[
                    "ticker",
                    "name",
                    "setup",
                    "score",
                    "current_price",
                    "quote_as_of",
                    "data_freshness",
                    "price_source",
                    "entry_price",
                    "stop_loss",
                    "target_1",
                    "target_2",
                    "short_return_pct",
                    "volume_ratio",
                    "atr_pct",
                    "rs_score",
                    "risk_reward_1",
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
                    "exit_rule",
                    "data_basis",
                    "reason",
                ]
            ),
            diagnostics,
        )

    result = pd.DataFrame(rows).sort_values(
        by=["score", "volume_ratio", "short_return_pct"], ascending=[False, False, False]
    ).head(top_n).reset_index(drop=True)
    return _with_diagnostics(result, diagnostics)


def build_high_risk_trade_candidates(
    market: str,
    top_n: int = 8,
    interval: str = "5m",
    min_score: int = 60,
    universe: list[dict[str, str]] | None = None,
    scan_limit: int | None = None,
    progress_callback: ScanProgressCallback | None = None,
    learning_adjustments: dict[tuple[str, str, str], LearningAdjustment] | None = None,
    event_adjustments: dict[str, ContextAdjustment] | None = None,
    news_adjustments: dict[str, ContextAdjustment] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fallback_rows: list[dict[str, object]] = []
    regime = classify_market_regime(market)
    candidates = universe if universe is not None else get_high_risk_universe(market)
    if scan_limit is not None:
        candidates = candidates[: max(1, int(scan_limit))]
    diagnostics = {
        "scanned": 0,
        "daily_pass": 0,
        "intraday_pass": 0,
        "selected": 0,
        "fallback_selected": 0,
        "errors": 0,
    }

    total = len(candidates)
    for index, item in enumerate(candidates, start=1):
        try:
            diagnostics["scanned"] += 1
            _emit_progress(
                progress_callback,
                market=market,
                scan_type="high_risk_trade",
                done=index,
                total=total,
                ticker=item["ticker"],
                stage="고위험 일봉 선별",
            )
            daily = get_stock_data(item["ticker"])
            if not is_recent_price_data(daily, max_age_days=3) or not _passes_high_risk_daily_prefilter(daily):
                continue
            diagnostics["daily_pass"] += 1
            daily_latest = daily.iloc[-1]
            daily_ts = latest_price_timestamp(daily)
            close = float(daily_latest.get("Close", 0) or 0)
            atr = float(daily_latest.get("atr14", 0) or 0)
            atr_floor = max(atr, close * 0.04)
            daily_score = 42
            daily_reasons: list[str] = ["분봉 확인 전 고위험 일봉 예비 후보입니다."]
            if float(daily_latest.get("return_20d", 0) or 0) >= 5:
                daily_score += 12
                daily_reasons.append("최근 상승 탄력이 있습니다.")
            if float(daily_latest.get("volume_ratio", 1) or 1) >= 1.25:
                daily_score += 12
                daily_reasons.append("일봉 거래량이 늘었습니다.")
            if float(daily_latest.get("atr_pct", 0) or 0) >= 4:
                daily_score += 8
                daily_reasons.append("변동성이 커 단타성 움직임이 나올 수 있습니다.")
            if float(daily_latest.get("rs_score", 0) or 0) >= 8:
                daily_score += 8
                daily_reasons.append("시장 대비 상대강도가 있습니다.")
            daily_score = min(100, max(0, int(daily_score + regime.adjustment)))
            if close > 0 and daily_score >= max(42, min_score - 18):
                fallback_rows.append(
                    {
                        "ticker": item["ticker"],
                        "name": item["name"],
                        "setup": "고위험일봉예비",
                        "score": daily_score,
                        "current_price": round(close, 2),
                        "quote_as_of": daily_ts.strftime("%Y-%m-%d") if daily_ts is not None else "",
                        "data_freshness": price_data_freshness_label(daily, intraday=False),
                        "price_source": price_source_label(daily, intraday=False),
                        "entry_price": round(close, 2),
                        "stop_loss": round(max(0.01, close - atr_floor * 1.0), 2),
                        "target_1": round(close + atr_floor * 2.0, 2),
                        "target_2": round(close + atr_floor * 3.2, 2),
                        "short_return_pct": round(float(daily_latest.get("return_5d", 0) or 0), 2),
                        "volume_ratio": round(float(daily_latest.get("volume_ratio", 1) or 1), 2),
                        "atr_pct": round(float(daily_latest.get("atr_pct", 0) or 0), 2),
                        "rs_score": round(float(daily_latest.get("rs_score", 0) or 0), 2),
                        "risk_reward_1": 2.0,
                        "regime": regime.regime,
                        "regime_delta": regime.adjustment,
                        "context_delta": 0,
                        "event_risk": "",
                        "event_note": "",
                        "earnings_date": "",
                        "ex_dividend_date": "",
                        "news_bias": "중립",
                        "news_score": 0,
                        "news_count": 0,
                        "learning_delta": 0,
                        "exit_rule": "분봉 확인 전에는 추격 금지, 관찰 우선",
                        "risk_level": "높음",
                        "data_basis": "일봉 예비",
                        "reason": " / ".join(([regime.note] if regime.note else []) + daily_reasons[:4]),
                    }
                )
            intraday = get_intraday_stock_data(item["ticker"], period="5d", interval=interval)
            if daily.empty or intraday.empty or len(intraday) < 25 or not is_recent_price_data(intraday, max_age_days=1):
                continue
            diagnostics["intraday_pass"] += 1

            daily_latest = daily.iloc[-1]
            intra_latest = intraday.iloc[-1]
            intra_ts = latest_price_timestamp(intraday)
            event_summary = get_stock_event_summary(item["ticker"])
            news_summary = get_stock_news_summary(item["ticker"])
            recent_high = float(intraday["session_high_20"].iloc[-2])
            recent_low = float(intraday["session_low_20"].iloc[-2])
            chase = _intraday_chase_risk(intraday)
            current_price = float(intra_latest["Close"])

            if str(chase["chase_label"]) in {"추격금지", "눌림대기"}:
                entry_price = min(max(float(chase["wait_price"]), recent_high * 0.98), current_price)
            else:
                entry_price = max(current_price, recent_high)
            stop_loss = min(float(intra_latest["vwap_proxy"]), recent_low)
            if stop_loss >= entry_price:
                stop_loss = entry_price * 0.95

            atr = float(daily_latest.get("atr14", 0) or 0)
            risk_per_share = max(entry_price - stop_loss, atr, entry_price * 0.015)
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

            if int(chase["chase_penalty"]) > 0:
                score -= int(chase["chase_penalty"]) + 4
                reasons.append(str(chase["note"]))

            setup = "고위험추격"
            if float(intra_latest["Close"]) >= recent_high and float(intra_latest["volume_ratio"]) >= 2.8:
                setup = "급등추격"
                exit_rule = "초강세 구간이라 짧게 보고 1차 목표가 근처에서 빠르게 익절"
            elif float(intra_latest["Close"]) > float(intra_latest["vwap_proxy"]):
                setup = "고위험VWAP지지"

            event_risk = str(event_summary.get("event_risk", "") or "")
            news_bias = str(news_summary.get("news_bias", "중립") or "중립")
            news_score = int(news_summary.get("news_score", 0) or 0)
            if event_risk == "높음":
                score -= 10
                reasons.append("이벤트가 가까워 변동성이 과하게 커질 수 있습니다.")
            elif event_risk == "중간":
                score -= 4

            if news_bias == "긍정":
                score += min(7, max(2, news_score * 2))
                reasons.append("최근 뉴스 흐름이 강세 쪽입니다.")
            elif news_bias == "부정":
                score -= min(7, max(2, abs(news_score) * 2))
                reasons.append("최근 뉴스 흐름이 약세 쪽이라 주의가 필요합니다.")

            score = min(100, max(0, int(score + regime.adjustment)))
            score, learning_delta, learning_note = apply_learning_adjustment(
                base_score=score,
                scan_type="high_risk_trade",
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
                    "current_price": round(current_price, 2),
                    "quote_as_of": intra_ts.strftime("%Y-%m-%d %H:%M") if intra_ts is not None else "",
                    "data_freshness": price_data_freshness_label(intraday, intraday=True),
                    "price_source": price_source_label(intraday, intraday=True),
                    "entry_price": round(entry_price, 2),
                    "stop_loss": round(stop_loss, 2),
                    "target_1": round(target_1, 2),
                    "target_2": round(target_2, 2),
                    "target_3": round(entry_price + risk_per_share * 5.0, 2),
                    "entry_range": f"{round(entry_price * 0.985, 2)} ~ {round(entry_price * 1.002, 2)}",
                    "short_return_pct": round(float(intra_latest["short_return_pct"]), 2),
                    "day_return_pct": chase["day_return_pct"],
                    "from_day_high_pct": chase["from_day_high_pct"],
                    "chase_risk": chase["chase_label"],
                    "chase_penalty": -(int(chase["chase_penalty"]) + (4 if int(chase["chase_penalty"]) > 0 else 0)),
                    "volume_ratio": round(float(intra_latest["volume_ratio"]), 2),
                    "atr_pct": round(float(daily_latest.get("atr_pct", 0) or 0), 2),
                    "rs_score": round(float(daily_latest.get("rs_score", 0) or 0), 2),
                    "risk_reward_1": round((target_1 - entry_price) / max(entry_price - stop_loss, 0.01), 2),
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
                    "exit_rule": exit_rule,
                    "risk_level": "매우높음",
                    "price_basis": (
                        f"현재가 {current_price:.2f}, 진입 {entry_price:.2f}, 손절 {stop_loss:.2f}. "
                        f"당일 {float(chase['day_return_pct']):+.2f}%, 고점대비 {float(chase['from_day_high_pct']):+.2f}%, "
                        f"추격위험 {chase['chase_label']}."
                    ),
                    "reason": " / ".join(
                        ([regime.note] if regime.note else [])
                        + ([context_note] if context_note else [])
                        + ([learning_note] if learning_note else [])
                        + reasons[:4]
                    ),
                }
            )
            diagnostics["selected"] += 1
        except Exception:
            diagnostics["errors"] += 1
            continue

    if not rows:
        if fallback_rows:
            fallback = pd.DataFrame(fallback_rows).sort_values(
                by=["score", "volume_ratio", "short_return_pct"], ascending=[False, False, False]
            ).head(top_n).reset_index(drop=True)
            diagnostics["fallback_selected"] = len(fallback)
            return _with_diagnostics(fallback, diagnostics)
        return _with_diagnostics(
            pd.DataFrame(
                columns=[
                    "ticker",
                    "name",
                    "setup",
                    "score",
                    "current_price",
                    "quote_as_of",
                    "data_freshness",
                    "price_source",
                    "entry_price",
                    "stop_loss",
                    "target_1",
                    "target_2",
                    "short_return_pct",
                    "volume_ratio",
                    "atr_pct",
                    "rs_score",
                    "risk_reward_1",
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
                    "exit_rule",
                    "risk_level",
                    "data_basis",
                    "reason",
                ]
            ),
            diagnostics,
        )

    result = pd.DataFrame(rows).sort_values(
        by=["score", "volume_ratio", "short_return_pct"], ascending=[False, False, False]
    ).head(top_n).reset_index(drop=True)
    return _with_diagnostics(result, diagnostics)
