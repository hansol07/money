from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.strategy.tracker import evaluate_scan_history


@dataclass(slots=True)
class LearningAdjustment:
    scan_type: str
    market: str
    setup: str
    adjustment: int
    confidence: float
    note: str
    samples: int


@dataclass(slots=True)
class ContextAdjustment:
    key: str
    adjustment: int
    confidence: float
    note: str
    samples: int


def _clip_adjustment(value: float, lower: int = -12, upper: int = 12) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return 0
    value = float(numeric)
    return int(max(lower, min(upper, round(value))))


def _safe_float(value: object, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return float(numeric)


def _safe_int(value: object, default: int = 0) -> int:
    return int(round(_safe_float(value, float(default))))


def _build_context_adjustments(
    detail: pd.DataFrame,
    *,
    min_samples: int,
) -> tuple[dict[str, ContextAdjustment], dict[str, ContextAdjustment], list[dict[str, object]]]:
    if detail.empty:
        return {}, {}, []

    mature = detail[detail["status"] == "평가완료"].copy()
    if mature.empty:
        return {}, {}, []
    defaults = {
        "event_risk": "",
        "news_bias": "",
        "news_score": 0.0,
        "ret_20d_pct": 0.0,
        "hit_20d": False,
        "path_5d": "",
    }
    for column, default in defaults.items():
        if column not in mature.columns:
            mature[column] = default

    event_adjustments: dict[str, ContextAdjustment] = {}
    news_adjustments: dict[str, ContextAdjustment] = {}
    rows: list[dict[str, object]] = []

    if "event_risk" in mature.columns:
        grouped = (
            mature.groupby("event_risk", dropna=False)
            .agg(
                picks=("ticker", "count"),
                avg_ret_20d_pct=("ret_20d_pct", "mean"),
                hit_rate_20d_pct=("hit_20d", lambda s: s.mean() * 100),
                target_first_5d_pct=("path_5d", lambda s: ((s == "목표가 선도달") | (s == "목표가 도달")).mean() * 100),
                stop_first_5d_pct=("path_5d", lambda s: ((s == "손절 선도달") | (s == "손절 도달")).mean() * 100),
            )
            .reset_index()
        )
        for _, row in grouped.iterrows():
            event_risk = str(row.get("event_risk", "") or "")
            picks = _safe_int(row.get("picks", 0))
            if not event_risk or picks < min_samples:
                continue
            raw = (
                _safe_float(row.get("avg_ret_20d_pct", 0)) * 0.18
                + (_safe_float(row.get("hit_rate_20d_pct", 0)) - 50) * 0.06
                + (_safe_float(row.get("target_first_5d_pct", 0)) - 45) * 0.03
                - _safe_float(row.get("stop_first_5d_pct", 0)) * 0.02
            )
            adjustment = _clip_adjustment(raw, lower=-6, upper=6)
            confidence = min(1.0, picks / 12)
            note = f"{event_risk} 이벤트 리스크 최근 성과 반영"
            event_adjustments[event_risk] = ContextAdjustment(
                key=event_risk,
                adjustment=adjustment,
                confidence=round(confidence, 2),
                note=note,
                samples=picks,
            )
            rows.append(
                {
                    "context_type": "event_risk",
                    "context_key": event_risk,
                    "samples": picks,
                    "adjustment": adjustment,
                    "confidence": round(confidence, 2),
                    "note": note,
                }
            )

    if "news_bias" in mature.columns:
        grouped = (
            mature.groupby("news_bias", dropna=False)
            .agg(
                picks=("ticker", "count"),
                avg_news_score=("news_score", "mean"),
                avg_ret_20d_pct=("ret_20d_pct", "mean"),
                hit_rate_20d_pct=("hit_20d", lambda s: s.mean() * 100),
                target_first_5d_pct=("path_5d", lambda s: ((s == "목표가 선도달") | (s == "목표가 도달")).mean() * 100),
                stop_first_5d_pct=("path_5d", lambda s: ((s == "손절 선도달") | (s == "손절 도달")).mean() * 100),
            )
            .reset_index()
        )
        for _, row in grouped.iterrows():
            news_bias = str(row.get("news_bias", "") or "")
            picks = _safe_int(row.get("picks", 0))
            if not news_bias or picks < min_samples:
                continue
            raw = (
                _safe_float(row.get("avg_ret_20d_pct", 0)) * 0.16
                + (_safe_float(row.get("hit_rate_20d_pct", 0)) - 50) * 0.05
                + (_safe_float(row.get("target_first_5d_pct", 0)) - 45) * 0.025
                - _safe_float(row.get("stop_first_5d_pct", 0)) * 0.018
            )
            adjustment = _clip_adjustment(raw, lower=-5, upper=5)
            confidence = min(1.0, picks / 12)
            note = f"{news_bias} 뉴스 흐름 최근 성과 반영"
            news_adjustments[news_bias] = ContextAdjustment(
                key=news_bias,
                adjustment=adjustment,
                confidence=round(confidence, 2),
                note=note,
                samples=picks,
            )
            rows.append(
                {
                    "context_type": "news_bias",
                    "context_key": news_bias,
                    "samples": picks,
                    "adjustment": adjustment,
                    "confidence": round(confidence, 2),
                    "note": note,
                }
            )

    return event_adjustments, news_adjustments, rows


def build_learning_adjustments(
    limit: int = 400,
    min_samples: int = 3,
) -> tuple[
    dict[tuple[str, str, str], LearningAdjustment],
    pd.DataFrame,
    dict[str, ContextAdjustment],
    dict[str, ContextAdjustment],
    pd.DataFrame,
]:
    detail, summary, _, _, pattern_stats = evaluate_scan_history(limit=limit)
    adjustments: dict[tuple[str, str, str], LearningAdjustment] = {}
    rows: list[dict[str, object]] = []

    if summary.empty:
        empty = pd.DataFrame()
        return adjustments, empty, {}, {}, empty

    market_bias_map: dict[tuple[str, str], float] = {}
    for _, row in summary.iterrows():
        scan_type = str(row.get("scan_type", ""))
        market = str(row.get("market", ""))
        picks = _safe_int(row.get("picks", 0))
        if picks < min_samples:
            continue

        avg_ret_20d = _safe_float(row.get("avg_ret_20d_pct", 0))
        hit_rate_20d = _safe_float(row.get("hit_rate_20d_pct", 0))
        market_bias = _clip_adjustment(avg_ret_20d * 0.25 + (hit_rate_20d - 50) * 0.08, lower=-6, upper=6)
        market_bias_map[(scan_type, market)] = market_bias

    if not pattern_stats.empty:
        for _, row in pattern_stats.iterrows():
            scan_type = str(row.get("scan_type", ""))
            market = str(row.get("market", ""))
            setup = str(row.get("setup", "") or "")
            picks = _safe_int(row.get("picks", 0))
            if picks < min_samples:
                continue

            avg_ret_20d = _safe_float(row.get("avg_ret_20d_pct", 0))
            hit_rate_20d = _safe_float(row.get("hit_rate_20d_pct", 0))
            strong_success = _safe_float(row.get("strong_success_20d", 0))
            market_bias = market_bias_map.get((scan_type, market), 0.0)
            raw = avg_ret_20d * 0.35 + (hit_rate_20d - 50) * 0.12 + strong_success * 0.03 + market_bias
            adjustment = _clip_adjustment(raw)
            confidence = min(1.0, picks / 12)

            if adjustment >= 5:
                note = "최근 성과가 좋아 가산점을 줍니다."
            elif adjustment <= -5:
                note = "최근 성과가 약해 보수적으로 봅니다."
            else:
                note = "최근 성과가 무난해 큰 보정은 없습니다."

            record = LearningAdjustment(
                scan_type=scan_type,
                market=market,
                setup=setup,
                adjustment=adjustment,
                confidence=round(confidence, 2),
                note=note,
                samples=picks,
            )
            adjustments[(scan_type, market, setup)] = record
            rows.append(
                {
                    "scan_type": scan_type,
                    "market": market,
                    "setup": setup,
                    "samples": picks,
                    "adjustment": adjustment,
                    "confidence": round(confidence, 2),
                    "note": note,
                }
            )

    adjustment_df = pd.DataFrame(rows)
    if not adjustment_df.empty:
        adjustment_df = adjustment_df.sort_values(
            by=["adjustment", "samples", "confidence"], ascending=[False, False, False]
        ).reset_index(drop=True)
    event_adjustments, news_adjustments, context_rows = _build_context_adjustments(detail, min_samples=min_samples)
    context_df = pd.DataFrame(context_rows)
    if not context_df.empty:
        context_df = context_df.sort_values(
            by=["context_type", "adjustment", "samples"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
    return adjustments, adjustment_df, event_adjustments, news_adjustments, context_df


def apply_learning_adjustment(
    *,
    base_score: int,
    scan_type: str,
    market: str,
    setup: str,
    adjustments: dict[tuple[str, str, str], LearningAdjustment] | None,
) -> tuple[int, int, str]:
    if not adjustments:
        return base_score, 0, ""

    key = (scan_type, market, setup or "")
    adjustment = adjustments.get(key)
    if adjustment is None:
        return base_score, 0, ""

    final_score = max(0, min(100, base_score + adjustment.adjustment))
    return final_score, adjustment.adjustment, adjustment.note


def apply_context_adjustment(
    *,
    base_score: int,
    event_risk: str,
    news_bias: str,
    event_adjustments: dict[str, ContextAdjustment] | None,
    news_adjustments: dict[str, ContextAdjustment] | None,
) -> tuple[int, int, str]:
    total_delta = 0
    notes: list[str] = []

    event_record = (event_adjustments or {}).get(event_risk or "")
    if event_record is not None:
        total_delta += int(event_record.adjustment)
        notes.append(event_record.note)

    news_record = (news_adjustments or {}).get(news_bias or "")
    if news_record is not None:
        total_delta += int(news_record.adjustment)
        notes.append(news_record.note)

    final_score = max(0, min(100, base_score + total_delta))
    return final_score, total_delta, " / ".join(notes[:2])
