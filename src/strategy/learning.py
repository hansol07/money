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


def _clip_adjustment(value: float, lower: int = -12, upper: int = 12) -> int:
    return int(max(lower, min(upper, round(value))))


def build_learning_adjustments(limit: int = 400, min_samples: int = 3) -> tuple[dict[tuple[str, str, str], LearningAdjustment], pd.DataFrame]:
    _, summary, _, _, pattern_stats = evaluate_scan_history(limit=limit)
    adjustments: dict[tuple[str, str, str], LearningAdjustment] = {}
    rows: list[dict[str, object]] = []

    if summary.empty:
        return adjustments, pd.DataFrame()

    market_bias_map: dict[tuple[str, str], float] = {}
    for _, row in summary.iterrows():
        scan_type = str(row.get("scan_type", ""))
        market = str(row.get("market", ""))
        picks = int(row.get("picks", 0) or 0)
        if picks < min_samples:
            continue

        avg_ret_20d = float(row.get("avg_ret_20d_pct", 0) or 0)
        hit_rate_20d = float(row.get("hit_rate_20d_pct", 0) or 0)
        market_bias = _clip_adjustment(avg_ret_20d * 0.25 + (hit_rate_20d - 50) * 0.08, lower=-6, upper=6)
        market_bias_map[(scan_type, market)] = market_bias

    if not pattern_stats.empty:
        for _, row in pattern_stats.iterrows():
            scan_type = str(row.get("scan_type", ""))
            market = str(row.get("market", ""))
            setup = str(row.get("setup", "") or "")
            picks = int(row.get("picks", 0) or 0)
            if picks < min_samples:
                continue

            avg_ret_20d = float(row.get("avg_ret_20d_pct", 0) or 0)
            hit_rate_20d = float(row.get("hit_rate_20d_pct", 0) or 0)
            strong_success = float(row.get("strong_success_20d", 0) or 0)
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
    return adjustments, adjustment_df


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
