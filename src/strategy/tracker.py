from __future__ import annotations

import pandas as pd

from src.data.fetch import get_stock_data
from src.storage.local_store import load_recent_scan_history
from src.strategy.universe import is_tradable_ticker


def _get_future_rows(data: pd.DataFrame, captured_at: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    captured_date = pd.to_datetime(captured_at, errors="coerce")
    if pd.isna(captured_date):
        return pd.DataFrame()

    date_series = pd.to_datetime(data["Date"], errors="coerce")
    compare_date = pd.Timestamp(captured_date).normalize().to_datetime64()
    rows = data[date_series.to_numpy(dtype="datetime64[ns]") >= compare_date].copy()
    if rows.empty:
        return pd.DataFrame()
    return rows.reset_index(drop=True)


def _find_forward_return(future_rows: pd.DataFrame, base_price: float, days: int) -> float | None:
    if future_rows.empty or base_price <= 0:
        return None
    if len(future_rows) <= days:
        return None
    future_price = float(future_rows.iloc[days]["Close"])
    return (future_price - base_price) / base_price * 100


def _find_forward_extremes(future_rows: pd.DataFrame, base_price: float, days: int) -> tuple[float | None, float | None]:
    if future_rows.empty or base_price <= 0:
        return None, None

    window = future_rows.iloc[: min(len(future_rows), days + 1)]
    if window.empty:
        return None, None

    max_forward = (float(window["High"].max()) - base_price) / base_price * 100
    min_forward = (float(window["Low"].min()) - base_price) / base_price * 100
    return max_forward, min_forward


def _find_trigger_path(
    future_rows: pd.DataFrame,
    stop_loss: float | None,
    target_1: float | None,
    days: int,
) -> tuple[bool, bool, str]:
    if future_rows.empty:
        return False, False, "평가대기"

    window = future_rows.iloc[: min(len(future_rows), days + 1)].reset_index(drop=True)
    if window.empty:
        return False, False, "평가대기"

    target_index: int | None = None
    stop_index: int | None = None

    if target_1 is not None and target_1 > 0:
        target_hits = window.index[window["High"] >= target_1].tolist()
        if target_hits:
            target_index = int(target_hits[0])

    if stop_loss is not None and stop_loss > 0:
        stop_hits = window.index[window["Low"] <= stop_loss].tolist()
        if stop_hits:
            stop_index = int(stop_hits[0])

    target_hit = target_index is not None
    stop_hit = stop_index is not None

    if target_hit and stop_hit:
        if target_index <= stop_index:
            return True, True, "목표가 선도달"
        return True, True, "손절 선도달"
    if target_hit:
        return True, False, "목표가 도달"
    if stop_hit:
        return False, True, "손절 도달"
    return False, False, "미도달"


def _label_period(ret_value: float | None, target_hit: bool, stop_hit: bool, path_label: str, strong_cutoff: float) -> str:
    if ret_value is None:
        return "평가대기"
    if path_label == "목표가 선도달":
        return "목표선도달"
    if path_label == "손절 선도달":
        return "손절선도달"
    if target_hit:
        return "목표달성"
    if stop_hit:
        return "손절"
    if ret_value >= strong_cutoff:
        return "강한성공"
    if ret_value > 0:
        return "성공"
    return "실패"


def _build_detail(limit: int = 200) -> pd.DataFrame:
    history = load_recent_scan_history(limit=limit)
    rows: list[dict[str, object]] = []

    for entry in history:
        scan_type = str(entry.get("scan_type", ""))
        market = str(entry.get("market", ""))
        snapshot_id = str(entry.get("snapshot_id", ""))
        saved_at = str(entry.get("saved_at", ""))

        for row in entry.get("rows", []):
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker or not is_tradable_ticker(ticker):
                continue

            data = get_stock_data(ticker)
            base_price = float(row.get("current_price", 0) or row.get("entry_price", 0) or 0)
            captured_at = str(row.get("captured_at", "") or saved_at)
            score = float(row.get("score", 0) or 0)
            stop_loss = pd.to_numeric(row.get("stop_loss", None), errors="coerce")
            target_1 = pd.to_numeric(row.get("target_1", None), errors="coerce")

            future_rows = _get_future_rows(data, captured_at)
            ret_1d = _find_forward_return(future_rows, base_price, 1)
            ret_3d = _find_forward_return(future_rows, base_price, 3)
            ret_5d = _find_forward_return(future_rows, base_price, 5)
            ret_20d = _find_forward_return(future_rows, base_price, 20)

            max_5d, min_5d = _find_forward_extremes(future_rows, base_price, 5)
            max_20d, min_20d = _find_forward_extremes(future_rows, base_price, 20)

            target_hit_5d, stop_hit_5d, path_5d = _find_trigger_path(
                future_rows,
                None if pd.isna(stop_loss) else float(stop_loss),
                None if pd.isna(target_1) else float(target_1),
                5,
            )
            target_hit_20d, stop_hit_20d, path_20d = _find_trigger_path(
                future_rows,
                None if pd.isna(stop_loss) else float(stop_loss),
                None if pd.isna(target_1) else float(target_1),
                20,
            )

            available_returns = [value for value in [ret_1d, ret_3d, ret_5d, ret_20d] if value is not None]
            best_forward = max(available_returns) if available_returns else None
            status = "평가대기" if ret_5d is None and ret_20d is None else "평가완료"
            hit_1d = ret_1d is not None and ret_1d > 0
            hit_5d = ret_5d is not None and ret_5d > 0
            hit_20d = ret_20d is not None and ret_20d > 0

            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "saved_at": saved_at,
                    "scan_type": scan_type,
                    "market": market,
                    "ticker": ticker,
                    "name": row.get("name", ""),
                    "setup": row.get("setup", ""),
                    "action": row.get("action", ""),
                    "event_risk": row.get("event_risk", ""),
                    "news_bias": row.get("news_bias", ""),
                    "score": round(score, 2),
                    "current_price": round(base_price, 2),
                    "entry_price": round(float(row.get("entry_price", 0) or 0), 2) if row.get("entry_price", "") != "" else None,
                    "stop_loss": round(float(stop_loss), 2) if not pd.isna(stop_loss) else None,
                    "target_1": round(float(target_1), 2) if not pd.isna(target_1) else None,
                    "captured_at": captured_at,
                    "ret_1d_pct": round(ret_1d, 2) if ret_1d is not None else None,
                    "ret_3d_pct": round(ret_3d, 2) if ret_3d is not None else None,
                    "ret_5d_pct": round(ret_5d, 2) if ret_5d is not None else None,
                    "ret_20d_pct": round(ret_20d, 2) if ret_20d is not None else None,
                    "max_5d_pct": round(max_5d, 2) if max_5d is not None else None,
                    "min_5d_pct": round(min_5d, 2) if min_5d is not None else None,
                    "max_20d_pct": round(max_20d, 2) if max_20d is not None else None,
                    "min_20d_pct": round(min_20d, 2) if min_20d is not None else None,
                    "best_forward_pct": round(best_forward, 2) if best_forward is not None else None,
                    "target_hit_5d": target_hit_5d,
                    "stop_hit_5d": stop_hit_5d,
                    "target_hit_20d": target_hit_20d,
                    "stop_hit_20d": stop_hit_20d,
                    "path_5d": path_5d,
                    "path_20d": path_20d,
                    "hit_1d": hit_1d,
                    "hit_5d": hit_5d,
                    "hit_20d": hit_20d,
                    "label_1d": "평가대기" if ret_1d is None else ("상승" if ret_1d > 0 else "하락"),
                    "label_5d": _label_period(ret_5d, target_hit_5d, stop_hit_5d, path_5d, 6),
                    "label_20d": _label_period(ret_20d, target_hit_20d, stop_hit_20d, path_20d, 10),
                    "status": status,
                }
            )

    return pd.DataFrame(rows)


def _summarize_by_bucket(detail: pd.DataFrame) -> pd.DataFrame:
    mature = detail[detail["status"] == "평가완료"].copy()
    if mature.empty:
        return pd.DataFrame()

    grouped = (
        mature.groupby(["scan_type", "market"], dropna=False)
        .agg(
            picks=("ticker", "count"),
            avg_score=("score", "mean"),
            avg_ret_1d_pct=("ret_1d_pct", "mean"),
            avg_ret_3d_pct=("ret_3d_pct", "mean"),
            avg_ret_5d_pct=("ret_5d_pct", "mean"),
            avg_ret_20d_pct=("ret_20d_pct", "mean"),
            avg_max_5d_pct=("max_5d_pct", "mean"),
            avg_min_5d_pct=("min_5d_pct", "mean"),
            best_ret_20d_pct=("ret_20d_pct", "max"),
        )
        .reset_index()
    )

    extra = (
        mature.groupby(["scan_type", "market"], dropna=False)
        .agg(
            hit_rate_5d_pct=("hit_5d", lambda s: s.mean() * 100),
            hit_rate_20d_pct=("hit_20d", lambda s: s.mean() * 100),
            target_first_5d_pct=("path_5d", lambda s: ((s == "목표가 선도달") | (s == "목표가 도달")).mean() * 100),
            stop_first_5d_pct=("path_5d", lambda s: ((s == "손절 선도달") | (s == "손절 도달")).mean() * 100),
        )
        .reset_index()
    )

    summary = grouped.merge(extra, on=["scan_type", "market"], how="left")
    return summary.round(2).sort_values(
        by=["avg_ret_20d_pct", "target_first_5d_pct", "hit_rate_20d_pct"], ascending=[False, False, False]
    ).reset_index(drop=True)


def _build_ticker_leaderboard(detail: pd.DataFrame) -> pd.DataFrame:
    mature = detail[detail["status"] == "평가완료"].copy()
    if mature.empty:
        return pd.DataFrame()

    leaderboard = (
        mature.groupby(["market", "ticker", "name"], dropna=False)
        .agg(
            appearances=("ticker", "count"),
            avg_score=("score", "mean"),
            avg_ret_5d_pct=("ret_5d_pct", "mean"),
            avg_ret_20d_pct=("ret_20d_pct", "mean"),
            target_first_5d_pct=("path_5d", lambda s: ((s == "목표가 선도달") | (s == "목표가 도달")).mean() * 100),
            stop_first_5d_pct=("path_5d", lambda s: ((s == "손절 선도달") | (s == "손절 도달")).mean() * 100),
            best_forward_pct=("best_forward_pct", "max"),
            latest_saved_at=("saved_at", "max"),
        )
        .reset_index()
    )

    hit_rate = (
        mature.groupby(["market", "ticker", "name"], dropna=False)
        .agg(hit_rate_20d_pct=("hit_20d", lambda s: s.mean() * 100))
        .reset_index()
    )
    leaderboard = leaderboard.merge(hit_rate, on=["market", "ticker", "name"], how="left")
    leaderboard["memory_score"] = (
        leaderboard["appearances"] * 6
        + leaderboard["avg_score"] * 0.45
        + leaderboard["avg_ret_20d_pct"].fillna(0) * 1.4
        + leaderboard["hit_rate_20d_pct"].fillna(0) * 0.25
        + leaderboard["target_first_5d_pct"].fillna(0) * 0.1
        - leaderboard["stop_first_5d_pct"].fillna(0) * 0.08
    ).round(2)
    return leaderboard.round(2).sort_values(
        by=["memory_score", "avg_ret_20d_pct", "appearances"], ascending=[False, False, False]
    ).reset_index(drop=True)


def _build_pending_status(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()

    pending = detail[detail["status"] == "평가대기"].copy()
    if pending.empty:
        return pd.DataFrame()

    return (
        pending.groupby(["scan_type", "market"], dropna=False)
        .agg(waiting=("ticker", "count"), latest_saved_at=("saved_at", "max"))
        .reset_index()
        .sort_values(by=["latest_saved_at", "waiting"], ascending=[False, False])
        .reset_index(drop=True)
    )


def _build_pattern_stats(detail: pd.DataFrame) -> pd.DataFrame:
    mature = detail[detail["status"] == "평가완료"].copy()
    if mature.empty:
        return pd.DataFrame()

    mature["setup"] = mature["setup"].replace("", "미분류")
    pattern_stats = (
        mature.groupby(["scan_type", "market", "setup"], dropna=False)
        .agg(
            picks=("ticker", "count"),
            avg_score=("score", "mean"),
            avg_ret_3d_pct=("ret_3d_pct", "mean"),
            avg_ret_5d_pct=("ret_5d_pct", "mean"),
            avg_ret_20d_pct=("ret_20d_pct", "mean"),
            avg_max_5d_pct=("max_5d_pct", "mean"),
            avg_min_5d_pct=("min_5d_pct", "mean"),
            strong_success_20d=("label_20d", lambda s: (s == "강한성공").mean() * 100),
            hit_rate_20d_pct=("hit_20d", lambda s: s.mean() * 100),
            target_first_5d_pct=("path_5d", lambda s: ((s == "목표가 선도달") | (s == "목표가 도달")).mean() * 100),
            stop_first_5d_pct=("path_5d", lambda s: ((s == "손절 선도달") | (s == "손절 도달")).mean() * 100),
        )
        .reset_index()
    )
    return pattern_stats.round(2).sort_values(
        by=["avg_ret_20d_pct", "target_first_5d_pct", "picks"], ascending=[False, False, False]
    ).reset_index(drop=True)


def evaluate_scan_history(limit: int = 200) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail = _build_detail(limit=limit)
    if detail.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    summary = _summarize_by_bucket(detail)
    leaderboard = _build_ticker_leaderboard(detail)
    pending = _build_pending_status(detail)
    pattern_stats = _build_pattern_stats(detail)

    detail = detail.sort_values(by=["saved_at", "score"], ascending=[False, False]).reset_index(drop=True)
    return detail, summary, leaderboard, pending, pattern_stats
