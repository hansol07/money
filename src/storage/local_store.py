from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.storage.sqlite_cache import (
    append_feature_log_entries,
    append_scan_snapshot,
    has_scan_snapshot,
    load_recent_feature_log_entries,
    load_recent_scan_snapshots,
    seed_feature_log_entries,
    seed_scan_snapshots,
)


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
WATCHLIST_FILE = DATA_DIR / "watchlists.json"
SCAN_HISTORY_FILE = DATA_DIR / "scan_history.json"
FEATURE_LOG_FILE = DATA_DIR / "candidate_feature_log.json"
MANUAL_TRACKING_FILE = DATA_DIR / "manual_tracking.json"
DAILY_BRIEFING_FILE = DATA_DIR / "daily_briefings.json"
DAILY_BRIEFING_ACTION_FILE = DATA_DIR / "daily_briefing_actions.json"
DECISION_LOG_FILE = DATA_DIR / "decision_log.json"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_portfolio(df: pd.DataFrame) -> None:
    ensure_data_dir()
    records = df.fillna("").to_dict("records")
    PORTFOLIO_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def load_portfolio() -> pd.DataFrame | None:
    if not PORTFOLIO_FILE.exists():
        return None
    records = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
    return pd.DataFrame(records)


def normalize_portfolio_frame(df: pd.DataFrame) -> pd.DataFrame:
    column_aliases = {
        "market": "market",
        "시장": "market",
        "국가": "market",
        "ticker": "ticker",
        "symbol": "ticker",
        "code": "ticker",
        "종목코드": "ticker",
        "티커": "ticker",
        "종목번호": "ticker",
        "name": "name",
        "종목명": "name",
        "종목": "name",
        "quantity": "quantity",
        "qty": "quantity",
        "shares": "quantity",
        "보유수량": "quantity",
        "수량": "quantity",
        "avg_price": "avg_price",
        "average_price": "avg_price",
        "buy_price": "avg_price",
        "평균단가": "avg_price",
        "매입가": "avg_price",
        "매수가": "avg_price",
        "cash_budget": "cash_budget",
        "budget": "cash_budget",
        "추가매수가능금액": "cash_budget",
        "추가매수금액": "cash_budget",
        "예수금": "cash_budget",
        "target_weight": "target_weight",
        "weight": "target_weight",
        "목표비중": "target_weight",
        "비중": "target_weight",
    }
    required_columns = ["market", "ticker", "name", "quantity", "avg_price", "cash_budget", "target_weight"]
    frame = df.copy()
    frame.columns = [column_aliases.get(str(column).strip(), str(column).strip()) for column in frame.columns]

    for column in required_columns:
        if column not in frame.columns:
            frame[column] = ""

    frame = frame[required_columns].fillna("")
    frame["market"] = frame["market"].astype(str).str.strip().str.upper().replace({"KOR": "KR", "USA": "US"})
    frame["market"] = frame["market"].replace(
        {
            "한국": "KR",
            "국장": "KR",
            "KOREA": "KR",
            "KRX": "KR",
            "미국": "US",
            "미장": "US",
            "USA": "US",
            "NASDAQ": "US",
            "NYSE": "US",
        }
    )
    frame["market"] = frame["market"].where(frame["market"].isin(["US", "KR"]), "US")
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["name"] = frame["name"].astype(str).str.strip()

    for numeric_column in ["quantity", "avg_price", "cash_budget", "target_weight"]:
        frame[numeric_column] = pd.to_numeric(frame[numeric_column], errors="coerce").fillna(0.0)

    frame = frame[frame["ticker"] != ""].drop_duplicates(subset=["market", "ticker"], keep="first")
    return frame.reset_index(drop=True)


def save_watchlists(watchlists: dict[str, list[dict[str, str]]]) -> None:
    ensure_data_dir()
    WATCHLIST_FILE.write_text(json.dumps(watchlists, ensure_ascii=False, indent=2), encoding="utf-8")


def load_watchlists() -> dict[str, list[dict[str, str]]] | None:
    if not WATCHLIST_FILE.exists():
        return None
    return json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))


def append_scan_history(scan_type: str, market: str, frame: pd.DataFrame) -> None:
    ensure_data_dir()
    history = load_scan_history()
    records = frame.fillna("").to_dict("records")
    saved_at = pd.Timestamp.now(tz="Asia/Seoul").isoformat(timespec="seconds")
    snapshot_id = f"{scan_type}_{market}_{uuid4().hex[:8]}"

    for record in records:
        record.setdefault("captured_at", saved_at)

    snapshot = {
        "snapshot_id": snapshot_id,
        "scan_type": scan_type,
        "market": market,
        "saved_at": saved_at,
        "row_count": len(records),
        "rows": records,
    }

    history.append(snapshot)
    SCAN_HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    append_scan_snapshot(snapshot)
    append_feature_log(snapshot_id=snapshot_id, scan_type=scan_type, market=market, saved_at=saved_at, rows=records)


def load_scan_history() -> list[dict[str, object]]:
    if not SCAN_HISTORY_FILE.exists():
        return []
    history = json.loads(SCAN_HISTORY_FILE.read_text(encoding="utf-8"))
    return _normalize_scan_history(history)


def _normalize_scan_history(history: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized_history: list[dict[str, object]] = []

    for entry in history:
        rows = entry.get("rows", [])
        first_captured_at = ""
        if rows:
            first_captured_at = str(rows[0].get("captured_at", "") or "")

        normalized_history.append(
            {
                "snapshot_id": entry.get("snapshot_id", f"{entry.get('scan_type', 'scan')}_{entry.get('market', 'ALL')}_legacy"),
                "scan_type": entry.get("scan_type", ""),
                "market": entry.get("market", ""),
                "saved_at": entry.get("saved_at", first_captured_at),
                "row_count": entry.get("row_count", len(rows)),
                "rows": rows,
            }
        )

    return normalized_history


def load_recent_scan_history(limit: int = 200) -> list[dict[str, object]]:
    cached = load_recent_scan_snapshots(limit=limit)
    if cached:
        return _normalize_scan_history(cached)

    if not SCAN_HISTORY_FILE.exists():
        return []
    try:
        history = json.loads(SCAN_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    normalized = _normalize_scan_history(history)
    seed_scan_snapshots(normalized)
    if limit > 0:
        normalized = normalized[-limit:]
    return normalized


def has_scan_snapshot_for_day(scan_type: str, market: str, day_key: str) -> bool:
    if has_scan_snapshot(scan_type, market, day_key):
        return True
    history = load_recent_scan_history(limit=1000)
    for entry in reversed(history):
        if str(entry.get("scan_type", "")) != scan_type:
            continue
        if str(entry.get("market", "")) != market:
            continue
        saved_at = str(entry.get("saved_at", "") or "")
        if saved_at.startswith(day_key):
            return True
    return False


def has_scan_snapshot_for_prefix(scan_type: str, market: str, prefix: str) -> bool:
    if has_scan_snapshot(scan_type, market, prefix):
        return True
    history = load_recent_scan_history(limit=1000)
    for entry in reversed(history):
        if str(entry.get("scan_type", "")) != scan_type:
            continue
        if str(entry.get("market", "")) != market:
            continue
        saved_at = str(entry.get("saved_at", "") or "")
        if saved_at.startswith(prefix):
            return True
    return False


def append_feature_log(
    *,
    snapshot_id: str,
    scan_type: str,
    market: str,
    saved_at: str,
    rows: list[dict[str, object]],
) -> None:
    ensure_data_dir()
    feature_rows: list[dict[str, object]] = []

    for row in rows:
        feature_rows.append(
            {
                "snapshot_id": snapshot_id,
                "scan_type": scan_type,
                "market": market,
                "saved_at": saved_at,
                "captured_at": row.get("captured_at", saved_at),
                "ticker": row.get("ticker", ""),
                "name": row.get("name", ""),
                "setup": row.get("setup", ""),
                "action": row.get("action", ""),
                "bucket": row.get("bucket", ""),
                "score": row.get("score", ""),
                "current_price": row.get("current_price", ""),
                "entry_price": row.get("entry_price", ""),
                "stop_loss": row.get("stop_loss", ""),
                "target_1": row.get("target_1", ""),
                "target_2": row.get("target_2", ""),
                "volume_ratio": row.get("volume_ratio", ""),
                "return_20d": row.get("return_20d", ""),
                "return_60d_pct": row.get("return_60d_pct", ""),
                "return_1y_pct": row.get("return_1y_pct", ""),
                "trend_score": row.get("trend_score", ""),
                "momentum_score": row.get("momentum_score", ""),
                "volume_score": row.get("volume_score", ""),
                "breakout_score": row.get("breakout_score", ""),
                "short_return_pct": row.get("short_return_pct", ""),
                "above_vwap": row.get("above_vwap", ""),
                "risk_reward_1": row.get("risk_reward_1", ""),
                "event_risk": row.get("event_risk", ""),
                "event_note": row.get("event_note", ""),
                "earnings_date": row.get("earnings_date", ""),
                "ex_dividend_date": row.get("ex_dividend_date", ""),
                "news_bias": row.get("news_bias", ""),
                "news_score": row.get("news_score", ""),
                "news_count": row.get("news_count", ""),
            }
        )

    try:
        existing = json.loads(FEATURE_LOG_FILE.read_text(encoding="utf-8")) if FEATURE_LOG_FILE.exists() else []
    except Exception:
        existing = []

    append_feature_log_entries(feature_rows)
    existing.extend(feature_rows)
    if len(existing) > 5000:
        existing = existing[-5000:]
    FEATURE_LOG_FILE.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def load_feature_log() -> list[dict[str, object]]:
    if not FEATURE_LOG_FILE.exists():
        return []
    return json.loads(FEATURE_LOG_FILE.read_text(encoding="utf-8"))


def load_recent_feature_log(limit: int = 1000) -> list[dict[str, object]]:
    cached = load_recent_feature_log_entries(limit=limit)
    if cached:
        return cached

    if not FEATURE_LOG_FILE.exists():
        return []
    try:
        records = json.loads(FEATURE_LOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    seed_feature_log_entries(records)
    if limit > 0:
        return records[-limit:]
    return records


def load_manual_tracking() -> pd.DataFrame:
    if not MANUAL_TRACKING_FILE.exists():
        return pd.DataFrame(
            columns=[
                "tracking_id",
                "created_at",
                "market",
                "ticker",
                "name",
                "source",
                "setup",
                "score",
                "current_price",
                "entry_price",
                "stop_loss",
                "target_1",
                "memo",
            ]
        )
    records = json.loads(MANUAL_TRACKING_FILE.read_text(encoding="utf-8"))
    return pd.DataFrame(records)


def save_manual_tracking(df: pd.DataFrame) -> None:
    ensure_data_dir()
    records = df.fillna("").to_dict("records")
    MANUAL_TRACKING_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def append_manual_tracking(row: dict[str, object]) -> None:
    current = load_manual_tracking()
    tracking_id = str(row.get("tracking_id", "") or f"manual_{uuid4().hex[:8]}")
    created_at = str(row.get("created_at", "") or pd.Timestamp.now(tz="Asia/Seoul").isoformat(timespec="seconds"))
    payload = {
        "tracking_id": tracking_id,
        "created_at": created_at,
        "market": row.get("market", ""),
        "ticker": str(row.get("ticker", "")).strip().upper(),
        "name": row.get("name", ""),
        "source": row.get("source", ""),
        "setup": row.get("setup", ""),
        "score": row.get("score", ""),
        "current_price": row.get("current_price", ""),
        "entry_price": row.get("entry_price", ""),
        "stop_loss": row.get("stop_loss", ""),
        "target_1": row.get("target_1", ""),
        "memo": row.get("memo", ""),
    }
    if current.empty:
        updated = pd.DataFrame([payload])
    else:
        current = current[~((current["market"] == payload["market"]) & (current["ticker"] == payload["ticker"]))]
        updated = pd.concat([current, pd.DataFrame([payload])], ignore_index=True)
    save_manual_tracking(updated)


def remove_manual_tracking(tracking_id: str) -> None:
    current = load_manual_tracking()
    if current.empty:
        return
    updated = current[current["tracking_id"] != tracking_id].reset_index(drop=True)
    save_manual_tracking(updated)


def load_daily_briefings() -> pd.DataFrame:
    if not DAILY_BRIEFING_FILE.exists():
        return pd.DataFrame(
            columns=[
                "briefing_id",
                "saved_at",
                "briefing_date",
                "mode",
                "us_budget",
                "kr_budget",
                "total_budget",
                "action_count",
                "headline",
                "top_actions",
                "notes",
            ]
        )
    records = json.loads(DAILY_BRIEFING_FILE.read_text(encoding="utf-8"))
    return pd.DataFrame(records)


def save_daily_briefings(df: pd.DataFrame) -> None:
    ensure_data_dir()
    records = df.fillna("").to_dict("records")
    DAILY_BRIEFING_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def append_daily_briefing(row: dict[str, object]) -> str:
    current = load_daily_briefings()
    payload = {
        "briefing_id": str(row.get("briefing_id", "") or f"brief_{uuid4().hex[:8]}"),
        "saved_at": str(row.get("saved_at", "") or pd.Timestamp.now(tz="Asia/Seoul").isoformat(timespec="seconds")),
        "briefing_date": str(row.get("briefing_date", "") or pd.Timestamp.now(tz="Asia/Seoul").date().isoformat()),
        "mode": str(row.get("mode", "") or ""),
        "us_budget": row.get("us_budget", 0),
        "kr_budget": row.get("kr_budget", 0),
        "total_budget": row.get("total_budget", 0),
        "action_count": row.get("action_count", 0),
        "headline": str(row.get("headline", "") or ""),
        "top_actions": str(row.get("top_actions", "") or ""),
        "notes": str(row.get("notes", "") or ""),
    }
    if current.empty:
        updated = pd.DataFrame([payload])
    else:
        current = current[~(
            (current["briefing_date"].astype(str) == payload["briefing_date"])
            & (current["mode"].astype(str) == payload["mode"])
        )]
        updated = pd.concat([current, pd.DataFrame([payload])], ignore_index=True)
    save_daily_briefings(updated)
    return payload["briefing_id"]


def load_daily_briefing_actions() -> pd.DataFrame:
    if not DAILY_BRIEFING_ACTION_FILE.exists():
        return pd.DataFrame(
            columns=[
                "briefing_id",
                "briefing_date",
                "saved_at",
                "market",
                "bucket",
                "ticker",
                "name",
                "setup",
                "score",
                "planned_amount",
                "ref_price",
                "estimated_units",
                "reason",
            ]
        )
    records = json.loads(DAILY_BRIEFING_ACTION_FILE.read_text(encoding="utf-8"))
    return pd.DataFrame(records)


def save_daily_briefing_actions(df: pd.DataFrame) -> None:
    ensure_data_dir()
    records = df.fillna("").to_dict("records")
    DAILY_BRIEFING_ACTION_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def append_daily_briefing_actions(briefing_id: str, briefing_date: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return

    current = load_daily_briefing_actions()
    payload = frame.copy()
    payload["briefing_id"] = briefing_id
    payload["briefing_date"] = briefing_date
    payload["saved_at"] = pd.Timestamp.now(tz="Asia/Seoul").isoformat(timespec="seconds")

    if not current.empty:
        current = current[current["briefing_id"].astype(str) != str(briefing_id)].copy()
        payload = pd.concat([current, payload], ignore_index=True)

    save_daily_briefing_actions(payload)


def load_decision_log() -> pd.DataFrame:
    if not DECISION_LOG_FILE.exists():
        return pd.DataFrame(
            columns=[
                "decision_id",
                "created_at",
                "decision_date",
                "decision",
                "market",
                "bucket",
                "ticker",
                "name",
                "setup",
                "score",
                "planned_amount",
                "execution_status",
                "confidence_view",
                "ref_price",
                "buy_now_limit",
                "stop_loss",
                "target_1",
                "source",
                "memo",
            ]
        )
    records = json.loads(DECISION_LOG_FILE.read_text(encoding="utf-8"))
    return pd.DataFrame(records)


def save_decision_log(df: pd.DataFrame) -> None:
    ensure_data_dir()
    records = df.fillna("").to_dict("records")
    DECISION_LOG_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def append_decision_log(row: dict[str, object]) -> None:
    current = load_decision_log()
    created_at = str(row.get("created_at", "") or pd.Timestamp.now(tz="Asia/Seoul").isoformat(timespec="seconds"))
    payload = {
        "decision_id": str(row.get("decision_id", "") or f"decision_{uuid4().hex[:8]}"),
        "created_at": created_at,
        "decision_date": str(row.get("decision_date", "") or pd.Timestamp.now(tz="Asia/Seoul").date().isoformat()),
        "decision": str(row.get("decision", "") or ""),
        "market": str(row.get("market", "") or ""),
        "bucket": str(row.get("bucket", "") or ""),
        "ticker": str(row.get("ticker", "") or "").strip().upper(),
        "name": str(row.get("name", "") or ""),
        "setup": str(row.get("setup", "") or ""),
        "score": row.get("score", ""),
        "planned_amount": row.get("planned_amount", ""),
        "execution_status": str(row.get("execution_status", "") or ""),
        "confidence_view": str(row.get("confidence_view", "") or ""),
        "ref_price": row.get("ref_price", ""),
        "buy_now_limit": row.get("buy_now_limit", ""),
        "stop_loss": row.get("stop_loss", ""),
        "target_1": row.get("target_1", ""),
        "source": str(row.get("source", "") or ""),
        "memo": str(row.get("memo", "") or ""),
    }
    updated = pd.concat([current, pd.DataFrame([payload])], ignore_index=True) if not current.empty else pd.DataFrame([payload])
    save_decision_log(updated)
