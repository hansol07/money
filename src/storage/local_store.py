from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
WATCHLIST_FILE = DATA_DIR / "watchlists.json"
SCAN_HISTORY_FILE = DATA_DIR / "scan_history.json"
FEATURE_LOG_FILE = DATA_DIR / "candidate_feature_log.json"


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

    history.append(
        {
            "snapshot_id": snapshot_id,
            "scan_type": scan_type,
            "market": market,
            "saved_at": saved_at,
            "row_count": len(records),
            "rows": records,
        }
    )
    SCAN_HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    append_feature_log(snapshot_id=snapshot_id, scan_type=scan_type, market=market, saved_at=saved_at, rows=records)


def load_scan_history() -> list[dict[str, object]]:
    if not SCAN_HISTORY_FILE.exists():
        return []
    history = json.loads(SCAN_HISTORY_FILE.read_text(encoding="utf-8"))
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


def has_scan_snapshot_for_day(scan_type: str, market: str, day_key: str) -> bool:
    history = load_scan_history()
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
    history = load_scan_history()
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
    existing = load_feature_log()
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
            }
        )

    existing.extend(feature_rows)
    FEATURE_LOG_FILE.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def load_feature_log() -> list[dict[str, object]]:
    if not FEATURE_LOG_FILE.exists():
        return []
    return json.loads(FEATURE_LOG_FILE.read_text(encoding="utf-8"))
