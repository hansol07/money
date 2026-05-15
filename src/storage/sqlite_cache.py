from __future__ import annotations

import json
import sqlite3
from io import StringIO
from pathlib import Path
from time import time
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CACHE_DB_FILE = DATA_DIR / "cache.sqlite3"
_SCHEMA_READY = False


def _connect() -> sqlite3.Connection:
    global _SCHEMA_READY
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_DB_FILE, timeout=10)
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    if _SCHEMA_READY:
        return conn
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cache_entries (
            cache_key TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            expires_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scan_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            scan_type TEXT NOT NULL,
            market TEXT NOT NULL,
            saved_at TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scan_snapshots_saved_at ON scan_snapshots(saved_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scan_snapshots_type_market ON scan_snapshots(scan_type, market, saved_at)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_log_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id TEXT NOT NULL,
            scan_type TEXT NOT NULL,
            market TEXT NOT NULL,
            saved_at TEXT NOT NULL,
            ticker TEXT NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feature_log_saved_at ON feature_log_entries(saved_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feature_log_snapshot ON feature_log_entries(snapshot_id)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_bars (
            ticker TEXT NOT NULL,
            market TEXT NOT NULL,
            interval TEXT NOT NULL,
            ts TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            source TEXT NOT NULL,
            captured_at TEXT NOT NULL,
            PRIMARY KEY (ticker, interval, ts)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_price_bars_ticker_interval_ts ON price_bars(ticker, interval, ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_price_bars_market_interval_ts ON price_bars(market, interval, ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_price_bars_interval_ticker_ts ON price_bars(interval, ticker, ts DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_price_bars_market_interval_ticker_ts ON price_bars(market, interval, ticker, ts DESC)")
    _SCHEMA_READY = True
    return conn


def _get_payload(cache_key: str, allow_expired: bool = False) -> str | None:
    now = time()
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT payload, expires_at FROM cache_entries WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            payload, expires_at = row
            if float(expires_at) < now:
                if allow_expired:
                    return str(payload)
                return None
            return str(payload)
    except Exception:
        return None


def _set_payload(cache_key: str, payload: str, ttl_seconds: int) -> None:
    now = time()
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries (cache_key, payload, expires_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (cache_key, payload, now + ttl_seconds, now),
            )
    except Exception:
        return


def get_cached_json(cache_key: str, allow_expired: bool = False) -> dict[str, Any] | None:
    payload = _get_payload(cache_key, allow_expired=allow_expired)
    if payload is None:
        return None
    try:
        value = json.loads(payload)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def set_cached_json(cache_key: str, value: dict[str, Any], ttl_seconds: int) -> None:
    _set_payload(cache_key, json.dumps(value, ensure_ascii=False), ttl_seconds)


def get_cached_frame(
    cache_key: str,
    date_columns: list[str] | None = None,
    allow_expired: bool = False,
) -> pd.DataFrame | None:
    payload = _get_payload(cache_key, allow_expired=allow_expired)
    if payload is None:
        return None
    try:
        frame = pd.read_json(StringIO(payload), orient="split")
    except Exception:
        return None
    for column in date_columns or []:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def set_cached_frame(cache_key: str, frame: pd.DataFrame, ttl_seconds: int) -> None:
    if frame.empty:
        return
    try:
        payload = frame.to_json(orient="split", date_format="iso")
    except Exception:
        return
    _set_payload(cache_key, payload, ttl_seconds)


def clear_sqlite_cache() -> None:
    try:
        with _connect() as conn:
            conn.execute("DELETE FROM cache_entries")
    except Exception:
        return


def get_sqlite_cache_stats() -> dict[str, Any]:
    try:
        with _connect() as conn:
            cache_entries = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
            expired_cache_entries = conn.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE expires_at < ?",
                (time(),),
            ).fetchone()[0]
            scan_snapshots = conn.execute("SELECT COUNT(*) FROM scan_snapshots").fetchone()[0]
            feature_entries = conn.execute("SELECT COUNT(*) FROM feature_log_entries").fetchone()[0]
            price_bars = conn.execute("SELECT COUNT(*) FROM price_bars").fetchone()[0]
            price_tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM price_bars").fetchone()[0]
            db_size_bytes = CACHE_DB_FILE.stat().st_size if CACHE_DB_FILE.exists() else 0
        return {
            "cache_entries": int(cache_entries or 0),
            "expired_cache_entries": int(expired_cache_entries or 0),
            "scan_snapshots": int(scan_snapshots or 0),
            "feature_entries": int(feature_entries or 0),
            "price_bars": int(price_bars or 0),
            "price_tickers": int(price_tickers or 0),
            "db_size_mb": round(db_size_bytes / 1024 / 1024, 2),
        }
    except Exception:
        return {
            "cache_entries": 0,
            "expired_cache_entries": 0,
            "scan_snapshots": 0,
            "feature_entries": 0,
            "price_bars": 0,
            "price_tickers": 0,
            "db_size_mb": 0.0,
        }


def _infer_market(ticker: str) -> str:
    symbol = str(ticker or "").upper()
    if symbol.endswith((".KS", ".KQ")):
        return "KR"
    return "US"


def set_price_bars(
    ticker: str,
    interval: str,
    frame: pd.DataFrame,
    *,
    source: str = "yfinance",
    market: str | None = None,
) -> int:
    if frame.empty:
        return 0
    ticker = str(ticker or "").strip().upper()
    interval = str(interval or "").strip()
    if not ticker or not interval:
        return 0

    time_column = "Date" if "Date" in frame.columns else "Datetime" if "Datetime" in frame.columns else None
    required = [time_column, "Open", "High", "Low", "Close", "Volume"]
    if time_column is None or any(column not in frame.columns for column in required if column):
        return 0

    captured_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    rows: list[tuple[object, ...]] = []
    working = frame[required].copy()
    working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
    working = working.dropna(subset=[time_column, "Close"])
    for row in working.to_dict("records"):
        ts = pd.Timestamp(row[time_column])
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        rows.append(
            (
                ticker,
                str(market or _infer_market(ticker)),
                interval,
                ts.isoformat(),
                float(pd.to_numeric(row.get("Open"), errors="coerce")) if not pd.isna(pd.to_numeric(row.get("Open"), errors="coerce")) else None,
                float(pd.to_numeric(row.get("High"), errors="coerce")) if not pd.isna(pd.to_numeric(row.get("High"), errors="coerce")) else None,
                float(pd.to_numeric(row.get("Low"), errors="coerce")) if not pd.isna(pd.to_numeric(row.get("Low"), errors="coerce")) else None,
                float(pd.to_numeric(row.get("Close"), errors="coerce")) if not pd.isna(pd.to_numeric(row.get("Close"), errors="coerce")) else None,
                float(pd.to_numeric(row.get("Volume"), errors="coerce")) if not pd.isna(pd.to_numeric(row.get("Volume"), errors="coerce")) else None,
                source,
                captured_at,
            )
        )
    if not rows:
        return 0
    try:
        with _connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO price_bars
                    (ticker, market, interval, ts, open, high, low, close, volume, source, captured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)
    except Exception:
        return 0


def get_price_bars(ticker: str, interval: str, *, limit: int | None = None) -> pd.DataFrame:
    ticker = str(ticker or "").strip().upper()
    interval = str(interval or "").strip()
    if not ticker or not interval:
        return pd.DataFrame()
    try:
        query = """
            SELECT ts, open, high, low, close, volume
            FROM price_bars
            WHERE ticker = ? AND interval = ?
            ORDER BY ts DESC
        """
        params: tuple[object, ...] = (ticker, interval)
        if limit is not None and int(limit) > 0:
            query += " LIMIT ?"
            params = (ticker, interval, int(limit))
        with _connect() as conn:
            rows = conn.execute(query, params).fetchall()
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return frame


def get_price_bars_for_tickers(tickers: list[str], interval: str, *, bars_per_ticker: int = 260) -> pd.DataFrame:
    cleaned = []
    seen: set[str] = set()
    for ticker in tickers:
        symbol = str(ticker or "").strip().upper()
        if symbol and symbol not in seen:
            cleaned.append(symbol)
            seen.add(symbol)
    if not cleaned:
        return pd.DataFrame()
    interval = str(interval or "").strip()
    if not interval:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    try:
        with _connect() as conn:
            for start in range(0, len(cleaned), 300):
                chunk = cleaned[start : start + 300]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""
                    SELECT ticker, ts, open, high, low, close, volume
                    FROM (
                        SELECT ticker, ts, open, high, low, close, volume,
                               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY ts DESC) AS rn
                        FROM price_bars
                        WHERE interval = ? AND ticker IN ({placeholders})
                    )
                    WHERE rn <= ?
                    ORDER BY ticker, ts
                    """,
                    (interval, *chunk, max(1, int(bars_per_ticker))),
                ).fetchall()
                if rows:
                    frames.append(pd.DataFrame(rows, columns=["ticker", "Date", "Open", "High", "Low", "Close", "Volume"]))
    except Exception:
        return pd.DataFrame()
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return frame.dropna(subset=["Date", "Close"]).sort_values(["ticker", "Date"]).reset_index(drop=True)


def get_price_warehouse_stats() -> pd.DataFrame:
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT market, interval, COUNT(*) AS bars, COUNT(DISTINCT ticker) AS tickers,
                       MIN(ts) AS first_ts, MAX(ts) AS last_ts
                FROM price_bars
                GROUP BY market, interval
                ORDER BY market, interval
                """
            ).fetchall()
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["market", "interval", "bars", "tickers", "first_ts", "last_ts"])


def get_price_warehouse_tickers(market: str | None = None, *, interval: str | None = None) -> pd.DataFrame:
    where: list[str] = []
    params: list[object] = []
    if market:
        where.append("market = ?")
        params.append(str(market).upper())
    if interval:
        where.append("interval = ?")
        params.append(str(interval))
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    try:
        with _connect() as conn:
            rows = conn.execute(
                f"""
                SELECT market, ticker, COUNT(*) AS bars, MIN(ts) AS first_ts, MAX(ts) AS last_ts
                FROM price_bars
                {where_sql}
                GROUP BY market, ticker
                ORDER BY MAX(ts) DESC, COUNT(*) DESC, ticker
                """,
                tuple(params),
            ).fetchall()
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["market", "ticker", "bars", "first_ts", "last_ts"])


def get_cache_key_status(cache_keys: list[str]) -> dict[str, int]:
    total = len(cache_keys)
    if total <= 0:
        return {"total": 0, "fresh": 0, "stale": 0, "missing": 0}

    now = time()
    seen: set[str] = set()
    fresh = 0
    stale = 0
    try:
        with _connect() as conn:
            for start in range(0, total, 500):
                chunk = cache_keys[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT cache_key, expires_at FROM cache_entries WHERE cache_key IN ({placeholders})",
                    chunk,
                ).fetchall()
                for cache_key, expires_at in rows:
                    key = str(cache_key)
                    if key in seen:
                        continue
                    seen.add(key)
                    if float(expires_at) >= now:
                        fresh += 1
                    else:
                        stale += 1
    except Exception:
        return {"total": total, "fresh": 0, "stale": 0, "missing": total}

    return {
        "total": total,
        "fresh": fresh,
        "stale": stale,
        "missing": max(0, total - len(seen)),
    }


def append_scan_snapshot(snapshot: dict[str, Any]) -> None:
    snapshot_id = str(snapshot.get("snapshot_id", "") or "")
    if not snapshot_id:
        return
    try:
        payload = json.dumps(snapshot, ensure_ascii=False)
        with _connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scan_snapshots (snapshot_id, scan_type, market, saved_at, row_count, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    str(snapshot.get("scan_type", "") or ""),
                    str(snapshot.get("market", "") or ""),
                    str(snapshot.get("saved_at", "") or ""),
                    int(snapshot.get("row_count", 0) or 0),
                    payload,
                ),
            )
    except Exception:
        return


def seed_scan_snapshots(snapshots: list[dict[str, Any]]) -> None:
    if not snapshots:
        return
    try:
        rows = [
            (
                str(snapshot.get("snapshot_id", "") or ""),
                str(snapshot.get("scan_type", "") or ""),
                str(snapshot.get("market", "") or ""),
                str(snapshot.get("saved_at", "") or ""),
                int(snapshot.get("row_count", 0) or 0),
                json.dumps(snapshot, ensure_ascii=False),
            )
            for snapshot in snapshots
            if str(snapshot.get("snapshot_id", "") or "")
        ]
        if not rows:
            return
        with _connect() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO scan_snapshots (snapshot_id, scan_type, market, saved_at, row_count, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
    except Exception:
        return


def load_recent_scan_snapshots(limit: int = 200) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM scan_snapshots
                ORDER BY saved_at DESC, snapshot_id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        snapshots: list[dict[str, Any]] = []
        for (payload,) in reversed(rows):
            value = json.loads(str(payload))
            if isinstance(value, dict):
                snapshots.append(value)
        return snapshots
    except Exception:
        return []


def has_scan_snapshot(scan_type: str, market: str, saved_at_prefix: str) -> bool:
    try:
        with _connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM scan_snapshots
                WHERE scan_type = ? AND market = ? AND saved_at LIKE ?
                LIMIT 1
                """,
                (scan_type, market, f"{saved_at_prefix}%"),
            ).fetchone()
        return row is not None
    except Exception:
        return False


def append_feature_log_entries(entries: list[dict[str, Any]]) -> None:
    if not entries:
        return
    try:
        rows = [
            (
                str(entry.get("snapshot_id", "") or ""),
                str(entry.get("scan_type", "") or ""),
                str(entry.get("market", "") or ""),
                str(entry.get("saved_at", "") or ""),
                str(entry.get("ticker", "") or ""),
                json.dumps(entry, ensure_ascii=False),
            )
            for entry in entries
            if str(entry.get("snapshot_id", "") or "") and str(entry.get("ticker", "") or "")
        ]
        if not rows:
            return
        with _connect() as conn:
            conn.executemany(
                """
                INSERT INTO feature_log_entries (snapshot_id, scan_type, market, saved_at, ticker, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
    except Exception:
        return


def seed_feature_log_entries(entries: list[dict[str, Any]]) -> None:
    if not entries:
        return
    try:
        rows = [
            (
                str(entry.get("snapshot_id", "") or ""),
                str(entry.get("scan_type", "") or ""),
                str(entry.get("market", "") or ""),
                str(entry.get("saved_at", "") or ""),
                str(entry.get("ticker", "") or ""),
                json.dumps(entry, ensure_ascii=False),
            )
            for entry in entries
            if str(entry.get("snapshot_id", "") or "") and str(entry.get("ticker", "") or "")
        ]
        if not rows:
            return
        with _connect() as conn:
            existing_count = conn.execute("SELECT COUNT(*) FROM feature_log_entries").fetchone()[0]
            if int(existing_count or 0) > 0:
                return
            conn.executemany(
                """
                INSERT INTO feature_log_entries (snapshot_id, scan_type, market, saved_at, ticker, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
    except Exception:
        return


def load_recent_feature_log_entries(limit: int = 1000) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM feature_log_entries
                ORDER BY saved_at DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        entries: list[dict[str, Any]] = []
        for (payload,) in reversed(rows):
            value = json.loads(str(payload))
            if isinstance(value, dict):
                entries.append(value)
        return entries
    except Exception:
        return []
