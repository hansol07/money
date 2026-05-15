from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.fetch import _download_yahoo_chart, is_recent_price_data, latest_price_timestamp
from src.storage.sqlite_cache import get_price_bars, get_price_warehouse_stats, set_price_bars
from src.strategy.universe import get_market_sweep_universe, is_tradable_ticker


@dataclass(slots=True)
class BackfillTask:
    market: str
    ticker: str
    name: str
    interval: str
    period: str | None
    start: datetime | None
    end: datetime | None
    max_age_days: int


@dataclass(slots=True)
class BackfillResult:
    market: str
    ticker: str
    interval: str
    status: str
    rows: int
    latest: str
    message: str = ""


def _unique_universe(markets: Iterable[str], limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for market in markets:
        for item in get_market_sweep_universe(market):
            ticker = str(item.get("ticker", "") or "").strip().upper()
            if not is_tradable_ticker(ticker):
                continue
            key = (market, ticker)
            if key in seen:
                continue
            rows.append({"market": market, "ticker": ticker, "name": str(item.get("name", "") or "")})
            seen.add(key)
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def _stored_is_fresh(ticker: str, interval: str, max_age_days: int) -> tuple[bool, str]:
    stored = get_price_bars(ticker, interval)
    latest = latest_price_timestamp(stored)
    latest_text = pd.Timestamp(latest).strftime("%Y-%m-%d %H:%M") if latest is not None else ""
    return bool(not stored.empty and is_recent_price_data(stored, max_age_days=max_age_days)), latest_text


def _run_task(task: BackfillTask, *, force: bool, retries: int, retry_sleep: float) -> BackfillResult:
    if not force:
        fresh, latest_text = _stored_is_fresh(task.ticker, task.interval, task.max_age_days)
        if fresh:
            return BackfillResult(task.market, task.ticker, task.interval, "skip_fresh", 0, latest_text)

    last_message = ""
    for attempt in range(max(1, retries)):
        frame = _download_yahoo_chart(
            task.ticker,
            start=task.start,
            end=task.end,
            period=task.period,
            interval=task.interval,
        )
        if not frame.empty:
            rows = set_price_bars(task.ticker, task.interval, frame, source="yahoo_chart_backfill")
            latest = latest_price_timestamp(frame)
            latest_text = pd.Timestamp(latest).strftime("%Y-%m-%d %H:%M") if latest is not None else ""
            return BackfillResult(task.market, task.ticker, task.interval, "saved", rows, latest_text)
        last_message = "empty response"
        if attempt + 1 < retries:
            sleep(retry_sleep)
    return BackfillResult(task.market, task.ticker, task.interval, "failed", 0, "", last_message)


def _build_tasks(args: argparse.Namespace) -> list[BackfillTask]:
    markets = [market.upper() for market in args.markets]
    universe = _unique_universe(markets, limit=args.limit)
    now = datetime.utcnow()
    daily_start = now - timedelta(days=int(args.daily_years * 365) + 30)
    tasks: list[BackfillTask] = []
    for item in universe:
        market = str(item["market"])
        ticker = str(item["ticker"])
        name = str(item["name"])
        if "daily" in args.intervals:
            tasks.append(
                BackfillTask(
                    market=market,
                    ticker=ticker,
                    name=name,
                    interval="1d",
                    period=None,
                    start=daily_start,
                    end=now,
                    max_age_days=3,
                )
            )
        if "intraday" in args.intervals:
            tasks.append(
                BackfillTask(
                    market=market,
                    ticker=ticker,
                    name=name,
                    interval=args.intraday_interval,
                    period=args.intraday_period,
                    start=None,
                    end=None,
                    max_age_days=1,
                )
            )
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill local price warehouse for the stock helper app.")
    parser.add_argument("--markets", nargs="+", default=["US", "KR"], choices=["US", "KR"])
    parser.add_argument("--intervals", nargs="+", default=["daily", "intraday"], choices=["daily", "intraday"])
    parser.add_argument("--daily-years", type=float, default=5.0)
    parser.add_argument("--intraday-period", default="5d")
    parser.add_argument("--intraday-interval", default="5m")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Optional total ticker limit for smoke tests.")
    parser.add_argument("--force", action="store_true", help="Refresh even when stored data is already fresh.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()

    tasks = _build_tasks(args)
    if not tasks:
        print("No tasks.")
        return 0

    print(f"Backfill start: tasks={len(tasks)} workers={args.workers} force={args.force}")
    print("Before:")
    stats = get_price_warehouse_stats()
    print(stats.to_string(index=False) if not stats.empty else "(empty)")

    counts: dict[str, int] = {"saved": 0, "skip_fresh": 0, "failed": 0}
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [
            executor.submit(_run_task, task, force=args.force, retries=args.retries, retry_sleep=args.retry_sleep)
            for task in tasks
        ]
        for future in as_completed(futures):
            result = future.result()
            counts[result.status] = counts.get(result.status, 0) + 1
            completed += 1
            if result.status != "skip_fresh":
                print(
                    f"{completed}/{len(tasks)} {result.status.upper()} "
                    f"{result.market} {result.ticker} {result.interval} rows={result.rows} latest={result.latest} {result.message}"
                )
            elif completed % max(1, int(args.progress_every)) == 0:
                print(f"{completed}/{len(tasks)} progress saved={counts.get('saved', 0)} skip={counts.get('skip_fresh', 0)} failed={counts.get('failed', 0)}")

    print("After:")
    stats = get_price_warehouse_stats()
    print(stats.to_string(index=False) if not stats.empty else "(empty)")
    print(f"Done: {counts}")
    return 0 if counts.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
