from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pandas as pd

from src.data.fetch import get_stock_data
from src.storage.local_store import append_scan_history, has_scan_snapshot_for_day, has_scan_snapshot_for_prefix
from src.strategy.auto_candidates import build_auto_candidate_sets
from src.strategy.learning import build_learning_adjustments
from src.strategy.profiles import build_high_risk_trade_candidates, build_short_term_trade_candidates
from src.strategy.realtime import scan_intraday_market


AUTO_SCAN_TYPES = ["monthly", "weekly", "daily", "next_day"]
AUTO_MARKETS = ["US", "KR"]
MARKET_TIMEZONES = {"US": "America/New_York", "KR": "Asia/Seoul"}
MARKET_CLOSE_HOUR = {"US": 17, "KR": 16}
MARKET_BENCHMARKS = {"US": "SPY", "KR": "005930.KS"}
MARKET_OPEN_MINUTES = {"US": 9 * 60 + 30, "KR": 9 * 60}
MARKET_CLOSE_MINUTES = {"US": 16 * 60, "KR": 15 * 60 + 30}
INTRADAY_SCAN_TYPES = ["realtime_scan", "short_term_trade", "high_risk_trade"]


@dataclass
class SchedulerStatus:
    running: bool = False
    interval_seconds: int = 1800
    last_run_at: str = ""
    last_result: str = "대기중"
    saved_snapshots: int = 0
    error_message: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "running": self.running,
                "interval_seconds": self.interval_seconds,
                "last_run_at": self.last_run_at,
                "last_result": self.last_result,
                "saved_snapshots": self.saved_snapshots,
                "error_message": self.error_message,
            }

    def update(self, **kwargs: object) -> None:
        with self.lock:
            for key, value in kwargs.items():
                setattr(self, key, value)


class BackgroundAnalyzer:
    def __init__(self, interval_seconds: int = 1800) -> None:
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.status = SchedulerStatus(interval_seconds=interval_seconds)

    def _market_now(self, market: str) -> pd.Timestamp:
        return pd.Timestamp.now(tz=MARKET_TIMEZONES.get(market, "Asia/Seoul"))

    def _is_market_session_ready(self, market: str) -> tuple[bool, str]:
        market_now = self._market_now(market)
        if market_now.weekday() >= 5:
            return False, "주말이라 자동 저장을 건너뜁니다."

        close_hour = MARKET_CLOSE_HOUR.get(market, 17)
        if market_now.hour < close_hour:
            return False, "아직 장마감 전이라 자동 저장을 기다립니다."

        benchmark = MARKET_BENCHMARKS.get(market, "")
        data = get_stock_data(benchmark)
        if data.empty:
            return False, "기준 종목 데이터를 못 불러와 저장을 건너뜁니다."

        latest_date = pd.to_datetime(data["Date"].iloc[-1], errors="coerce")
        if pd.isna(latest_date):
            return False, "최신 거래일 확인에 실패했습니다."

        if latest_date.date() != market_now.date():
            return False, "오늘은 해당 시장 휴장일이라 자동 저장을 건너뜁니다."

        return True, ""

    def _is_market_open_now(self, market: str) -> tuple[bool, str]:
        market_now = self._market_now(market)
        if market_now.weekday() >= 5:
            return False, "주말이라 장중 저장을 건너뜁니다."

        benchmark = MARKET_BENCHMARKS.get(market, "")
        data = get_stock_data(benchmark)
        if data.empty:
            return False, "기준 종목 데이터를 못 불러와 장중 저장을 건너뜁니다."

        latest_date = pd.to_datetime(data["Date"].iloc[-1], errors="coerce")
        if pd.isna(latest_date) or latest_date.date() != market_now.date():
            return False, "오늘은 해당 시장 휴장일이라 장중 저장을 건너뜁니다."

        current_minutes = market_now.hour * 60 + market_now.minute
        open_minutes = MARKET_OPEN_MINUTES.get(market, 9 * 60)
        close_minutes = MARKET_CLOSE_MINUTES.get(market, 16 * 60)
        if current_minutes < open_minutes or current_minutes > close_minutes:
            return False, "장중 시간이 아니라 장중 저장을 건너뜁니다."

        return True, ""

    def _half_hour_slot_key(self, market: str) -> str:
        market_now = self._market_now(market)
        minute_bucket = 0 if market_now.minute < 30 else 30
        slot = market_now.replace(minute=minute_bucket, second=0, microsecond=0)
        return slot.strftime("%Y-%m-%dT%H:%M")

    def _save_intraday_snapshots(self, market: str) -> tuple[int, list[str]]:
        ready, note = self._is_market_open_now(market)
        if not ready:
            return 0, [f"{market}: {note}"]

        slot_key = self._half_hour_slot_key(market)
        learning_adjustments, _, event_adjustments, news_adjustments, _ = build_learning_adjustments(limit=400, min_samples=3)
        saved_count = 0
        notes: list[str] = []

        realtime_frame = scan_intraday_market(
            market,
            interval="5m",
            min_score=60,
            learning_adjustments=learning_adjustments,
            event_adjustments=event_adjustments,
            news_adjustments=news_adjustments,
        )
        short_term_frame = build_short_term_trade_candidates(
            market,
            top_n=8,
            interval="5m",
            min_score=65,
            learning_adjustments=learning_adjustments,
            event_adjustments=event_adjustments,
            news_adjustments=news_adjustments,
        )
        high_risk_frame = build_high_risk_trade_candidates(
            market,
            top_n=8,
            interval="5m",
            min_score=60,
            learning_adjustments=learning_adjustments,
            event_adjustments=event_adjustments,
            news_adjustments=news_adjustments,
        )

        for scan_type, frame in [
            ("realtime_scan", realtime_frame),
            ("short_term_trade", short_term_frame),
            ("high_risk_trade", high_risk_frame),
        ]:
            if has_scan_snapshot_for_prefix(scan_type, market, slot_key):
                notes.append(f"{market}: {scan_type}는 현재 슬롯 저장분이 이미 있습니다.")
                continue
            if frame.empty:
                notes.append(f"{market}: {scan_type} 후보가 없습니다.")
                continue
            append_scan_history(scan_type, market, frame)
            saved_count += 1

        return saved_count, notes

    def _run_cycle(self) -> int:
        now = pd.Timestamp.now(tz="Asia/Seoul")
        notes: list[str] = []
        saved_count = 0

        for market in AUTO_MARKETS:
            market_now = self._market_now(market)
            day_key = market_now.strftime("%Y-%m-%d")
            ready, note = self._is_market_session_ready(market)
            if not ready:
                notes.append(f"{market}: {note}")
                continue

            candidate_sets = build_auto_candidate_sets(market, top_n=12)
            for scan_type in AUTO_SCAN_TYPES:
                if has_scan_snapshot_for_day(scan_type, market, day_key):
                    continue
                frame = candidate_sets.get(scan_type, pd.DataFrame())
                if frame.empty:
                    continue
                append_scan_history(scan_type, market, frame)
                saved_count += 1

            intraday_saved, intraday_notes = self._save_intraday_snapshots(market)
            saved_count += intraday_saved
            notes.extend(intraday_notes)

        result_note = "자동 저장 완료" if saved_count > 0 else "오늘 저장분 이미 있거나 휴장일입니다."
        if notes and saved_count == 0:
            result_note = " / ".join(notes[:2])

        self.status.update(
            last_run_at=now.isoformat(timespec="seconds"),
            last_result=result_note,
            saved_snapshots=saved_count,
            error_message="",
        )
        return saved_count

    def run_once(self) -> int:
        try:
            return self._run_cycle()
        except Exception as error:
            self.status.update(
                last_result="오류",
                error_message=str(error),
                last_run_at=pd.Timestamp.now(tz="Asia/Seoul").isoformat(timespec="seconds"),
            )
            return 0

    def _loop(self) -> None:
        self.status.update(running=True)
        while not self.stop_event.is_set():
            self.run_once()
            self.stop_event.wait(self.interval_seconds)
        self.status.update(running=False)

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, name="background-analyzer", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        self.status.update(running=False)

    def get_status(self) -> dict[str, object]:
        return self.status.snapshot()
