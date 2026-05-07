from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from time import sleep
import tempfile

import pandas as pd
import yfinance as yf
import streamlit as st

from src.indicators.technical import add_indicators
from src.storage.sqlite_cache import (
    get_cached_frame,
    get_cached_json,
    get_price_bars,
    set_cached_frame,
    set_cached_json,
    set_price_bars,
)


YF_DOWNLOAD_TIMEOUT = 8
YF_DOWNLOAD_RETRIES = 2
# Keep yfinance's sqlite/tz cache outside the Korean workspace path. On Windows,
# curl_cffi/yfinance can fail with "unable to open database file" when the cache
# path contains non-ASCII segments, which previously forced the app onto stale
# local stock data.
YF_CACHE_DIR = Path(tempfile.gettempdir()) / "stock_decision_helper_yfinance_cache"


def is_recent_price_data(data: pd.DataFrame, *, max_age_days: int = 3) -> bool:
    if data.empty:
        return False
    date_column = "Date" if "Date" in data.columns else "Datetime" if "Datetime" in data.columns else None
    if date_column is None:
        return True
    latest = pd.to_datetime(data[date_column].iloc[-1], errors="coerce")
    if pd.isna(latest):
        return False
    if getattr(latest, "tzinfo", None) is not None:
        latest = latest.tz_convert(None)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return (now - pd.Timestamp(latest)).days <= max_age_days


def latest_price_timestamp(data: pd.DataFrame) -> pd.Timestamp | None:
    if data.empty:
        return None
    date_column = "Date" if "Date" in data.columns else "Datetime" if "Datetime" in data.columns else None
    if date_column is None:
        return None
    latest = pd.to_datetime(data[date_column].iloc[-1], errors="coerce")
    if pd.isna(latest):
        return None
    if getattr(latest, "tzinfo", None) is not None:
        latest = latest.tz_convert(None)
    return pd.Timestamp(latest)


def price_data_freshness_label(data: pd.DataFrame, *, intraday: bool = False) -> str:
    latest = latest_price_timestamp(data)
    if latest is None:
        return "기준없음"
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    age_days = (now - latest).days
    if intraday:
        if age_days <= 0:
            return "최신"
        if age_days <= 1:
            return "약간 지연"
        return "오래됨"
    if age_days <= 1:
        return "최신"
    if age_days <= 3:
        return "약간 지연"
    return "오래됨"


def price_source_label(data: pd.DataFrame, *, intraday: bool = False) -> str:
    if data.empty:
        return "기준없음"
    if intraday:
        return "분봉"
    return "일봉"


def _max_age_for_interval(interval: str) -> int:
    return 3 if str(interval).lower() in {"1d", "1wk", "1mo"} else 1


def _warehouse_or_empty(ticker: str, interval: str, *, intraday: bool = False) -> pd.DataFrame:
    stored = get_price_bars(ticker, interval)
    if stored.empty:
        return pd.DataFrame()
    if intraday:
        stored = stored.rename(columns={"Date": "Datetime"})
        stored["price_change_pct"] = stored["Close"].pct_change() * 100
        stored["volume_ma20"] = stored["Volume"].rolling(20).mean()
        stored["volume_ratio"] = stored["Volume"] / stored["volume_ma20"]
        stored["short_return_pct"] = stored["Close"].pct_change(3) * 100
        stored["session_high_20"] = stored["High"].rolling(20).max()
        stored["session_low_20"] = stored["Low"].rolling(20).min()
        typical_price = (stored["High"] + stored["Low"] + stored["Close"]) / 3
        cumulative_volume = stored["Volume"].cumsum().replace(0, pd.NA)
        stored["vwap_proxy"] = (typical_price * stored["Volume"]).cumsum() / cumulative_volume
        return stored.dropna().reset_index(drop=True)
    return add_indicators(stored)

try:
    YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(YF_CACHE_DIR))
    if hasattr(yf, "cache") and hasattr(yf.cache, "set_cache_location"):
        yf.cache.set_cache_location(str(YF_CACHE_DIR))
except Exception:
    pass


def _download_yf(ticker: str, **kwargs: object) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(YF_DOWNLOAD_RETRIES):
        try:
            return yf.download(
                ticker,
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=YF_DOWNLOAD_TIMEOUT,
                **kwargs,
            )
        except Exception as exc:
            last_error = exc
            if attempt + 1 < YF_DOWNLOAD_RETRIES:
                sleep(0.25)
    if last_error is not None:
        raise last_error
    return pd.DataFrame()


def _cache_only_mode() -> bool:
    try:
        return bool(st.session_state.get("cache_only_mode", False))
    except Exception:
        return False


def _stale_frame_or_empty(cache_key: str, date_columns: list[str] | None = None) -> pd.DataFrame:
    stale = get_cached_frame(cache_key, date_columns=date_columns, allow_expired=True)
    if stale is not None and not stale.empty:
        return stale
    return pd.DataFrame()


def _stale_json_or_default(cache_key: str, default: dict[str, object]) -> dict[str, object]:
    stale = get_cached_json(cache_key, allow_expired=True)
    if stale is not None:
        return {**default, **stale}
    return default


def _normalize_timestamp(value: object) -> pd.Timestamp | None:
    if value in (None, "", 0):
        return None
    try:
        ts = pd.to_datetime(value, unit="s", errors="coerce")
    except Exception:
        ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is not None:
        try:
            ts = ts.tz_convert(None)
        except Exception:
            ts = ts.tz_localize(None)
    return pd.Timestamp(ts)


def _quote_is_recent(quote: dict[str, object]) -> bool:
    as_of = str(quote.get("as_of", "") or "").strip()
    if not as_of:
        return False
    source = str(quote.get("source", "") or "").strip().lower()
    max_age_days = 1 if source and source != "1d" else 3
    ts = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(ts):
        return False
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return (now - pd.Timestamp(ts)).days <= max_age_days


def _classify_event_risk(*values: object) -> str:
    has_medium = False
    for value in values:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            continue
        days_left = float(numeric)
        if 0 <= days_left <= 7:
            return "높음"
        if 8 <= days_left <= 21:
            has_medium = True
    return "중간" if has_medium else "낮음"


def _normalize_series_index(series: pd.Series) -> pd.Series:
    cleaned = series.copy()
    if cleaned.empty:
        return cleaned
    index = pd.to_datetime(cleaned.index, errors="coerce")
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert(None)
    cleaned.index = index
    cleaned = cleaned[~cleaned.index.isna()]
    return cleaned.sort_index()


@st.cache_data(ttl=900, show_spinner=False)
def get_stock_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    ticker = ticker.strip().upper()
    if not ticker:
        return pd.DataFrame()

    cache_key = f"stock_data:{ticker}:{period}:{interval}"
    max_age_days = _max_age_for_interval(interval)
    cached = get_cached_frame(cache_key, date_columns=["Date"])
    if cached is not None and not cached.empty and is_recent_price_data(cached, max_age_days=max_age_days):
        set_price_bars(ticker, interval, cached, source="streamlit_cache")
        return cached
    if _cache_only_mode():
        warehouse = _warehouse_or_empty(ticker, interval, intraday=False)
        if not warehouse.empty:
            return warehouse
        return _stale_frame_or_empty(cache_key, date_columns=["Date"])

    end = datetime.utcnow()
    start = end - timedelta(days=365 * 5 + 30)

    try:
        df = _download_yf(
            ticker,
            start=start,
            end=end,
            interval=interval,
        )
    except Exception:
        return _stale_frame_or_empty(cache_key, date_columns=["Date"])

    if df.empty:
        return _stale_frame_or_empty(cache_key, date_columns=["Date"])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename_axis("Date").reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
    result = add_indicators(df)
    set_cached_frame(cache_key, result, ttl_seconds=60 * 60 * 6)
    set_price_bars(ticker, interval, df, source="yfinance")
    return result


@st.cache_data(ttl=120, show_spinner=False)
def get_intraday_stock_data(
    ticker: str,
    period: str = "5d",
    interval: str = "5m",
    force_refresh: bool = False,
) -> pd.DataFrame:
    ticker = ticker.strip().upper()
    if not ticker:
        return pd.DataFrame()

    cache_key = f"intraday:{ticker}:{period}:{interval}"
    cached = None if force_refresh else get_cached_frame(cache_key, date_columns=["Datetime"])
    if cached is not None and not cached.empty and is_recent_price_data(cached, max_age_days=1):
        cache_for_bars = cached.rename(columns={"Datetime": "Date"}) if "Datetime" in cached.columns else cached
        set_price_bars(ticker, interval, cache_for_bars, source="streamlit_cache")
        return cached
    if _cache_only_mode():
        warehouse = _warehouse_or_empty(ticker, interval, intraday=True)
        if not warehouse.empty:
            return warehouse
        return _stale_frame_or_empty(cache_key, date_columns=["Datetime"])

    try:
        df = _download_yf(
            ticker,
            period=period,
            interval=interval,
        )
    except Exception:
        return _stale_frame_or_empty(cache_key, date_columns=["Datetime"])

    if df.empty:
        return _stale_frame_or_empty(cache_key, date_columns=["Datetime"])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename_axis("Datetime").reset_index()
    time_column = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={time_column: "Datetime"})
    df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]].dropna()
    df["price_change_pct"] = df["Close"].pct_change() * 100
    df["volume_ma20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma20"]
    df["short_return_pct"] = df["Close"].pct_change(3) * 100
    df["session_high_20"] = df["High"].rolling(20).max()
    df["session_low_20"] = df["Low"].rolling(20).min()
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cumulative_volume = df["Volume"].cumsum().replace(0, pd.NA)
    df["vwap_proxy"] = (typical_price * df["Volume"]).cumsum() / cumulative_volume
    result = df.dropna().reset_index(drop=True)
    set_cached_frame(cache_key, result, ttl_seconds=120)
    set_price_bars(ticker, interval, df, source="yfinance")
    return result


@st.cache_data(ttl=90, show_spinner=False)
def get_latest_quote(ticker: str, force_refresh: bool = False) -> dict[str, object]:
    ticker = ticker.strip().upper()
    empty_result = {
        "current_price": None,
        "prev_close": None,
        "change_pct": None,
        "as_of": "",
        "source": "",
    }
    if not ticker:
        return empty_result

    cache_key = f"latest_quote:{ticker}"
    cached = None if force_refresh else get_cached_json(cache_key)
    if cached is not None and _quote_is_recent(cached):
        return cached
    if _cache_only_mode():
        return _stale_json_or_default(cache_key, empty_result)

    try:
        intraday = _download_yf(
            ticker,
            period="1d",
            interval="5m",
        )
    except Exception:
        intraday = pd.DataFrame()

    if not intraday.empty:
        if isinstance(intraday.columns, pd.MultiIndex):
            intraday.columns = intraday.columns.get_level_values(0)
        intraday = intraday.rename_axis("Datetime").reset_index()
        time_column = "Datetime" if "Datetime" in intraday.columns else "Date"
        intraday = intraday.rename(columns={time_column: "Datetime"})
        intraday = intraday.dropna(subset=["Close"])
        if not intraday.empty:
            current_price = float(intraday["Close"].iloc[-1])
            as_of = pd.to_datetime(intraday["Datetime"].iloc[-1], errors="coerce")
            prev_close = None
            change_pct = None
            try:
                daily = _download_yf(
                    ticker,
                    period="5d",
                    interval="1d",
                )
            except Exception:
                daily = pd.DataFrame()
            if not daily.empty:
                if isinstance(daily.columns, pd.MultiIndex):
                    daily.columns = daily.columns.get_level_values(0)
                daily = daily.dropna(subset=["Close"])
                if len(daily) >= 2:
                    prev_close = float(daily["Close"].iloc[-2])
                    if prev_close > 0:
                        change_pct = (current_price - prev_close) / prev_close * 100
            result = {
                "current_price": round(current_price, 4),
                "prev_close": round(prev_close, 4) if prev_close is not None else None,
                "change_pct": round(change_pct, 2) if change_pct is not None else None,
                "as_of": pd.Timestamp(as_of).strftime("%Y-%m-%d %H:%M") if not pd.isna(as_of) else "",
                "source": "5m",
            }
            set_cached_json(cache_key, result, ttl_seconds=90)
            return result

    try:
        daily = _download_yf(
            ticker,
            period="5d",
            interval="1d",
        )
    except Exception:
        daily = pd.DataFrame()

    if daily.empty:
        return _stale_json_or_default(cache_key, empty_result)

    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)
    daily = daily.rename_axis("Date").reset_index().dropna(subset=["Close"])
    if daily.empty:
        return _stale_json_or_default(cache_key, empty_result)

    current_price = float(daily["Close"].iloc[-1])
    prev_close = float(daily["Close"].iloc[-2]) if len(daily) >= 2 else None
    change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close and prev_close > 0 else None
    as_of = pd.to_datetime(daily["Date"].iloc[-1], errors="coerce")
    result = {
        "current_price": round(current_price, 4),
        "prev_close": round(prev_close, 4) if prev_close is not None else None,
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "as_of": pd.Timestamp(as_of).strftime("%Y-%m-%d") if not pd.isna(as_of) else "",
        "source": "1d",
    }
    set_cached_json(cache_key, result, ttl_seconds=90)
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_dividend_yield(ticker: str) -> float:
    ticker = ticker.strip().upper()
    if not ticker:
        return 0.0

    cache_key = f"dividend_yield:v2:{ticker}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return float(cached.get("dividend_yield_pct", 0.0) or 0.0)
    if _cache_only_mode():
        stale = get_cached_json(cache_key, allow_expired=True)
        return float(stale.get("dividend_yield_pct", 0.0) or 0.0) if stale is not None else 0.0

    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="1y", auto_adjust=True)
        dividends = stock.dividends
    except Exception:
        stale = get_cached_json(cache_key, allow_expired=True)
        return float(stale.get("dividend_yield_pct", 0.0) or 0.0) if stale is not None else 0.0

    if history.empty or dividends is None or len(dividends) == 0:
        stale = get_cached_json(cache_key, allow_expired=True)
        return float(stale.get("dividend_yield_pct", 0.0) or 0.0) if stale is not None else 0.0

    latest_price = float(history["Close"].iloc[-1])
    trailing_dividend = float(dividends[dividends.index >= (dividends.index.max() - pd.Timedelta(days=365))].sum())
    if latest_price <= 0:
        stale = get_cached_json(cache_key, allow_expired=True)
        return float(stale.get("dividend_yield_pct", 0.0) or 0.0) if stale is not None else 0.0
    result = trailing_dividend / latest_price * 100
    set_cached_json(cache_key, {"dividend_yield_pct": round(float(result), 4)}, ttl_seconds=60 * 60 * 24)
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_dividend_details(ticker: str) -> dict[str, object]:
    ticker = ticker.strip().upper()
    empty_result = {
        "dividend_yield_pct": 0.0,
        "annual_dividend": 0.0,
        "dividend_growth_1y_pct": 0.0,
        "dividend_growth_3y_pct": 0.0,
        "dividend_events_1y": 0,
        "ex_dividend_date": "",
    }
    if not ticker:
        return empty_result

    cache_key = f"dividend_details:v2:{ticker}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return {**empty_result, **cached}
    if _cache_only_mode():
        return _stale_json_or_default(cache_key, empty_result)

    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="5y", auto_adjust=True)
        dividends = _normalize_series_index(stock.dividends)
        info = getattr(stock, "info", {}) or {}
    except Exception:
        return _stale_json_or_default(cache_key, empty_result)

    if history.empty or dividends.empty:
        return _stale_json_or_default(cache_key, empty_result)

    latest_price = float(history["Close"].iloc[-1])
    if latest_price <= 0:
        return _stale_json_or_default(cache_key, empty_result)

    today = pd.Timestamp.utcnow().tz_localize(None)
    last_1y_mask = dividends.index >= (today - pd.Timedelta(days=365))
    prev_1y_mask = (dividends.index >= (today - pd.Timedelta(days=730))) & (dividends.index < (today - pd.Timedelta(days=365)))

    annual_dividend = float(dividends[last_1y_mask].sum())
    previous_annual_dividend = float(dividends[prev_1y_mask].sum())
    dividend_yield_pct = annual_dividend / latest_price * 100 if latest_price > 0 else 0.0
    dividend_growth_1y_pct = 0.0
    if previous_annual_dividend > 0:
        dividend_growth_1y_pct = (annual_dividend / previous_annual_dividend - 1) * 100

    yearly_dividends = dividends.groupby(dividends.index.year).sum().sort_index()
    dividend_growth_3y_pct = 0.0
    if len(yearly_dividends) >= 4:
        start_value = float(yearly_dividends.iloc[-4])
        end_value = float(yearly_dividends.iloc[-1])
        if start_value > 0 and end_value > 0:
            dividend_growth_3y_pct = ((end_value / start_value) ** (1 / 3) - 1) * 100

    ex_dividend_ts = _normalize_timestamp(info.get("exDividendDate"))
    if ex_dividend_ts is None and not dividends.empty:
        ex_dividend_ts = pd.Timestamp(dividends.index.max())

    result = {
        "dividend_yield_pct": round(dividend_yield_pct, 2),
        "annual_dividend": round(annual_dividend, 4),
        "dividend_growth_1y_pct": round(dividend_growth_1y_pct, 2),
        "dividend_growth_3y_pct": round(dividend_growth_3y_pct, 2),
        "dividend_events_1y": int(dividends[last_1y_mask].shape[0]),
        "ex_dividend_date": ex_dividend_ts.strftime("%Y-%m-%d") if ex_dividend_ts is not None else "",
    }
    set_cached_json(cache_key, result, ttl_seconds=60 * 60 * 24)
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_profile(ticker: str) -> dict[str, str]:
    ticker = ticker.strip().upper()
    if not ticker:
        return {"sector": "", "industry": ""}

    cache_key = f"stock_profile:v2:{ticker}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return {
            "sector": str(cached.get("sector", "") or ""),
            "industry": str(cached.get("industry", "") or ""),
        }
    if _cache_only_mode():
        stale = get_cached_json(cache_key, allow_expired=True)
        if stale is not None:
            return {
                "sector": str(stale.get("sector", "") or ""),
                "industry": str(stale.get("industry", "") or ""),
            }
        return {"sector": "", "industry": ""}

    try:
        stock = yf.Ticker(ticker)
        info = getattr(stock, "info", {}) or {}
    except Exception:
        stale = get_cached_json(cache_key, allow_expired=True)
        if stale is not None:
            return {
                "sector": str(stale.get("sector", "") or ""),
                "industry": str(stale.get("industry", "") or ""),
            }
        return {"sector": "", "industry": ""}

    result = {
        "sector": str(info.get("sector", "") or ""),
        "industry": str(info.get("industry", "") or ""),
    }
    if result["sector"] or result["industry"]:
        set_cached_json(cache_key, result, ttl_seconds=60 * 60 * 24)
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_event_summary(ticker: str) -> dict[str, object]:
    ticker = ticker.strip().upper()
    empty_result = {
        "earnings_date": "",
        "earnings_days_left": None,
        "ex_dividend_date": "",
        "ex_dividend_days_left": None,
        "event_risk": "낮음",
        "event_note": "",
    }
    if not ticker:
        return empty_result

    cache_key = f"event_summary:{ticker}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        result = {**empty_result, **cached}
        result["event_risk"] = str(
            result.get("event_risk")
            or _classify_event_risk(result.get("earnings_days_left"), result.get("ex_dividend_days_left"))
        )
        return result
    if _cache_only_mode():
        result = _stale_json_or_default(cache_key, empty_result)
        result["event_risk"] = str(
            result.get("event_risk")
            or _classify_event_risk(result.get("earnings_days_left"), result.get("ex_dividend_days_left"))
        )
        return result

    try:
        stock = yf.Ticker(ticker)
        calendar = getattr(stock, "calendar", None)
        info = getattr(stock, "info", {}) or {}
    except Exception:
        result = _stale_json_or_default(cache_key, empty_result)
        result["event_risk"] = str(
            result.get("event_risk")
            or _classify_event_risk(result.get("earnings_days_left"), result.get("ex_dividend_days_left"))
        )
        return result

    today = pd.Timestamp.now().normalize()
    earnings_ts: pd.Timestamp | None = None
    ex_dividend_ts = _normalize_timestamp(info.get("exDividendDate"))

    try:
        if calendar is not None and not getattr(calendar, "empty", True):
            if isinstance(calendar, pd.DataFrame):
                values = calendar.values.flatten().tolist()
            elif isinstance(calendar, dict):
                values = list(calendar.values())
            else:
                values = []
            for value in values:
                ts = _normalize_timestamp(value)
                if ts is not None:
                    earnings_ts = ts.normalize()
                    break
    except Exception:
        earnings_ts = None

    event_notes: list[str] = []
    earnings_days_left = None
    ex_dividend_days_left = None

    if earnings_ts is not None:
        earnings_days_left = int((earnings_ts - today).days)
        if 0 <= earnings_days_left <= 14:
            event_notes.append(f"실적 발표 {earnings_days_left}일 전")

    if ex_dividend_ts is not None:
        ex_dividend_ts = ex_dividend_ts.normalize()
        ex_dividend_days_left = int((ex_dividend_ts - today).days)
        if 0 <= ex_dividend_days_left <= 14:
            event_notes.append(f"배당락 {ex_dividend_days_left}일 전")

    event_note = " / ".join(event_notes) if event_notes else "가까운 핵심 일정은 아직 크지 않습니다."
    event_risk = _classify_event_risk(earnings_days_left, ex_dividend_days_left)
    result = {
        "earnings_date": earnings_ts.strftime("%Y-%m-%d") if earnings_ts is not None else "",
        "earnings_days_left": earnings_days_left,
        "ex_dividend_date": ex_dividend_ts.strftime("%Y-%m-%d") if ex_dividend_ts is not None else "",
        "ex_dividend_days_left": ex_dividend_days_left,
        "event_risk": event_risk,
        "event_note": event_note,
    }
    set_cached_json(cache_key, result, ttl_seconds=60 * 60 * 6)
    return result


@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_news_summary(ticker: str) -> dict[str, object]:
    ticker = ticker.strip().upper()
    empty_result = {
        "news_count": 0,
        "news_score": 0,
        "news_bias": "중립",
        "headline": "",
        "news_note": "최근 뉴스 데이터가 많지 않습니다.",
    }
    if not ticker:
        return empty_result

    cache_key = f"news_summary:{ticker}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return cached
    if _cache_only_mode():
        return _stale_json_or_default(cache_key, empty_result)

    try:
        stock = yf.Ticker(ticker)
        news_items = getattr(stock, "news", []) or []
    except Exception:
        return _stale_json_or_default(cache_key, empty_result)

    if not news_items:
        return _stale_json_or_default(cache_key, empty_result)

    positive_keywords = ["beat", "upgrade", "surge", "growth", "record", "partnership", "strong", "raises", "gain"]
    negative_keywords = ["miss", "downgrade", "lawsuit", "probe", "delay", "weak", "fall", "cuts", "drop"]

    score = 0
    first_title = ""
    for index, item in enumerate(news_items[:10]):
        title = str(item.get("title", "") or "")
        if index == 0:
            first_title = title
        lowered = title.lower()
        score += sum(1 for keyword in positive_keywords if keyword in lowered)
        score -= sum(1 for keyword in negative_keywords if keyword in lowered)

    if score >= 2:
        bias = "긍정"
        note = "최근 뉴스 흐름이 비교적 우호적입니다."
    elif score <= -2:
        bias = "부정"
        note = "최근 뉴스 흐름이 다소 부담스럽습니다."
    else:
        bias = "중립"
        note = "최근 뉴스 흐름이 한쪽으로 강하게 쏠리진 않습니다."

    result = {
        "news_count": min(len(news_items), 10),
        "news_score": int(score),
        "news_bias": bias,
        "headline": first_title,
        "news_note": note,
    }
    set_cached_json(cache_key, result, ttl_seconds=60 * 30)
    return result
