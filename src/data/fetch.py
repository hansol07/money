from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
import streamlit as st

from src.indicators.technical import add_indicators


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

    end = datetime.utcnow()
    start = end - timedelta(days=365 * 5 + 30)

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename_axis("Date").reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
    return add_indicators(df)


@st.cache_data(ttl=120, show_spinner=False)
def get_intraday_stock_data(ticker: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    ticker = ticker.strip().upper()
    if not ticker:
        return pd.DataFrame()

    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

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
    return df.dropna().reset_index(drop=True)


@st.cache_data(ttl=90, show_spinner=False)
def get_latest_quote(ticker: str) -> dict[str, object]:
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

    try:
        intraday = yf.download(
            ticker,
            period="1d",
            interval="5m",
            auto_adjust=True,
            progress=False,
            threads=False,
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
                daily = yf.download(
                    ticker,
                    period="5d",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
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
            return {
                "current_price": round(current_price, 4),
                "prev_close": round(prev_close, 4) if prev_close is not None else None,
                "change_pct": round(change_pct, 2) if change_pct is not None else None,
                "as_of": pd.Timestamp(as_of).strftime("%Y-%m-%d %H:%M") if not pd.isna(as_of) else "",
                "source": "5m",
            }

    try:
        daily = yf.download(
            ticker,
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        daily = pd.DataFrame()

    if daily.empty:
        return empty_result

    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)
    daily = daily.rename_axis("Date").reset_index().dropna(subset=["Close"])
    if daily.empty:
        return empty_result

    current_price = float(daily["Close"].iloc[-1])
    prev_close = float(daily["Close"].iloc[-2]) if len(daily) >= 2 else None
    change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close and prev_close > 0 else None
    as_of = pd.to_datetime(daily["Date"].iloc[-1], errors="coerce")
    return {
        "current_price": round(current_price, 4),
        "prev_close": round(prev_close, 4) if prev_close is not None else None,
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "as_of": pd.Timestamp(as_of).strftime("%Y-%m-%d") if not pd.isna(as_of) else "",
        "source": "1d",
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_dividend_yield(ticker: str) -> float:
    ticker = ticker.strip().upper()
    if not ticker:
        return 0.0

    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="1y", auto_adjust=True)
        dividends = stock.dividends
    except Exception:
        return 0.0

    if history.empty or dividends is None or len(dividends) == 0:
        return 0.0

    latest_price = float(history["Close"].iloc[-1])
    trailing_dividend = float(dividends[dividends.index >= (dividends.index.max() - pd.Timedelta(days=365))].sum())
    if latest_price <= 0:
        return 0.0
    return trailing_dividend / latest_price * 100


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

    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="5y", auto_adjust=True)
        dividends = _normalize_series_index(stock.dividends)
        info = getattr(stock, "info", {}) or {}
    except Exception:
        return empty_result

    if history.empty or dividends.empty:
        return empty_result

    latest_price = float(history["Close"].iloc[-1])
    if latest_price <= 0:
        return empty_result

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

    return {
        "dividend_yield_pct": round(dividend_yield_pct, 2),
        "annual_dividend": round(annual_dividend, 4),
        "dividend_growth_1y_pct": round(dividend_growth_1y_pct, 2),
        "dividend_growth_3y_pct": round(dividend_growth_3y_pct, 2),
        "dividend_events_1y": int(dividends[last_1y_mask].shape[0]),
        "ex_dividend_date": ex_dividend_ts.strftime("%Y-%m-%d") if ex_dividend_ts is not None else "",
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_profile(ticker: str) -> dict[str, str]:
    ticker = ticker.strip().upper()
    if not ticker:
        return {"sector": "", "industry": ""}

    try:
        stock = yf.Ticker(ticker)
        info = getattr(stock, "info", {}) or {}
    except Exception:
        return {"sector": "", "industry": ""}

    return {
        "sector": str(info.get("sector", "") or ""),
        "industry": str(info.get("industry", "") or ""),
    }
