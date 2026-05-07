from __future__ import annotations

from html import escape
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.automation.scheduler import BackgroundAnalyzer
from src.backtest.engine import run_backtest
from src.data.fetch import (
    get_intraday_stock_data,
    is_recent_price_data,
    latest_price_timestamp,
    get_latest_quote,
    get_stock_event_summary,
    get_stock_news_summary,
    get_stock_data,
)
from src.data.universe_refresh import refresh_universe_files
from src.portfolio.analytics import analyze_portfolio, build_portfolio_outlook, build_rebalance_suggestions
from src.portfolio.models import PositionInput
from src.storage.local_store import (
    append_daily_briefing,
    append_daily_briefing_actions,
    append_decision_log,
    append_scan_history,
    append_manual_tracking,
    has_scan_snapshot_for_day,
    has_scan_snapshot_for_prefix,
    load_daily_briefing_actions,
    load_daily_briefings,
    load_decision_log,
    load_manual_tracking,
    load_portfolio,
    load_recent_feature_log,
    load_recent_scan_history,
    load_watchlists,
    normalize_portfolio_frame,
    remove_manual_tracking,
    save_portfolio,
    save_watchlists,
)
from src.storage.sqlite_cache import clear_sqlite_cache, get_cache_key_status, get_price_warehouse_stats, get_sqlite_cache_stats
from src.strategy.auto_candidates import build_auto_candidate_sets, build_compounder_candidates
from src.strategy.dividend import build_dividend_profiles
from src.strategy.learning import apply_context_adjustment, build_learning_adjustments
from src.strategy.profiles import build_high_risk_trade_candidates, build_short_term_trade_candidates, build_strategy_profiles
from src.strategy.regime import classify_market_regime
from src.strategy.recommendation import analyze_position
from src.strategy.realtime import scan_intraday_market
from src.strategy.scanner import scan_market
from src.strategy.tracker import evaluate_scan_history
from src.strategy.universe import get_default_watchlists, get_market_sweep_universe, is_tradable_ticker, normalize_watchlist_frame
from src.ui.charts import build_price_chart


st.set_page_config(
    page_title="Stock Decision Helper",
    page_icon="📈",
    layout="wide",
)


def _inject_ui_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f5f2ea;
            --panel: #fffdf8;
            --panel-strong: #f7efe1;
            --ink: #17212f;
            --muted: #667085;
            --line: #e3d8c8;
            --accent: #1f6f5b;
            --accent-soft: #dceee8;
            --danger: #b54708;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 12% 0%, rgba(31, 111, 91, 0.12), transparent 30rem),
                linear-gradient(135deg, #f8f4eb 0%, #eef3ef 48%, #f7f0e4 100%);
            color: var(--ink);
        }
        [data-testid="stHeader"] {
            background: rgba(245, 242, 234, 0.72);
            backdrop-filter: blur(12px);
            min-height: 2.75rem;
        }
        .block-container {
            max-width: 1480px;
            padding-top: 2.6rem;
            padding-bottom: 3rem;
            overflow: visible;
        }
        h1, h2, h3 {
            letter-spacing: -0.035em;
            color: var(--ink);
        }
        h1 {
            font-size: 2rem !important;
            margin-bottom: 0.15rem !important;
        }
        h2, h3 {
            margin-top: 0.55rem !important;
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #17212f 0%, #223326 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        div[data-testid="stSidebar"] label,
        div[data-testid="stSidebar"] p,
        div[data-testid="stSidebar"] span,
        div[data-testid="stSidebar"] small {
            color: #f8f4eb;
        }
        div[data-testid="stSidebar"] .stCaptionContainer p {
            color: rgba(248, 244, 235, 0.74);
        }
        div[data-testid="stSidebar"] [data-testid="stExpander"] {
            border-color: rgba(255, 255, 255, 0.16);
            background: rgba(255, 255, 255, 0.06);
        }
        div[data-testid="stAppViewContainer"] section.main div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {
            gap: 1rem;
        }
        [data-testid="stSelectbox"],
        [data-testid="stNumberInput"],
        [data-testid="stTextInput"],
        [data-testid="stSlider"],
        [data-testid="stFileUploader"],
        [data-testid="stDataEditor"],
        [data-testid="stDataFrame"] {
            margin-bottom: 0.35rem;
        }
        div[role="radiogroup"] {
            gap: 0.5rem;
            padding: 0.2rem 0 0.6rem 0;
            flex-wrap: wrap;
        }
        div[role="radiogroup"] label {
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.38rem 0.8rem;
            background: rgba(255, 253, 248, 0.84);
            box-shadow: 0 1px 0 rgba(23, 33, 47, 0.03);
        }
        div[role="radiogroup"] label:hover {
            border-color: var(--accent);
            background: var(--accent-soft);
            color: var(--accent);
        }
        [data-testid="stPills"] {
            margin-top: 0.15rem;
            margin-bottom: 1.1rem;
        }
        [data-testid="stPills"] button {
            border-radius: 999px;
            border-color: rgba(31, 111, 91, 0.18);
            background: rgba(255, 253, 248, 0.86);
            box-shadow: 0 1px 0 rgba(23, 33, 47, 0.03);
        }
        [data-testid="stPills"] button[aria-pressed="true"] {
            border-color: rgba(31, 111, 91, 0.45);
            background: #dceee8;
            color: #1f6f5b;
            font-weight: 800;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(31, 111, 91, 0.16);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            background: linear-gradient(180deg, rgba(255, 253, 248, 0.96), rgba(248, 243, 233, 0.92));
            box-shadow: 0 10px 24px rgba(23, 33, 47, 0.06);
            overflow: visible;
            min-height: 5.35rem;
        }
        div[data-testid="stMetric"] label {
            color: var(--muted);
            font-size: 0.78rem;
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink);
            letter-spacing: -0.03em;
            line-height: 1.08;
            overflow: visible;
        }
        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {
            border: 1px solid rgba(31, 111, 91, 0.3);
            border-radius: 12px;
            background: linear-gradient(180deg, #276f5e, #1e5b4d);
            color: #fffdf8;
            font-weight: 750;
            box-shadow: 0 8px 18px rgba(31, 111, 91, 0.18);
        }
        div[data-testid="stButton"] button:hover,
        div[data-testid="stDownloadButton"] button:hover {
            border-color: #174a3f;
            background: linear-gradient(180deg, #2c7b68, #1f6f5b);
            color: #fffdf8;
        }
        div[data-testid="stTabs"] button {
            border-radius: 999px;
            padding: 0.45rem 0.85rem;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: var(--accent-soft);
            color: var(--accent);
            font-weight: 750;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 16px;
            overflow: hidden;
            background: var(--panel);
            box-shadow: 0 10px 26px rgba(23, 33, 47, 0.045);
        }
        [data-testid="stAlert"] {
            border-radius: 15px;
            border: 1px solid rgba(31, 111, 91, 0.13);
            box-shadow: 0 7px 18px rgba(23, 33, 47, 0.04);
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 16px;
            background: rgba(255, 253, 248, 0.68);
        }
        hr {
            margin: 1.1rem 0;
            border-color: rgba(23, 33, 47, 0.08);
        }
        .stCaptionContainer p {
            color: var(--muted);
            line-height: 1.45;
        }
        .ux-hero {
            border: 1px solid rgba(31, 111, 91, 0.16);
            border-radius: 24px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 0.9rem;
            background:
                linear-gradient(135deg, rgba(255, 253, 248, 0.96), rgba(232, 243, 237, 0.86)),
                radial-gradient(circle at 90% 10%, rgba(181, 71, 8, 0.12), transparent 18rem);
            box-shadow: 0 16px 36px rgba(23, 33, 47, 0.07);
        }
        .ux-hero-title {
            font-size: 1.75rem;
            font-weight: 850;
            letter-spacing: -0.045em;
            margin: 0;
            color: var(--ink);
        }
        .ux-hero-sub {
            margin: 0.25rem 0 0 0;
            color: var(--muted);
        }
        .ux-nav-card {
            border: 1px solid rgba(31, 111, 91, 0.14);
            border-radius: 18px;
            padding: 0.8rem 0.9rem 0.55rem 0.9rem;
            background: rgba(255, 253, 248, 0.78);
            box-shadow: 0 8px 24px rgba(23, 33, 47, 0.05);
            margin-bottom: 0.85rem;
        }
        .ux-nav-label {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 750;
            margin: 0.05rem 0 0.25rem 0;
        }
        .ux-section-card {
            border: 1px solid var(--line);
            border-radius: 18px;
            background: rgba(255, 253, 248, 0.68);
            padding: 0.85rem 1rem;
        }
        .ux-action-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 0.85rem;
            margin: 0.55rem 0 1rem 0;
        }
        .ux-action-card {
            border: 1px solid rgba(31, 111, 91, 0.16);
            border-radius: 18px;
            padding: 0.95rem;
            background:
                linear-gradient(180deg, rgba(255, 253, 248, 0.98), rgba(247, 239, 225, 0.8));
            box-shadow: 0 12px 28px rgba(23, 33, 47, 0.065);
        }
        .ux-decision-card {
            border: 1px solid rgba(23, 33, 47, 0.10);
            border-radius: 20px;
            padding: 1rem;
            background:
                linear-gradient(180deg, rgba(255, 253, 248, 0.98), rgba(245, 242, 234, 0.88));
            box-shadow: 0 14px 30px rgba(23, 33, 47, 0.075);
        }
        .ux-decision-top {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: flex-start;
            margin-bottom: 0.8rem;
        }
        .ux-decision-title {
            color: var(--ink);
            font-size: 1.18rem;
            line-height: 1.2;
            font-weight: 850;
            letter-spacing: -0.035em;
        }
        .ux-decision-sub {
            color: var(--muted);
            font-size: 0.8rem;
            margin-top: 0.2rem;
        }
        .ux-decision-badge {
            border-radius: 999px;
            padding: 0.24rem 0.55rem;
            background: #dceee8;
            color: #1f6f5b;
            font-size: 0.76rem;
            font-weight: 850;
            white-space: nowrap;
        }
        .ux-decision-levels {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.55rem;
            margin-bottom: 0.75rem;
        }
        .ux-level {
            border: 1px solid rgba(23, 33, 47, 0.08);
            border-radius: 14px;
            padding: 0.55rem;
            background: rgba(255, 253, 248, 0.72);
        }
        .ux-level-label {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 750;
            margin-bottom: 0.18rem;
        }
        .ux-level-value {
            color: var(--ink);
            font-size: 0.98rem;
            font-weight: 850;
        }
        .ux-decision-reason {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.45;
            min-height: 2.35rem;
        }
        .ux-action-kicker {
            color: var(--accent);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }
        .ux-action-title {
            color: var(--ink);
            font-size: 1.18rem;
            font-weight: 850;
            letter-spacing: -0.035em;
            margin-bottom: 0.25rem;
        }
        .ux-action-meta {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.45;
            min-height: 2.2rem;
        }
        .ux-action-price {
            margin-top: 0.65rem;
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
        }
        .ux-pill {
            border: 1px solid rgba(31, 111, 91, 0.18);
            border-radius: 999px;
            padding: 0.24rem 0.5rem;
            background: rgba(220, 238, 232, 0.58);
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 750;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


PAGE_GROUPS: dict[str, list[str]] = {
    "매일 보는 화면": ["대시보드", "예산 플래너", "오늘 추천", "단타", "실시간"],
    "보유 관리": ["보유 종목", "종목 분석", "관심 추적", "뉴스/이벤트"],
    "전략/검증": ["전략별 추천", "배당주", "자동 후보군", "추적", "백테스트"],
    "설정": ["후보군 관리"],
}

MAIN_PAGES: list[str] = [
    "대시보드",
    "신규 추천",
    "보유 운용",
    "종목 분석",
    "학습·관리",
]


PAGE_ALIASES: dict[str, str] = {
    "예산 플래너": "신규 추천",
    "오늘 추천": "신규 추천",
    "단타": "신규 추천",
    "실시간": "신규 추천",
    "전략별 추천": "신규 추천",
    "배당주": "신규 추천",
    "보유 종목": "보유 운용",
    "관심 추적": "학습·관리",
    "뉴스/이벤트": "학습·관리",
    "자동 후보군": "학습·관리",
    "추적": "학습·관리",
    "백테스트": "학습·관리",
    "후보군 관리": "학습·관리",
}


SCAN_PRESETS: dict[str, dict[str, object]] = {
    "빠른 확인": {
        "min_score": 68,
        "top_n": 6,
        "scan_limit": 12,
        "market_sweep_limit": 80,
        "realtime_min_score": 65,
        "interval": "15m",
    },
    "균형 추천": {
        "min_score": 65,
        "top_n": 8,
        "scan_limit": 24,
        "market_sweep_limit": 140,
        "realtime_min_score": 60,
        "interval": "5m",
    },
    "깊게 탐색": {
        "min_score": 60,
        "top_n": 12,
        "scan_limit": 48,
        "market_sweep_limit": 260,
        "realtime_min_score": 55,
        "interval": "5m",
    },
    "공격 단타": {
        "min_score": 58,
        "top_n": 12,
        "scan_limit": 36,
        "market_sweep_limit": 220,
        "realtime_min_score": 52,
        "interval": "1m",
    },
}

FULL_COLLECTION_STATE_FILE = Path(__file__).resolve().parent / "data" / "full_collection_state.json"


def _compact_timestamp(value: object) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d %H:%M")


def _infer_market_from_ticker(ticker: object) -> str:
    symbol = str(ticker or "").strip().upper()
    if symbol.endswith((".KS", ".KQ")):
        return "KR"
    return ""


def _format_price(value: object, market: str | None = None, ticker: object = None) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    market_code = str(market or "").upper()
    if market_code in {"NAN", "NONE", "NULL"}:
        market_code = ""
    if not market_code:
        market_code = _infer_market_from_ticker(ticker)
    if market_code == "KR":
        return f"{int(round(float(numeric), 0)):,}원"
    return f"${float(numeric):,.2f}"


def _format_mixed_currency(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    return f"{float(numeric):,.0f}"


def _safe_float(value: object, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return float(numeric)


def _safe_int(value: object, default: int = 0) -> int:
    return int(round(_safe_float(value, float(default))))


def _is_market_color_column(column: object) -> bool:
    name = str(column).lower()
    excluded_tokens = [
        "weight",
        "allocation",
        "dividend",
        "volatility",
        "atr",
        "risk_reward",
        "hit_rate",
        "target_rate",
        "confidence",
    ]
    if any(token in name for token in excluded_tokens):
        return False
    if name in {"change_pct", "return_pct", "pnl_pct", "current_return_pct", "short_return_pct", "drawdown_pct"}:
        return True
    if name.startswith(("ret_", "avg_ret_", "best_ret_", "max_", "min_")) and name.endswith("_pct"):
        return True
    if name.startswith("return_") and name.endswith("_pct"):
        return True
    if name == "return_20d":
        return True
    return False


def _market_color(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    if float(numeric) > 0:
        return "color: #d32f2f; font-weight: 700"
    if float(numeric) < 0:
        return "color: #1565c0; font-weight: 700"
    return ""


def _apply_market_table_style(prepared: pd.DataFrame):
    color_columns = [column for column in prepared.columns if _is_market_color_column(column)]
    if not color_columns:
        return prepared
    try:
        return prepared.style.map(_market_color, subset=color_columns)
    except Exception:
        return prepared


def _prepare_table(
    df: pd.DataFrame,
    datetime_columns: list[str] | None = None,
    currency_columns: list[str] | None = None,
    plain_numeric_columns: list[str] | None = None,
    default_market: str | None = None,
) -> pd.DataFrame:
    frame = df.copy()
    for column in datetime_columns or []:
        if column in frame.columns:
            frame[column] = frame[column].apply(_compact_timestamp)
    for column in currency_columns or []:
        if column in frame.columns:
            if "market" in frame.columns or "ticker" in frame.columns:
                frame[column] = frame.apply(
                    lambda row: _format_price(
                        row.get(column),
                        str(row.get("market", default_market or "")).upper(),
                        row.get("ticker", ""),
                    ),
                    axis=1,
                )
            else:
                frame[column] = frame[column].apply(lambda value: _format_price(value, default_market))
    for column in plain_numeric_columns or []:
        if column in frame.columns:
            frame[column] = frame[column].apply(_format_mixed_currency)
    return frame


def _attach_latest_quotes(
    frame: pd.DataFrame,
    *,
    default_market: str | None = None,
    ticker_column: str = "ticker",
    market_column: str = "market",
    force_refresh: bool = False,
) -> pd.DataFrame:
    if frame.empty or ticker_column not in frame.columns:
        return frame

    enriched = frame.copy()
    current_prices: list[float | None] = []
    change_pcts: list[float | None] = []
    quote_as_ofs: list[str] = []
    data_freshness_values: list[str] = []
    price_sources: list[str] = []

    for _, row in enriched.iterrows():
        quote = get_latest_quote(str(row.get(ticker_column, "")), force_refresh=force_refresh)
        current_price = pd.to_numeric(quote.get("current_price", None), errors="coerce")
        change_pct = pd.to_numeric(quote.get("change_pct", None), errors="coerce")
        market = default_market
        if market_column in enriched.columns:
            market = str(row.get(market_column, default_market or "")).upper()

        current_prices.append(None if pd.isna(current_price) else float(current_price))
        change_pcts.append(None if pd.isna(change_pct) else float(change_pct))
        quote_source = str(quote.get("source", "") or "")
        quote_as_of = str(quote.get("as_of", "") or "")
        quote_as_ofs.append(quote_as_of)
        data_freshness_values.append(_quote_freshness_label(quote_as_of) if quote_as_of else str(row.get("data_freshness", "")))
        price_sources.append("분봉" if quote_source == "5m" else "일봉" if quote_source == "1d" else str(row.get("price_source", "")))

        if "current_price" not in enriched.columns or pd.isna(pd.to_numeric(row.get("current_price", None), errors="coerce")):
            continue

        existing_price = pd.to_numeric(row.get("current_price", None), errors="coerce")
        if not pd.isna(existing_price) and not pd.isna(current_price):
            current_prices[-1] = float(current_price)
        elif not pd.isna(existing_price):
            current_prices[-1] = float(existing_price)

    enriched["current_price"] = current_prices
    enriched["change_pct"] = change_pcts
    enriched["quote_as_of"] = quote_as_ofs
    enriched["data_freshness"] = data_freshness_values
    enriched["price_source"] = price_sources
    return enriched


def _show_table(
    df: pd.DataFrame,
    *,
    datetime_columns: list[str] | None = None,
    currency_columns: list[str] | None = None,
    plain_numeric_columns: list[str] | None = None,
    default_market: str | None = None,
    column_config: dict[str, object] | None = None,
    hide_index: bool = True,
) -> None:
    prepared = _prepare_table(
        df,
        datetime_columns=datetime_columns,
        currency_columns=currency_columns,
        plain_numeric_columns=plain_numeric_columns,
        default_market=default_market,
    )
    display_data = _apply_market_table_style(prepared)
    st.dataframe(
        display_data,
        width="stretch",
        hide_index=hide_index,
        column_config=column_config,
        height=min(760, max(160, 42 + len(prepared) * 38)),
    )


def _action_card_html(
    *,
    kicker: str,
    title: str,
    meta: str,
    pills: list[str],
) -> str:
    pill_html = "".join(f'<span class="ux-pill">{escape(str(pill))}</span>' for pill in pills if str(pill).strip())
    return (
        '<div class="ux-action-card">'
        f'<div class="ux-action-kicker">{escape(kicker)}</div>'
        f'<div class="ux-action-title">{escape(title)}</div>'
        f'<div class="ux-action-meta">{escape(meta)}</div>'
        f'<div class="ux-action-price">{pill_html}</div>'
        "</div>"
    )


def _render_action_deck(frame: pd.DataFrame, *, title: str, limit: int = 4) -> None:
    if frame.empty:
        return
    st.markdown(f"#### {title}")
    cards: list[str] = []
    for _, row in frame.head(limit).iterrows():
        market = str(row.get("market", "") or "")
        ticker = str(row.get("ticker", "") or "")
        name = str(row.get("name", "") or "")
        bucket = str(row.get("bucket", row.get("scan_type", "")) or "")
        setup = str(row.get("setup", "") or "")
        score = pd.to_numeric(row.get("score", None), errors="coerce")
        planned = pd.to_numeric(row.get("planned_amount", None), errors="coerce")
        entry = row.get("buy_now_limit", row.get("entry_price", row.get("current_price", None)))
        stop = row.get("stop_loss", None)
        current = row.get("current_price", row.get("ref_price", None))
        target = row.get("target_1", None)
        target_2 = row.get("target_2", None)
        target_3 = row.get("target_3", None)
        entry_num = pd.to_numeric(entry, errors="coerce")
        current_num = pd.to_numeric(current, errors="coerce")
        if pd.isna(entry_num) and not pd.isna(current_num):
            entry = float(current_num)
            entry_num = current_num
        stop_num = pd.to_numeric(stop, errors="coerce")
        if pd.isna(stop_num) and not pd.isna(entry_num):
            stop = float(entry_num) * 0.92
            stop_num = pd.to_numeric(stop, errors="coerce")
        target_num = pd.to_numeric(target, errors="coerce")
        if pd.isna(target_num) and not pd.isna(entry_num):
            target = float(entry_num) * 1.08
            target_num = pd.to_numeric(target, errors="coerce")
        if pd.isna(pd.to_numeric(target_2, errors="coerce")) and not pd.isna(entry_num):
            target_2 = float(entry_num) * 1.14
        if pd.isna(pd.to_numeric(target_3, errors="coerce")) and not pd.isna(entry_num):
            target_3 = float(entry_num) * 1.22
        action = str(row.get("execution_status", row.get("action", row.get("score_view", "확인"))) or "확인")
        timing_action = str(row.get("timing_action", "") or "")
        exit_timing = str(row.get("exit_timing", "") or "")
        add_timing = str(row.get("add_timing", "") or "")
        scale_timing = str(row.get("scale_timing", "") or "")
        sell_plan = str(row.get("sell_plan", "") or "")
        if not any([timing_action, exit_timing, add_timing, scale_timing, sell_plan]):
            playbook = _timing_playbook(row)
            timing_action = playbook["timing_action"]
            exit_timing = playbook["exit_timing"]
            add_timing = playbook["add_timing"]
            scale_timing = playbook["scale_timing"]
            sell_plan = playbook["sell_plan"]
        reason = str(row.get("action_note", row.get("reason", row.get("allocation_reason", ""))) or "")
        confidence = str(row.get("confidence_detail", row.get("confidence_view", "")) or "")
        price_basis = str(row.get("price_basis", row.get("price_rule", "")) or "")
        samples = _safe_int(row.get("recent_sample_count", 0))
        hit_rate = pd.to_numeric(row.get("recent_hit_rate_20d", None), errors="coerce")
        target_rate = pd.to_numeric(row.get("recent_target_rate_5d", None), errors="coerce")
        learning_delta = pd.to_numeric(row.get("learning_delta", None), errors="coerce")
        score_text = f"{_safe_int(score)}점" if not pd.isna(score) else "점수 확인"
        planned_text = f" · {float(planned):,.0f} 배정" if not pd.isna(planned) else ""
        if not pd.isna(score):
            score_text = f"{_safe_int(score)}점"
        entry_range = str(row.get("entry_range", "") or "").strip()
        if not entry_range and not pd.isna(entry_num):
            entry_range = f"{_format_price(float(entry_num) * 0.985, market)} ~ {_format_price(float(entry_num) * 1.01, market)}"
        entry_text = entry_range or (_format_price(entry, market) if entry not in (None, "") else "가격 확인")
        current_text = _format_price(current, market) if current not in (None, "") else "현재가 확인"
        stop_text = _format_price(stop, market) if stop not in (None, "") else "손절 기준 확인"
        target_text = _format_price(target, market) if target not in (None, "") else "목표 확인"
        target_2_text = _format_price(target_2, market) if target_2 not in (None, "") else "2차 확인"
        target_3_text = _format_price(target_3, market) if target_3 not in (None, "") else "3차 확인"
        subtitle_parts = [part for part in [market, bucket, setup, score_text] if str(part).strip()]
        reason_text = reason[:135] if reason else "조건 충족 여부와 가격 기준을 확인하세요."
        trust_parts = []
        if confidence:
            trust_parts.append(confidence)
        if samples > 0:
            trust_parts.append(f"표본 {samples}개")
        if not pd.isna(hit_rate) and float(hit_rate) > 0:
            trust_parts.append(f"20일 적중 {float(hit_rate):.0f}%")
        if not pd.isna(target_rate) and float(target_rate) > 0:
            trust_parts.append(f"5일 목표 {float(target_rate):.0f}%")
        if not pd.isna(learning_delta) and float(learning_delta) != 0:
            trust_parts.append(f"학습 {float(learning_delta):+.0f}")
        trust_text = " · ".join(trust_parts) if trust_parts else "학습 표본 부족: 결과 누적 중"
        basis_text = price_basis or "가격 기준: 현재가/ATR/손익비 기반"
        timing_text = " · ".join(part for part in [timing_action, exit_timing] if part) or "타이밍 기준은 표에서 확인"
        add_scale_text = " / ".join(part for part in [add_timing, scale_timing] if part)
        cards.append(
            (
                '<div class="ux-decision-card">'
                '<div class="ux-decision-top">'
                '<div>'
                f'<div class="ux-decision-title">{escape(f"{ticker} {name}".strip() or "후보")}</div>'
                f'<div class="ux-decision-sub">{" · ".join(escape(str(part)) for part in subtitle_parts)}{escape(planned_text)}</div>'
                '</div>'
                f'<div class="ux-decision-badge">{escape(action)}</div>'
                '</div>'
                '<div class="ux-decision-levels">'
                '<div class="ux-level"><div class="ux-level-label">현재가</div>'
                f'<div class="ux-level-value">{escape(current_text)}</div></div>'
                '<div class="ux-level"><div class="ux-level-label">진입가/구간</div>'
                f'<div class="ux-level-value">{escape(entry_text)}</div></div>'
                '<div class="ux-level"><div class="ux-level-label">손절가</div>'
                f'<div class="ux-level-value">{escape(stop_text)}</div></div>'
                '<div class="ux-level"><div class="ux-level-label">1차 매도</div>'
                f'<div class="ux-level-value">{escape(target_text)}</div></div>'
                '<div class="ux-level"><div class="ux-level-label">2차 매도</div>'
                f'<div class="ux-level-value">{escape(target_2_text)}</div></div>'
                '<div class="ux-level"><div class="ux-level-label">3차 매도</div>'
                f'<div class="ux-level-value">{escape(target_3_text)}</div></div>'
                '</div>'
                f'<div class="ux-decision-reason"><b>신뢰 근거</b> {escape(trust_text)}</div>'
                f'<div class="ux-decision-reason"><b>가격 근거</b> {escape(basis_text)}</div>'
                f'<div class="ux-decision-reason"><b>타이밍</b> {escape(timing_text)}</div>'
                f'<div class="ux-decision-reason"><b>추가매수</b> {escape(add_scale_text or "조건 미충족 시 금지")}</div>'
                f'<div class="ux-decision-reason"><b>매도 계획</b> {escape(sell_plan or "1~3차 목표 확인")}</div>'
                f'<div class="ux-decision-reason">{escape(reason_text)}</div>'
                '</div>'
            )
        )
    st.markdown(f'<div class="ux-action-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _candidate_column_config() -> dict[str, object]:
    return {
        "market": st.column_config.TextColumn("시장", width="small"),
        "ticker": st.column_config.TextColumn("티커", width="small"),
        "name": st.column_config.TextColumn("종목명", width="small"),
        "setup": st.column_config.TextColumn("세팅", width="small"),
        "action": st.column_config.TextColumn("액션", width="small"),
        "timing_action": st.column_config.TextColumn("지금행동", width="small"),
        "score_view": st.column_config.TextColumn("점수판단", width="small"),
        "score": st.column_config.NumberColumn("점수", format="%d", width="small"),
        "score_band": st.column_config.TextColumn("등급", width="small"),
        "prediction_view": st.column_config.TextColumn("예상", width="small"),
        "delta_view": st.column_config.TextColumn("보정합계", width="small"),
        "recent_hit_rate_20d": st.column_config.NumberColumn("최근적중률", format="%.1f", width="small"),
        "recent_target_rate_5d": st.column_config.NumberColumn("최근목표도달", format="%.1f", width="small"),
        "recent_sample_count": st.column_config.NumberColumn("누적표본", format="%d", width="small"),
        "confidence_view": st.column_config.TextColumn("신뢰도", width="small"),
        "current_price": st.column_config.TextColumn("현재가", width="small"),
        "change_pct": st.column_config.NumberColumn("당일등락", format="%.2f", width="small"),
        "entry_range": st.column_config.TextColumn("진입구간", width="medium"),
        "entry_price": st.column_config.TextColumn("진입가", width="small"),
        "stop_loss": st.column_config.TextColumn("손절가", width="small"),
        "target_1": st.column_config.TextColumn("1차목표", width="small"),
        "target_2": st.column_config.TextColumn("2차목표", width="small"),
        "target_3": st.column_config.TextColumn("3차목표", width="small"),
        "confidence_score": st.column_config.NumberColumn("신뢰도", format="%d", width="small"),
        "confidence_detail": st.column_config.TextColumn("신뢰근거", width="medium"),
        "price_basis": st.column_config.TextColumn("가격근거", width="large"),
        "quote_as_of": st.column_config.TextColumn("시세기준", width="small"),
        "data_freshness": st.column_config.TextColumn("데이터", width="small"),
        "price_source": st.column_config.TextColumn("시세출처", width="small"),
        "regime": st.column_config.TextColumn("장세", width="small"),
        "regime_delta": st.column_config.NumberColumn("장세보정", format="%d", width="small"),
        "context_delta": st.column_config.NumberColumn("이벤트보정", format="%d", width="small"),
        "event_risk": st.column_config.TextColumn("이벤트리스크", width="small"),
        "earnings_date": st.column_config.TextColumn("실적예정", width="small"),
        "ex_dividend_date": st.column_config.TextColumn("배당락일", width="small"),
        "news_bias": st.column_config.TextColumn("뉴스흐름", width="small"),
        "news_count": st.column_config.NumberColumn("뉴스건수", format="%d", width="small"),
        "learning_delta": st.column_config.NumberColumn("학습보정", format="%d", width="small"),
        "trend_score": st.column_config.NumberColumn("추세", format="%d", width="small"),
        "momentum_score": st.column_config.NumberColumn("모멘텀", format="%d", width="small"),
        "volume_score": st.column_config.NumberColumn("거래량", format="%d", width="small"),
        "breakout_score": st.column_config.NumberColumn("돌파", format="%d", width="small"),
        "volume_ratio": st.column_config.NumberColumn("거래량배수", format="%.2f", width="small"),
        "return_20d": st.column_config.NumberColumn("20일수익률", format="%.2f", width="small"),
        "rs_score": st.column_config.NumberColumn("상대강도", format="%.2f", width="small"),
        "atr_pct": st.column_config.NumberColumn("ATR%", format="%.2f", width="small"),
        "from_52w_high_pct": st.column_config.NumberColumn("52주고점대비", format="%.2f", width="small"),
        "return_60d_pct": st.column_config.NumberColumn("60일수익률", format="%.2f", width="small"),
        "return_1y_pct": st.column_config.NumberColumn("1년수익률", format="%.2f", width="small"),
        "reason": st.column_config.TextColumn("핵심 사유", width="large"),
    }


def _history_column_config() -> dict[str, object]:
    return {
        "snapshot_id": st.column_config.TextColumn("스냅샷ID", width="small"),
        "scan_type": st.column_config.TextColumn("유형", width="small"),
        "market": st.column_config.TextColumn("시장", width="small"),
        "row_count": st.column_config.NumberColumn("종목수", format="%d", width="small"),
        "saved_at": st.column_config.TextColumn("저장시각", width="small"),
    }


def _tracking_column_config() -> dict[str, object]:
    return {
        "scan_type": st.column_config.TextColumn("유형", width="small"),
        "market": st.column_config.TextColumn("시장", width="small"),
        "ticker": st.column_config.TextColumn("티커", width="small"),
        "name": st.column_config.TextColumn("종목명", width="small"),
        "score": st.column_config.NumberColumn("점수", format="%.0f", width="small"),
        "current_price": st.column_config.TextColumn("기준가", width="small"),
        "entry_price": st.column_config.TextColumn("진입가", width="small"),
        "stop_loss": st.column_config.TextColumn("손절가", width="small"),
        "target_1": st.column_config.TextColumn("목표가", width="small"),
        "event_risk": st.column_config.TextColumn("이벤트리스크", width="small"),
        "news_bias": st.column_config.TextColumn("뉴스흐름", width="small"),
        "ret_1d_pct": st.column_config.NumberColumn("1일", format="%.2f", width="small"),
        "ret_3d_pct": st.column_config.NumberColumn("3일", format="%.2f", width="small"),
        "ret_5d_pct": st.column_config.NumberColumn("5일", format="%.2f", width="small"),
        "ret_20d_pct": st.column_config.NumberColumn("20일", format="%.2f", width="small"),
        "max_5d_pct": st.column_config.NumberColumn("5일최고", format="%.2f", width="small"),
        "min_5d_pct": st.column_config.NumberColumn("5일최저", format="%.2f", width="small"),
        "path_5d": st.column_config.TextColumn("5일경로", width="small"),
        "path_20d": st.column_config.TextColumn("20일경로", width="small"),
        "best_forward_pct": st.column_config.NumberColumn("최고전진수익", format="%.2f", width="small"),
        "status": st.column_config.TextColumn("상태", width="small"),
        "saved_at": st.column_config.TextColumn("저장시각", width="small"),
        "captured_at": st.column_config.TextColumn("캡처시각", width="small"),
    }


def _tracking_summary_column_config() -> dict[str, object]:
    return {
        "scan_type": st.column_config.TextColumn("유형", width="small"),
        "market": st.column_config.TextColumn("시장", width="small"),
        "picks": st.column_config.NumberColumn("건수", format="%d", width="small"),
        "avg_score": st.column_config.NumberColumn("평균점수", format="%.1f", width="small"),
        "avg_ret_1d_pct": st.column_config.NumberColumn("1일평균", format="%.2f", width="small"),
        "avg_ret_3d_pct": st.column_config.NumberColumn("3일평균", format="%.2f", width="small"),
        "avg_ret_5d_pct": st.column_config.NumberColumn("5일평균", format="%.2f", width="small"),
        "avg_ret_20d_pct": st.column_config.NumberColumn("20일평균", format="%.2f", width="small"),
        "avg_max_5d_pct": st.column_config.NumberColumn("5일최고평균", format="%.2f", width="small"),
        "avg_min_5d_pct": st.column_config.NumberColumn("5일최저평균", format="%.2f", width="small"),
        "best_ret_20d_pct": st.column_config.NumberColumn("최고20일", format="%.2f", width="small"),
        "hit_rate_5d_pct": st.column_config.NumberColumn("5일적중률", format="%.1f", width="small"),
        "hit_rate_20d_pct": st.column_config.NumberColumn("20일적중률", format="%.1f", width="small"),
        "target_first_5d_pct": st.column_config.NumberColumn("목표도달률", format="%.1f", width="small"),
        "stop_first_5d_pct": st.column_config.NumberColumn("손절도달률", format="%.1f", width="small"),
    }


def _pattern_stats_column_config() -> dict[str, object]:
    return {
        "scan_type": st.column_config.TextColumn("유형", width="small"),
        "market": st.column_config.TextColumn("시장", width="small"),
        "setup": st.column_config.TextColumn("패턴", width="small"),
        "picks": st.column_config.NumberColumn("건수", format="%d", width="small"),
        "avg_score": st.column_config.NumberColumn("평균점수", format="%.1f", width="small"),
        "avg_ret_3d_pct": st.column_config.NumberColumn("3일평균", format="%.2f", width="small"),
        "avg_ret_5d_pct": st.column_config.NumberColumn("5일평균", format="%.2f", width="small"),
        "avg_ret_20d_pct": st.column_config.NumberColumn("20일평균", format="%.2f", width="small"),
        "avg_max_5d_pct": st.column_config.NumberColumn("5일최고평균", format="%.2f", width="small"),
        "avg_min_5d_pct": st.column_config.NumberColumn("5일최저평균", format="%.2f", width="small"),
        "hit_rate_20d_pct": st.column_config.NumberColumn("20일적중률", format="%.1f", width="small"),
        "strong_success_20d": st.column_config.NumberColumn("강한성공률", format="%.1f", width="small"),
        "target_first_5d_pct": st.column_config.NumberColumn("목표도달률", format="%.1f", width="small"),
        "stop_first_5d_pct": st.column_config.NumberColumn("손절도달률", format="%.1f", width="small"),
    }


def _leaderboard_column_config() -> dict[str, object]:
    return {
        "market": st.column_config.TextColumn("시장", width="small"),
        "ticker": st.column_config.TextColumn("티커", width="small"),
        "name": st.column_config.TextColumn("종목명", width="small"),
        "appearances": st.column_config.NumberColumn("등장횟수", format="%d", width="small"),
        "avg_score": st.column_config.NumberColumn("평균점수", format="%.1f", width="small"),
        "avg_ret_5d_pct": st.column_config.NumberColumn("5일평균", format="%.2f", width="small"),
        "avg_ret_20d_pct": st.column_config.NumberColumn("20일평균", format="%.2f", width="small"),
        "target_first_5d_pct": st.column_config.NumberColumn("목표도달률", format="%.1f", width="small"),
        "stop_first_5d_pct": st.column_config.NumberColumn("손절도달률", format="%.1f", width="small"),
        "hit_rate_20d_pct": st.column_config.NumberColumn("20일적중률", format="%.1f", width="small"),
        "best_forward_pct": st.column_config.NumberColumn("최고전진수익", format="%.2f", width="small"),
        "latest_saved_at": st.column_config.TextColumn("최근저장", width="small"),
        "memory_score": st.column_config.NumberColumn("기억점수", format="%.2f", width="small"),
    }


def _trade_column_config() -> dict[str, object]:
    return {
        "ticker": st.column_config.TextColumn("티커", width="small"),
        "name": st.column_config.TextColumn("종목명", width="small"),
        "risk_level": st.column_config.TextColumn("위험도", width="small"),
        "timing_action": st.column_config.TextColumn("지금행동", width="small"),
        "exit_timing": st.column_config.TextColumn("손절기준", width="small"),
        "add_timing": st.column_config.TextColumn("추가매수", width="small"),
        "scale_timing": st.column_config.TextColumn("불타기", width="small"),
        "sell_plan": st.column_config.TextColumn("분할매도", width="medium"),
        "confidence_score": st.column_config.NumberColumn("신뢰도", format="%d", width="small"),
        "confidence_detail": st.column_config.TextColumn("신뢰근거", width="medium"),
        "setup": st.column_config.TextColumn("세팅", width="small"),
        "score_view": st.column_config.TextColumn("점수판단", width="small"),
        "score": st.column_config.NumberColumn("점수", format="%d", width="small"),
        "score_band": st.column_config.TextColumn("등급", width="small"),
        "prediction_view": st.column_config.TextColumn("예상", width="small"),
        "delta_view": st.column_config.TextColumn("보정합계", width="small"),
        "recent_hit_rate_20d": st.column_config.NumberColumn("최근적중률", format="%.1f", width="small"),
        "recent_target_rate_5d": st.column_config.NumberColumn("최근목표도달", format="%.1f", width="small"),
        "recent_sample_count": st.column_config.NumberColumn("누적표본", format="%d", width="small"),
        "confidence_view": st.column_config.TextColumn("신뢰도", width="small"),
        "current_price": st.column_config.TextColumn("현재가", width="small"),
        "change_pct": st.column_config.NumberColumn("당일등락", format="%.2f", width="small"),
        "entry_price": st.column_config.TextColumn("진입가", width="small"),
        "stop_loss": st.column_config.TextColumn("손절가", width="small"),
        "target_1": st.column_config.TextColumn("1차목표", width="small"),
        "target_2": st.column_config.TextColumn("2차목표", width="small"),
        "target_3": st.column_config.TextColumn("3차목표", width="small"),
        "price_basis": st.column_config.TextColumn("가격근거", width="large"),
        "quote_as_of": st.column_config.TextColumn("시세기준", width="small"),
        "data_freshness": st.column_config.TextColumn("데이터", width="small"),
        "price_source": st.column_config.TextColumn("시세출처", width="small"),
        "regime": st.column_config.TextColumn("장세", width="small"),
        "regime_delta": st.column_config.NumberColumn("장세보정", format="%d", width="small"),
        "context_delta": st.column_config.NumberColumn("이벤트보정", format="%d", width="small"),
        "event_risk": st.column_config.TextColumn("이벤트리스크", width="small"),
        "earnings_date": st.column_config.TextColumn("실적예정", width="small"),
        "ex_dividend_date": st.column_config.TextColumn("배당락일", width="small"),
        "news_bias": st.column_config.TextColumn("뉴스흐름", width="small"),
        "news_count": st.column_config.NumberColumn("뉴스건수", format="%d", width="small"),
        "learning_delta": st.column_config.NumberColumn("학습보정", format="%d", width="small"),
        "short_return_pct": st.column_config.NumberColumn("단기탄력", format="%.2f", width="small"),
        "volume_ratio": st.column_config.NumberColumn("거래량배수", format="%.2f", width="small"),
        "atr_pct": st.column_config.NumberColumn("ATR%", format="%.2f", width="small"),
        "rs_score": st.column_config.NumberColumn("상대강도", format="%.2f", width="small"),
        "risk_reward_1": st.column_config.NumberColumn("1차손익비", format="%.2f", width="small"),
        "exit_rule": st.column_config.TextColumn("청산기준", width="medium"),
        "reason": st.column_config.TextColumn("핵심 사유", width="large"),
    }


def _render_recommendation_definition(kind: str) -> None:
    definitions = {
        "today": (
            "오늘추천의 의미",
            "오늘 진입 후보입니다. 최근 차트/거래량/상대강도가 좋아 이번 주 또는 근시일 안에 위로 움직일 가능성을 보는 추천입니다.",
            "필수 확인: 현재가, 진입가/진입구간, 손절가, 1~3차 매도가, 데이터 최신성, 신뢰도.",
        ),
        "trade": (
            "단타의 의미",
            "지금 들어가 짧은 구간의 탄력과 거래량을 먹고 나오는 실행 플랜입니다. 분봉/VWAP/거래량 기준이 중요하고 손절은 더 엄격하게 봅니다.",
            "필수 확인: 현재가, 진입가, 손절가, 1~3차 매도가, 청산기준, 손익비, 분봉 최신성.",
        ),
        "long": (
            "장기투자의 의미",
            "차트 추세, 뉴스/이벤트, 장세와 상대강도를 종합해 계속 모아갈 후보를 고르는 추천입니다. 단기 급등보다 누적 매수 가능성과 장기 우상향을 봅니다.",
            "필수 확인: 현재가, 분할 진입가, 이탈 기준, 1~3차 목표, 장세/뉴스/데이터 최신성.",
        ),
    }
    title, meaning, checklist = definitions[kind]
    st.info(f"**{title}**\n\n{meaning}\n\n{checklist}")


def _timing_playbook(row: pd.Series) -> dict[str, str]:
    market = str(row.get("market", "") or "")
    current = pd.to_numeric(row.get("current_price", None), errors="coerce")
    entry = pd.to_numeric(row.get("entry_price", None), errors="coerce")
    stop = pd.to_numeric(row.get("stop_loss", None), errors="coerce")
    target_1 = pd.to_numeric(row.get("target_1", None), errors="coerce")
    target_2 = pd.to_numeric(row.get("target_2", None), errors="coerce")
    target_3 = pd.to_numeric(row.get("target_3", None), errors="coerce")
    confidence = _safe_int(row.get("confidence_score", 0))
    score = _safe_int(row.get("score", 0))

    if pd.isna(current):
        current = entry
    if pd.isna(current) or pd.isna(entry) or pd.isna(stop):
        return {
            "timing_action": "가격확인",
            "exit_timing": "가격 확인 전 금지",
            "add_timing": "가격 확인 후",
            "scale_timing": "가격 확인 후",
            "sell_plan": "목표 확인",
        }

    current_f = float(current)
    entry_f = float(entry)
    stop_f = float(stop)
    t1 = float(target_1) if not pd.isna(target_1) else entry_f + max(entry_f - stop_f, entry_f * 0.03) * 1.5
    t2 = float(target_2) if not pd.isna(target_2) else entry_f + max(entry_f - stop_f, entry_f * 0.03) * 2.5
    t3 = float(target_3) if not pd.isna(target_3) else entry_f + max(entry_f - stop_f, entry_f * 0.03) * 3.5
    risk = max(entry_f - stop_f, entry_f * 0.01)

    if current_f <= stop_f:
        action = "손절/제외"
    elif current_f <= entry_f * 1.01 and score >= 65 and confidence >= 45:
        action = "진입검토"
    elif current_f > entry_f * 1.04:
        action = "추격금지"
    else:
        action = "대기"

    pullback_zone = max(stop_f, entry_f - risk * 0.45)
    add_rule = f"{_format_price(pullback_zone, market)} 반등 확인"
    scale_rule = f"{_format_price(t1, market)} 돌파+거래량"
    exit_rule = f"{_format_price(stop_f, market)} 이탈 시 정리"
    sell_plan = (
        f"{_format_price(t1, market)} 40% / "
        f"{_format_price(t2, market)} 30% / "
        f"{_format_price(t3, market)} 30%"
    )
    return {
        "timing_action": action,
        "exit_timing": exit_rule,
        "add_timing": add_rule,
        "scale_timing": scale_rule,
        "sell_plan": sell_plan,
    }


def _prepare_trade_execution_view(frame: pd.DataFrame, *, include_risk_level: bool = False) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    view = frame.copy()
    for column in [
        "risk_level",
        "confidence_score",
        "confidence_detail",
        "price_basis",
        "data_freshness",
        "price_source",
        "current_price",
        "entry_price",
        "stop_loss",
        "target_1",
        "target_2",
        "target_3",
        "risk_reward_1",
        "timing_action",
        "exit_timing",
        "add_timing",
        "scale_timing",
        "sell_plan",
    ]:
        if column not in view.columns:
            view[column] = None if column != "risk_level" else ""

    numeric_current = pd.to_numeric(view["current_price"], errors="coerce")
    numeric_entry = pd.to_numeric(view["entry_price"], errors="coerce")
    view["current_price"] = numeric_current.fillna(numeric_entry)

    entry = pd.to_numeric(view["entry_price"], errors="coerce")
    stop = pd.to_numeric(view["stop_loss"], errors="coerce")
    target_1 = pd.to_numeric(view["target_1"], errors="coerce")
    target_2 = pd.to_numeric(view["target_2"], errors="coerce")
    risk = (entry - stop).where((entry > 0) & (stop > 0), entry * 0.03)
    risk = risk.fillna(entry * 0.03).clip(lower=0.01)
    view["target_1"] = target_1.fillna(entry + risk * 1.5)
    view["target_2"] = target_2.fillna(entry + risk * 2.5)
    view["target_3"] = pd.to_numeric(view["target_3"], errors="coerce").fillna(entry + risk * 3.5)
    view["risk_reward_1"] = pd.to_numeric(view["risk_reward_1"], errors="coerce").fillna(
        (pd.to_numeric(view["target_1"], errors="coerce") - entry) / risk
    )
    if "confidence_score" in view.columns:
        view["confidence_score"] = pd.to_numeric(view["confidence_score"], errors="coerce").fillna(0).astype(int)
    playbook = view.apply(_timing_playbook, axis=1, result_type="expand")
    for column in ["timing_action", "exit_timing", "add_timing", "scale_timing", "sell_plan"]:
        view[column] = playbook[column]

    ordered = [
        "ticker",
        "name",
    ]
    if include_risk_level:
        ordered.append("risk_level")
    ordered += [
        "timing_action",
        "current_price",
        "entry_price",
        "stop_loss",
        "target_1",
        "target_2",
        "target_3",
        "exit_timing",
        "add_timing",
        "scale_timing",
        "sell_plan",
        "risk_reward_1",
        "confidence_score",
        "confidence_detail",
        "setup",
        "score_view",
        "score_band",
        "prediction_view",
        "delta_view",
        "recent_hit_rate_20d",
        "recent_target_rate_5d",
        "recent_sample_count",
        "score",
        "change_pct",
        "quote_as_of",
        "data_freshness",
        "price_source",
        "event_risk",
        "earnings_date",
        "ex_dividend_date",
        "news_bias",
        "news_count",
        "atr_pct",
        "rs_score",
        "regime",
        "context_delta",
        "learning_delta",
        "price_basis",
        "reason",
    ]
    for column in ordered:
        if column not in view.columns:
            view[column] = ""
    return view[ordered]


def _prepare_recommendation_execution_view(
    frame: pd.DataFrame,
    *,
    include_market: bool = True,
    default_market: str | None = None,
    include_source: bool = False,
    include_targets: bool = True,
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    view = frame.copy()
    for column in [
        "market",
        "ticker",
        "name",
        "scan_type",
        "data_basis",
        "setup",
        "action",
        "score",
        "score_view",
        "score_band",
        "confidence_score",
        "confidence_detail",
        "confidence_view",
        "current_price",
        "entry_price",
        "stop_loss",
        "target_1",
        "target_2",
        "target_3",
        "timing_action",
        "entry_range",
        "price_basis",
        "change_pct",
        "quote_as_of",
        "data_freshness",
        "price_source",
        "event_risk",
        "earnings_date",
        "ex_dividend_date",
        "news_bias",
        "reason",
    ]:
        if column not in view.columns:
            view[column] = "" if column in {"market", "ticker", "name", "scan_type", "data_basis"} else None

    market_values = view["market"].astype(str).str.upper().replace({"NONE": "", "NAN": ""})
    if default_market:
        market_values = market_values.mask(market_values.str.strip() == "", default_market.upper())
    view["market"] = market_values
    if include_market and market_values.replace("", pd.NA).isna().all():
        view["market"] = ""

    current = pd.to_numeric(view["current_price"], errors="coerce")
    entry = pd.to_numeric(view["entry_price"], errors="coerce")
    if "buy_now_limit" in view.columns:
        buy_now_limit = pd.to_numeric(view["buy_now_limit"], errors="coerce")
        entry = entry.fillna(buy_now_limit)
    current = current.fillna(entry)
    entry = entry.fillna(current)
    view["current_price"] = current
    view["entry_price"] = entry

    stop = pd.to_numeric(view["stop_loss"], errors="coerce").fillna(entry * 0.92)
    target_1 = pd.to_numeric(view["target_1"], errors="coerce").fillna(entry * 1.08)
    target_2 = pd.to_numeric(view["target_2"], errors="coerce").fillna(entry * 1.14)
    target_3 = pd.to_numeric(view["target_3"], errors="coerce").fillna(entry * 1.22)
    view["stop_loss"] = stop
    view["target_1"] = target_1
    view["target_2"] = target_2
    view["target_3"] = target_3

    def _entry_range_for_row(row: pd.Series) -> str:
        market = str(row.get("market", "") or "").upper()
        entry_price = pd.to_numeric(row.get("entry_price", None), errors="coerce")
        current_price = pd.to_numeric(row.get("current_price", None), errors="coerce")
        if pd.isna(entry_price):
            entry_price = current_price
        if pd.isna(entry_price):
            return "가격 확인"
        lower = float(entry_price) * 0.985
        upper = float(entry_price) * 1.01
        if "buy_now_limit" in row.index:
            buy_limit = pd.to_numeric(row.get("buy_now_limit", None), errors="coerce")
            if not pd.isna(buy_limit):
                upper = min(upper, float(buy_limit))
        if upper < lower:
            lower, upper = upper, lower
        return f"{_format_price(lower, market)} ~ {_format_price(upper, market)}"

    view["entry_range"] = view.apply(_entry_range_for_row, axis=1)
    if "confidence_score" in view.columns:
        view["confidence_score"] = pd.to_numeric(view["confidence_score"], errors="coerce").fillna(
            pd.to_numeric(view.get("score", 0), errors="coerce").fillna(0) * 0.65
        ).clip(lower=0, upper=100).astype(int)
        freshness_penalty = view["data_freshness"].astype(str).map(
            {"약간 지연": 5, "오래됨": 14, "기준없음": 20}
        ).fillna(0)
        view["confidence_score"] = (view["confidence_score"] - freshness_penalty).clip(lower=0, upper=100).astype(int)

    missing_confidence = view["confidence_detail"].astype(str).str.strip().isin(["", "None", "nan"])
    if missing_confidence.any():
        fallback = view.get("confidence_view", "").astype(str)
        view.loc[missing_confidence, "confidence_detail"] = fallback.where(fallback.str.strip() != "", "학습 표본 누적 중")

    missing_basis = view["price_basis"].astype(str).str.strip().isin(["", "None", "nan"])
    view.loc[missing_basis, "price_basis"] = "현재가 기준 진입구간, 손절가, 1차 목표 자동 산정"

    playbook = view.apply(_timing_playbook, axis=1, result_type="expand")
    view["timing_action"] = playbook["timing_action"]

    ordered = []
    if include_market:
        ordered.append("market")
    ordered += ["ticker", "name"]
    if include_source:
        ordered += ["scan_type", "data_basis"]
    ordered += [
        "timing_action",
        "current_price",
        "entry_range",
        "entry_price",
        "stop_loss",
        "target_1",
    ]
    if include_targets:
        ordered += ["target_2", "target_3"]
    ordered += [
        "confidence_score",
        "confidence_detail",
        "score_view",
        "score",
        "setup",
        "action",
        "change_pct",
        "quote_as_of",
        "data_freshness",
        "price_source",
        "event_risk",
        "earnings_date",
        "ex_dividend_date",
        "news_bias",
    ]
    ordered += [column for column in extra_columns or [] if column in view.columns and column not in ordered]
    ordered += [
        "price_basis",
        "reason",
    ]
    for column in ordered:
        if column not in view.columns:
            view[column] = ""
    return view[ordered]


def _manual_tracking_column_config() -> dict[str, object]:
    return {
        "market": st.column_config.TextColumn("시장", width="small"),
        "ticker": st.column_config.TextColumn("티커", width="small"),
        "name": st.column_config.TextColumn("종목명", width="small"),
        "source": st.column_config.TextColumn("출처", width="small"),
        "setup": st.column_config.TextColumn("세팅", width="small"),
        "score_view": st.column_config.TextColumn("점수판단", width="small"),
        "recent_hit_rate_20d": st.column_config.NumberColumn("최근적중률", format="%.1f", width="small"),
        "recent_target_rate_5d": st.column_config.NumberColumn("최근목표도달", format="%.1f", width="small"),
        "recent_sample_count": st.column_config.NumberColumn("누적표본", format="%d", width="small"),
        "confidence_view": st.column_config.TextColumn("신뢰도", width="small"),
        "current_price": st.column_config.TextColumn("현재가", width="small"),
        "change_pct": st.column_config.NumberColumn("당일등락", format="%.2f", width="small"),
        "current_return_pct": st.column_config.NumberColumn("현재수익률", format="%.2f", width="small"),
        "entry_price": st.column_config.TextColumn("진입가", width="small"),
        "stop_loss": st.column_config.TextColumn("손절가", width="small"),
        "target_1": st.column_config.TextColumn("목표가", width="small"),
        "ret_3d_pct": st.column_config.NumberColumn("3일", format="%.2f", width="small"),
        "ret_5d_pct": st.column_config.NumberColumn("5일", format="%.2f", width="small"),
        "ret_20d_pct": st.column_config.NumberColumn("20일", format="%.2f", width="small"),
        "path_5d": st.column_config.TextColumn("5일경로", width="small"),
        "path_20d": st.column_config.TextColumn("20일경로", width="small"),
        "alert_status": st.column_config.TextColumn("현재상태", width="small"),
        "quote_as_of": st.column_config.TextColumn("시세기준", width="small"),
        "memo": st.column_config.TextColumn("메모", width="medium"),
        "created_at": st.column_config.TextColumn("추가시각", width="small"),
    }


def _dividend_column_config() -> dict[str, object]:
    return {
        "ticker": st.column_config.TextColumn("티커", width="small"),
        "name": st.column_config.TextColumn("종목명", width="small"),
        "style": st.column_config.TextColumn("유형", width="small"),
        "score": st.column_config.NumberColumn("점수", format="%d", width="small"),
        "current_price": st.column_config.TextColumn("현재가", width="small"),
        "annual_dividend": st.column_config.TextColumn("연배당", width="small"),
        "dividend_yield_pct": st.column_config.NumberColumn("배당수익률", format="%.2f", width="small"),
        "dividend_growth_1y_pct": st.column_config.NumberColumn("1년배당성장", format="%.2f", width="small"),
        "dividend_growth_3y_pct": st.column_config.NumberColumn("3년배당성장", format="%.2f", width="small"),
        "dividend_events_1y": st.column_config.NumberColumn("연간횟수", format="%d", width="small"),
        "ex_dividend_date": st.column_config.TextColumn("배당락일", width="small"),
        "accumulate_low": st.column_config.TextColumn("모으기하단", width="small"),
        "accumulate_high": st.column_config.TextColumn("모으기상단", width="small"),
        "pullback_pct": st.column_config.NumberColumn("고점대비", format="%.2f", width="small"),
        "action": st.column_config.TextColumn("지금 판단", width="small"),
        "reason": st.column_config.TextColumn("핵심 사유", width="large"),
    }


def _merge_watchlists_with_defaults(
    current: dict[str, list[dict[str, str]]] | None,
) -> dict[str, list[dict[str, str]]]:
    defaults = get_default_watchlists()
    merged: dict[str, list[dict[str, str]]] = {}

    for market in ["US", "KR"]:
        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in (current or {}).get(market, []) + defaults.get(market, []):
            ticker = str(item.get("ticker", "") or "").strip().upper()
            if not ticker or ticker in seen or not is_tradable_ticker(ticker):
                continue
            rows.append({"ticker": ticker, "name": str(item.get("name", "") or "")})
            seen.add(ticker)
        merged[market] = rows
    return merged


def init_state() -> None:
    if "portfolio" not in st.session_state:
        loaded_portfolio = load_portfolio()
        st.session_state.portfolio = loaded_portfolio if loaded_portfolio is not None else pd.DataFrame(
            [
                {
                    "market": "US",
                    "ticker": "AAPL",
                    "name": "Apple",
                    "quantity": 10,
                    "avg_price": 180.0,
                    "cash_budget": 1000.0,
                    "target_weight": 0.15,
                },
                {
                    "market": "KR",
                    "ticker": "005930.KS",
                    "name": "Samsung Electronics",
                    "quantity": 15,
                    "avg_price": 71000.0,
                    "cash_budget": 500000.0,
                    "target_weight": 0.2,
                },
            ]
        )
    if "watchlists" not in st.session_state:
        st.session_state.watchlists = _merge_watchlists_with_defaults(load_watchlists())
    else:
        st.session_state.watchlists = _merge_watchlists_with_defaults(st.session_state.watchlists)
    if "scanner_settings" not in st.session_state:
        st.session_state.scanner_settings = {
            "min_score": 65,
            "top_n": 8,
            "scan_limit": 24,
            "market_sweep_limit": 140,
        }
    else:
        st.session_state.scanner_settings.setdefault("scan_limit", 24)
        st.session_state.scanner_settings.setdefault("market_sweep_limit", 140)
    if "realtime_settings" not in st.session_state:
        st.session_state.realtime_settings = {
            "min_score": 60,
            "interval": "5m",
        }
    st.session_state.setdefault("active_scan_preset", st.session_state.get("scan_preset", "균형 추천"))
    st.session_state.setdefault("cache_only_mode", False)


@st.cache_resource
def get_background_analyzer() -> BackgroundAnalyzer:
    analyzer = BackgroundAnalyzer(interval_seconds=1800)
    analyzer.start()
    return analyzer


def render_sidebar() -> tuple[str, str]:
    st.sidebar.header("조회 설정")
    market = st.sidebar.selectbox("시장", ["US", "KR"], index=0)
    default_ticker = "AAPL" if market == "US" else "005930.KS"
    ticker = st.sidebar.text_input("종목 코드", value=default_ticker).strip().upper()
    st.sidebar.caption("한국 종목은 Yahoo Finance 형식으로 `.KS` 또는 `.KQ`를 붙여 주세요.")
    st.sidebar.divider()
    st.sidebar.subheader("작동 모드")
    preset_name = st.sidebar.selectbox(
        "스캔 프리셋",
        list(SCAN_PRESETS.keys()),
        index=list(SCAN_PRESETS.keys()).index(st.session_state.get("active_scan_preset", "균형 추천"))
        if st.session_state.get("active_scan_preset", "균형 추천") in SCAN_PRESETS
        else 1,
        key="scan_preset_select",
    )
    if st.sidebar.button("프리셋 적용", width="stretch"):
        preset = SCAN_PRESETS[preset_name]
        st.session_state["active_scan_preset"] = preset_name
        st.session_state.scanner_settings["min_score"] = int(preset["min_score"])
        st.session_state.scanner_settings["top_n"] = int(preset["top_n"])
        st.session_state.scanner_settings["scan_limit"] = int(preset["scan_limit"])
        st.session_state.scanner_settings["market_sweep_limit"] = int(preset["market_sweep_limit"])
        st.session_state.realtime_settings["min_score"] = int(preset["realtime_min_score"])
        st.session_state.realtime_settings["interval"] = str(preset["interval"])
        get_today_scan_state.clear()
        get_auto_candidate_sets_state.clear()
        st.sidebar.success(f"{preset_name} 모드를 적용했습니다.")
        st.rerun()
    st.sidebar.caption("빠른 확인은 가볍게, 깊게 탐색은 넓게, 공격 단타는 분봉 민감도를 올립니다.")
    online_lookup = st.sidebar.toggle(
        "온라인 새 데이터 조회",
        value=not bool(st.session_state.get("cache_only_mode", False)),
        help="끄면 로컬 DB만 사용합니다. 단, 오래된 데이터는 추천에서 차단됩니다.",
    )
    st.session_state["cache_only_mode"] = not online_lookup
    if st.session_state["cache_only_mode"]:
        st.sidebar.caption("현재: 로컬 DB 모드. 최신성이 부족하면 추천은 차단됩니다.")
    else:
        st.sidebar.caption("현재: 온라인 조회 허용. 최신성은 높지만 느리거나 실패할 수 있습니다.")
    st.sidebar.divider()
    st.sidebar.subheader("추천 기준")
    st.session_state.scanner_settings["min_score"] = st.sidebar.slider(
        "최소 추천 점수",
        min_value=50,
        max_value=90,
        value=int(st.session_state.scanner_settings["min_score"]),
        step=1,
    )
    st.session_state.scanner_settings["top_n"] = st.sidebar.slider(
        "상위 표시 개수",
        min_value=3,
        max_value=20,
        value=int(st.session_state.scanner_settings["top_n"]),
        step=1,
    )
    st.session_state.scanner_settings["scan_limit"] = st.sidebar.slider(
        "스캔 종목 수",
        min_value=5,
        max_value=80,
        value=min(80, max(5, int(st.session_state.scanner_settings["scan_limit"]))),
        step=5,
        help="오늘 추천은 빠르게 보여주는 것이 우선이라 기본값을 작게 둡니다. 상세 탐색은 별도 버튼으로 실행합니다.",
    )
    st.session_state.scanner_settings["market_sweep_limit"] = st.sidebar.slider(
        "단타/예산 시장 탐색 수",
        min_value=40,
        max_value=500,
        value=min(500, max(40, int(st.session_state.scanner_settings["market_sweep_limit"]))),
        step=20,
        help="단타와 예산 플래너에서 시장을 얼마나 넓게 훑을지 정합니다. CSV 유니버스를 넣으면 수천 종목도 이 범위 안에서 탐색합니다.",
    )
    st.sidebar.divider()
    st.sidebar.subheader("실시간 기준")
    st.session_state.realtime_settings["min_score"] = st.sidebar.slider(
        "실시간 최소 점수",
        min_value=45,
        max_value=90,
        value=int(st.session_state.realtime_settings["min_score"]),
        step=1,
    )
    st.session_state.realtime_settings["interval"] = st.sidebar.selectbox(
        "분봉 간격",
        options=["1m", "5m", "15m"],
        index=["1m", "5m", "15m"].index(st.session_state.realtime_settings["interval"]),
    )

    st.sidebar.divider()
    st.sidebar.subheader("자동 분석 엔진")
    auto_engine_enabled = st.sidebar.toggle(
        "자동 분석 엔진 실행",
        value=bool(st.session_state.get("auto_engine_enabled", False)),
        help="켜면 서버가 켜져 있는 동안 백그라운드 스캔을 합니다. 화면 반응성을 위해 기본은 꺼둡니다.",
    )
    st.session_state["auto_engine_enabled"] = auto_engine_enabled
    if auto_engine_enabled:
        analyzer = get_background_analyzer()
        status = analyzer.get_status()
        st.sidebar.caption("서버가 켜져 있는 동안 자동 후보군을 주기적으로 저장합니다.")
        st.sidebar.write(f"상태: {'실행중' if status['running'] else '중지'}")
        st.sidebar.write(f"주기: {int(status['interval_seconds']) // 60}분")
        if status["last_run_at"]:
            st.sidebar.write(f"최근 실행: {_compact_timestamp(status['last_run_at'])}")
        st.sidebar.write(f"최근 결과: {status['last_result']}")
        if status.get("error_message"):
            st.sidebar.error(str(status["error_message"]))
        if st.sidebar.button("지금 자동 분석 실행", width="stretch"):
            saved_count = analyzer.run_once()
            st.sidebar.success(f"자동 분석을 실행했고 {saved_count}개 스냅샷을 저장했습니다.")
    else:
        st.sidebar.caption("현재: 꺼짐. 버튼을 누른 분석만 실행해서 화면 반응성을 우선합니다.")

    with st.sidebar.expander("로컬 DB 상태", expanded=False):
        db_stats = get_sqlite_cache_stats()
        us_sweep_count = len(get_market_sweep_universe("US"))
        kr_sweep_count = len(get_market_sweep_universe("KR"))
        readiness_limit = int(st.session_state.scanner_settings["market_sweep_limit"])
        us_ready = _daily_cache_readiness("US", readiness_limit)
        kr_ready = _daily_cache_readiness("KR", readiness_limit)
        st.write(f"시세/프로필 캐시: {db_stats['cache_entries']:,}개")
        st.write(f"만료되어 추천 차단: {db_stats['expired_cache_entries']:,}개")
        st.write(f"스냅샷: {db_stats['scan_snapshots']:,}개")
        st.write(f"피처 로그: {db_stats['feature_entries']:,}개")
        st.write(f"가격 원본 바: {int(db_stats.get('price_bars', 0) or 0):,}개")
        st.write(f"가격 저장 종목: {int(db_stats.get('price_tickers', 0) or 0):,}개")
        st.write(f"시장 탐색 유니버스: US {us_sweep_count:,}개 / KR {kr_sweep_count:,}개")
        st.write(f"DB 크기: {db_stats['db_size_mb']:.2f} MB")
        st.caption(f"현재 탐색 수 {readiness_limit:,}개 기준 최신 데이터 준비도입니다. 만료 캐시는 추천에 쓰지 않습니다.")
        for ready in [us_ready, kr_ready]:
            progress_value = min(1.0, max(0.0, float(ready["coverage_pct"]) / 100))
            st.progress(progress_value)
            st.caption(
                f"{ready['market']} {ready['label']} · 추천 가능 {int(ready['usable']):,}/{int(ready['total']):,}개 "
                f"(신선 {int(ready['fresh']):,}, 만료 {int(ready['stale']):,}, 없음 {int(ready['missing']):,})"
            )
        warehouse_stats = get_price_warehouse_stats()
        if not warehouse_stats.empty:
            st.caption("가격 원본 저장소")
            _show_table(
                warehouse_stats,
                column_config={
                    "market": st.column_config.TextColumn("시장", width="small"),
                    "interval": st.column_config.TextColumn("간격", width="small"),
                    "bars": st.column_config.NumberColumn("바수", format="%d", width="small"),
                    "tickers": st.column_config.NumberColumn("종목수", format="%d", width="small"),
                    "first_ts": st.column_config.TextColumn("시작", width="small"),
                    "last_ts": st.column_config.TextColumn("최근", width="small"),
                },
            )
        if st.button("시장 종목 목록 갱신", width="stretch"):
            try:
                counts = refresh_universe_files()
                st.success(f"시장 종목 목록 갱신 완료: US {counts['US']:,}개 / KR {counts['KR']:,}개")
                st.rerun()
            except Exception as exc:
                st.warning(str(exc))
        warm_limit = st.number_input(
            "일봉 캐시 예열 수",
            min_value=10,
            max_value=300,
            value=min(300, max(50, readiness_limit)),
            step=10,
            help="시장 탐색 앞쪽 종목의 일봉 데이터를 미리 받아 단타/예산 스캔을 덜 답답하게 만듭니다.",
        )
        if st.button("일봉 캐시 예열", width="stretch"):
            with st.spinner("US/KR 일봉 캐시를 채우는 중입니다. 첫 실행은 시간이 걸릴 수 있습니다."):
                counts = _warm_daily_cache(int(warm_limit))
            st.success(f"캐시 예열 완료: US {counts['US']:,}개 / KR {counts['KR']:,}개")

    return market, ticker


@st.cache_data(ttl=1800, show_spinner=False)
def get_learning_state() -> tuple[
    dict[tuple[str, str, str], object],
    pd.DataFrame,
    dict[str, object],
    dict[str, object],
    pd.DataFrame,
]:
    return build_learning_adjustments(limit=400, min_samples=3)


@st.cache_data(ttl=1800, show_spinner=False)
def get_tracking_state() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        return evaluate_scan_history(limit=400)
    except Exception:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty


def _clear_all_cached_data() -> None:
    clear_sqlite_cache()
    get_stock_data.clear()
    get_intraday_stock_data.clear()
    get_latest_quote.clear()
    get_learning_state.clear()
    get_tracking_state.clear()
    get_today_scan_state.clear()
    get_strategy_profiles_state.clear()
    get_dividend_profiles_state.clear()
    get_compounder_state.clear()
    get_auto_candidate_sets_state.clear()
    get_portfolio_analysis_state.clear()
    get_portfolio_outlook_state.clear()


def _warm_daily_cache(limit_per_market: int) -> dict[str, int]:
    counts = {"US": 0, "KR": 0}
    limit = max(1, int(limit_per_market))
    for market in ["US", "KR"]:
        for item in get_market_sweep_universe(market)[:limit]:
            data = get_stock_data(str(item.get("ticker", "")))
            if not data.empty:
                counts[market] += 1
    return counts


def _daily_cache_readiness(market: str, limit: int) -> dict[str, object]:
    limit = max(1, int(limit))
    universe = get_market_sweep_universe(market)[:limit]
    keys = [f"stock_data:{str(item.get('ticker', '')).strip().upper()}:5y:1d" for item in universe]
    keys = [key for key in keys if key != "stock_data::5y:1d"]
    status = get_cache_key_status(keys)
    total = int(status.get("total", 0) or 0)
    fresh = int(status.get("fresh", 0) or 0)
    stale = int(status.get("stale", 0) or 0)
    missing = int(status.get("missing", 0) or 0)
    usable = fresh
    coverage_pct = round((usable / total) * 100, 1) if total else 0.0
    fresh_pct = round((fresh / total) * 100, 1) if total else 0.0
    if coverage_pct >= 80:
        label = "즉시 스캔 가능"
        action = "넓게 돌려도 체감 속도가 좋습니다."
    elif coverage_pct >= 35:
        label = "부분 준비"
        action = "조금만 예열하면 단타/예산 플랜이 더 빨라집니다."
    else:
        label = "예열 권장"
        action = "첫 넓은 탐색은 느릴 수 있어 일봉 캐시 예열을 추천합니다."
    return {
        "market": market,
        "label": label,
        "action": action,
        "total": total,
        "fresh": fresh,
        "stale": stale,
        "missing": missing,
        "usable": usable,
        "coverage_pct": coverage_pct,
        "fresh_pct": fresh_pct,
    }


def _latest_data_date(data: pd.DataFrame) -> str:
    if data.empty:
        return ""
    column = "Date" if "Date" in data.columns else "Datetime" if "Datetime" in data.columns else ""
    if not column:
        return ""
    latest = pd.to_datetime(data[column].iloc[-1], errors="coerce")
    if pd.isna(latest):
        return ""
    return pd.Timestamp(latest).strftime("%Y-%m-%d")


def _format_data_basis_timestamp(ts: pd.Timestamp | None, *, intraday: bool) -> str:
    if ts is None:
        return "없음"
    if intraday:
        return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


@st.cache_data(ttl=120, show_spinner=False)
def _build_market_data_basis(include_intraday: bool = False) -> pd.DataFrame:
    probes = [
        {"market": "US", "ticker": "SPY", "name": "S&P500 ETF"},
        {"market": "KR", "ticker": "005930.KS", "name": "Samsung Electronics"},
    ]
    rows: list[dict[str, object]] = []
    for probe in probes:
        ticker = str(probe["ticker"])
        try:
            daily = get_stock_data(ticker)
        except Exception:
            daily = pd.DataFrame()
        daily_ok = is_recent_price_data(daily, max_age_days=3)
        rows.append(
            {
                "market": probe["market"],
                "kind": "일봉",
                "ticker": ticker,
                "basis_time": _format_data_basis_timestamp(latest_price_timestamp(daily), intraday=False),
                "status": "추천 가능" if daily_ok else "추천 차단",
            }
        )
        if include_intraday:
            try:
                intraday = get_intraday_stock_data(ticker)
            except Exception:
                intraday = pd.DataFrame()
            intraday_ok = is_recent_price_data(intraday, max_age_days=1)
            rows.append(
                {
                    "market": probe["market"],
                    "kind": "분봉",
                    "ticker": ticker,
                    "basis_time": _format_data_basis_timestamp(latest_price_timestamp(intraday), intraday=True),
                    "status": "추천 가능" if intraday_ok else "추천 차단",
                }
            )
    return pd.DataFrame(rows)


def _render_data_freshness_gate(*, purpose: str, require_intraday: bool = False) -> bool:
    basis = _build_market_data_basis(include_intraday=require_intraday)
    if basis.empty:
        st.error("가격 데이터 기준시각을 확인하지 못했습니다. 추천을 차단합니다.")
        return False

    blocked = basis[basis["status"] != "추천 가능"]
    st.markdown("#### 데이터 기준시각")
    _show_table(
        basis,
        column_config={
            "market": st.column_config.TextColumn("시장", width="small"),
            "kind": st.column_config.TextColumn("데이터", width="small"),
            "ticker": st.column_config.TextColumn("검증티커", width="small"),
            "basis_time": st.column_config.TextColumn("기준시각", width="small"),
            "status": st.column_config.TextColumn("상태", width="small"),
        },
    )
    required_kind = "분봉" if require_intraday else "일봉"
    required = basis[basis["kind"] == required_kind]
    has_usable_market = bool((required["status"] == "추천 가능").any())
    if blocked.empty:
        st.success(f"{purpose}: 최신성 검사를 통과했습니다. 그래도 투자 전 증권사 호가로 최종 확인하세요.")
    elif has_usable_market:
        st.warning(
            f"{purpose}: 일부 시장 데이터가 오래되었거나 비어 있습니다. 차단된 시장/종목은 추천에서 제외하고, "
            "표의 기준시각을 먼저 확인하세요."
        )
    else:
        st.error(
            f"{purpose}: 필요한 {required_kind} 데이터가 최신이 아니라 추천을 중단합니다. "
            "온라인 조회가 켜져 있는지 확인하고 새로고침하세요."
        )
    return has_usable_market


def _audit_market_data_sample(market: str, limit: int = 80) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in get_market_sweep_universe(market)[: max(1, int(limit))]:
        ticker = str(item.get("ticker", "") or "").strip().upper()
        if not ticker:
            continue
        try:
            data = get_stock_data(ticker)
        except Exception:
            data = pd.DataFrame()
        recent = is_recent_price_data(data, max_age_days=3)
        rows.append(
            {
                "market": market,
                "ticker": ticker,
                "name": item.get("name", ""),
                "latest_date": _latest_data_date(data),
                "rows": int(len(data)),
                "status": "정상" if recent else ("데이터 없음" if data.empty else "오래됨/거래중단 의심"),
            }
        )
    return pd.DataFrame(rows)


def _build_data_quality_summary(sample_limit: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = pd.concat(
        [_audit_market_data_sample("US", sample_limit), _audit_market_data_sample("KR", sample_limit)],
        ignore_index=True,
    )
    if detail.empty:
        return pd.DataFrame(), detail
    summary = (
        detail.groupby(["market", "status"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(by=["market", "status"])
        .reset_index(drop=True)
    )
    return summary, detail


def _combined_cache_readiness(limit: int) -> dict[str, object]:
    us_ready = _daily_cache_readiness("US", limit)
    kr_ready = _daily_cache_readiness("KR", limit)
    total = int(us_ready["total"]) + int(kr_ready["total"])
    usable = int(us_ready["usable"]) + int(kr_ready["usable"])
    fresh = int(us_ready["fresh"]) + int(kr_ready["fresh"])
    stale = int(us_ready["stale"]) + int(kr_ready["stale"])
    missing = int(us_ready["missing"]) + int(kr_ready["missing"])
    coverage_pct = round((usable / total) * 100, 1) if total else 0.0
    fresh_pct = round((fresh / total) * 100, 1) if total else 0.0
    return {
        "markets": [us_ready, kr_ready],
        "total": total,
        "usable": usable,
        "fresh": fresh,
        "stale": stale,
        "missing": missing,
        "coverage_pct": coverage_pct,
        "fresh_pct": fresh_pct,
    }


def _recommend_scan_preset(readiness: dict[str, object], *, purpose: str) -> tuple[str, str]:
    coverage = float(readiness.get("coverage_pct", 0) or 0)
    fresh_pct = float(readiness.get("fresh_pct", 0) or 0)
    missing = int(readiness.get("missing", 0) or 0)

    if purpose == "short_term":
        if coverage >= 82 and missing <= 120:
            return "공격 단타", "캐시가 충분해서 분봉 민감도를 올리고 넓게 훑어도 괜찮습니다."
        if coverage >= 50:
            return "균형 추천", "일봉 후보 압축은 빠르게 가능하지만, 분봉 조회는 적당히 제한하는 편이 좋습니다."
        return "빠른 확인", "첫 단타 스캔은 느릴 수 있어 작은 범위로 감을 잡는 편이 안전합니다."

    if coverage >= 82 and fresh_pct >= 15:
        return "깊게 탐색", "저장 데이터가 충분해 예산 플랜 후보군을 넓게 보는 쪽이 좋습니다."
    if coverage >= 45:
        return "균형 추천", "속도와 탐색 폭의 균형이 좋아 예산 플랜 기본값으로 적합합니다."
    return "빠른 확인", "캐시가 아직 얕아서 먼저 빠르게 후보를 보고 필요한 만큼 예열하는 편이 좋습니다."


def _apply_scan_preset(preset_name: str) -> None:
    preset = SCAN_PRESETS[preset_name]
    st.session_state["active_scan_preset"] = preset_name
    st.session_state.scanner_settings["min_score"] = int(preset["min_score"])
    st.session_state.scanner_settings["top_n"] = int(preset["top_n"])
    st.session_state.scanner_settings["scan_limit"] = int(preset["scan_limit"])
    st.session_state.scanner_settings["market_sweep_limit"] = int(preset["market_sweep_limit"])
    st.session_state.realtime_settings["min_score"] = int(preset["realtime_min_score"])
    st.session_state.realtime_settings["interval"] = str(preset["interval"])
    get_today_scan_state.clear()
    get_auto_candidate_sets_state.clear()


def _render_scan_advisor(*, purpose: str, key_prefix: str) -> dict[str, object]:
    limit = int(st.session_state.scanner_settings["market_sweep_limit"])
    readiness = _combined_cache_readiness(limit)
    recommended, reason = _recommend_scan_preset(readiness, purpose=purpose)
    current = str(st.session_state.get("active_scan_preset", "균형 추천"))
    tone = st.success if recommended == current else st.info
    tone(
        f"추천 실행 모드: `{recommended}` · 현재 `{current}` · "
        f"최신 데이터 준비도 {float(readiness['coverage_pct']):.0f}% "
        f"({int(readiness['fresh']):,}/{int(readiness['total']):,}개 추천 가능). {reason}"
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("최신 준비도", f"{float(readiness['coverage_pct']):.0f}%")
    metric_cols[1].metric("신선 캐시", f"{int(readiness['fresh']):,}")
    metric_cols[2].metric("오래됨 차단", f"{int(readiness['stale']):,}")
    metric_cols[3].metric("예열 필요", f"{int(readiness['missing']):,}")

    action_cols = st.columns(2)
    with action_cols[0]:
        if recommended != current and st.button(
            f"{recommended} 모드로 맞추기",
            key=f"{key_prefix}_apply_recommended_preset",
            width="stretch",
        ):
            _apply_scan_preset(recommended)
            st.rerun()
    warm_count = min(300, max(50, int(readiness.get("missing", 0) or 0) // 2))
    with action_cols[1]:
        if int(readiness.get("missing", 0) or 0) > 0 and st.button(
            f"부족 캐시 {warm_count}개 예열",
            key=f"{key_prefix}_warm_missing_cache",
            width="stretch",
        ):
            with st.spinner(f"US/KR 각 {warm_count:,}개까지 일봉 캐시를 예열하는 중입니다."):
                counts = _warm_daily_cache(warm_count)
            st.success(f"예열 완료: US {counts['US']:,}개 / KR {counts['KR']:,}개")
            st.rerun()
    return readiness


def _score_band(score: float) -> str:
    if score >= 85:
        return "매우강함"
    if score >= 75:
        return "강함"
    if score >= 65:
        return "관심"
    if score >= 55:
        return "보통"
    return "주의"


def _score_view(score: float) -> str:
    band = _score_band(score)
    if band == "매우강함":
        return f"{int(round(score))} / 바로확인"
    if band == "강함":
        return f"{int(round(score))} / 꽤좋음"
    if band == "관심":
        return f"{int(round(score))} / 볼만함"
    if band == "보통":
        return f"{int(round(score))} / 애매함"
    return f"{int(round(score))} / 조심"


def _prediction_view(score: float, total_delta: float) -> str:
    if score >= 82 and total_delta >= 3:
        return "상승쪽 우세"
    if score >= 72:
        return "우상향 기대"
    if score >= 58:
        return "관찰 가능"
    if total_delta <= -4 or score < 48:
        return "하방 주의"
    return "애매함"


def _delta_view(delta: float) -> str:
    if delta >= 6:
        return f"+{int(round(delta))} 강한가산"
    if delta > 0:
        return f"+{int(round(delta))} 가산"
    if delta <= -6:
        return f"{int(round(delta))} 강한감산"
    if delta < 0:
        return f"{int(round(delta))} 감산"
    return "0 변화적음"


def _build_pattern_lookup(pattern_stats: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, float]]:
    if pattern_stats.empty:
        return {}

    lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    for _, row in pattern_stats.iterrows():
        key = (
            str(row.get("scan_type", "")),
            str(row.get("market", "")),
            str(row.get("setup", "")),
        )
        lookup[key] = {
            "hit_rate_20d_pct": float(row.get("hit_rate_20d_pct", 0) or 0),
            "target_first_5d_pct": float(row.get("target_first_5d_pct", 0) or 0),
            "picks": float(row.get("picks", 0) or 0),
        }
    return lookup


def _build_event_learning_stats(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if detail.empty:
        empty = pd.DataFrame()
        return empty, empty

    feature_log = pd.DataFrame(load_recent_feature_log(limit=3000))
    if feature_log.empty:
        empty = pd.DataFrame()
        return empty, empty

    merge_columns = ["snapshot_id", "scan_type", "market", "ticker"]
    feature_columns = merge_columns + ["event_risk", "news_bias", "news_score", "news_count"]
    for column, default in {"event_risk": "", "news_bias": "", "news_score": 0.0, "news_count": 0}.items():
        if column not in feature_log.columns:
            feature_log[column] = default
    merged = detail.merge(
        feature_log[feature_columns].drop_duplicates(subset=merge_columns, keep="last"),
        on=merge_columns,
        how="left",
        suffixes=("", "_feature"),
    )
    for column, default in {
        "event_risk": "",
        "news_bias": "",
        "news_score": 0.0,
        "news_count": 0,
        "ret_5d_pct": 0.0,
        "ret_20d_pct": 0.0,
        "hit_20d": False,
        "path_5d": "",
    }.items():
        feature_column = f"{column}_feature"
        if column not in merged.columns:
            merged[column] = default
        if feature_column in merged.columns:
            merged[column] = merged[column].where(merged[column].notna(), merged[feature_column])
        merged[column] = merged[column].fillna(default)
    for numeric_column in ["news_score", "news_count", "ret_5d_pct", "ret_20d_pct"]:
        merged[numeric_column] = pd.to_numeric(merged[numeric_column], errors="coerce").fillna(0.0)
    merged["hit_20d"] = pd.to_numeric(merged["hit_20d"], errors="coerce").fillna(0.0)
    mature = merged[merged["status"] == "평가완료"].copy()
    if mature.empty:
        empty = pd.DataFrame()
        return empty, empty

    event_stats = (
        mature.groupby("event_risk", dropna=False)
        .agg(
            picks=("ticker", "count"),
            avg_ret_5d_pct=("ret_5d_pct", "mean"),
            avg_ret_20d_pct=("ret_20d_pct", "mean"),
            hit_rate_20d_pct=("hit_20d", lambda s: s.mean() * 100),
            target_first_5d_pct=("path_5d", lambda s: ((s == "목표가 선도달") | (s == "목표가 도달")).mean() * 100),
        )
        .reset_index()
        .sort_values(by=["event_risk"], ascending=[True])
        .reset_index(drop=True)
    )

    news_stats = (
        mature.groupby("news_bias", dropna=False)
        .agg(
            picks=("ticker", "count"),
            avg_news_score=("news_score", "mean"),
            avg_news_count=("news_count", "mean"),
            avg_ret_5d_pct=("ret_5d_pct", "mean"),
            avg_ret_20d_pct=("ret_20d_pct", "mean"),
            hit_rate_20d_pct=("hit_20d", lambda s: s.mean() * 100),
        )
        .reset_index()
        .sort_values(by=["avg_ret_20d_pct", "picks"], ascending=[False, False])
        .reset_index(drop=True)
    )

    return event_stats.round(2), news_stats.round(2)


def _enrich_recommendation_frame(
    frame: pd.DataFrame,
    *,
    scan_type: str,
    market: str,
    pattern_lookup: dict[tuple[str, str, str], dict[str, float]],
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    enriched = frame.copy()
    defaults: dict[str, object] = {
        "setup": "",
        "action": "",
        "current_price": None,
        "change_pct": None,
        "quote_as_of": "",
        "data_freshness": "",
        "price_source": "",
        "entry_price": None,
        "stop_loss": None,
        "target_1": None,
        "target_2": None,
        "target_3": None,
        "regime": "",
        "regime_delta": 0,
        "context_delta": 0,
        "learning_delta": 0,
        "trend_score": 0,
        "momentum_score": 0,
        "volume_score": 0,
        "breakout_score": 0,
        "volume_ratio": None,
        "return_20d": None,
        "short_return_pct": None,
        "above_vwap": None,
        "rs_score": None,
        "atr_pct": None,
        "from_52w_high_pct": None,
        "risk_level": "",
        "risk_reward_1": None,
        "exit_rule": "",
        "event_risk": "",
        "event_note": "",
        "earnings_date": "",
        "ex_dividend_date": "",
        "news_bias": "중립",
        "news_score": 0,
        "news_count": 0,
    }
    for column, default_value in defaults.items():
        if column not in enriched.columns:
            enriched[column] = default_value
    enriched["score_band"] = enriched["score"].apply(lambda value: _score_band(float(value)))
    enriched["score_view"] = enriched["score"].apply(lambda value: _score_view(float(value)))
    enriched["total_delta"] = (
        pd.to_numeric(enriched.get("regime_delta", 0), errors="coerce").fillna(0)
        + pd.to_numeric(enriched.get("learning_delta", 0), errors="coerce").fillna(0)
        + pd.to_numeric(enriched.get("context_delta", 0), errors="coerce").fillna(0)
    )
    enriched["delta_view"] = enriched["total_delta"].apply(lambda value: _delta_view(float(value)))
    enriched["prediction_view"] = enriched.apply(
        lambda row: _prediction_view(
            float(pd.to_numeric(row.get("score", 0), errors="coerce") or 0),
            float(pd.to_numeric(row.get("total_delta", 0), errors="coerce") or 0),
        ),
        axis=1,
    )

    def _price_basis(row: pd.Series) -> str:
        entry = pd.to_numeric(row.get("entry_price", None), errors="coerce")
        stop = pd.to_numeric(row.get("stop_loss", None), errors="coerce")
        target = pd.to_numeric(row.get("target_1", None), errors="coerce")
        current = pd.to_numeric(row.get("current_price", None), errors="coerce")
        atr_pct = pd.to_numeric(row.get("atr_pct", None), errors="coerce")
        if pd.isna(entry) or pd.isna(stop) or pd.isna(target):
            return "가격 근거 부족: 시세/지표 확인 필요"
        risk_pct = ((float(entry) - float(stop)) / float(entry) * 100) if float(entry) > 0 else 0
        reward_pct = ((float(target) - float(entry)) / float(entry) * 100) if float(entry) > 0 else 0
        basis = f"진입 {float(entry):.2f}, 손절 -{risk_pct:.1f}%, 목표 +{reward_pct:.1f}%"
        if not pd.isna(current) and float(current) > 0:
            basis += " / 현재가 기준"
        if not pd.isna(atr_pct) and float(atr_pct) > 0:
            basis += f" / ATR {float(atr_pct):.1f}%"
        return basis

    recent_hits: list[float] = []
    recent_targets: list[float] = []
    recent_samples: list[int] = []
    confidence_views: list[str] = []
    confidence_scores: list[int] = []
    confidence_details: list[str] = []
    for _, row in enriched.iterrows():
        setup = str(row.get("setup", "") or "미분류")
        stats = pattern_lookup.get((scan_type, market, setup), {})
        hit_rate = round(float(stats.get("hit_rate_20d_pct", 0.0)), 1)
        target_rate = round(float(stats.get("target_first_5d_pct", 0.0)), 1)
        samples = _safe_int(stats.get("picks", 0.0))
        learning_delta = _safe_int(row.get("learning_delta", 0))
        context_delta = _safe_int(row.get("context_delta", 0))
        base_confidence = 42 + min(24, samples * 2) + max(-12, min(12, learning_delta + context_delta))
        data_freshness = str(row.get("data_freshness", "") or "")
        if data_freshness == "약간 지연":
            base_confidence -= 5
        elif data_freshness == "오래됨":
            base_confidence -= 14
        elif data_freshness == "기준없음":
            base_confidence -= 20
        if hit_rate > 0:
            base_confidence += int((hit_rate - 50) * 0.35)
        if target_rate > 0:
            base_confidence += int((target_rate - 35) * 0.2)
        confidence_score = int(max(15, min(92, base_confidence)))
        recent_hits.append(hit_rate)
        recent_targets.append(target_rate)
        recent_samples.append(samples)
        confidence_scores.append(confidence_score)
        if samples >= 12 and hit_rate >= 55 and confidence_score >= 65:
            confidence_views.append("높음")
        elif samples >= 5 and confidence_score >= 50:
            confidence_views.append("보통")
        elif samples > 0:
            confidence_views.append("낮음")
        else:
            confidence_views.append("신규")
        if samples > 0:
            confidence_details.append(
                f"{confidence_views[-1]} {confidence_score}점 · 표본 {samples} · 20일적중 {hit_rate:.0f}% · 5일목표 {target_rate:.0f}%"
            )
        else:
            confidence_details.append(f"신규 {confidence_score}점 · 아직 이 패턴의 성과 표본이 부족")

    enriched["recent_hit_rate_20d"] = recent_hits
    enriched["recent_target_rate_5d"] = recent_targets
    enriched["recent_sample_count"] = recent_samples
    enriched["confidence_view"] = confidence_views
    enriched["confidence_score"] = confidence_scores
    enriched["confidence_detail"] = confidence_details
    enriched["price_basis"] = enriched.apply(_price_basis, axis=1)
    entry = pd.to_numeric(enriched["entry_price"], errors="coerce")
    stop = pd.to_numeric(enriched["stop_loss"], errors="coerce")
    target_2 = pd.to_numeric(enriched["target_2"], errors="coerce")
    risk = (entry - stop).where((entry > 0) & (stop > 0), entry * 0.03)
    risk = risk.fillna(entry * 0.03).clip(lower=0.01)
    enriched["target_2"] = target_2.fillna(entry + risk * 2.5)
    enriched["target_3"] = pd.to_numeric(enriched["target_3"], errors="coerce").fillna(entry + risk * 3.5)
    return enriched


def _watchlist_cache_key(market: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(item.get("ticker", "")).strip().upper(), str(item.get("name", "") or ""))
        for item in st.session_state.watchlists.get(market, [])
        if is_tradable_ticker(str(item.get("ticker", "")).strip().upper())
    )


def _portfolio_cache_key(portfolio: pd.DataFrame) -> tuple[tuple[object, ...], ...]:
    if portfolio.empty:
        return tuple()
    columns = ["market", "ticker", "name", "quantity", "avg_price", "cash_budget", "target_weight"]
    frame = portfolio.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[columns].fillna("")
    rows: list[tuple[object, ...]] = []
    for _, row in frame.iterrows():
        rows.append(
            (
                str(row.get("market", "")).strip().upper(),
                str(row.get("ticker", "")).strip().upper(),
                str(row.get("name", "") or ""),
                round(float(pd.to_numeric(row.get("quantity", 0), errors="coerce") or 0), 6),
                round(float(pd.to_numeric(row.get("avg_price", 0), errors="coerce") or 0), 6),
                round(float(pd.to_numeric(row.get("cash_budget", 0), errors="coerce") or 0), 6),
                round(float(pd.to_numeric(row.get("target_weight", 0), errors="coerce") or 0), 6),
            )
        )
    return tuple(rows)


def _portfolio_from_cache_key(portfolio_key: tuple[tuple[object, ...], ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market": row[0],
                "ticker": row[1],
                "name": row[2],
                "quantity": row[3],
                "avg_price": row[4],
                "cash_budget": row[5],
                "target_weight": row[6],
            }
            for row in portfolio_key
        ]
    )


@st.cache_data(ttl=600, show_spinner=False)
def get_portfolio_analysis_state(
    portfolio_key: tuple[tuple[object, ...], ...],
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    return analyze_portfolio(_portfolio_from_cache_key(portfolio_key))


@st.cache_data(ttl=600, show_spinner=False)
def get_portfolio_outlook_state(portfolio_key: tuple[tuple[object, ...], ...]) -> pd.DataFrame:
    return build_portfolio_outlook(_portfolio_from_cache_key(portfolio_key))


@st.cache_data(ttl=600, show_spinner=False)
def get_today_scan_state(
    *,
    market: str,
    watchlist_key: tuple[tuple[str, str], ...],
    min_score: int,
    scan_limit: int,
) -> pd.DataFrame:
    watchlist = [{"ticker": ticker, "name": name} for ticker, name in watchlist_key][: max(1, int(scan_limit))]
    learning_adjustments, _, event_adjustments, news_adjustments, _ = get_learning_state()
    _, _, _, _, pattern_stats = get_tracking_state()
    pattern_lookup = _build_pattern_lookup(pattern_stats)
    scan = scan_market(
        market,
        watchlist,
        min_score=min_score,
        learning_adjustments=learning_adjustments,
        event_adjustments=event_adjustments,
        news_adjustments=news_adjustments,
    )
    scan = _enrich_recommendation_frame(scan, scan_type="today_scan", market=market, pattern_lookup=pattern_lookup)
    return _attach_latest_quotes(scan, default_market=market)


@st.cache_data(ttl=1800, show_spinner=False)
def get_strategy_profiles_state(market: str, top_n: int) -> dict[str, pd.DataFrame]:
    return build_strategy_profiles(market, top_n=top_n)


@st.cache_data(ttl=1800, show_spinner=False)
def get_dividend_profiles_state(market: str, top_n: int) -> dict[str, pd.DataFrame]:
    return build_dividend_profiles(market, top_n=top_n)


@st.cache_data(ttl=1800, show_spinner=False)
def get_compounder_state(market: str, top_n: int) -> pd.DataFrame:
    return build_compounder_candidates(market, top_n=top_n)


@st.cache_data(ttl=1800, show_spinner=False)
def get_auto_candidate_sets_state(market: str, top_n: int) -> dict[str, pd.DataFrame]:
    return build_auto_candidate_sets(market, top_n=top_n)


def _enrich_outlook_frame(
    frame: pd.DataFrame,
    *,
    event_adjustments: dict[str, object],
    news_adjustments: dict[str, object],
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    enriched = frame.copy()
    deltas: list[int] = []
    delta_views: list[str] = []
    predictions: list[str] = []
    learning_reasons: list[str] = []
    adjusted_scores: list[int] = []
    adjusted_outlooks: list[str] = []

    for _, row in enriched.iterrows():
        base_score = int(pd.to_numeric(row.get("outlook_score", 0), errors="coerce") or 0)
        adjusted_score, context_delta, context_note = apply_context_adjustment(
            base_score=base_score,
            event_risk=str(row.get("event_risk", "") or ""),
            news_bias=str(row.get("news_bias", "중립") or "중립"),
            event_adjustments=event_adjustments,
            news_adjustments=news_adjustments,
        )
        adjusted_score = max(0, min(100, int(round(adjusted_score))))
        deltas.append(int(context_delta))
        delta_views.append(_delta_view(float(context_delta)))
        predictions.append(_prediction_view(float(adjusted_score), float(context_delta)))
        learning_reasons.append(str(context_note or ""))
        adjusted_scores.append(adjusted_score)
        if adjusted_score >= 75:
            adjusted_outlooks.append("상승 기대")
        elif adjusted_score >= 58:
            adjusted_outlooks.append("완만한 우상향")
        elif adjusted_score >= 43:
            adjusted_outlooks.append("중립/관찰")
        else:
            adjusted_outlooks.append("하방 주의")

    enriched["context_delta"] = deltas
    enriched["delta_view"] = delta_views
    enriched["prediction_view"] = predictions
    enriched["learning_note"] = learning_reasons
    enriched["outlook_score"] = adjusted_scores
    enriched["outlook"] = adjusted_outlooks
    if "reason" in enriched.columns:
        enriched["reason"] = enriched.apply(
            lambda row: f"{row.get('reason', '')} / {row.get('learning_note', '')}".strip(" /")
            if str(row.get("learning_note", "")).strip()
            else str(row.get("reason", "")),
            axis=1,
        )
    return enriched


def _build_portfolio_daily_report(outlook: pd.DataFrame) -> dict[str, object]:
    if outlook.empty:
        return {
            "headline": "",
            "add_candidates": pd.DataFrame(),
            "trim_candidates": pd.DataFrame(),
            "event_watch": pd.DataFrame(),
        }

    add_candidates = outlook[
        outlook["action_hint"].isin(["눌림 추가 가능", "보유 유지"])
        & outlook["outlook"].isin(["상승 기대", "완만한 우상향"])
    ].copy()
    trim_candidates = outlook[
        outlook["action_hint"].isin(["손실관리 우선", "비중 축소 검토", "일부 차익 실현"])
    ].copy()
    event_watch = outlook[outlook["event_risk"] == "높음"].copy()

    strong_count = int(outlook["outlook"].isin(["상승 기대", "완만한 우상향"]).sum())
    weak_count = int((outlook["outlook"] == "하방 주의").sum())
    event_count = int((outlook["event_risk"] == "높음").sum())

    if weak_count >= 2:
        headline = "오늘은 방어 쪽 점검이 먼저입니다."
    elif event_count >= 2:
        headline = "가까운 이벤트가 많아 일정 확인이 중요합니다."
    elif strong_count >= max(2, len(outlook) // 2):
        headline = "계좌 흐름은 대체로 우호적입니다."
    else:
        headline = "추가와 축소를 함께 보는 중립 구간입니다."

    add_candidates = add_candidates.sort_values(
        by=["outlook_score", "weight_gap_pct", "ticker"],
        ascending=[False, True, True],
    ).head(5)
    trim_candidates = trim_candidates.sort_values(
        by=["priority_rank", "outlook_score", "ticker"],
        ascending=[True, True, True],
    ).head(5)
    event_watch = event_watch.sort_values(
        by=["outlook_score", "ticker"],
        ascending=[True, True],
    ).head(5)

    return {
        "headline": headline,
        "add_candidates": add_candidates,
        "trim_candidates": trim_candidates,
        "event_watch": event_watch,
    }


def _portfolio_summary_defaults(summary: dict[str, object] | None = None) -> dict[str, object]:
    defaults: dict[str, object] = {
        "total_value": 0.0,
        "top_weight": 0.0,
        "us_weight": 0.0,
        "kr_weight": 0.0,
        "growth_weight": 0.0,
        "dividend_weight": 0.0,
        "high_risk_weight": 0.0,
    }
    defaults.update(summary or {})
    return defaults


def _finalize_budget_mix(mix: list[dict[str, object]]) -> list[dict[str, object]]:
    cleaned: list[dict[str, object]] = []
    for item in mix:
        weight = max(0.0, float(item.get("weight", 0) or 0))
        cleaned.append({**item, "weight": weight})
    total = sum(float(item["weight"]) for item in cleaned)
    if total <= 0:
        return cleaned
    for item in cleaned:
        item["weight"] = float(item["weight"]) / total
    return cleaned


def _build_budget_mix(mode: str, portfolio_summary: dict[str, object]) -> list[dict[str, object]]:
    growth_weight = float(portfolio_summary.get("growth_weight", 0) or 0)
    dividend_weight = float(portfolio_summary.get("dividend_weight", 0) or 0)
    high_risk_weight = float(portfolio_summary.get("high_risk_weight", 0) or 0)

    if mode == "단타 중심":
        return _finalize_budget_mix([
            {"bucket": "초단타", "weight": 0.18},
            {"bucket": "일반 단타", "weight": 0.35},
            {"bucket": "고위험 단타", "weight": 0.12 if high_risk_weight < 35 else 0.06},
            {"bucket": "성장", "weight": 0.20},
            {"bucket": "장기", "weight": 0.15},
        ])
    if mode == "장기 중심":
        return _finalize_budget_mix([
            {"bucket": "장기", "weight": 0.40},
            {"bucket": "성장", "weight": 0.25},
            {"bucket": "배당", "weight": 0.20},
            {"bucket": "일반 단타", "weight": 0.15},
        ])
    if mode == "배당 중심":
        return _finalize_budget_mix([
            {"bucket": "배당", "weight": 0.45},
            {"bucket": "장기", "weight": 0.25},
            {"bucket": "성장", "weight": 0.15},
            {"bucket": "일반 단타", "weight": 0.15},
        ])
    if mode == "성장 중심":
        return _finalize_budget_mix([
            {"bucket": "성장", "weight": 0.45},
            {"bucket": "장기", "weight": 0.25},
            {"bucket": "일반 단타", "weight": 0.20},
            {"bucket": "배당", "weight": 0.10},
        ])
    if mode == "방어 중심":
        return _finalize_budget_mix([
            {"bucket": "배당", "weight": 0.40},
            {"bucket": "장기", "weight": 0.30},
            {"bucket": "성장", "weight": 0.15},
            {"bucket": "일반 단타", "weight": 0.10},
            {"bucket": "고위험 단타", "weight": 0.05},
        ])

    mix = [
        {"bucket": "장기", "weight": 0.30},
        {"bucket": "성장", "weight": 0.25},
        {"bucket": "배당", "weight": 0.20},
        {"bucket": "초단타", "weight": 0.08},
        {"bucket": "일반 단타", "weight": 0.15},
        {"bucket": "고위험 단타", "weight": 0.02},
    ]
    if growth_weight < 20:
        for item in mix:
            if item["bucket"] == "성장":
                item["weight"] += 0.08
            if item["bucket"] == "장기":
                item["weight"] -= 0.04
            if item["bucket"] == "배당":
                item["weight"] -= 0.04
    if dividend_weight < 10:
        for item in mix:
            if item["bucket"] == "배당":
                item["weight"] += 0.08
            if item["bucket"] == "고위험 단타":
                item["weight"] -= 0.04
            if item["bucket"] == "일반 단타":
                item["weight"] -= 0.04
    if high_risk_weight >= 40:
        for item in mix:
            if item["bucket"] == "고위험 단타":
                item["weight"] = max(0.03, item["weight"] - 0.05)
            if item["bucket"] == "장기":
                item["weight"] += 0.03
            if item["bucket"] == "배당":
                item["weight"] += 0.02
    return _finalize_budget_mix(mix)


def _small_budget_threshold(market: str) -> float:
    return 300.0 if market.upper() == "US" else 100000.0


def _build_small_budget_mix(mode: str) -> list[dict[str, object]]:
    if mode == "배당 중심":
        bucket = "배당"
    elif mode == "장기 중심":
        bucket = "장기"
    elif mode == "성장 중심":
        bucket = "성장"
    elif mode == "방어 중심":
        bucket = "장기"
    else:
        bucket = "예산맞춤"
    return [{"bucket": bucket, "weight": 1.0}]


def _build_split_plan(bucket: str, amount: float) -> tuple[float, float, float]:
    if amount <= 0:
        return 0.0, 0.0, 0.0

    if bucket == "초단타":
        weights = (0.6, 0.3, 0.1)
    elif bucket == "고위험 단타":
        weights = (0.5, 0.3, 0.2)
    elif bucket == "일반 단타":
        weights = (0.5, 0.3, 0.2)
    elif bucket == "성장":
        weights = (0.4, 0.3, 0.3)
    elif bucket == "배당":
        weights = (0.35, 0.3, 0.35)
    else:
        weights = (0.35, 0.3, 0.35)

    first = round(amount * weights[0], 2)
    second = round(amount * weights[1], 2)
    third = round(max(0.0, amount - first - second), 2)
    return first, second, third


def _bucket_cash_reserve_pct(bucket: str) -> float:
    if bucket == "예산맞춤":
        return 0.08
    if bucket in {"초단타", "고위험 단타"}:
        return 0.35
    if bucket == "일반 단타":
        return 0.25
    if bucket == "성장":
        return 0.20
    if bucket == "배당":
        return 0.15
    return 0.18


def _bucket_execution_rule(bucket: str) -> str:
    if bucket == "예산맞춤":
        return "공통추천 우선, 예산 내 1주"
    if bucket == "예산맞춤":
        weights = (0.7, 0.2, 0.1)
    elif bucket == "초단타":
        return "힘 꺾이면 패스"
    if bucket == "고위험 단타":
        return "소액만, 손절 이탈 금지"
    if bucket == "일반 단타":
        return "진입가 근처만"
    if bucket == "성장":
        return "1차만 먼저"
    if bucket == "배당":
        return "분할, 배당락 확인"
    if bucket == "장기":
        return "추격 금지, 분할"
    return "약하면 현금 보류"


def _derive_budget_execution_levels(bucket: str, candidate: pd.Series, ref_price: float) -> dict[str, object]:
    entry_price = pd.to_numeric(candidate.get("entry_price", ref_price), errors="coerce")
    stop_loss = pd.to_numeric(candidate.get("stop_loss", None), errors="coerce")
    target_1 = pd.to_numeric(candidate.get("target_1", None), errors="coerce")
    target_2 = pd.to_numeric(candidate.get("target_2", None), errors="coerce")
    current_price = pd.to_numeric(candidate.get("current_price", ref_price), errors="coerce")

    base_price = float(entry_price) if not pd.isna(entry_price) and float(entry_price) > 0 else ref_price
    current = float(current_price) if not pd.isna(current_price) and float(current_price) > 0 else ref_price

    if pd.isna(stop_loss) or float(stop_loss) <= 0:
        if bucket in {"초단타", "고위험 단타"}:
            stop_loss = base_price * 0.965
        elif bucket == "일반 단타":
            stop_loss = base_price * 0.94
        elif bucket == "배당":
            stop_loss = base_price * 0.90
        else:
            stop_loss = base_price * 0.92

    if pd.isna(target_1) or float(target_1) <= 0:
        if bucket in {"초단타", "고위험 단타"}:
            target_1 = base_price * 1.04
        elif bucket == "일반 단타":
            target_1 = base_price * 1.06
        elif bucket == "배당":
            target_1 = base_price * 1.08
        else:
            target_1 = base_price * 1.10

    if pd.isna(target_2) or float(target_2) <= 0:
        target_2 = float(target_1) * 1.05

    if bucket in {"초단타", "고위험 단타", "일반 단타"}:
        buy_now_limit = base_price * 1.01
        add_on_pullback = base_price * 0.985
        reentry_above = base_price * 1.018
        invalidation = float(stop_loss)
    elif bucket == "배당":
        buy_now_limit = min(current, base_price * 1.005)
        add_on_pullback = base_price * 0.97
        reentry_above = base_price * 1.03
        invalidation = float(stop_loss)
    else:
        buy_now_limit = min(current, base_price * 1.01)
        add_on_pullback = base_price * 0.975
        reentry_above = base_price * 1.035
        invalidation = float(stop_loss)

    return {
        "buy_now_limit": round(float(buy_now_limit), 2),
        "add_on_pullback": round(float(add_on_pullback), 2),
        "stop_loss": round(float(invalidation), 2),
        "reentry_above": round(float(reentry_above), 2),
        "target_1": round(float(target_1), 2),
        "target_2": round(float(target_2), 2),
        "price_basis": (
            f"현재가 {base_price:.2f} / 진입가 {buy_now_limit:.2f} / "
            f"손절 {float(invalidation):.2f} / 1차목표 {float(target_1):.2f}"
        ),
    }


def _budget_confidence_label(score: object, candidate_count: int) -> str:
    score_value = pd.to_numeric(score, errors="coerce")
    if pd.isna(score_value):
        score_value = 0
    score_float = float(score_value)
    if candidate_count <= 1:
        return "낮음: 대체후보 부족"
    if score_float >= 78 and candidate_count >= 3:
        return "높음: 점수+후보 충분"
    if score_float >= 65:
        return "보통: 실행 전 가격확인"
    return "낮음: 소액/관찰"


def _budget_execution_status(bucket: str, ref_price: object, buy_now_limit: object, stop_loss: object) -> str:
    price = pd.to_numeric(ref_price, errors="coerce")
    buy_limit = pd.to_numeric(buy_now_limit, errors="coerce")
    stop = pd.to_numeric(stop_loss, errors="coerce")
    if pd.isna(price) or float(price) <= 0:
        return "가격확인 필요"
    if not pd.isna(stop) and float(price) <= float(stop):
        return "보류: 방어선 근접"
    if not pd.isna(buy_limit) and float(price) <= float(buy_limit):
        return "지금 검토 가능" if bucket not in {"초단타", "고위험 단타"} else "짧게 검토 가능"
    return "대기: 진입가 초과"


def _budget_affordability_status(planned_amount: object, ref_price: object, estimated_units: object) -> str | None:
    amount = pd.to_numeric(planned_amount, errors="coerce")
    price = pd.to_numeric(ref_price, errors="coerce")
    units = pd.to_numeric(estimated_units, errors="coerce")
    if pd.isna(amount) or pd.isna(price) or float(price) <= 0:
        return None
    if not pd.isna(units) and float(units) >= 1:
        return None
    if float(amount) < float(price):
        return "보류: 예산 부족"
    return None


def _append_budget_note(base: object, note: str) -> str:
    text = str(base or "").strip()
    return f"{text} {note}".strip() if text else note


def _budget_allocation_reason(bucket: str, score: object, score_weight_pct: float, candidate_count: int) -> str:
    score_value = pd.to_numeric(score, errors="coerce")
    score_text = f"{int(round(float(score_value)))}점" if not pd.isna(score_value) else "점수 없음"
    weight_text = f"후보 내 비중 {score_weight_pct:.0f}%"
    if candidate_count <= 1:
        return f"{bucket} 후보가 적어 보수 배정. {score_text}, {weight_text}."
    return f"{bucket} 내 상대점수 기준 배정. {score_text}, {weight_text}, 후보 {candidate_count}개."


def _pick_plan_candidates(market: str, top_n: int = 3) -> dict[str, pd.DataFrame]:
    learning_adjustments, _, event_adjustments, news_adjustments, _ = get_learning_state()
    interval = str(st.session_state.realtime_settings["interval"])
    market_sweep_limit = int(st.session_state.scanner_settings["market_sweep_limit"])
    market_universe = get_market_sweep_universe(market)[: max(1, market_sweep_limit)]

    today = scan_market(
        market,
        market_universe,
        min_score=int(st.session_state.scanner_settings["min_score"]),
        learning_adjustments=learning_adjustments,
        event_adjustments=event_adjustments,
        news_adjustments=news_adjustments,
    ).head(top_n)
    short_term = build_short_term_trade_candidates(
        market,
        top_n=top_n,
        interval=interval,
        min_score=max(60, int(st.session_state.realtime_settings["min_score"])),
        universe=market_universe,
        learning_adjustments=learning_adjustments,
        event_adjustments=event_adjustments,
        news_adjustments=news_adjustments,
    ).head(top_n)
    short_term_fast = short_term[
        short_term["setup"].isin(["초강세추격", "급등추격"])
    ].head(top_n)
    short_term_swing = short_term[
        short_term["setup"].isin(["장중돌파", "VWAP지지", "단타돌파"])
    ].head(top_n)
    high_risk = build_high_risk_trade_candidates(
        market,
        top_n=top_n,
        interval=interval,
        min_score=60,
        universe=market_universe,
        learning_adjustments=learning_adjustments,
        event_adjustments=event_adjustments,
        news_adjustments=news_adjustments,
    ).head(top_n)
    strategy_profiles = get_strategy_profiles_state(market, top_n)
    dividend_profiles = get_dividend_profiles_state(market, top_n)
    compounders = get_compounder_state(market, top_n)

    return {
        "budget": today,
        "today": today,
        "short_term": short_term,
        "short_term_fast": short_term_fast,
        "short_term_swing": short_term_swing,
        "high_risk": high_risk,
        "stable": strategy_profiles.get("stable", pd.DataFrame()).head(top_n),
        "growth": strategy_profiles.get("growth", pd.DataFrame()).head(top_n),
        "dividend": dividend_profiles.get("stable", pd.DataFrame()).head(top_n),
        "compounder": compounders.head(top_n),
    }


def _empty_plan_candidates() -> dict[str, pd.DataFrame]:
    return {
        "budget": pd.DataFrame(),
        "today": pd.DataFrame(),
        "short_term": pd.DataFrame(),
        "short_term_fast": pd.DataFrame(),
        "short_term_swing": pd.DataFrame(),
        "high_risk": pd.DataFrame(),
        "stable": pd.DataFrame(),
        "growth": pd.DataFrame(),
        "dividend": pd.DataFrame(),
        "compounder": pd.DataFrame(),
    }


def _recent_plan_candidates(market: str, top_n: int = 3) -> dict[str, pd.DataFrame]:
    candidates = _empty_plan_candidates()
    recent = _build_recent_snapshot_candidates(limit=80)
    if recent.empty:
        return candidates

    frame = recent[recent["market"].astype(str).str.upper() == market.upper()].copy()
    if frame.empty:
        return candidates

    for column in ["setup", "score", "current_price", "entry_price", "stop_loss", "target_1", "reason"]:
        if column not in frame.columns:
            frame[column] = "" if column in {"setup", "reason"} else None
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0)
    frame = frame.sort_values(by=["score", "ticker"], ascending=[False, True]).head(max(top_n, 6)).reset_index(drop=True)

    short_term_mask = frame["scan_type"].astype(str).isin(["realtime_scan", "short_term_trade", "next_day"])
    high_risk_mask = frame["scan_type"].astype(str).isin(["high_risk_trade"])
    long_term_mask = frame["scan_type"].astype(str).isin(["weekly", "monthly", "daily"])

    candidates["today"] = frame.head(top_n)
    candidates["short_term"] = frame[short_term_mask].head(top_n)
    candidates["short_term_swing"] = candidates["short_term"]
    candidates["short_term_fast"] = frame[short_term_mask & (frame["score"] >= 75)].head(top_n)
    candidates["high_risk"] = frame[high_risk_mask].head(top_n)
    candidates["stable"] = frame[long_term_mask].head(top_n)
    candidates["growth"] = frame[long_term_mask | (frame["score"] >= 75)].head(top_n)
    candidates["compounder"] = frame[long_term_mask | (frame["score"] >= 70)].head(top_n)
    return candidates


@st.cache_data(ttl=900, show_spinner=False)
def _build_budget_affordable_universe_candidates(market: str, max_price: float, limit: int = 900) -> pd.DataFrame:
    if max_price <= 0:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for item in get_market_sweep_universe(market)[: max(1, limit)]:
        ticker = str(item.get("ticker", "") or "").strip().upper()
        if not ticker:
            continue
        try:
            data = get_stock_data(ticker)
        except Exception:
            continue
        if data.empty or "Close" not in data.columns or not is_recent_price_data(data, max_age_days=3):
            continue

        latest = data.iloc[-1]
        current_price = pd.to_numeric(latest.get("Close", None), errors="coerce")
        if pd.isna(current_price) or float(current_price) <= 0 or float(current_price) > max_price:
            continue

        ma20 = pd.to_numeric(latest.get("ma20", current_price), errors="coerce")
        ma60 = pd.to_numeric(latest.get("ma60", current_price), errors="coerce")
        rsi = pd.to_numeric(latest.get("rsi", 50), errors="coerce")
        volume_ratio = pd.to_numeric(latest.get("volume_ratio", 1), errors="coerce")
        return_20d = pd.to_numeric(latest.get("return_20d", 0), errors="coerce")
        rs_score = pd.to_numeric(latest.get("rs_score", 0), errors="coerce")
        atr_pct = pd.to_numeric(latest.get("atr_pct", 0), errors="coerce")
        from_high = pd.to_numeric(latest.get("from_52w_high_pct", -50), errors="coerce")

        price = float(current_price)
        trend_bonus = 10 if not pd.isna(ma20) and not pd.isna(ma60) and price > float(ma20) > float(ma60) else 0
        momentum_bonus = min(18, max(0, float(return_20d) * 0.8 if not pd.isna(return_20d) else 0))
        rs_bonus = min(14, max(0, float(rs_score) * 0.6 if not pd.isna(rs_score) else 0))
        volume_bonus = 8 if not pd.isna(volume_ratio) and float(volume_ratio) >= 1.2 else 0
        rsi_penalty = -8 if not pd.isna(rsi) and (float(rsi) >= 78 or float(rsi) <= 25) else 0
        high_bonus = 6 if not pd.isna(from_high) and float(from_high) >= -20 else 0
        score = round(max(35, min(95, 55 + trend_bonus + momentum_bonus + rs_bonus + volume_bonus + high_bonus + rsi_penalty)), 1)

        stop_loss = round(price * (0.94 if market.upper() == "US" else 0.93), 2)
        target_1 = round(price * (1.08 if market.upper() == "US" else 1.10), 2)
        setup = "예산맞춤"
        if momentum_bonus + volume_bonus >= 14:
            setup = "예산맞춤 강세"
        elif trend_bonus:
            setup = "예산맞춤 추세"

        full_units = int(max_price // price) if price > 0 else 0
        budget_use_pct = min(100.0, (full_units * price / max(max_price, 1)) * 100)
        multi_unit_bonus = min(10, full_units) * 1.2
        budget_value_score = round(score * 0.72 + budget_use_pct * 0.18 + multi_unit_bonus, 1)

        rows.append(
            {
                "ticker": ticker,
                "name": str(item.get("name", "") or ""),
                "action": "예산맞춤후보",
                "setup": setup,
                "score": score,
                "current_price": round(price, 2),
                "entry_price": round(price, 2),
                "stop_loss": stop_loss,
                "target_1": target_1,
                "volume_ratio": round(float(volume_ratio), 2) if not pd.isna(volume_ratio) else None,
                "return_20d": round(float(return_20d), 2) if not pd.isna(return_20d) else None,
                "rs_score": round(float(rs_score), 2) if not pd.isna(rs_score) else None,
                "atr_pct": round(float(atr_pct), 2) if not pd.isna(atr_pct) else None,
                "budget_fit_pct": round(price / max_price * 100, 1),
                "budget_use_pct": round(budget_use_pct, 1),
                "budget_max_units": full_units,
                "budget_value_score": budget_value_score,
                "reason": (
                    f"예산 {max_price:,.0f} 이하에서 매수 가능. "
                    f"최대 {full_units}주, 예산활용 {budget_use_pct:.0f}% 후보."
                ),
            }
        )

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(by=["budget_value_score", "score", "budget_use_pct", "ticker"], ascending=[False, False, False, True])
        .reset_index(drop=True)
    )


def _merge_budget_affordable_candidates(
    candidate_sets: dict[str, pd.DataFrame],
    affordable: pd.DataFrame,
    *,
    top_n: int = 24,
) -> dict[str, pd.DataFrame]:
    if affordable.empty:
        return candidate_sets
    enriched = {key: value.copy() for key, value in candidate_sets.items()}
    target_keys = ["budget", "today", "short_term", "short_term_fast", "short_term_swing", "growth", "stable", "compounder"]
    for key in target_keys:
        base = enriched.get(key, pd.DataFrame())
        combined = pd.concat([affordable, base], ignore_index=True) if not base.empty else affordable.copy()
        if "ticker" in combined.columns:
            combined = combined.drop_duplicates(subset=["ticker"], keep="first")
        if "score" in combined.columns:
            combined["_budget_sort_score"] = pd.to_numeric(combined["score"], errors="coerce").fillna(0)
            combined["_budget_sort_fit"] = pd.to_numeric(combined.get("budget_use_pct", 0), errors="coerce").fillna(
                pd.to_numeric(combined.get("budget_fit_pct", 0), errors="coerce").fillna(0)
            )
            combined["_budget_sort_units"] = pd.to_numeric(combined.get("budget_max_units", 0), errors="coerce").fillna(0).clip(upper=10)
            combined = combined.sort_values(
                by=["_budget_sort_score", "_budget_sort_fit", "_budget_sort_units", "ticker"],
                ascending=[False, False, False, True],
            ).drop(
                columns=["_budget_sort_score"],
                errors="ignore",
            )
            combined = combined.drop(columns=["_budget_sort_fit", "_budget_sort_units"], errors="ignore")
        enriched[key] = combined.head(top_n).reset_index(drop=True)
    if enriched.get("high_risk", pd.DataFrame()).empty:
        enriched["high_risk"] = affordable.head(min(8, top_n)).copy()
    return enriched


def _merge_common_recommendations_for_budget(
    candidate_sets: dict[str, pd.DataFrame],
    common: pd.DataFrame,
    *,
    budget: float,
    top_n: int = 24,
) -> dict[str, pd.DataFrame]:
    if common.empty:
        return candidate_sets
    enriched = {key: value.copy() for key, value in candidate_sets.items()}
    source = common.copy()
    if "entry_price" not in source.columns:
        source["entry_price"] = source.get("current_price", None)
    if "current_price" in source.columns:
        price = pd.to_numeric(source["current_price"], errors="coerce")
    else:
        price = pd.to_numeric(source["entry_price"], errors="coerce")
    source["_budget_affordable"] = (price > 0) & (price <= budget)
    source["_budget_score"] = pd.to_numeric(source.get("score", 0), errors="coerce").fillna(0)
    source["_budget_source_rank"] = 100
    source["budget_source"] = "공통추천"
    source = source.sort_values(
        by=["_budget_affordable", "_budget_score", "ticker"],
        ascending=[False, False, True],
    ).drop(columns=["_budget_score", "_budget_source_rank"], errors="ignore")

    budget_base = enriched.get("budget", pd.DataFrame())
    budget_combined = pd.concat([source, budget_base], ignore_index=True) if not budget_base.empty else source
    if "ticker" in budget_combined.columns:
        budget_combined = budget_combined.drop_duplicates(subset=["ticker"], keep="first")
    enriched["budget"] = budget_combined.head(top_n).reset_index(drop=True)

    today_base = enriched.get("today", pd.DataFrame())
    today_combined = pd.concat([source, today_base], ignore_index=True) if not today_base.empty else source
    if "ticker" in today_combined.columns:
        today_combined = today_combined.drop_duplicates(subset=["ticker"], keep="first")
    enriched["today"] = today_combined.head(top_n).reset_index(drop=True)
    return enriched


def _summarize_candidate_names(frame: pd.DataFrame, *, fallback_col: str = "ticker") -> str:
    if frame.empty:
        return "후보 부족"
    parts: list[str] = []
    for _, row in frame.head(3).iterrows():
        ticker = str(row.get("ticker", "") or "")
        name = str(row.get("name", "") or row.get(fallback_col, "") or ticker)
        parts.append(f"{ticker} {name}".strip())
    return ", ".join(parts)


def _get_bucket_source(candidate_sets: dict[str, pd.DataFrame], bucket: str) -> tuple[pd.DataFrame, str]:
    if bucket == "예산맞춤":
        source = candidate_sets.get("budget", pd.DataFrame())
        if source.empty:
            source = candidate_sets.get("today", pd.DataFrame())
        if source.empty:
            source = candidate_sets.get("short_term", pd.DataFrame())
        return source, "entry_price"
    if bucket == "일반 단타":
        return candidate_sets["short_term"], "entry_price"
    if bucket == "초단타":
        source = candidate_sets["short_term_fast"] if not candidate_sets["short_term_fast"].empty else candidate_sets["short_term"]
        return source, "entry_price"
    if bucket == "고위험 단타":
        return candidate_sets["high_risk"], "entry_price"
    if bucket == "성장":
        source = candidate_sets["growth"] if not candidate_sets["growth"].empty else candidate_sets["today"]
        return source, "current_price"
    if bucket == "배당":
        source = candidate_sets["dividend"] if not candidate_sets["dividend"].empty else candidate_sets["stable"]
        return source, "accumulate_low"
    if bucket == "장기":
        source = candidate_sets["compounder"] if not candidate_sets["compounder"].empty else candidate_sets["stable"]
        if source.empty:
            source = candidate_sets["growth"] if not candidate_sets["growth"].empty else candidate_sets["today"]
        return source, "current_price"
    return candidate_sets["today"], "current_price"


def _candidate_set_health(candidate_sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    labels = {
        "budget": "예산맞춤",
        "today": "오늘추천",
        "short_term": "일반단타",
        "short_term_fast": "초단타",
        "short_term_swing": "스윙단타",
        "high_risk": "고위험",
        "stable": "안정",
        "growth": "성장",
        "dividend": "배당",
        "compounder": "장기",
    }
    rows: list[dict[str, object]] = []
    for key, label in labels.items():
        frame = candidate_sets.get(key, pd.DataFrame())
        diagnostics = frame.attrs.get("diagnostics", {}) if hasattr(frame, "attrs") else {}
        top_score = None
        top_ticker = ""
        if not frame.empty:
            top_score = pd.to_numeric(frame.get("score", pd.Series([None])).iloc[0], errors="coerce")
            top_score = None if pd.isna(top_score) else float(top_score)
            top_ticker = str(frame.iloc[0].get("ticker", "") or "")
        count = int(len(frame))
        rows.append(
            {
                "bucket": label,
                "candidate_count": count,
                "top_ticker": top_ticker,
                "top_score": top_score,
                "scanned": int(diagnostics.get("scanned", 0) or 0),
                "daily_pass": int(diagnostics.get("daily_pass", 0) or 0),
                "intraday_pass": int(diagnostics.get("intraday_pass", 0) or 0),
                "errors": int(diagnostics.get("errors", 0) or 0),
                "status": "충분" if count >= 2 else ("부족" if count == 1 else "없음"),
            }
        )
    return pd.DataFrame(rows)


def _normalize_budget_plan_amounts(frame: pd.DataFrame, budget: float) -> pd.DataFrame:
    if frame.empty or budget <= 0:
        return frame
    normalized = frame.copy()
    total = float(pd.to_numeric(normalized["planned_amount"], errors="coerce").fillna(0).sum())
    if total <= 0 or total <= budget:
        return normalized
    scale = budget / total
    for column in ["planned_amount", "executable_amount", "reserve_amount", "split_1", "split_2", "split_3"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0).apply(
                lambda value: round(float(value) * scale, 2)
            )
    if "unit_price" in normalized.columns:
        prices = pd.to_numeric(normalized["unit_price"], errors="coerce")
        normalized["estimated_units"] = [
            int(float(amount) // float(price)) if not pd.isna(price) and float(price) > 0 else None
            for amount, price in zip(normalized["planned_amount"], prices, strict=False)
        ]
        planned = pd.to_numeric(normalized["planned_amount"], errors="coerce").fillna(0)
        units = pd.to_numeric(normalized["estimated_units"], errors="coerce").fillna(0)
        shortage_mask = prices.notna() & (prices > planned) & (units < 1)
        if shortage_mask.any():
            normalized.loc[shortage_mask, "executable_amount"] = 0.0
            normalized.loc[shortage_mask, "reserve_amount"] = planned[shortage_mask]
            for column in ["split_1", "split_2", "split_3"]:
                if column in normalized.columns:
                    normalized.loc[shortage_mask, column] = 0.0
            normalized.loc[shortage_mask, "note"] = normalized.loc[shortage_mask, "note"].apply(
                lambda value: _append_budget_note(value, "1주 매수 불가, 현금 보류.")
            )
    normalized["note"] = normalized["note"].astype(str) + " 예산한도 정규화."
    return normalized


def _build_budget_candidate_detail(plan_df: pd.DataFrame, candidate_sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if plan_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, plan_row in plan_df.iterrows():
        bucket = str(plan_row.get("bucket", "") or "")
        amount = float(pd.to_numeric(plan_row.get("planned_amount", 0), errors="coerce") or 0)
        source, price_col = _get_bucket_source(candidate_sets, bucket)
        if source.empty or amount <= 0:
            continue

        detail_source = source.copy()
        if price_col in detail_source.columns:
            ref_prices = pd.to_numeric(detail_source[price_col], errors="coerce")
        elif "current_price" in detail_source.columns:
            ref_prices = pd.to_numeric(detail_source["current_price"], errors="coerce")
        else:
            ref_prices = pd.Series([pd.NA] * len(detail_source), index=detail_source.index)
        current_prices = pd.to_numeric(detail_source.get("current_price", pd.Series([pd.NA] * len(detail_source), index=detail_source.index)), errors="coerce")
        detail_source["_budget_ref_price"] = ref_prices.fillna(current_prices)
        detail_source["_budget_score"] = pd.to_numeric(detail_source.get("score", 0), errors="coerce").fillna(0)
        detail_source["_budget_fit"] = (
            pd.to_numeric(detail_source["_budget_ref_price"], errors="coerce").fillna(0).clip(lower=0) / max(amount, 1)
        ).clip(upper=1)
        ref_numeric = pd.to_numeric(detail_source["_budget_ref_price"], errors="coerce")
        detail_source["_budget_full_units"] = [
            int(amount // float(price)) if not pd.isna(price) and float(price) > 0 else 0
            for price in ref_numeric
        ]
        detail_source["_budget_use_full"] = (
            ref_numeric.fillna(0).clip(lower=0) * detail_source["_budget_full_units"] / max(amount, 1)
        ).clip(upper=1)
        detail_source["_budget_multi_unit"] = pd.to_numeric(
            detail_source["_budget_full_units"],
            errors="coerce",
        ).fillna(0).clip(upper=10) / 10
        detail_source["_budget_common_rank"] = detail_source.get(
            "budget_source",
            pd.Series([""] * len(detail_source), index=detail_source.index),
        ).astype(str).eq("공통추천").astype(int)
        detail_source["_budget_value_score"] = (
            detail_source["_budget_score"] * 0.58
            + detail_source["_budget_use_full"] * 24
            + detail_source["_budget_multi_unit"] * 14
            + detail_source["_budget_common_rank"] * 5
        )
        detail_source.loc[detail_source["_budget_full_units"] < 1, "_budget_value_score"] -= 80
        affordable = detail_source[
            detail_source["_budget_ref_price"].notna()
            & (detail_source["_budget_ref_price"] > 0)
            & (detail_source["_budget_ref_price"] <= amount)
        ]
        market = str(plan_row.get("market", "") or "")
        small_budget_detail = amount <= _small_budget_threshold(market) if market else amount <= 300
        if not affordable.empty:
            detail_pool = affordable
            if small_budget_detail:
                cheap_pool = affordable[affordable["_budget_ref_price"] <= max(amount * 0.5, 1)].copy()
                if not cheap_pool.empty:
                    detail_pool = cheap_pool
            detail = detail_pool.sort_values(
                by=["_budget_value_score", "_budget_score", "_budget_use_full", "ticker"],
                ascending=[False, False, False, True],
            ).head(4 if small_budget_detail else 3).copy()
        else:
            detail = detail_source.sort_values(
                by=["_budget_value_score", "_budget_ref_price", "_budget_score"],
                ascending=[False, True, False],
                na_position="last",
            ).head(3).copy()

        score_series = pd.to_numeric(detail.get("_budget_value_score", detail.get("score", 0)), errors="coerce").fillna(0).clip(lower=1)
        score_sum = float(score_series.sum())
        if score_sum <= 0 or len(detail) == 0:
            weights = [1 / len(detail)] * len(detail)
        else:
            weights = [float(value) / score_sum for value in score_series]
        candidate_count = len(detail)

        for idx, (_, row) in enumerate(detail.iterrows()):
            alloc_amount = round(amount * weights[idx], 2)
            score_weight_pct = round(float(weights[idx]) * 100, 1)
            ref_price = pd.to_numeric(row.get(price_col, row.get("current_price", None)), errors="coerce")
            if pd.isna(ref_price):
                ref_price = pd.to_numeric(row.get("current_price", None), errors="coerce")
            units = int(alloc_amount // float(ref_price)) if not pd.isna(ref_price) and float(ref_price) > 0 else None
            shortage_status = _budget_affordability_status(
                alloc_amount,
                None if pd.isna(ref_price) else float(ref_price),
                units,
            )
            first_amount, second_amount, third_amount = (
                (0.0, 0.0, 0.0) if shortage_status else _build_split_plan(bucket, alloc_amount)
            )
            levels = (
                _derive_budget_execution_levels(bucket, row, float(ref_price))
                if not pd.isna(ref_price) and float(ref_price) > 0
                else {
                    "buy_now_limit": None,
                    "add_on_pullback": None,
                    "stop_loss": None,
                    "reentry_above": None,
                    "target_1": None,
                    "target_2": None,
                    "price_basis": "가격 데이터 부족",
                }
            )

            rows.append(
                {
                    "bucket": bucket,
                    "ticker": row.get("ticker", ""),
                    "name": row.get("name", ""),
                    "setup": row.get("setup", row.get("style", "")),
                    "budget_source": row.get("budget_source", row.get("scan_type", "공통/예산")),
                    "score": row.get("score", None),
                    "planned_amount": alloc_amount,
                    "ref_price": None if pd.isna(ref_price) else round(float(ref_price), 2),
                    "estimated_units": units,
                    "max_units_if_all_in": int(row.get("_budget_full_units", 0) or 0),
                    "budget_use_pct": round(float(row.get("_budget_use_full", 0) or 0) * 100, 1),
                    "first_amount": first_amount,
                    "second_amount": second_amount,
                    "third_amount": third_amount,
                    "buy_now_limit": levels["buy_now_limit"],
                    "add_on_pullback": levels["add_on_pullback"],
                    "stop_loss": levels["stop_loss"],
                    "reentry_above": levels["reentry_above"],
                    "target_1": levels["target_1"],
                    "target_2": levels["target_2"],
                    "execution_rule": _bucket_execution_rule(bucket),
                    "execution_status": shortage_status or _budget_execution_status(
                        bucket,
                        None if pd.isna(ref_price) else float(ref_price),
                        levels["buy_now_limit"],
                        levels["stop_loss"],
                    ),
                    "confidence_view": _budget_confidence_label(row.get("score", None), candidate_count),
                    "confidence_score": min(88, max(25, int(_safe_float(row.get("score", 0)) * 0.75) + (8 if candidate_count >= 3 else 0))),
                    "confidence_detail": (
                        f"{_budget_confidence_label(row.get('score', None), candidate_count)} · "
                        f"후보 {candidate_count}개 · 점수 {_safe_int(row.get('score', 0))}"
                    ),
                    "score_weight_pct": score_weight_pct,
                    "price_basis": levels.get("price_basis", ""),
                    "allocation_reason": _append_budget_note(
                        _budget_allocation_reason(
                            bucket,
                            row.get("score", None),
                            score_weight_pct,
                            candidate_count,
                        ),
                        "배정금액이 1주 가격보다 작아 실행이 아니라 현금 보류입니다." if shortage_status else "",
                    ),
                    "quality_note": _append_budget_note(
                        "대체 후보 충분" if candidate_count >= 3 else "대체 후보가 적어 가격 조건 엄격",
                        "소액예산: 1주 매수 불가" if shortage_status else "",
                    ),
                    "reason": row.get("reason", ""),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _build_budget_plan_rows(
    *,
    market: str,
    budget: float,
    mode: str,
    portfolio_summary: dict[str, object],
    outlook: pd.DataFrame | None = None,
    candidate_sets: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if budget <= 0:
        return pd.DataFrame()

    candidate_sets = candidate_sets or _pick_plan_candidates(market, top_n=3)
    small_budget_mode = 0 < budget <= _small_budget_threshold(market)
    mix = _build_small_budget_mix(mode) if small_budget_mode else _build_budget_mix(mode, portfolio_summary)
    rows: list[dict[str, object]] = []

    for item in mix:
        bucket = str(item["bucket"])
        weight = float(item["weight"])
        amount = round(budget * weight, 2)
        bucket_outlook = pd.DataFrame()

        if bucket == "예산맞춤":
            source, price_col = _get_bucket_source(candidate_sets, bucket)
            horizon = "오늘~2주"
            note = "다른 추천 탭과 같은 공통추천 우선."
            plan_type = "예산맞춤"
        elif bucket == "일반 단타":
            source, price_col = _get_bucket_source(candidate_sets, bucket)
            horizon = "3일~10일"
            note = "짧게. 손절 엄격."
            plan_type = "스윙단타"
        elif bucket == "초단타":
            source, price_col = _get_bucket_source(candidate_sets, bucket)
            horizon = "당일~3일"
            note = "탄력 우선. 빠른 손절."
            plan_type = "초단타"
        elif bucket == "고위험 단타":
            source, price_col = _get_bucket_source(candidate_sets, bucket)
            horizon = "당일~5일"
            note = "소액만. 물타기 금지."
            plan_type = "고위험단타"
        elif bucket == "성장":
            source, price_col = _get_bucket_source(candidate_sets, bucket)
            horizon = "1개월~6개월"
            note = "성장+상대강도 우선."
            plan_type = "성장"
            if outlook is not None and not outlook.empty:
                bucket_outlook = outlook[outlook["action_hint"].isin(["눌림 추가 가능", "보유 유지"])]
        elif bucket == "배당":
            source, price_col = _get_bucket_source(candidate_sets, bucket)
            horizon = "6개월~3년"
            note = "배당+방어력."
            plan_type = "배당"
            if outlook is not None and not outlook.empty:
                bucket_outlook = outlook[outlook["action_hint"].isin(["눌림 추가 가능", "보유 유지"])]
        elif bucket == "장기":
            source, price_col = _get_bucket_source(candidate_sets, bucket)
            horizon = "6개월~3년+"
            note = "장기 우상향 우선."
            plan_type = "장기"
            if outlook is not None and not outlook.empty:
                bucket_outlook = outlook[outlook["action_hint"].isin(["눌림 추가 가능", "보유 유지"])]
        else:
            source, price_col = _get_bucket_source(candidate_sets, bucket)
            horizon = "유동적"
            note = "지금 강한 후보."
            plan_type = bucket

        if not bucket_outlook.empty:
            avg_gap = pd.to_numeric(bucket_outlook.get("weight_gap_pct", None), errors="coerce").dropna()
            if not avg_gap.empty:
                mean_gap = float(avg_gap.mean())
                if mean_gap <= -3:
                    amount = round(amount * 1.15, 2)
                    note += " 비중 낮음."
                elif mean_gap >= 3:
                    amount = round(amount * 0.85, 2)
                    note += " 비중 높음."
        if small_budget_mode:
            note += " 소액예산: 가격대별 후보를 같이 비교."

        best_price = None
        share_count = None
        if not source.empty:
            source_for_budget = source.copy()
            if price_col in source_for_budget.columns:
                source_prices = pd.to_numeric(source_for_budget[price_col], errors="coerce")
            elif "current_price" in source_for_budget.columns:
                source_prices = pd.to_numeric(source_for_budget["current_price"], errors="coerce")
            else:
                source_prices = pd.Series([pd.NA] * len(source_for_budget), index=source_for_budget.index)
            current_prices = pd.to_numeric(
                source_for_budget.get("current_price", pd.Series([pd.NA] * len(source_for_budget), index=source_for_budget.index)),
                errors="coerce",
            )
            source_for_budget["_budget_ref_price"] = source_prices.fillna(current_prices)
            affordable_source = source_for_budget[
                source_for_budget["_budget_ref_price"].notna()
                & (source_for_budget["_budget_ref_price"] > 0)
                & (source_for_budget["_budget_ref_price"] <= amount)
            ]
            sample_row = affordable_source.iloc[0] if not affordable_source.empty else source_for_budget.iloc[0]
            sample_price = pd.to_numeric(sample_row.get(price_col, sample_row.get("current_price", None)), errors="coerce")
            if pd.isna(sample_price):
                sample_price = pd.to_numeric(sample_row.get("current_price", None), errors="coerce")
            if not pd.isna(sample_price) and float(sample_price) > 0:
                best_price = round(float(sample_price), 2)
                share_count = int(amount // float(sample_price)) if amount > 0 else 0

        split_1, split_2, split_3 = _build_split_plan(bucket, amount)
        reserve_pct = _bucket_cash_reserve_pct(bucket)
        reserve_amount = round(amount * reserve_pct, 2)
        executable_amount = round(max(0.0, amount - reserve_amount), 2)
        if source.empty:
            reserve_amount = amount
            executable_amount = 0.0
            note += " 후보 부족, 현금."
        elif share_count == 0 and best_price and best_price > amount:
            reserve_amount = amount
            executable_amount = 0.0
            split_1, split_2, split_3 = 0.0, 0.0, 0.0
            note += " 1주 매수 불가, 현금 보류."

        rows.append(
            {
                "market": market,
                "bucket": bucket,
                "plan_type": plan_type,
                "allocation_pct": round(weight * 100, 1),
                "planned_amount": amount,
                "executable_amount": executable_amount,
                "reserve_amount": reserve_amount,
                "horizon": horizon,
                "unit_price": best_price,
                "estimated_units": share_count,
                "split_1": split_1,
                "split_2": split_2,
                "split_3": split_3,
                "candidates": _summarize_candidate_names(source),
                "execution_rule": _bucket_execution_rule(bucket),
                "note": note,
            }
        )

    frame = pd.DataFrame(rows)
    frame = _normalize_budget_plan_amounts(frame, budget)
    return frame.sort_values(by=["allocation_pct", "bucket"], ascending=[False, True]).reset_index(drop=True)


def _build_combined_budget_actions(
    *,
    us_plan: pd.DataFrame,
    kr_plan: pd.DataFrame,
    us_detail: pd.DataFrame,
    kr_detail: pd.DataFrame,
) -> pd.DataFrame:
    detail_frames: list[pd.DataFrame] = []
    if not us_detail.empty:
        detail_frames.append(us_detail.assign(market="US"))
    if not kr_detail.empty:
        detail_frames.append(kr_detail.assign(market="KR"))

    if not detail_frames:
        return pd.DataFrame()

    detail = pd.concat(detail_frames, ignore_index=True)
    if detail.empty:
        return pd.DataFrame()

    plan_frames: list[pd.DataFrame] = []
    if not us_plan.empty:
        plan_frames.append(us_plan[["bucket", "allocation_pct", "planned_amount", "executable_amount", "reserve_amount", "execution_rule", "note"]].assign(market="US"))
    if not kr_plan.empty:
        plan_frames.append(kr_plan[["bucket", "allocation_pct", "planned_amount", "executable_amount", "reserve_amount", "execution_rule", "note"]].assign(market="KR"))
    plan_lookup = pd.concat(plan_frames, ignore_index=True) if plan_frames else pd.DataFrame()

    merged = detail.merge(
        plan_lookup,
        on=["market", "bucket"],
        how="left",
        suffixes=("", "_plan"),
    )

    planned_numeric = pd.to_numeric(merged.get("planned_amount", 0), errors="coerce").fillna(0)
    ref_price_numeric = pd.to_numeric(merged.get("ref_price", 0), errors="coerce").fillna(0)
    units_numeric = pd.to_numeric(merged.get("estimated_units", 0), errors="coerce").fillna(0)
    budget_short_mask = (planned_numeric > 0) & (ref_price_numeric > planned_numeric) & (units_numeric < 1)
    if budget_short_mask.any():
        merged.loc[budget_short_mask, "execution_status"] = "보류: 예산 부족"
        merged.loc[budget_short_mask, "executable_amount"] = 0.0
        merged.loc[budget_short_mask, "reserve_amount"] = planned_numeric[budget_short_mask]
        merged.loc[budget_short_mask, "first_amount"] = 0.0
        merged.loc[budget_short_mask, "second_amount"] = 0.0
        merged.loc[budget_short_mask, "third_amount"] = 0.0
        merged.loc[budget_short_mask, "allocation_reason"] = merged.loc[budget_short_mask, "allocation_reason"].apply(
            lambda value: _append_budget_note(value, "1주 가격보다 배정금액이 작아 매수 대신 현금 보류.")
        )
        merged.loc[budget_short_mask, "quality_note"] = merged.loc[budget_short_mask, "quality_note"].apply(
            lambda value: _append_budget_note(value, "소액예산: 1주 매수 불가")
        )

    merged["score_value"] = pd.to_numeric(merged.get("score", 0), errors="coerce").fillna(0)
    merged["planned_amount_value"] = pd.to_numeric(merged.get("planned_amount", 0), errors="coerce").fillna(0)
    merged["allocation_value"] = pd.to_numeric(merged.get("allocation_pct", 0), errors="coerce").fillna(0)
    merged["estimated_units_value"] = pd.to_numeric(merged.get("estimated_units", 0), errors="coerce").fillna(0)
    status_bonus = merged.get("execution_status", pd.Series("", index=merged.index)).astype(str).map(
        {
            "지금 검토 가능": 6,
            "짧게 검토 가능": 4,
            "대기: 진입가 초과": -2,
            "보류: 방어선 근접": -8,
            "보류: 예산 부족": -20,
            "가격확인 필요": -4,
        }
    ).fillna(0)
    merged["priority_score"] = (
        merged["score_value"] * 0.6
        + merged["allocation_value"] * 0.25
        + merged["estimated_units_value"].clip(upper=20) * 0.5
        + status_bonus
    )
    merged["action_headline"] = merged.apply(
        lambda row: (
            f"{row.get('market', '')} {row.get('ticker', '')} · "
            f"{row.get('execution_status', '상태확인')} · {row.get('planned_amount', '')} 배정"
        ),
        axis=1,
    )
    merged["action_note"] = merged.apply(
        lambda row: (
            f"{row.get('bucket', '')} / {row.get('setup', '') or '기본세팅'} / "
            f"{row.get('confidence_view', '')} / {row.get('execution_rule', '')} / {row.get('note', '')}"
        ),
        axis=1,
    )
    merged = merged.sort_values(
        by=["priority_score", "score_value", "planned_amount_value"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return merged[
        [
            "market",
            "bucket",
            "ticker",
            "name",
            "setup",
            "budget_source",
            "score",
            "planned_amount",
            "executable_amount",
            "reserve_amount",
            "ref_price",
            "estimated_units",
            "max_units_if_all_in",
            "budget_use_pct",
            "first_amount",
            "second_amount",
            "third_amount",
            "buy_now_limit",
            "add_on_pullback",
            "stop_loss",
            "reentry_above",
            "target_1",
            "target_2",
            "execution_status",
            "confidence_view",
            "confidence_score",
            "confidence_detail",
            "score_weight_pct",
            "priority_score",
            "action_headline",
            "action_note",
            "allocation_reason",
            "quality_note",
            "price_basis",
            "execution_rule",
            "reason",
        ]
    ].head(10)


def _budget_actions_execution_view(actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return actions.copy()
    view = actions.copy()
    view["current_price"] = pd.to_numeric(view.get("ref_price", None), errors="coerce")
    view["entry_price"] = pd.to_numeric(view.get("buy_now_limit", None), errors="coerce")
    view["stop_loss"] = pd.to_numeric(view.get("stop_loss", None), errors="coerce")
    view["target_1"] = pd.to_numeric(view.get("target_1", None), errors="coerce")
    view["target_2"] = pd.to_numeric(view.get("target_2", None), errors="coerce")
    view["target_3"] = pd.to_numeric(view.get("target_3", view["target_2"] * 1.06), errors="coerce")

    current = pd.to_numeric(view["current_price"], errors="coerce")
    entry = pd.to_numeric(view["entry_price"], errors="coerce")
    stop = pd.to_numeric(view["stop_loss"], errors="coerce")
    view["entry_gap_pct"] = ((entry - current) / current.replace(0, pd.NA) * 100).round(2)
    view["risk_pct"] = ((current - stop) / current.replace(0, pd.NA) * 100).round(2)
    view["entry_decision"] = view.apply(
        lambda row: (
            "진입 가능"
            if str(row.get("execution_status", "")).startswith(("지금", "짧게"))
            else ("대기" if "대기" in str(row.get("execution_status", "")) else str(row.get("execution_status", "확인")))
        ),
        axis=1,
    )

    ordered = [
        "market",
        "bucket",
        "ticker",
        "name",
        "entry_decision",
        "current_price",
        "entry_price",
        "stop_loss",
        "target_1",
        "target_2",
        "entry_gap_pct",
        "risk_pct",
        "estimated_units",
        "max_units_if_all_in",
        "budget_use_pct",
        "planned_amount",
        "executable_amount",
        "reserve_amount",
        "score",
        "confidence_score",
        "confidence_detail",
        "setup",
        "budget_source",
        "allocation_reason",
        "price_basis",
        "reason",
    ]
    for column in ordered:
        if column not in view.columns:
            view[column] = ""
    return view[ordered]


def _build_budget_summary_text(combined_actions: pd.DataFrame) -> str:
    if combined_actions.empty:
        return "후보가 부족합니다. 오늘은 현금 보류 쪽이 낫습니다."

    top = combined_actions.iloc[0]
    top_market = str(top.get("market", "") or "")
    top_ticker = str(top.get("ticker", "") or "")
    top_bucket = str(top.get("bucket", "") or "")
    score = _safe_int(top.get("score", 0))

    market_mix = (
        combined_actions.groupby("market")["planned_amount"]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum())
        .to_dict()
    )
    us_weight = float(market_mix.get("US", 0.0))
    kr_weight = float(market_mix.get("KR", 0.0))

    if us_weight > kr_weight * 1.3 and us_weight > 0:
        flow_note = "미국 우세"
    elif kr_weight > us_weight * 1.3 and kr_weight > 0:
        flow_note = "한국 우세"
    else:
        flow_note = "미국/한국 균형"

    buy_limit = pd.to_numeric(top.get("buy_now_limit", None), errors="coerce")
    stop_loss = pd.to_numeric(top.get("stop_loss", None), errors="coerce")
    buy_text = f"진입 상한은 {_format_price(buy_limit, top_market)}" if not pd.isna(buy_limit) else "진입가는 별도 확인"
    stop_text = f"방어선은 {_format_price(stop_loss, top_market)}" if not pd.isna(stop_loss) else "방어선은 별도 확인"
    status = str(top.get("execution_status", "상태확인") or "상태확인")
    confidence = str(top.get("confidence_view", "신뢰도 확인") or "신뢰도 확인")

    return (
        f"1순위: `{top_market} {top_ticker}` / {top_bucket} / {score}점 / {status}. "
        f"{buy_text}, {stop_text}. 흐름은 {flow_note}. 상한 넘으면 현금 보류."
        f" 신뢰도: {confidence}."
    )


def _remember_recommendation_snapshot(scan_type: str, market: str, frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    now = pd.Timestamp.now(tz="Asia/Seoul")
    if scan_type in {"realtime_scan", "short_term_trade", "high_risk_trade"}:
        minute_bucket = 0 if now.minute < 30 else 30
        snapshot_key = now.replace(minute=minute_bucket, second=0, microsecond=0).isoformat(timespec="minutes")
    else:
        snapshot_key = now.date().isoformat()
    try:
        if scan_type in {"realtime_scan", "short_term_trade", "high_risk_trade"}:
            if has_scan_snapshot_for_prefix(scan_type, market, snapshot_key):
                return False
        elif has_scan_snapshot_for_day(scan_type, market, snapshot_key):
            return False
        save_frame = frame.copy().head(20)
        if "current_price" not in save_frame.columns and "ref_price" in save_frame.columns:
            save_frame["current_price"] = save_frame["ref_price"]
        if "entry_price" not in save_frame.columns and "buy_now_limit" in save_frame.columns:
            save_frame["entry_price"] = save_frame["buy_now_limit"]
        append_scan_history(scan_type, market, save_frame)
        get_learning_state.clear()
        get_tracking_state.clear()
        return True
    except Exception:
        return False


def _record_collection_result(
    rows: list[dict[str, object]],
    *,
    scan_type: str,
    market: str,
    frame: pd.DataFrame,
    saved: bool,
) -> None:
    rows.append(
        {
            "scan_type": scan_type,
            "market": market,
            "candidates": int(len(frame)) if frame is not None else 0,
            "saved": "저장" if saved else ("후보없음" if frame is None or frame.empty else "이미저장/스킵"),
        }
    )


def _load_full_collection_state() -> dict[str, object]:
    try:
        if FULL_COLLECTION_STATE_FILE.exists():
            data = json.loads(FULL_COLLECTION_STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {"US": 0, "KR": 0, "last_run_at": "", "completed": False}


def _save_full_collection_state(state: dict[str, object]) -> None:
    try:
        FULL_COLLECTION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        FULL_COLLECTION_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _reset_full_collection_state() -> None:
    _save_full_collection_state({"US": 0, "KR": 0, "last_run_at": "", "completed": False})


def _collect_full_market_batch(batch_size: int = 120) -> pd.DataFrame:
    state = _load_full_collection_state()
    rows: list[dict[str, object]] = []
    learning_adjustments, _, event_adjustments, news_adjustments, _ = get_learning_state()
    _, _, _, _, pattern_stats = get_tracking_state()
    pattern_lookup = _build_pattern_lookup(pattern_stats)
    min_score = int(st.session_state.scanner_settings["min_score"])
    size = max(20, int(batch_size))

    for market in ["US", "KR"]:
        universe = get_market_sweep_universe(market)
        start = int(state.get(market, 0) or 0)
        end = min(len(universe), start + size)
        chunk = universe[start:end]
        fresh_count = 0
        missing_count = 0
        for item in chunk:
            ticker = str(item.get("ticker", "") or "").strip().upper()
            if not ticker or not is_tradable_ticker(ticker):
                continue
            try:
                data = get_stock_data(ticker)
            except Exception:
                data = pd.DataFrame()
            if data.empty or not is_recent_price_data(data, max_age_days=3):
                missing_count += 1
            else:
                fresh_count += 1

        scan = scan_market(
            market,
            chunk,
            min_score=min_score,
            learning_adjustments=learning_adjustments,
            event_adjustments=event_adjustments,
            news_adjustments=news_adjustments,
        )
        scan = _enrich_recommendation_frame(scan, scan_type="full_today_scan", market=market, pattern_lookup=pattern_lookup)
        saved_today = _remember_recommendation_snapshot("full_today_scan", market, scan)
        long_term = scan[pd.to_numeric(scan.get("score", 0), errors="coerce").fillna(0) >= max(68, min_score)].copy()
        if not long_term.empty:
            long_term["setup"] = long_term.get("setup", "장기후보")
        saved_long = _remember_recommendation_snapshot("full_long_term", market, long_term)

        rows.append(
            {
                "scan_type": "full_price_cache",
                "market": market,
                "range": f"{start + 1}-{end}/{len(universe)}" if universe else "0/0",
                "candidates": len(chunk),
                "fresh_prices": fresh_count,
                "missing_prices": missing_count,
                "saved": "완료" if end >= len(universe) else "진행중",
            }
        )
        rows.append(
            {
                "scan_type": "full_today_scan",
                "market": market,
                "range": f"{start + 1}-{end}/{len(universe)}" if universe else "0/0",
                "candidates": len(scan),
                "fresh_prices": fresh_count,
                "missing_prices": missing_count,
                "saved": "저장" if saved_today else ("후보없음" if scan.empty else "이미저장/스킵"),
            }
        )
        rows.append(
            {
                "scan_type": "full_long_term",
                "market": market,
                "range": f"{start + 1}-{end}/{len(universe)}" if universe else "0/0",
                "candidates": len(long_term),
                "fresh_prices": fresh_count,
                "missing_prices": missing_count,
                "saved": "저장" if saved_long else ("후보없음" if long_term.empty else "이미저장/스킵"),
            }
        )
        state[market] = 0 if end >= len(universe) else end

    state["last_run_at"] = pd.Timestamp.now(tz="Asia/Seoul").isoformat(timespec="seconds")
    state["completed"] = bool(int(state.get("US", 0) or 0) == 0 and int(state.get("KR", 0) or 0) == 0)
    _save_full_collection_state(state)
    get_learning_state.clear()
    get_tracking_state.clear()
    return pd.DataFrame(rows)


def _collect_daily_recommendation_snapshots() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    learning_adjustments, _, event_adjustments, news_adjustments, _ = get_learning_state()
    _, _, _, _, pattern_stats = get_tracking_state()
    pattern_lookup = _build_pattern_lookup(pattern_stats)
    interval = str(st.session_state.realtime_settings["interval"])
    min_score = int(st.session_state.scanner_settings["min_score"])
    realtime_min_score = int(st.session_state.realtime_settings["min_score"])
    top_n = min(5, int(st.session_state.scanner_settings["top_n"]))
    today_limit = min(10, int(st.session_state.scanner_settings["scan_limit"]))
    realtime_limit = min(8, int(st.session_state.scanner_settings["scan_limit"]))
    trade_limit = min(16, int(st.session_state.scanner_settings["market_sweep_limit"]))

    get_today_scan_state.clear()
    get_intraday_stock_data.clear()
    get_latest_quote.clear()

    today_frames: dict[str, pd.DataFrame] = {}
    for market in ["US", "KR"]:
        today = get_today_scan_state(
            market=market,
            watchlist_key=_watchlist_cache_key(market),
            min_score=min_score,
            scan_limit=today_limit,
        )
        today_frames[market] = today
        saved = _remember_recommendation_snapshot("today_scan", market, today)
        _record_collection_result(rows, scan_type="today_scan", market=market, frame=today, saved=saved)

        budget_frame = today.head(top_n).copy()
        if not budget_frame.empty:
            budget_frame["budget_source"] = "한방수집: 오늘추천 기반"
        saved = _remember_recommendation_snapshot("budget_plan", market, budget_frame)
        _record_collection_result(rows, scan_type="budget_plan", market=market, frame=budget_frame, saved=saved)

    for market in ["US", "KR"]:
        universe = st.session_state.watchlists[market][:realtime_limit]
        realtime = scan_intraday_market(
            market,
            universe,
            interval=interval,
            min_score=realtime_min_score,
            force_refresh=True,
            learning_adjustments=learning_adjustments,
            event_adjustments=event_adjustments,
            news_adjustments=news_adjustments,
        )
        realtime = _enrich_recommendation_frame(realtime, scan_type="realtime_scan", market=market, pattern_lookup=pattern_lookup)
        realtime = _attach_latest_quotes(realtime, default_market=market, force_refresh=True)
        saved = _remember_recommendation_snapshot("realtime_scan", market, realtime)
        _record_collection_result(rows, scan_type="realtime_scan", market=market, frame=realtime, saved=saved)

    for market in ["US", "KR"]:
        trade_universe = get_market_sweep_universe(market)[:trade_limit]
        short_term = build_short_term_trade_candidates(
            market,
            top_n=top_n,
            interval=interval,
            min_score=max(60, realtime_min_score),
            universe=trade_universe,
            scan_limit=trade_limit,
            learning_adjustments=learning_adjustments,
            event_adjustments=event_adjustments,
            news_adjustments=news_adjustments,
        )
        short_term = _enrich_recommendation_frame(short_term, scan_type="short_term_trade", market=market, pattern_lookup=pattern_lookup)
        short_term = _attach_latest_quotes(short_term, default_market=market)
        saved = _remember_recommendation_snapshot("short_term_trade", market, short_term)
        _record_collection_result(rows, scan_type="short_term_trade", market=market, frame=short_term, saved=saved)

        high_risk = build_high_risk_trade_candidates(
            market,
            top_n=top_n,
            interval=interval,
            min_score=60,
            universe=trade_universe,
            scan_limit=trade_limit,
            learning_adjustments=learning_adjustments,
            event_adjustments=event_adjustments,
            news_adjustments=news_adjustments,
        )
        high_risk = _enrich_recommendation_frame(high_risk, scan_type="high_risk_trade", market=market, pattern_lookup=pattern_lookup)
        high_risk = _attach_latest_quotes(high_risk, default_market=market)
        saved = _remember_recommendation_snapshot("high_risk_trade", market, high_risk)
        _record_collection_result(rows, scan_type="high_risk_trade", market=market, frame=high_risk, saved=saved)

    get_learning_state.clear()
    get_tracking_state.clear()
    return pd.DataFrame(rows)


def _serialize_top_budget_actions(combined_actions: pd.DataFrame, limit: int = 5) -> str:
    if combined_actions.empty:
        return ""
    parts: list[str] = []
    for _, row in combined_actions.head(limit).iterrows():
        parts.append(
            f"{row.get('market', '')} {row.get('ticker', '')} {row.get('bucket', '')} "
            f"{float(pd.to_numeric(row.get('planned_amount', 0), errors='coerce') or 0):,.0f}"
        )
    return " | ".join(parts)


def _build_briefing_action_tracking(actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, row in actions.iterrows():
        ticker = str(row.get("ticker", "") or "").strip().upper()
        market = str(row.get("market", "") or "").strip().upper()
        if not ticker:
            continue

        quote = get_latest_quote(ticker)
        current_price = pd.to_numeric(quote.get("current_price", None), errors="coerce")
        ref_price = pd.to_numeric(row.get("ref_price", None), errors="coerce")
        current_return = None
        if not pd.isna(current_price) and not pd.isna(ref_price) and float(ref_price) > 0:
            current_return = (float(current_price) / float(ref_price) - 1) * 100

        status = "관찰중"
        if current_return is not None:
            if current_return >= 5:
                status = "좋은 흐름"
            elif current_return <= -4:
                status = "주의"
            elif current_return >= 1:
                status = "천천히 우위"

        rows.append(
            {
                "briefing_date": row.get("briefing_date", ""),
                "saved_at": row.get("saved_at", ""),
                "market": market,
                "bucket": row.get("bucket", ""),
                "ticker": ticker,
                "name": row.get("name", ""),
                "setup": row.get("setup", ""),
                "score": row.get("score", None),
                "planned_amount": row.get("planned_amount", None),
                "ref_price": None if pd.isna(ref_price) else round(float(ref_price), 2),
                "current_price": None if pd.isna(current_price) else round(float(current_price), 2),
                "current_return_pct": None if current_return is None else round(float(current_return), 2),
                "status": status,
                "quote_as_of": str(quote.get("as_of", "")),
                "reason": row.get("reason", ""),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        by=["briefing_date", "current_return_pct"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def _build_selected_budget_actions(
    *,
    market: str,
    plan_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    selection_map: dict[str, str],
) -> pd.DataFrame:
    if plan_df.empty or detail_df.empty or not selection_map:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, plan_row in plan_df.iterrows():
        bucket = str(plan_row.get("bucket", "") or "")
        ticker = str(selection_map.get(bucket, "") or "").strip().upper()
        if not ticker:
            continue

        match = detail_df[
            (detail_df["bucket"].astype(str) == bucket)
            & (detail_df["ticker"].astype(str).str.upper() == ticker)
        ]
        if match.empty:
            continue

        picked = match.iloc[0]
        rows.append(
            {
                "market": market,
                "bucket": bucket,
                "ticker": picked.get("ticker", ""),
                "name": picked.get("name", ""),
                "setup": picked.get("setup", ""),
                "score": picked.get("score", None),
                "planned_amount": plan_row.get("planned_amount", None),
                "executable_amount": plan_row.get("executable_amount", None),
                "reserve_amount": plan_row.get("reserve_amount", None),
                "split_1": plan_row.get("split_1", None),
                "split_2": plan_row.get("split_2", None),
                "split_3": plan_row.get("split_3", None),
                "ref_price": picked.get("ref_price", None),
                "estimated_units": picked.get("estimated_units", None),
                "buy_now_limit": picked.get("buy_now_limit", None),
                "add_on_pullback": picked.get("add_on_pullback", None),
                "stop_loss": picked.get("stop_loss", None),
                "reentry_above": picked.get("reentry_above", None),
                "target_1": picked.get("target_1", None),
                "target_2": picked.get("target_2", None),
                "horizon": plan_row.get("horizon", ""),
                "execution_rule": picked.get("execution_rule", plan_row.get("execution_rule", "")),
                "execution_status": picked.get("execution_status", ""),
                "confidence_view": picked.get("confidence_view", ""),
                "allocation_reason": picked.get("allocation_reason", ""),
                "quality_note": picked.get("quality_note", ""),
                "note": plan_row.get("note", ""),
                "reason": picked.get("reason", ""),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _budget_action_option_label(row: pd.Series) -> str:
    score = pd.to_numeric(row.get("score", None), errors="coerce")
    score_text = f"{int(round(float(score)))}점" if not pd.isna(score) else "점수없음"
    amount = pd.to_numeric(row.get("planned_amount", 0), errors="coerce")
    amount_text = f"{float(amount):,.0f}" if not pd.isna(amount) else "0"
    return (
        f"{row.get('market', '')} {row.get('ticker', '')} | {row.get('bucket', '')} | "
        f"{row.get('execution_status', '상태확인')} | {score_text} | {amount_text}"
    )


def _append_budget_action_decision(row: pd.Series, decision: str, memo: str) -> None:
    payload = {
        "decision": decision,
        "market": row.get("market", ""),
        "bucket": row.get("bucket", ""),
        "ticker": row.get("ticker", ""),
        "name": row.get("name", ""),
        "setup": row.get("setup", ""),
        "score": row.get("score", ""),
        "planned_amount": row.get("planned_amount", ""),
        "execution_status": row.get("execution_status", ""),
        "confidence_view": row.get("confidence_view", ""),
        "ref_price": row.get("ref_price", ""),
        "buy_now_limit": row.get("buy_now_limit", ""),
        "stop_loss": row.get("stop_loss", ""),
        "target_1": row.get("target_1", ""),
        "source": "예산 플래너",
        "memo": memo,
    }
    append_decision_log(payload)
    if decision == "관심추가":
        append_manual_tracking(
            {
                "market": payload["market"],
                "ticker": payload["ticker"],
                "name": payload["name"],
                "source": "예산 플래너 결정",
                "setup": payload["setup"],
                "score": payload["score"],
                "current_price": payload["ref_price"],
                "entry_price": payload["buy_now_limit"],
                "stop_loss": payload["stop_loss"],
                "target_1": payload["target_1"],
                "memo": memo or "예산 플래너에서 관심추가",
            }
        )


def _render_budget_decision_panel(combined_actions: pd.DataFrame) -> None:
    if combined_actions.empty:
        return

    st.markdown("#### 내 결정 기록")
    option_map = {
        _budget_action_option_label(row): idx
        for idx, row in combined_actions.head(10).iterrows()
    }
    if not option_map:
        return

    pick_col, decision_col = st.columns([2.2, 1])
    with pick_col:
        picked_label = st.selectbox("결정할 실행안", options=list(option_map.keys()), key="budget_decision_pick")
    with decision_col:
        decision = st.selectbox("내 결정", ["매수함", "보류", "관심추가", "제외"], key="budget_decision_value")
    memo = st.text_input("결정 메모", value="", placeholder="예: 진입가 근처까지 대기, 절반만 실행", key="budget_decision_memo")
    if st.button("이 결정 저장", width="stretch", key="budget_decision_save"):
        row = combined_actions.loc[option_map[picked_label]]
        _append_budget_action_decision(row, decision, memo)
        st.success(f"{row.get('market', '')} {row.get('ticker', '')} 결정을 저장했습니다.")

    decisions = load_decision_log()
    if not decisions.empty:
        recent = decisions.sort_values(by=["created_at"], ascending=False).head(12).copy()
        with st.expander("최근 결정 기록", expanded=False):
            _show_table(
                recent[
                    [
                        "created_at",
                        "decision",
                        "market",
                        "bucket",
                        "ticker",
                        "name",
                        "score",
                        "planned_amount",
                        "execution_status",
                        "confidence_view",
                        "memo",
                    ]
                ],
                plain_numeric_columns=["planned_amount"],
                datetime_columns=["created_at"],
                column_config={
                    "created_at": st.column_config.TextColumn("기록시각", width="small"),
                    "decision": st.column_config.TextColumn("결정", width="small"),
                    "market": st.column_config.TextColumn("시장", width="small"),
                    "bucket": st.column_config.TextColumn("유형", width="small"),
                    "ticker": st.column_config.TextColumn("티커", width="small"),
                    "name": st.column_config.TextColumn("종목명", width="small"),
                    "score": st.column_config.NumberColumn("점수", format="%d", width="small"),
                    "planned_amount": st.column_config.TextColumn("배정금액", width="small"),
                    "execution_status": st.column_config.TextColumn("집행상태", width="small"),
                    "confidence_view": st.column_config.TextColumn("신뢰도", width="small"),
                    "memo": st.column_config.TextColumn("메모", width="large"),
                },
            )


def _decision_log_summary(limit: int = 80) -> tuple[pd.DataFrame, dict[str, object]]:
    decisions = load_decision_log()
    if decisions.empty:
        return decisions, {
            "total": 0,
            "buy_count": 0,
            "hold_count": 0,
            "watch_count": 0,
            "skip_count": 0,
            "buy_amount": 0.0,
            "latest": "",
        }

    recent = decisions.sort_values(by=["created_at"], ascending=False).head(limit).copy()
    decision_series = recent.get("decision", pd.Series(dtype=str)).astype(str)
    amount_series = pd.to_numeric(recent.get("planned_amount", pd.Series(dtype=float)), errors="coerce").fillna(0)
    buy_mask = decision_series == "매수함"
    summary = {
        "total": int(len(recent)),
        "buy_count": int(buy_mask.sum()),
        "hold_count": int((decision_series == "보류").sum()),
        "watch_count": int((decision_series == "관심추가").sum()),
        "skip_count": int((decision_series == "제외").sum()),
        "buy_amount": float(amount_series[buy_mask].sum()),
        "latest": str(recent.iloc[0].get("created_at", "") or "") if not recent.empty else "",
    }
    return recent, summary


def _decision_log_note(summary: dict[str, object]) -> str:
    total = int(summary.get("total", 0) or 0)
    if total <= 0:
        return "아직 저장된 내 결정이 없습니다. 예산 플래너에서 매수/보류/관심추가를 남기면 다음 판단이 더 선명해집니다."
    buy_count = int(summary.get("buy_count", 0) or 0)
    hold_count = int(summary.get("hold_count", 0) or 0)
    watch_count = int(summary.get("watch_count", 0) or 0)
    if buy_count >= max(3, hold_count + watch_count):
        return "최근 결정은 실행 쪽으로 기울었습니다. 새 플랜은 진입가와 현금 보류를 더 엄격히 확인하세요."
    if hold_count + watch_count > buy_count:
        return "최근 결정은 관망/추적이 많습니다. 오늘은 신뢰도 높은 후보만 좁혀 보는 흐름이 좋습니다."
    return "최근 결정은 균형적입니다. 점수보다 집행상태와 가격 조건을 같이 보세요."


def _render_decision_log_overview(*, compact: bool = False) -> None:
    recent, summary = _decision_log_summary(limit=60)
    if compact:
        st.caption(_decision_log_note(summary))
    else:
        st.markdown("#### 최근 내 결정 흐름")
        st.info(_decision_log_note(summary))

    cols = st.columns(4)
    cols[0].metric("매수함", f"{int(summary['buy_count'])}")
    cols[1].metric("보류", f"{int(summary['hold_count'])}")
    cols[2].metric("관심추가", f"{int(summary['watch_count'])}")
    cols[3].metric("매수 배정액", f"{float(summary['buy_amount']):,.0f}")

    if recent.empty:
        return
    with st.expander("최근 결정 상세", expanded=not compact):
        view = recent.head(15).copy()
        _show_table(
            view[
                [
                    "created_at",
                    "decision",
                    "market",
                    "bucket",
                    "ticker",
                    "name",
                    "score",
                    "planned_amount",
                    "execution_status",
                    "confidence_view",
                    "memo",
                ]
            ],
            plain_numeric_columns=["planned_amount"],
            datetime_columns=["created_at"],
            column_config={
                "created_at": st.column_config.TextColumn("기록시각", width="small"),
                "decision": st.column_config.TextColumn("결정", width="small"),
                "market": st.column_config.TextColumn("시장", width="small"),
                "bucket": st.column_config.TextColumn("유형", width="small"),
                "ticker": st.column_config.TextColumn("티커", width="small"),
                "name": st.column_config.TextColumn("종목명", width="small"),
                "score": st.column_config.NumberColumn("점수", format="%d", width="small"),
                "planned_amount": st.column_config.TextColumn("배정금액", width="small"),
                "execution_status": st.column_config.TextColumn("집행상태", width="small"),
                "confidence_view": st.column_config.TextColumn("신뢰도", width="small"),
                "memo": st.column_config.TextColumn("메모", width="large"),
            },
        )


def _build_manual_tracking_pool() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    learning_adjustments, _, event_adjustments, news_adjustments, _ = get_learning_state()
    _, _, _, _, pattern_stats = get_tracking_state()
    pattern_lookup = _build_pattern_lookup(pattern_stats)
    min_score = int(st.session_state.scanner_settings["min_score"])
    interval = str(st.session_state.realtime_settings["interval"])
    realtime_min_score = int(st.session_state.realtime_settings["min_score"])
    top_n = int(st.session_state.scanner_settings["top_n"])

    sources: list[tuple[str, str, pd.DataFrame]] = [
        (
            "오늘추천",
            "today_scan",
            pd.concat(
                [
                    _enrich_recommendation_frame(
                        scan_market("US", st.session_state.watchlists["US"], min_score=min_score, learning_adjustments=learning_adjustments, event_adjustments=event_adjustments, news_adjustments=news_adjustments),
                        scan_type="today_scan",
                        market="US",
                        pattern_lookup=pattern_lookup,
                    ).assign(market="US"),
                    _enrich_recommendation_frame(
                        scan_market("KR", st.session_state.watchlists["KR"], min_score=min_score, learning_adjustments=learning_adjustments, event_adjustments=event_adjustments, news_adjustments=news_adjustments),
                        scan_type="today_scan",
                        market="KR",
                        pattern_lookup=pattern_lookup,
                    ).assign(market="KR"),
                ],
                ignore_index=True,
            ),
        ),
        (
            "실시간",
            "realtime_scan",
            pd.concat(
                [
                    _enrich_recommendation_frame(
                        scan_intraday_market(
                            "US",
                            st.session_state.watchlists["US"],
                            interval=interval,
                            min_score=realtime_min_score,
                            learning_adjustments=learning_adjustments,
                            event_adjustments=event_adjustments,
                            news_adjustments=news_adjustments,
                        ),
                        scan_type="realtime_scan",
                        market="US",
                        pattern_lookup=pattern_lookup,
                    ).assign(market="US"),
                    _enrich_recommendation_frame(
                        scan_intraday_market(
                            "KR",
                            st.session_state.watchlists["KR"],
                            interval=interval,
                            min_score=realtime_min_score,
                            learning_adjustments=learning_adjustments,
                            event_adjustments=event_adjustments,
                            news_adjustments=news_adjustments,
                        ),
                        scan_type="realtime_scan",
                        market="KR",
                        pattern_lookup=pattern_lookup,
                    ).assign(market="KR"),
                ],
                ignore_index=True,
            ),
        ),
        (
            "일반단타",
            "short_term_trade",
            pd.concat(
                [
                    _enrich_recommendation_frame(
                        build_short_term_trade_candidates("US", top_n=top_n, interval=interval, min_score=max(60, realtime_min_score), learning_adjustments=learning_adjustments, event_adjustments=event_adjustments, news_adjustments=news_adjustments),
                        scan_type="short_term_trade",
                        market="US",
                        pattern_lookup=pattern_lookup,
                    ).assign(market="US"),
                    _enrich_recommendation_frame(
                        build_short_term_trade_candidates("KR", top_n=top_n, interval=interval, min_score=max(60, realtime_min_score), learning_adjustments=learning_adjustments, event_adjustments=event_adjustments, news_adjustments=news_adjustments),
                        scan_type="short_term_trade",
                        market="KR",
                        pattern_lookup=pattern_lookup,
                    ).assign(market="KR"),
                ],
                ignore_index=True,
            ),
        ),
        (
            "고위험단타",
            "high_risk_trade",
            pd.concat(
                [
                    _enrich_recommendation_frame(
                        build_high_risk_trade_candidates("US", top_n=top_n, interval=interval, min_score=60, learning_adjustments=learning_adjustments, event_adjustments=event_adjustments, news_adjustments=news_adjustments),
                        scan_type="high_risk_trade",
                        market="US",
                        pattern_lookup=pattern_lookup,
                    ).assign(market="US"),
                    _enrich_recommendation_frame(
                        build_high_risk_trade_candidates("KR", top_n=top_n, interval=interval, min_score=60, learning_adjustments=learning_adjustments, event_adjustments=event_adjustments, news_adjustments=news_adjustments),
                        scan_type="high_risk_trade",
                        market="KR",
                        pattern_lookup=pattern_lookup,
                    ).assign(market="KR"),
                ],
                ignore_index=True,
            ),
        ),
    ]

    for source_name, scan_type, frame in sources:
        if frame.empty:
            continue
        normalized = frame.copy()
        normalized["source"] = source_name
        normalized["scan_type"] = scan_type
        for column in ["current_price", "entry_price", "stop_loss", "target_1", "recent_hit_rate_20d", "recent_target_rate_5d", "score_view"]:
            if column not in normalized.columns:
                normalized[column] = ""
        frames.append(
            normalized[
                [
                    "market",
                    "ticker",
                    "name",
                    "source",
                    "scan_type",
                    "setup",
                    "score",
                    "score_view",
                    "current_price",
                    "entry_price",
                    "stop_loss",
                    "target_1",
                    "recent_hit_rate_20d",
                    "recent_target_rate_5d",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["market", "ticker", "source"], keep="first").reset_index(drop=True)


def _render_manual_tracking_quick_add(
    frame: pd.DataFrame,
    *,
    source_label: str,
    key_prefix: str,
    default_market: str | None = None,
) -> None:
    if frame.empty:
        return

    working = frame.copy()
    if "market" not in working.columns and default_market is not None:
        working["market"] = default_market
    if "source" not in working.columns:
        working["source"] = source_label

    labels = {
        f"{row.get('market', default_market or '')} | {row.get('ticker', '')} | {row.get('name', '')}": row
        for _, row in working.iterrows()
    }
    if not labels:
        return

    st.caption("이 표에서 바로 관심 추적에 추가할 수 있습니다.")
    selected_label = st.selectbox(
        "관심 추적 추가",
        options=list(labels.keys()),
        key=f"{key_prefix}_tracking_select",
        label_visibility="collapsed",
    )
    memo = st.text_input(
        "메모",
        key=f"{key_prefix}_tracking_memo",
        placeholder="왜 보는 종목인지 짧게 남겨도 됩니다.",
        label_visibility="collapsed",
    )
    if st.button("이 종목 관심 추적 추가", key=f"{key_prefix}_tracking_button", width="stretch"):
        selected = dict(labels[selected_label])
        selected["market"] = selected.get("market", default_market or "")
        selected["source"] = selected.get("source", source_label)
        selected["memo"] = memo.strip()
        append_manual_tracking(selected)
        append_scan_history("manual_track", str(selected["market"]), pd.DataFrame([selected]))
        st.success(f"{selected['ticker']}를 관심 추적에 추가했습니다.")


def _manual_alert_status(current_price: float | None, stop_loss: float | None, target_1: float | None) -> str:
    if current_price is None or pd.isna(current_price):
        return "시세대기"
    if stop_loss is not None and not pd.isna(stop_loss):
        if current_price <= stop_loss:
            return "손절구간"
        if current_price <= stop_loss * 1.02:
            return "손절근접"
    if target_1 is not None and not pd.isna(target_1):
        if current_price >= target_1:
            return "목표도달"
        if current_price >= target_1 * 0.98:
            return "목표근접"
    return "추적중"


def render_portfolio_editor() -> None:
    st.subheader("보유 종목 입력")
    st.caption("평단/수량 입력 후 판단합니다.")

    upload = st.file_uploader("CSV로 보유 종목 업로드", type=["csv"], key="portfolio_csv_uploader")
    if upload is not None:
        try:
            imported = pd.read_csv(upload)
        except UnicodeDecodeError:
            upload.seek(0)
            imported = pd.read_csv(upload, encoding="cp949")
        except Exception as error:
            st.error(f"CSV를 읽는 중 문제가 생겼습니다: {error}")
            imported = None

        if imported is not None:
            normalized = normalize_portfolio_frame(imported)
            st.caption("업로드 미리보기")
            st.dataframe(normalized, width="stretch", hide_index=True)

            left, right = st.columns([1, 1])
            with left:
                if st.button("CSV 가져오기", width="stretch"):
                    if normalized.empty:
                        st.warning("가져올 수 있는 행이 없습니다. 컬럼명을 확인해 주세요.")
                    else:
                        st.session_state.portfolio = normalized
                        st.success(f"{len(normalized)}개 종목을 불러왔습니다.")
                        st.rerun()
            with right:
                st.caption("지원 예: 시장/종목코드/종목명/보유수량/평균단가")

    edited = st.data_editor(
        st.session_state.portfolio,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "market": st.column_config.SelectboxColumn("시장", options=["US", "KR"], required=True),
            "ticker": st.column_config.TextColumn("티커", required=True),
            "name": st.column_config.TextColumn("종목명"),
            "quantity": st.column_config.NumberColumn("보유수량", min_value=0.0, step=1.0),
            "avg_price": st.column_config.NumberColumn("평균단가", min_value=0.0, step=0.01),
            "cash_budget": st.column_config.NumberColumn("추가매수 가능금액", min_value=0.0, step=0.01),
            "target_weight": st.column_config.NumberColumn("목표비중", min_value=0.0, max_value=1.0, step=0.01),
        },
        key="portfolio_editor",
    )
    st.session_state.portfolio = edited

    sample_csv = pd.DataFrame(
        [
            {
                "시장": "US",
                "종목코드": "AAPL",
                "종목명": "Apple",
                "보유수량": 10,
                "평균단가": 180,
                "추가매수금액": 1000,
                "목표비중": 0.15,
            },
            {
                "시장": "KR",
                "종목코드": "005930.KS",
                "종목명": "Samsung Electronics",
                "보유수량": 15,
                "평균단가": 71000,
                "추가매수금액": 500000,
                "목표비중": 0.2,
            },
        ]
    )
    st.download_button(
        "CSV 샘플 다운로드",
        data=sample_csv.to_csv(index=False).encode("utf-8-sig"),
        file_name="portfolio_sample.csv",
        mime="text/csv",
        width="stretch",
    )

    left, right = st.columns([1, 1])
    with left:
        if st.button("보유 종목 저장", width="stretch"):
            save_portfolio(st.session_state.portfolio)
            st.success("보유 종목을 로컬 파일에 저장했습니다.")
    with right:
        if st.button("보유 종목 다시 불러오기", width="stretch"):
            loaded = load_portfolio()
            if loaded is None:
                st.warning("저장된 보유 종목 파일이 아직 없습니다.")
            else:
                st.session_state.portfolio = loaded
                st.rerun()


def render_selected_analysis(ticker: str) -> None:
    st.subheader(f"{ticker} 분석")
    data = get_stock_data(ticker)

    if data.empty:
        st.error("데이터를 불러오지 못했습니다. 티커 형식과 네트워크 상태를 확인해 주세요.")
        return

    latest = data.iloc[-1]
    recommendation = analyze_position(
        data=data,
        position=PositionInput(
            ticker=ticker,
            quantity=0,
            avg_price=0,
            cash_budget=0,
            target_weight=0,
        ),
    )
    market = "KR" if ticker.endswith((".KS", ".KQ")) else "US"
    stop_candidate = pd.to_numeric(latest.get("ma60", None), errors="coerce")
    stop_loss = float(stop_candidate) if not pd.isna(stop_candidate) else float(latest["Close"]) * 0.92
    target_1 = float(latest["Close"]) + max(0.0, float(latest["Close"]) - stop_loss) * 1.6
    _render_action_deck(
        pd.DataFrame(
            [
                {
                    "market": market,
                    "ticker": ticker,
                    "name": "",
                    "bucket": "종목분석",
                    "setup": recommendation.action,
                    "action": recommendation.action,
                    "score": recommendation.score,
                    "current_price": float(latest["Close"]),
                    "entry_price": float(latest["Close"]),
                    "stop_loss": stop_loss,
                    "target_1": target_1,
                    "reason": " / ".join(recommendation.reasons[:3]),
                }
            ]
        ),
        title="종목 판단 카드",
        limit=1,
    )

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(build_price_chart(data), width="stretch")
    with right:
        st.metric("현재가", _format_price(latest["Close"], market))
        st.metric("RSI", f"{latest['rsi']:.1f}")
        st.metric("MACD", f"{latest['macd']:.2f}")
        st.metric("액션", recommendation.action)
        st.metric("제안 매수 비중", f"{recommendation.suggested_buy_pct}%")
        st.metric("제안 매도 비중", f"{recommendation.suggested_sell_pct}%")
        st.write("사유")
        for reason in recommendation.reasons:
            st.write(f"- {reason}")

    st.dataframe(data.tail(20), width="stretch")


def _build_trade_plan(
    *,
    market: str,
    row: dict[str, object],
    data: pd.DataFrame,
    result,
    current_price: float,
    change_pct: object,
    quote_as_of: str,
) -> dict[str, object]:
    latest = data.iloc[-1]
    avg_price = float(row.get("avg_price", 0) or 0)
    quantity = float(row.get("quantity", 0) or 0)
    atr14 = float(pd.to_numeric(latest.get("atr14", 0), errors="coerce") or 0)
    ma20 = float(pd.to_numeric(latest.get("ma20", current_price), errors="coerce") or current_price)
    ma60 = float(pd.to_numeric(latest.get("ma60", current_price), errors="coerce") or current_price)
    rsi = float(pd.to_numeric(latest.get("rsi", 50), errors="coerce") or 50)
    atr_floor = atr14 if atr14 > 0 else current_price * 0.025
    return_pct = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else None

    if result.score >= 75:
        operation = "보유 우선"
        add_below = min(ma20, current_price - atr_floor * 0.6)
        reduce_below = ma60
        stop_loss = max(ma60 - atr_floor * 0.7, avg_price * 0.92 if avg_price > 0 else current_price * 0.92)
        reentry_above = max(ma20, current_price + atr_floor * 0.5)
        today_rule = "눌림에서만 소액 추가"
    elif result.score >= 55:
        operation = "보유/확인"
        add_below = min(ma20, current_price - atr_floor * 0.8) if rsi < 65 else None
        reduce_below = ma60
        stop_loss = max(ma60 - atr_floor, avg_price * 0.9 if avg_price > 0 else current_price * 0.9)
        reentry_above = max(ma20, current_price + atr_floor * 0.7)
        today_rule = "큰 추가매수 보류"
    elif result.score >= 40:
        operation = "관망/감량 준비"
        add_below = None
        reduce_below = min(ma60, current_price - atr_floor * 0.6)
        stop_loss = max(ma60 - atr_floor, avg_price * 0.9 if avg_price > 0 else current_price * 0.9)
        reentry_above = max(ma20, current_price + atr_floor)
        today_rule = "추가매수 금지"
    else:
        operation = "리스크 축소"
        add_below = None
        reduce_below = current_price
        stop_loss = max(current_price - atr_floor, avg_price * 0.88 if avg_price > 0 else current_price * 0.88)
        reentry_above = max(ma20, ma60, current_price + atr_floor * 1.2)
        today_rule = "방어 우선"

    if stop_loss >= current_price:
        stop_loss = current_price - atr_floor * 0.8
    if reduce_below and reduce_below >= current_price:
        reduce_below = current_price - atr_floor * 0.4

    if return_pct is not None and return_pct >= 20:
        take_profit_1 = max(current_price + atr_floor * 1.2, current_price * 1.04)
        take_profit_2 = max(current_price + atr_floor * 2.4, current_price * 1.08)
        if result.suggested_sell_pct < 20:
            today_rule = "목표가 근처 일부 익절"
    else:
        take_profit_1 = max(current_price + atr_floor * 1.8, current_price * 1.06)
        take_profit_2 = max(current_price + atr_floor * 3.0, current_price * 1.10)

    if quantity <= 0:
        today_rule = "신규 후보로 관찰"

    return {
        "market": market,
        "ticker": str(row.get("ticker", "")),
        "name": row.get("name", ""),
        "operation": operation,
        "action": result.action,
        "score": result.score,
        "current_price": round(current_price, 2),
        "change_pct": change_pct,
        "return_pct": round(return_pct, 2) if return_pct is not None else None,
        "buy_pct": result.suggested_buy_pct,
        "sell_pct": result.suggested_sell_pct,
        "add_below": round(add_below, 2) if add_below and add_below > 0 else None,
        "reduce_below": round(reduce_below, 2) if reduce_below and reduce_below > 0 else None,
        "stop_loss": round(stop_loss, 2) if stop_loss and stop_loss > 0 else None,
        "reentry_above": round(reentry_above, 2) if reentry_above and reentry_above > 0 else None,
        "take_profit_1": round(take_profit_1, 2) if take_profit_1 > 0 else None,
        "take_profit_2": round(take_profit_2, 2) if take_profit_2 > 0 else None,
        "today_rule": today_rule,
        "quote_as_of": quote_as_of,
        "reason": " / ".join(result.reasons[:3]),
    }


def render_portfolio_analysis() -> None:
    st.subheader("오늘 매매 운영표")
    st.caption("오늘 가격 기준만 봅니다. 추가, 감량, 손절, 재진입.")

    rows: list[dict[str, object]] = []
    for row in st.session_state.portfolio.to_dict("records"):
        market = str(row.get("market", "")).strip().upper()
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue

        data = get_stock_data(ticker)
        if data.empty:
            rows.append(
                {
                    "market": market,
                    "ticker": ticker,
                    "name": row.get("name", ""),
                    "operation": "데이터 없음",
                    "action": "데이터 없음",
                    "score": None,
                    "current_price": None,
                    "change_pct": None,
                    "return_pct": None,
                    "quote_as_of": "",
                    "today_rule": "가격 없음",
                    "reason": "데이터 확인 필요",
                }
            )
            continue

        result = analyze_position(
            data=data,
            position=PositionInput(
                ticker=ticker,
                quantity=float(row.get("quantity", 0) or 0),
                avg_price=float(row.get("avg_price", 0) or 0),
                cash_budget=float(row.get("cash_budget", 0) or 0),
                target_weight=float(row.get("target_weight", 0) or 0),
            ),
        )
        quote = get_latest_quote(ticker)
        live_price = pd.to_numeric(quote.get("current_price", None), errors="coerce")
        current_price = float(live_price) if not pd.isna(live_price) else float(result.current_price)
        rows.append(
            _build_trade_plan(
                market=market,
                row={**row, "ticker": ticker},
                data=data,
                result=result,
                current_price=current_price,
                change_pct=pd.to_numeric(quote.get("change_pct", None), errors="coerce"),
                quote_as_of=str(quote.get("as_of", "")),
            )
        )

    if not rows:
        st.info("보유 종목을 하나 이상 입력해 주세요.")
        return

    summary = pd.DataFrame(rows)
    urgent = int(summary["operation"].isin(["리스크 축소", "관망/감량 준비"]).sum()) if "operation" in summary.columns else 0
    add_ready = int((pd.to_numeric(summary.get("buy_pct", 0), errors="coerce").fillna(0) > 0).sum()) if "buy_pct" in summary.columns else 0
    sell_ready = int((pd.to_numeric(summary.get("sell_pct", 0), errors="coerce").fillna(0) > 0).sum()) if "sell_pct" in summary.columns else 0
    a, b, c = st.columns(3)
    a.metric("방어/감량 체크", urgent)
    b.metric("추가매수 후보", add_ready)
    c.metric("일부매도 후보", sell_ready)

    _show_table(
        summary[
            [
                "market",
                "ticker",
                "name",
                "operation",
                "action",
                "score",
                "current_price",
                "change_pct",
                "return_pct",
                "buy_pct",
                "sell_pct",
                "add_below",
                "reduce_below",
                "stop_loss",
                "reentry_above",
                "take_profit_1",
                "take_profit_2",
                "today_rule",
                "quote_as_of",
                "reason",
            ]
        ],
        datetime_columns=["quote_as_of"],
        currency_columns=[
            "current_price",
            "add_below",
            "reduce_below",
            "stop_loss",
            "reentry_above",
            "take_profit_1",
            "take_profit_2",
        ],
        column_config={
            "market": st.column_config.TextColumn("시장", width="small"),
            "ticker": st.column_config.TextColumn("티커", width="small"),
            "name": st.column_config.TextColumn("종목명", width="small"),
            "operation": st.column_config.TextColumn("운영모드", width="small"),
            "action": st.column_config.TextColumn("현재 판단", width="small"),
            "score": st.column_config.NumberColumn("점수", format="%d", width="small"),
            "current_price": st.column_config.TextColumn("현재가", width="small"),
            "change_pct": st.column_config.NumberColumn("당일등락", format="%.2f", width="small"),
            "return_pct": st.column_config.NumberColumn("현재수익률", format="%.2f", width="small"),
            "buy_pct": st.column_config.NumberColumn("추가매수%", format="%d", width="small"),
            "sell_pct": st.column_config.NumberColumn("축소%", format="%d", width="small"),
            "add_below": st.column_config.TextColumn("이하 눌림매수", width="small"),
            "reduce_below": st.column_config.TextColumn("이탈시 감량", width="small"),
            "stop_loss": st.column_config.TextColumn("최종 방어선", width="small"),
            "reentry_above": st.column_config.TextColumn("재진입 회복가", width="small"),
            "take_profit_1": st.column_config.TextColumn("1차 익절", width="small"),
            "take_profit_2": st.column_config.TextColumn("2차 익절", width="small"),
            "today_rule": st.column_config.TextColumn("오늘 운영", width="large"),
            "quote_as_of": st.column_config.TextColumn("시세기준", width="small"),
            "reason": st.column_config.TextColumn("핵심 사유", width="large"),
        },
    )


def render_rebalance_hint() -> None:
    st.subheader("보유 종목 비중 조절 힌트")

    recommendations: list[dict[str, object]] = []
    for row in st.session_state.portfolio.to_dict("records"):
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue

        data = get_stock_data(ticker)
        if data.empty:
            continue

        result = analyze_position(
            data=data,
            position=PositionInput(
                ticker=ticker,
                quantity=float(row.get("quantity", 0) or 0),
                avg_price=float(row.get("avg_price", 0) or 0),
                cash_budget=float(row.get("cash_budget", 0) or 0),
                target_weight=float(row.get("target_weight", 0) or 0),
            ),
        )

        if result.suggested_buy_pct <= 0 and result.suggested_sell_pct <= 0:
            continue

        recommendations.append(
            {
                "ticker": ticker,
                "action": result.action,
                "buy_pct": result.suggested_buy_pct,
                "sell_pct": result.suggested_sell_pct,
                "reason": result.reasons[0],
            }
        )

    if not recommendations:
        st.info("지금은 강한 비중 조절 신호가 없습니다.")
        return

    st.dataframe(pd.DataFrame(recommendations), width="stretch", hide_index=True)


def render_rebalance_suggestions() -> None:
    st.subheader("리밸런싱 제안")
    st.caption("비중, 섹터, 위험 쏠림을 봅니다.")

    analysis, summary, sector_summary, _ = get_portfolio_analysis_state(_portfolio_cache_key(st.session_state.portfolio))
    suggestions = build_rebalance_suggestions(analysis, summary, sector_summary)
    if suggestions.empty:
        st.info("리밸런싱 제안을 만들 데이터가 부족합니다.")
        return

    st.dataframe(suggestions, width="stretch", hide_index=True)


def render_portfolio_insights() -> None:
    st.subheader("포트폴리오 분석")
    st.caption("위험, 성장, 배당, 집중도.")

    analysis, summary, sector_summary, correlation = get_portfolio_analysis_state(_portfolio_cache_key(st.session_state.portfolio))
    if analysis.empty:
        st.info("분석할 보유 종목 데이터가 없습니다.")
        return

    a, b, c, d = st.columns(4)
    a.metric("총 평가금액", f"{summary['total_value']:,.0f}")
    b.metric("미국 비중", f"{summary['us_weight']:.1f}%")
    c.metric("한국 비중", f"{summary['kr_weight']:.1f}%")
    d.metric("최대 집중 비중", f"{summary['top_weight']:.1f}%")

    e, f, g, h = st.columns(4)
    e.metric("평균 배당수익률", f"{summary['avg_dividend_yield']:.2f}%")
    f.metric("평균 변동성", f"{summary['avg_volatility']:.2f}%")
    g.metric("성장 스타일 비중", f"{summary['growth_weight']:.1f}%")
    h.metric("고위험 비중", f"{summary['high_risk_weight']:.1f}%")

    st.markdown("#### 스타일 요약")
    style_summary = pd.DataFrame(
        [
            {"style": "성장", "weight_pct": round(summary["growth_weight"], 2)},
            {"style": "배당", "weight_pct": round(summary["dividend_weight"], 2)},
            {"style": "안정", "weight_pct": round(summary["defensive_weight"], 2)},
        ]
    )
    st.dataframe(style_summary, width="stretch", hide_index=True)

    if not sector_summary.empty:
        st.markdown("#### 섹터 비중")
        st.dataframe(sector_summary, width="stretch", hide_index=True)

    warnings: list[str] = []
    if summary["top_weight"] >= 35:
        warnings.append("한 종목 비중이 35%를 넘어 집중 리스크가 큽니다.")
    if not sector_summary.empty and float(sector_summary.iloc[0]["sector_weight_pct"]) >= 50:
        warnings.append("가장 큰 섹터 비중이 50%를 넘어 섹터 쏠림이 있습니다.")
    if summary["high_risk_weight"] >= 45:
        warnings.append("고위험 종목 비중이 높아 변동성이 크게 나올 수 있습니다.")

    st.markdown("#### 포트폴리오 경고")
    if warnings:
        for item in warnings:
            st.write(f"- {item}")
    else:
        st.write("- 현재 기준으로 과도한 집중 경고는 크지 않습니다.")

    if not correlation.empty:
        st.markdown("#### 종목 간 상관관계")
        st.dataframe(correlation, width="stretch")

    st.markdown("#### 보유 종목 상세 스타일")
    analysis_view = _attach_latest_quotes(analysis, market_column="market")
    _show_table(
        analysis_view[
            [
                "market",
                "ticker",
                "name",
                "current_price",
                "change_pct",
                "weight_pct",
                "pnl_pct",
                "volatility_pct",
                "drawdown_pct",
                "return_6m_pct",
                "return_1y_pct",
                "dividend_yield_pct",
                "risk_level",
                "style",
                "sector",
                "quote_as_of",
            ]
        ],
        datetime_columns=["quote_as_of"],
        currency_columns=["current_price"],
        column_config={
            "market": st.column_config.TextColumn("시장", width="small"),
            "ticker": st.column_config.TextColumn("티커", width="small"),
            "name": st.column_config.TextColumn("종목명", width="small"),
            "current_price": st.column_config.TextColumn("현재가", width="small"),
            "change_pct": st.column_config.NumberColumn("당일등락", format="%.2f", width="small"),
            "weight_pct": st.column_config.NumberColumn("비중", format="%.2f", width="small"),
            "pnl_pct": st.column_config.NumberColumn("손익%", format="%.2f", width="small"),
            "volatility_pct": st.column_config.NumberColumn("변동성", format="%.2f", width="small"),
            "drawdown_pct": st.column_config.NumberColumn("낙폭", format="%.2f", width="small"),
            "return_6m_pct": st.column_config.NumberColumn("6개월", format="%.2f", width="small"),
            "return_1y_pct": st.column_config.NumberColumn("1년", format="%.2f", width="small"),
            "dividend_yield_pct": st.column_config.NumberColumn("배당수익률", format="%.2f", width="small"),
            "risk_level": st.column_config.TextColumn("위험도", width="small"),
            "style": st.column_config.TextColumn("스타일", width="small"),
            "sector": st.column_config.TextColumn("섹터", width="small"),
            "quote_as_of": st.column_config.TextColumn("시세기준", width="small"),
        },
    )


def render_portfolio_outlook() -> None:
    st.subheader("포트폴리오 전망 모니터링")
    st.caption("내 종목이 위쪽인지, 위험한지, 이벤트가 있는지 봅니다.")

    outlook = get_portfolio_outlook_state(_portfolio_cache_key(st.session_state.portfolio))
    _, _, event_adjustments, news_adjustments, _ = get_learning_state()
    outlook = _enrich_outlook_frame(
        outlook,
        event_adjustments=event_adjustments,
        news_adjustments=news_adjustments,
    )
    if outlook.empty:
        st.info("전망을 만들 보유 종목 데이터가 아직 부족합니다.")
        return

    strong_count = int(outlook["outlook"].isin(["상승 기대", "완만한 우상향"]).sum())
    caution_count = int((outlook["outlook"] == "하방 주의").sum())
    event_high_count = int((outlook["event_risk"] == "높음").sum())
    action_add_count = int((outlook["action_hint"] == "눌림 추가 가능").sum())
    action_reduce_count = int(outlook["action_hint"].isin(["비중 축소 검토", "일부 차익 실현", "손실관리 우선"]).sum())
    a, b, c, d, e = st.columns(5)
    a.metric("우호적 전망", f"{strong_count}")
    b.metric("하방 주의", f"{caution_count}")
    c.metric("가까운 이벤트", f"{event_high_count}")
    d.metric("추가 후보", f"{action_add_count}")
    e.metric("축소/관리", f"{action_reduce_count}")

    st.markdown("#### 한눈에 보는 보유종목 전망")
    top_outlook = outlook.sort_values(
        by=["outlook_score", "ticker"],
        ascending=[False, True],
    ).head(6)
    for _, row in top_outlook.iterrows():
        ticker_name = f"{row.get('ticker', '')} {row.get('name', '')}".strip()
        summary_text = str(row.get("reason_summary", "") or row.get("reason", "") or "")
        next_step = str(row.get("next_step", "") or "보유하며 추적")
        if str(row.get("outlook", "")) == "상승 기대":
            st.success(f"{ticker_name}: 흐름이 좋은 편입니다. {summary_text} → {next_step}")
        elif str(row.get("outlook", "")) == "완만한 우상향":
            st.info(f"{ticker_name}: 크게 나쁘지 않고 천천히 볼 만합니다. {summary_text} → {next_step}")
        elif str(row.get("outlook", "")) == "하방 주의":
            st.warning(f"{ticker_name}: 방어 기준을 먼저 봐야 합니다. {summary_text} → {next_step}")
        else:
            st.write(f"{ticker_name}: 아직 방향이 애매합니다. {summary_text} → {next_step}")

    alert_rows: list[dict[str, object]] = []
    for _, row in outlook.iterrows():
        action_hint = str(row.get("action_hint", "") or "")
        event_risk = str(row.get("event_risk", "") or "")
        news_bias = str(row.get("news_bias", "") or "")
        outlook_name = str(row.get("outlook", "") or "")

        alert_type = ""
        alert_message = ""
        if action_hint in {"손실관리 우선", "비중 축소 검토"}:
            alert_type = "즉시 점검"
            alert_message = "비중이나 손실 기준을 먼저 점검할 구간입니다."
        elif action_hint == "일부 차익 실현":
            alert_type = "수익 보호"
            alert_message = "수익이 난 구간이라 일부 정리도 볼 만합니다."
        elif action_hint == "눌림 추가 가능":
            alert_type = "추가 후보"
            alert_message = "눌림이 오면 천천히 모아볼 만합니다."
        elif event_risk == "높음":
            alert_type = "이벤트 경계"
            alert_message = "가까운 일정 전후로 흔들림을 체크할 필요가 있습니다."
        elif news_bias == "부정" and outlook_name != "상승 기대":
            alert_type = "뉴스 주의"
            alert_message = "최근 뉴스 흐름이 부담이라 해석을 더 봐야 합니다."
        elif news_bias == "긍정" and outlook_name in {"상승 기대", "완만한 우상향"}:
            alert_type = "긍정 추적"
            alert_message = "좋은 흐름이 이어지는지 추적해볼 만합니다."

        if alert_type:
            alert_rows.append(
                {
                    "market": row.get("market", ""),
                    "ticker": row.get("ticker", ""),
                    "name": row.get("name", ""),
                    "alert_type": alert_type,
                    "action_hint": action_hint,
                    "outlook": outlook_name,
                    "current_price": row.get("current_price", None),
                    "return_pct": row.get("return_pct", None),
                    "current_weight_pct": row.get("current_weight_pct", None),
                    "target_weight_pct": row.get("target_weight_pct", None),
                    "next_step": row.get("next_step", ""),
                    "momentum_state": row.get("momentum_state", ""),
                    "risk_level": row.get("risk_level", ""),
                    "weight_status": row.get("weight_status", ""),
                    "data_freshness": row.get("data_freshness", ""),
                    "price_source": row.get("price_source", ""),
                    "event_risk": event_risk,
                    "news_bias": news_bias,
                    "alert_message": alert_message,
                }
            )

    alert_df = pd.DataFrame(alert_rows)
    if not alert_df.empty:
        st.markdown("#### 오늘 브리핑")
        immediate_count = int((alert_df["alert_type"] == "즉시 점검").sum())
        add_count = int((alert_df["alert_type"] == "추가 후보").sum())
        protect_count = int((alert_df["alert_type"] == "수익 보호").sum())
        x, y, z = st.columns(3)
        x.metric("즉시 점검", immediate_count)
        y.metric("추가 후보", add_count)
        z.metric("수익 보호", protect_count)

        top_brief = alert_df.copy()
        top_brief["brief_rank"] = top_brief["alert_type"].map(
            {
                "즉시 점검": 1,
                "이벤트 경계": 2,
                "수익 보호": 3,
                "추가 후보": 4,
                "뉴스 주의": 5,
                "긍정 추적": 6,
            }
        ).fillna(9)
        top_brief = top_brief.sort_values(by=["brief_rank", "ticker"], ascending=[True, True]).head(6)
        for _, row in top_brief.iterrows():
            st.write(
                f"- `{row['alert_type']}` {row['ticker']} {row['name']} | "
                f"{row['action_hint']} | {row['alert_message']}"
            )

        with st.expander("브리핑 상세"):
            _show_table(
                alert_df[
                    [
                        "market",
                        "ticker",
                        "name",
                        "alert_type",
                        "action_hint",
                        "outlook",
                        "current_price",
                        "return_pct",
                        "current_weight_pct",
                        "target_weight_pct",
                        "next_step",
                        "momentum_state",
                        "risk_level",
                        "weight_status",
                        "data_freshness",
                        "price_source",
                        "event_risk",
                        "news_bias",
                        "alert_message",
                    ]
                ],
                currency_columns=["current_price"],
                column_config={
                    "market": st.column_config.TextColumn("시장", width="small"),
                    "ticker": st.column_config.TextColumn("티커", width="small"),
                    "name": st.column_config.TextColumn("종목명", width="small"),
                    "alert_type": st.column_config.TextColumn("브리핑", width="small"),
                    "action_hint": st.column_config.TextColumn("현재 판단", width="small"),
                    "outlook": st.column_config.TextColumn("전망", width="small"),
                    "current_price": st.column_config.TextColumn("현재가", width="small"),
                    "return_pct": st.column_config.NumberColumn("현재수익률", format="%.2f", width="small"),
                    "current_weight_pct": st.column_config.NumberColumn("현재비중", format="%.2f", width="small"),
                    "target_weight_pct": st.column_config.NumberColumn("목표비중", format="%.2f", width="small"),
                    "next_step": st.column_config.TextColumn("다음 행동", width="medium"),
                    "momentum_state": st.column_config.TextColumn("차트상태", width="small"),
                    "risk_level": st.column_config.TextColumn("위험도", width="small"),
                    "weight_status": st.column_config.TextColumn("비중상태", width="small"),
                    "data_freshness": st.column_config.TextColumn("데이터", width="small"),
                    "price_source": st.column_config.TextColumn("시세출처", width="small"),
                    "event_risk": st.column_config.TextColumn("이벤트", width="small"),
                    "news_bias": st.column_config.TextColumn("뉴스", width="small"),
                    "alert_message": st.column_config.TextColumn("요약", width="large"),
                },
            )

    daily_report = _build_portfolio_daily_report(outlook)
    if str(daily_report.get("headline", "")).strip():
        st.markdown("#### 일일 리포트")
        st.info(str(daily_report["headline"]))

        add_candidates = daily_report["add_candidates"]
        trim_candidates = daily_report["trim_candidates"]
        event_watch = daily_report["event_watch"]

        col_add, col_trim, col_event = st.columns(3)
        with col_add:
            st.markdown("##### 추가 검토")
            if add_candidates.empty:
                st.caption("지금은 뚜렷한 추가 후보가 없습니다.")
            else:
                for _, row in add_candidates.iterrows():
                    st.write(
                        f"- `{row['ticker']}` {row['name']} | {row['action_hint']} | "
                        f"눌림구간 { _format_price(row.get('accumulate_price'), str(row.get('market', ''))) }"
                    )

        with col_trim:
            st.markdown("##### 축소/관리")
            if trim_candidates.empty:
                st.caption("지금은 급한 축소 후보가 많지 않습니다.")
            else:
                for _, row in trim_candidates.iterrows():
                    st.write(
                        f"- `{row['ticker']}` {row['name']} | {row['action_hint']} | "
                        f"주의가격 { _format_price(row.get('caution_price'), str(row.get('market', ''))) }"
                    )

        with col_event:
            st.markdown("##### 이벤트 임박")
            if event_watch.empty:
                st.caption("가까운 핵심 일정 종목이 많지 않습니다.")
            else:
                for _, row in event_watch.iterrows():
                    date_text = str(row.get("earnings_date", "") or row.get("ex_dividend_date", "") or "-")
                    st.write(
                        f"- `{row['ticker']}` {row['name']} | {row['event_note']} | {date_text}"
                    )

    priority_rows: list[dict[str, object]] = []
    for _, row in outlook.iterrows():
        priority = 0
        if str(row.get("event_risk", "")) == "높음":
            priority += 4
        if str(row.get("outlook", "")) == "하방 주의":
            priority += 4
        if str(row.get("news_bias", "")) == "부정":
            priority += 2
        if str(row.get("outlook", "")) == "상승 기대":
            priority += 1
        priority_rows.append({**row.to_dict(), "priority_score": priority})

    priority_df = pd.DataFrame(priority_rows).sort_values(
        by=["priority_score", "outlook_score", "ticker"],
        ascending=[False, True, True],
    ).head(5)
    if not priority_df.empty:
        st.markdown("#### 우선 확인 종목")
        for _, row in priority_df.iterrows():
            badge = "주의" if str(row.get("outlook", "")) == "하방 주의" else "체크"
            target_weight = pd.to_numeric(row.get("target_weight_pct", None), errors="coerce")
            target_weight_text = f"{float(target_weight):.1f}%" if not pd.isna(target_weight) else "-"
            st.write(
                f"- `{badge}` {row['ticker']} {row['name']} | {row['action_hint']} | {row.get('next_step', '')} | "
                f"현재 {float(row.get('current_weight_pct', 0) or 0):.1f}% / 목표 {target_weight_text} | "
                f"{row.get('reason_summary', '')}"
            )

    _show_table(
        outlook[
            [
                "market",
                "ticker",
                "name",
                "outlook",
                "outlook_score",
                "outlook_label",
                "prediction_view",
                "delta_view",
                "action_hint",
                "next_step",
                "current_price",
                "change_pct",
                "avg_price",
                "return_pct",
                "current_weight_pct",
                "target_weight_pct",
                "weight_gap_pct",
                "weight_status",
                "style",
                "risk_level",
                "momentum_state",
                "return_6m_pct",
                "return_1y_pct",
                "dividend_yield_pct",
                "accumulate_price",
                "caution_price",
                "target_price",
                "event_risk",
                "earnings_date",
                "ex_dividend_date",
                "news_bias",
                "news_count",
                "quote_as_of",
                "reason_summary",
                "reason",
                "event_note",
            ]
        ],
        datetime_columns=["quote_as_of"],
        currency_columns=["current_price", "avg_price", "accumulate_price", "caution_price", "target_price"],
        column_config={
            "market": st.column_config.TextColumn("시장", width="small"),
            "ticker": st.column_config.TextColumn("티커", width="small"),
            "name": st.column_config.TextColumn("종목명", width="small"),
            "outlook": st.column_config.TextColumn("전망", width="small"),
            "outlook_score": st.column_config.NumberColumn("전망점수", format="%d", width="small"),
            "outlook_label": st.column_config.TextColumn("점수해석", width="small"),
            "prediction_view": st.column_config.TextColumn("예상", width="small"),
            "delta_view": st.column_config.TextColumn("학습보정", width="small"),
            "action_hint": st.column_config.TextColumn("현재 판단", width="small"),
            "next_step": st.column_config.TextColumn("다음 행동", width="medium"),
            "current_price": st.column_config.TextColumn("현재가", width="small"),
            "change_pct": st.column_config.NumberColumn("당일등락", format="%.2f", width="small"),
            "avg_price": st.column_config.TextColumn("평균단가", width="small"),
            "return_pct": st.column_config.NumberColumn("현재수익률", format="%.2f", width="small"),
            "current_weight_pct": st.column_config.NumberColumn("현재비중", format="%.2f", width="small"),
            "target_weight_pct": st.column_config.NumberColumn("목표비중", format="%.2f", width="small"),
            "weight_gap_pct": st.column_config.NumberColumn("비중차이", format="%.2f", width="small"),
            "weight_status": st.column_config.TextColumn("비중상태", width="small"),
            "style": st.column_config.TextColumn("성격", width="small"),
            "risk_level": st.column_config.TextColumn("위험도", width="small"),
            "momentum_state": st.column_config.TextColumn("차트상태", width="small"),
            "return_6m_pct": st.column_config.NumberColumn("6개월", format="%.2f", width="small"),
            "return_1y_pct": st.column_config.NumberColumn("1년", format="%.2f", width="small"),
            "dividend_yield_pct": st.column_config.NumberColumn("배당률", format="%.2f", width="small"),
            "accumulate_price": st.column_config.TextColumn("눌림구간", width="small"),
            "caution_price": st.column_config.TextColumn("주의가격", width="small"),
            "target_price": st.column_config.TextColumn("기대목표", width="small"),
            "event_risk": st.column_config.TextColumn("이벤트리스크", width="small"),
            "earnings_date": st.column_config.TextColumn("실적예정", width="small"),
            "ex_dividend_date": st.column_config.TextColumn("배당락일", width="small"),
            "news_bias": st.column_config.TextColumn("뉴스흐름", width="small"),
            "news_count": st.column_config.NumberColumn("뉴스건수", format="%d", width="small"),
            "quote_as_of": st.column_config.TextColumn("시세기준", width="small"),
            "reason_summary": st.column_config.TextColumn("한줄 요약", width="medium"),
            "reason": st.column_config.TextColumn("핵심 사유", width="large"),
            "event_note": st.column_config.TextColumn("이벤트 메모", width="medium"),
        },
    )

    with st.expander("뉴스/이벤트 메모"):
        for _, row in outlook.iterrows():
            st.markdown(f"**{row['ticker']} {row['name']}**")
            st.write(f"- 이벤트: {row['event_note']}")
            if str(row.get('headline', '')).strip():
                st.write(f"- 최근 헤드라인: {row['headline']}")
            st.write(f"- 뉴스 해석: {row['news_note']}")


def render_budget_planner() -> None:
    st.subheader("예산 투입 플래너")
    st.caption("오늘 넣을 돈을 단타/성장/배당/장기로 나눕니다.")

    summary = _portfolio_summary_defaults()
    if st.session_state.portfolio.empty:
        st.info("보유 종목이 없어도 사용할 수 있습니다. 시장 탐색 후보를 기준으로 예산 플랜을 만듭니다.")

    left, center, right = st.columns(3)
    with left:
        us_budget = st.number_input("미국 추가 예산", min_value=0.0, value=3000.0, step=100.0)
    with center:
        kr_budget = st.number_input("한국 추가 예산", min_value=0.0, value=1000000.0, step=10000.0)
    with right:
        mode = st.selectbox(
            "배분 모드",
            ["자동 추천", "단타 중심", "장기 중심", "배당 중심", "성장 중심", "방어 중심"],
            index=0,
        )

    st.caption(
        f"현재 계좌 기준: 성장 {summary['growth_weight']:.1f}% / 배당 {summary['dividend_weight']:.1f}% / 고위험 {summary['high_risk_weight']:.1f}%"
    )
    if 0 < float(us_budget) < 300:
        st.warning("미국 소액 예산 모드: 로컬 유니버스에서 예산 안에 들어오는 1주 매수 가능 후보를 우선 탐색합니다.")
    if 0 < float(kr_budget) < 50000:
        st.warning("한국 소액 예산 모드: 로컬 유니버스에서 예산 안에 들어오는 1주 매수 가능 후보를 우선 탐색합니다.")

    _render_decision_log_overview(compact=True)

    if not _render_data_freshness_gate(purpose="예산 플래너", require_intraday=False):
        st.info("오래된 가격으로 예산 배분을 만들지 않도록 계산을 멈췄습니다.")
        return

    st.markdown("#### 실행 모드 제안")
    _render_scan_advisor(purpose="budget", key_prefix="budget")

    calc_col, scan_col, deep_col, hint_col = st.columns([1, 1, 1, 2])
    with calc_col:
        if st.button("예산 플랜 계산/새로고침", width="stretch", key="budget_plan_refresh"):
            st.session_state["budget_plan_ready"] = True
    with scan_col:
        include_live_candidates = st.toggle("시장 전체 새 탐색", value=False, key="budget_include_live_candidates")
    with deep_col:
        deep_budget_search = st.toggle("깊게 예산 탐색", value=False, key="budget_deep_affordable_search")
    with hint_col:
        affordable_limit = 300 if deep_budget_search else 80
        st.caption(
            f"예산맞춤 후보는 기본 시장별 `{affordable_limit}`개만 확인합니다. "
            "느리면 깊게 탐색을 끄고, 전체 수집은 대시보드에서 따로 돌리세요."
        )

    if float(us_budget) + float(kr_budget) <= 0:
        st.info("미국 또는 한국 추가 예산을 입력하면 플랜을 계산합니다.")
        briefings = load_daily_briefings()
        if not briefings.empty:
            with st.expander("최근 저장된 일일 브리핑"):
                _show_table(
                    briefings.sort_values(by=["saved_at"], ascending=[False]).head(10)[
                        ["briefing_date", "saved_at", "mode", "total_budget", "headline", "top_actions"]
                    ],
                    plain_numeric_columns=["total_budget"],
                    datetime_columns=["saved_at"],
                    column_config={
                        "briefing_date": st.column_config.TextColumn("날짜", width="small"),
                        "saved_at": st.column_config.TextColumn("저장시각", width="small"),
                        "mode": st.column_config.TextColumn("모드", width="small"),
                        "total_budget": st.column_config.TextColumn("총예산", width="small"),
                        "headline": st.column_config.TextColumn("브리핑", width="large"),
                        "top_actions": st.column_config.TextColumn("상위 실행안", width="large"),
                    },
                )
        return

    if not st.session_state.get("budget_plan_ready", False):
        st.info("예산과 모드를 정한 뒤 `예산 플랜 계산/새로고침`을 누르면 시장 탐색 기준으로 플랜을 만듭니다.")
        recent_candidates = pd.concat(
            [
                _recent_plan_candidates("US", top_n=3)["today"].assign(market="US"),
                _recent_plan_candidates("KR", top_n=3)["today"].assign(market="KR"),
            ],
            ignore_index=True,
        )
        if not recent_candidates.empty:
            with st.expander("최근 저장 후보 미리보기"):
                _show_table(
                    recent_candidates.head(8),
                    currency_columns=["current_price", "entry_price", "stop_loss", "target_1"],
                    column_config=_candidate_column_config(),
                )
        return

    spinner_text = (
        "예산 플랜 계산 중입니다. 시장 탐색 후보와 시세 데이터를 확인하고 있어요."
        if include_live_candidates
        else "예산 플랜 계산 중입니다. 저장된 후보와 예산 배분을 정리하고 있어요."
    )
    with st.spinner(spinner_text):
        if include_live_candidates:
            us_candidates = _pick_plan_candidates("US", top_n=3)
            kr_candidates = _pick_plan_candidates("KR", top_n=3)
        else:
            us_candidates = _recent_plan_candidates("US", top_n=3)
            kr_candidates = _recent_plan_candidates("KR", top_n=3)
        progress = st.progress(0.0, text="예산맞춤 후보 확인 준비")
        affordable_limit = 300 if deep_budget_search else 80
        progress.progress(0.15, text=f"미국 예산맞춤 후보 {affordable_limit}개 확인")
        us_affordable = _build_budget_affordable_universe_candidates("US", float(us_budget), limit=affordable_limit)
        progress.progress(0.45, text=f"한국 예산맞춤 후보 {affordable_limit}개 확인")
        kr_affordable = _build_budget_affordable_universe_candidates("KR", float(kr_budget), limit=affordable_limit)
        progress.progress(0.70, text="공통 추천과 예산 후보 병합")
        us_candidates = _merge_budget_affordable_candidates(us_candidates, us_affordable)
        kr_candidates = _merge_budget_affordable_candidates(kr_candidates, kr_affordable)
        us_common = get_today_scan_state(
            market="US",
            watchlist_key=_watchlist_cache_key("US"),
            min_score=int(st.session_state.scanner_settings["min_score"]),
            scan_limit=min(12, int(st.session_state.scanner_settings["scan_limit"])),
        )
        kr_common = get_today_scan_state(
            market="KR",
            watchlist_key=_watchlist_cache_key("KR"),
            min_score=int(st.session_state.scanner_settings["min_score"]),
            scan_limit=min(12, int(st.session_state.scanner_settings["scan_limit"])),
        )
        us_candidates = _merge_common_recommendations_for_budget(us_candidates, us_common, budget=float(us_budget))
        kr_candidates = _merge_common_recommendations_for_budget(kr_candidates, kr_common, budget=float(kr_budget))
        progress.progress(0.90, text="예산 실행안 산출")
        us_candidate_health = _candidate_set_health(us_candidates).assign(market="US")
        kr_candidate_health = _candidate_set_health(kr_candidates).assign(market="KR")
        us_outlook = pd.DataFrame()
        kr_outlook = pd.DataFrame()

        us_plan = _build_budget_plan_rows(
            market="US",
            budget=float(us_budget),
            mode=mode,
            portfolio_summary=summary,
            outlook=us_outlook,
            candidate_sets=us_candidates,
        )
        kr_plan = _build_budget_plan_rows(
            market="KR",
            budget=float(kr_budget),
            mode=mode,
            portfolio_summary=summary,
            outlook=kr_outlook,
            candidate_sets=kr_candidates,
        )

        us_detail = _build_budget_candidate_detail(us_plan, us_candidates) if not us_plan.empty else pd.DataFrame()
        kr_detail = _build_budget_candidate_detail(kr_plan, kr_candidates) if not kr_plan.empty else pd.DataFrame()
        combined_actions = _build_combined_budget_actions(
            us_plan=us_plan,
            kr_plan=kr_plan,
            us_detail=us_detail,
            kr_detail=kr_detail,
        )
        progress.progress(1.0, text="예산 플랜 계산 완료")
    st.success("예산 플랜 계산 완료")
    if not include_live_candidates:
        st.caption("빠른 모드: 저장된 스냅샷과 로컬 가격 캐시에서 예산 안에 들어오는 후보를 먼저 찾습니다. 최신 시장 후보까지 보려면 `시장 전체 새 탐색`을 켜세요.")
    st.caption("추천 일치 기준: 예산 플래너도 오늘 추천 탭의 공통추천을 먼저 사용하고, 예산으로 1주 매수가 어려울 때만 예산맞춤 후보를 보조로 섞습니다.")
    if float(us_budget) > 0:
        st.caption(f"미국 예산 맞춤 로컬 후보: {len(us_affordable)}개")
    if float(kr_budget) > 0:
        st.caption(f"한국 예산 맞춤 로컬 후보: {len(kr_affordable)}개")
    with st.expander("예산 후보 풀 진단", expanded=False):
        health = pd.concat([us_candidate_health, kr_candidate_health], ignore_index=True)
        _show_table(
            health,
            column_config={
                "market": st.column_config.TextColumn("시장", width="small"),
                "bucket": st.column_config.TextColumn("유형", width="small"),
                "candidate_count": st.column_config.NumberColumn("후보수", format="%d", width="small"),
                "top_ticker": st.column_config.TextColumn("상위종목", width="small"),
                "top_score": st.column_config.NumberColumn("상위점수", format="%.1f", width="small"),
                "scanned": st.column_config.NumberColumn("조회", format="%d", width="small"),
                "daily_pass": st.column_config.NumberColumn("일봉통과", format="%d", width="small"),
                "intraday_pass": st.column_config.NumberColumn("분봉통과", format="%d", width="small"),
                "errors": st.column_config.NumberColumn("오류", format="%d", width="small"),
                "status": st.column_config.TextColumn("상태", width="small"),
            },
        )

    total_budget = float(us_budget) + float(kr_budget)
    total_action_budget = float(
        pd.to_numeric(combined_actions.get("planned_amount", []), errors="coerce").fillna(0).sum()
    ) if not combined_actions.empty else 0.0
    plan_frames_for_metrics = [frame for frame in [us_plan, kr_plan] if not frame.empty]
    plan_metric_frame = pd.concat(plan_frames_for_metrics, ignore_index=True) if plan_frames_for_metrics else pd.DataFrame()
    executable_budget = float(
        pd.to_numeric(plan_metric_frame.get("executable_amount", []), errors="coerce").fillna(0).sum()
    ) if not plan_metric_frame.empty else 0.0
    reserve_budget = float(
        pd.to_numeric(plan_metric_frame.get("reserve_amount", []), errors="coerce").fillna(0).sum()
    ) if not plan_metric_frame.empty else 0.0
    st.markdown("#### 오늘 실행 요약")
    x, y, z, w = st.columns(4)
    x.metric("전체 추가 예산", f"{total_budget:,.0f}")
    y.metric("실행 후보 수", f"{len(combined_actions)}")
    z.metric("실제 집행 후보", f"{executable_budget:,.0f}")
    w.metric("현금 보류", f"{reserve_budget:,.0f}")
    if not combined_actions.empty and "execution_status" in combined_actions.columns:
        status_counts = combined_actions["execution_status"].astype(str).value_counts()
        s1, s2, s3 = st.columns(3)
        s1.metric("지금 검토", f"{int(status_counts.get('지금 검토 가능', 0) + status_counts.get('짧게 검토 가능', 0))}")
        s2.metric("가격 대기", f"{int(status_counts.get('대기: 진입가 초과', 0))}")
        s3.metric(
            "보류/확인",
            f"{int(status_counts.get('보류: 방어선 근접', 0) + status_counts.get('보류: 예산 부족', 0) + status_counts.get('가격확인 필요', 0))}",
        )

    summary_text = _build_budget_summary_text(combined_actions)
    st.info(summary_text)
    if not combined_actions.empty:
        remembered_markets: list[str] = []
        for market_name in ["US", "KR"]:
            market_actions = combined_actions[combined_actions["market"].astype(str).str.upper() == market_name].copy()
            if _remember_recommendation_snapshot("budget_plan", market_name, market_actions):
                remembered_markets.append(market_name)
        if remembered_markets:
            st.caption(f"학습 메모리 저장: {', '.join(remembered_markets)} 예산 플랜을 오늘 결과 추적 대상으로 기록했습니다.")
        _render_action_deck(combined_actions, title="오늘 바로 실행 카드", limit=4)
        execution_view = _budget_actions_execution_view(combined_actions)
        _show_table(
            execution_view,
            plain_numeric_columns=["planned_amount", "executable_amount", "reserve_amount"],
            currency_columns=[
                "current_price",
                "entry_price",
                "stop_loss",
                "target_1",
                "target_2",
                "target_3",
            ],
            column_config={
                "market": st.column_config.TextColumn("시장", width="small"),
                "bucket": st.column_config.TextColumn("유형", width="small"),
                "ticker": st.column_config.TextColumn("티커", width="small"),
                "name": st.column_config.TextColumn("종목명", width="small"),
                "entry_decision": st.column_config.TextColumn("지금판단", width="small"),
                "current_price": st.column_config.TextColumn("현재가", width="small"),
                "entry_price": st.column_config.TextColumn("진입가", width="small"),
                "stop_loss": st.column_config.TextColumn("손절가", width="small"),
                "target_1": st.column_config.TextColumn("1차목표", width="small"),
                "target_2": st.column_config.TextColumn("2차목표", width="small"),
                "target_3": st.column_config.TextColumn("3차목표", width="small"),
                "entry_gap_pct": st.column_config.NumberColumn("진입여유%", format="%.2f", width="small"),
                "risk_pct": st.column_config.NumberColumn("손절폭%", format="%.2f", width="small"),
                "estimated_units": st.column_config.NumberColumn("매수수량", format="%d", width="small"),
                "max_units_if_all_in": st.column_config.NumberColumn("전액시수량", format="%d", width="small"),
                "budget_use_pct": st.column_config.NumberColumn("전액활용", format="%.1f", width="small"),
                "planned_amount": st.column_config.TextColumn("배정금액", width="small"),
                "executable_amount": st.column_config.TextColumn("집행금액", width="small"),
                "reserve_amount": st.column_config.TextColumn("현금보류", width="small"),
                "score": st.column_config.NumberColumn("점수", format="%d", width="small"),
                "confidence_score": st.column_config.NumberColumn("신뢰도", format="%d", width="small"),
                "confidence_detail": st.column_config.TextColumn("신뢰근거", width="medium"),
                "setup": st.column_config.TextColumn("세팅", width="small"),
                "budget_source": st.column_config.TextColumn("추천출처", width="small"),
                "allocation_reason": st.column_config.TextColumn("배정근거", width="large"),
                "price_basis": st.column_config.TextColumn("가격근거", width="large"),
                "reason": st.column_config.TextColumn("핵심 사유", width="large"),
            },
        )

        top_actions = combined_actions.head(5)
        st.markdown("#### 무엇부터 할까")
        for _, row in top_actions.iterrows():
            st.write(
                f"- `{row['market']} {row['ticker']}` | {row['bucket']} | "
                f"{row.get('execution_status', '상태확인')} | {row['planned_amount']:,.0f} 배정 | "
                f"현재가 {_format_price(row.get('ref_price'), str(row.get('market', '')))} | "
                f"진입가 {_format_price(row.get('buy_now_limit'), str(row.get('market', '')))} | "
                f"손절가 {_format_price(row.get('stop_loss'), str(row.get('market', '')))} | {row['reason']}"
            )
        _render_budget_decision_panel(combined_actions)

    brief_left, brief_right = st.columns([1, 2])
    with brief_left:
        if st.button("오늘 브리핑 저장", width="stretch"):
            briefing_date = pd.Timestamp.now(tz="Asia/Seoul").date().isoformat()
            briefing_id = append_daily_briefing(
                {
                    "briefing_date": briefing_date,
                    "mode": mode,
                    "us_budget": float(us_budget),
                    "kr_budget": float(kr_budget),
                    "total_budget": total_budget,
                    "action_count": len(combined_actions),
                    "headline": summary_text,
                    "top_actions": _serialize_top_budget_actions(combined_actions),
                    "notes": (
                        f"성장 {summary['growth_weight']:.1f}% / 배당 {summary['dividend_weight']:.1f}% / "
                        f"고위험 {summary['high_risk_weight']:.1f}%"
                    ),
                }
            )
            if not combined_actions.empty:
                append_daily_briefing_actions(
                    briefing_id=briefing_id,
                    briefing_date=briefing_date,
                    frame=combined_actions.head(10).copy(),
                )
            st.success("오늘 브리핑을 저장했습니다.")
    with brief_right:
        st.caption("오늘 계획을 저장합니다.")

    briefings = load_daily_briefings()
    if not briefings.empty:
        with st.expander("저장된 일일 브리핑"):
            briefing_view = briefings.sort_values(by=["saved_at"], ascending=[False]).head(20).copy()
            _show_table(
                briefing_view[
                    [
                        "briefing_date",
                        "saved_at",
                        "mode",
                        "total_budget",
                        "action_count",
                        "headline",
                        "top_actions",
                        "notes",
                    ]
                ],
                plain_numeric_columns=["total_budget"],
                datetime_columns=["saved_at"],
                column_config={
                    "briefing_date": st.column_config.TextColumn("날짜", width="small"),
                    "saved_at": st.column_config.TextColumn("저장시각", width="small"),
                    "mode": st.column_config.TextColumn("모드", width="small"),
                    "total_budget": st.column_config.TextColumn("총예산", width="small"),
                    "action_count": st.column_config.NumberColumn("후보수", format="%d", width="small"),
                    "headline": st.column_config.TextColumn("브리핑", width="large"),
                    "top_actions": st.column_config.TextColumn("상위 실행안", width="large"),
                    "notes": st.column_config.TextColumn("계좌 메모", width="large"),
                },
            )

    briefing_actions = load_daily_briefing_actions()
    if not briefing_actions.empty:
        tracked_actions = _build_briefing_action_tracking(briefing_actions)
        if not tracked_actions.empty:
            st.markdown("#### 브리핑 실행안 추적")
            good_flow = int((tracked_actions["status"] == "좋은 흐름").sum())
            caution_flow = int((tracked_actions["status"] == "주의").sum())
            pending_flow = int((tracked_actions["status"] == "관찰중").sum())
            a1, a2, a3 = st.columns(3)
            a1.metric("좋은 흐름", f"{good_flow}")
            a2.metric("주의", f"{caution_flow}")
            a3.metric("관찰중", f"{pending_flow}")
            _show_table(
                tracked_actions.head(30),
                plain_numeric_columns=["planned_amount"],
                currency_columns=["ref_price", "current_price"],
                datetime_columns=["saved_at", "quote_as_of"],
                column_config={
                    "briefing_date": st.column_config.TextColumn("브리핑일", width="small"),
                    "saved_at": st.column_config.TextColumn("저장시각", width="small"),
                    "market": st.column_config.TextColumn("시장", width="small"),
                    "bucket": st.column_config.TextColumn("유형", width="small"),
                    "ticker": st.column_config.TextColumn("티커", width="small"),
                    "name": st.column_config.TextColumn("종목명", width="small"),
                    "setup": st.column_config.TextColumn("세팅", width="small"),
                    "score": st.column_config.NumberColumn("점수", format="%d", width="small"),
                    "planned_amount": st.column_config.TextColumn("배정금액", width="small"),
                    "ref_price": st.column_config.TextColumn("저장가격", width="small"),
                    "current_price": st.column_config.TextColumn("현재가", width="small"),
                    "current_return_pct": st.column_config.NumberColumn("현재수익률", format="%.2f", width="small"),
                    "status": st.column_config.TextColumn("현재상태", width="small"),
                    "quote_as_of": st.column_config.TextColumn("시세기준", width="small"),
                    "reason": st.column_config.TextColumn("핵심 사유", width="large"),
                },
            )

    us_tab, kr_tab = st.tabs(["미국 예산 플랜", "한국 예산 플랜"])
    with us_tab:
        if us_plan.empty:
            st.info("미국 예산이 0원이거나 플랜을 만들 데이터가 부족합니다.")
        else:
            _show_table(
                us_plan,
                plain_numeric_columns=["planned_amount", "executable_amount", "reserve_amount"],
                currency_columns=["unit_price", "split_1", "split_2", "split_3"],
                default_market="US",
                column_config={
                    "market": st.column_config.TextColumn("시장", width="small"),
                    "bucket": st.column_config.TextColumn("유형", width="small"),
                    "plan_type": st.column_config.TextColumn("세부유형", width="small"),
                    "allocation_pct": st.column_config.NumberColumn("배분%", format="%.1f", width="small"),
                    "planned_amount": st.column_config.TextColumn("투입금액", width="small"),
                    "executable_amount": st.column_config.TextColumn("집행금액", width="small"),
                    "reserve_amount": st.column_config.TextColumn("현금보류", width="small"),
                    "horizon": st.column_config.TextColumn("기간", width="small"),
                    "unit_price": st.column_config.TextColumn("현재가", width="small"),
                    "estimated_units": st.column_config.NumberColumn("예상수량", format="%d", width="small"),
                    "split_1": st.column_config.TextColumn("1차", width="small"),
                    "split_2": st.column_config.TextColumn("2차", width="small"),
                    "split_3": st.column_config.TextColumn("3차", width="small"),
                    "candidates": st.column_config.TextColumn("추천 후보", width="large"),
                    "execution_rule": st.column_config.TextColumn("실행 원칙", width="large"),
                    "note": st.column_config.TextColumn("운용 메모", width="large"),
                },
            )
            if not us_detail.empty:
                with st.expander("미국 세부 배분 후보"):
                    _show_table(
                        us_detail,
                        currency_columns=[
                            "planned_amount",
                            "ref_price",
                            "first_amount",
                            "second_amount",
                            "third_amount",
                            "buy_now_limit",
                            "add_on_pullback",
                            "stop_loss",
                            "reentry_above",
                            "target_1",
                            "target_2",
                        ],
                        default_market="US",
                        column_config={
                            "bucket": st.column_config.TextColumn("유형", width="small"),
                            "ticker": st.column_config.TextColumn("티커", width="small"),
                            "name": st.column_config.TextColumn("종목명", width="small"),
                            "setup": st.column_config.TextColumn("세팅", width="small"),
                            "score": st.column_config.NumberColumn("점수", format="%d", width="small"),
                            "confidence_view": st.column_config.TextColumn("신뢰도", width="small"),
                            "execution_status": st.column_config.TextColumn("집행상태", width="small"),
                            "score_weight_pct": st.column_config.NumberColumn("후보내비중", format="%.1f", width="small"),
                            "planned_amount": st.column_config.TextColumn("배정금액", width="small"),
                            "ref_price": st.column_config.TextColumn("현재가", width="small"),
                            "estimated_units": st.column_config.NumberColumn("예상수량", format="%d", width="small"),
                            "first_amount": st.column_config.TextColumn("1차금액", width="small"),
                            "second_amount": st.column_config.TextColumn("2차금액", width="small"),
                            "third_amount": st.column_config.TextColumn("3차금액", width="small"),
                            "buy_now_limit": st.column_config.TextColumn("진입가", width="small"),
                            "add_on_pullback": st.column_config.TextColumn("눌림추가", width="small"),
                            "stop_loss": st.column_config.TextColumn("손절/보류", width="small"),
                            "reentry_above": st.column_config.TextColumn("재진입", width="small"),
                            "target_1": st.column_config.TextColumn("1차목표", width="small"),
                            "target_2": st.column_config.TextColumn("2차목표", width="small"),
                            "execution_rule": st.column_config.TextColumn("실행 원칙", width="large"),
                            "allocation_reason": st.column_config.TextColumn("배정근거", width="large"),
                            "quality_note": st.column_config.TextColumn("품질메모", width="medium"),
                            "reason": st.column_config.TextColumn("핵심 사유", width="large"),
                            },
                        )
                with st.expander("미국 직접 선택 플랜"):
                    st.caption("유형별로 직접 고릅니다.")
                    us_selection_map: dict[str, str] = {}
                    for bucket in us_plan["bucket"].astype(str).tolist():
                        bucket_detail = us_detail[us_detail["bucket"].astype(str) == bucket].copy()
                        if bucket_detail.empty:
                            continue
                        option_labels = {
                            f"{str(row.get('ticker', ''))} | {str(row.get('name', ''))} | "
                            f"{str(row.get('setup', '')) or '기본'} | {_safe_int(row.get('score', 0))}점": str(row.get("ticker", ""))
                            for _, row in bucket_detail.iterrows()
                        }
                        default_label = next(iter(option_labels.keys()))
                        picked_label = st.selectbox(
                            f"{bucket} 직접 선택",
                            options=list(option_labels.keys()),
                            index=0,
                            key=f"budget_pick_us_{bucket}",
                        )
                        us_selection_map[bucket] = option_labels.get(picked_label, option_labels[default_label])

                    picked_us = _build_selected_budget_actions(
                        market="US",
                        plan_df=us_plan,
                        detail_df=us_detail,
                        selection_map=us_selection_map,
                    )
                    if not picked_us.empty:
                        _show_table(
                            picked_us,
                            plain_numeric_columns=["planned_amount", "executable_amount", "reserve_amount"],
                            currency_columns=[
                                "ref_price",
                                "split_1",
                                "split_2",
                                "split_3",
                                "buy_now_limit",
                                "add_on_pullback",
                                "stop_loss",
                                "reentry_above",
                                "target_1",
                                "target_2",
                            ],
                            default_market="US",
                            column_config={
                                "market": st.column_config.TextColumn("시장", width="small"),
                                "bucket": st.column_config.TextColumn("유형", width="small"),
                                "ticker": st.column_config.TextColumn("티커", width="small"),
                                "name": st.column_config.TextColumn("종목명", width="small"),
                                "setup": st.column_config.TextColumn("세팅", width="small"),
                            "score": st.column_config.NumberColumn("점수", format="%d", width="small"),
                            "confidence_view": st.column_config.TextColumn("신뢰도", width="small"),
                            "execution_status": st.column_config.TextColumn("집행상태", width="small"),
                            "planned_amount": st.column_config.TextColumn("배정금액", width="small"),
                            "executable_amount": st.column_config.TextColumn("집행금액", width="small"),
                                "reserve_amount": st.column_config.TextColumn("현금보류", width="small"),
                                "ref_price": st.column_config.TextColumn("현재가", width="small"),
                                "estimated_units": st.column_config.NumberColumn("예상수량", format="%d", width="small"),
                                "split_1": st.column_config.TextColumn("1차", width="small"),
                                "split_2": st.column_config.TextColumn("2차", width="small"),
                                "split_3": st.column_config.TextColumn("3차", width="small"),
                                "buy_now_limit": st.column_config.TextColumn("진입가", width="small"),
                                "add_on_pullback": st.column_config.TextColumn("눌림추가", width="small"),
                                "stop_loss": st.column_config.TextColumn("손절/보류", width="small"),
                                "reentry_above": st.column_config.TextColumn("재진입", width="small"),
                                "target_1": st.column_config.TextColumn("1차목표", width="small"),
                                "target_2": st.column_config.TextColumn("2차목표", width="small"),
                            "horizon": st.column_config.TextColumn("기간", width="small"),
                            "execution_rule": st.column_config.TextColumn("실행 원칙", width="large"),
                            "allocation_reason": st.column_config.TextColumn("배정근거", width="large"),
                            "quality_note": st.column_config.TextColumn("품질메모", width="medium"),
                            "note": st.column_config.TextColumn("운용 메모", width="large"),
                            "reason": st.column_config.TextColumn("핵심 사유", width="large"),
                        },
                        )

    with kr_tab:
        if kr_plan.empty:
            st.info("한국 예산이 0원이거나 플랜을 만들 데이터가 부족합니다.")
        else:
            _show_table(
                kr_plan,
                plain_numeric_columns=["planned_amount", "executable_amount", "reserve_amount"],
                currency_columns=["unit_price", "split_1", "split_2", "split_3"],
                default_market="KR",
                column_config={
                    "market": st.column_config.TextColumn("시장", width="small"),
                    "bucket": st.column_config.TextColumn("유형", width="small"),
                    "plan_type": st.column_config.TextColumn("세부유형", width="small"),
                    "allocation_pct": st.column_config.NumberColumn("배분%", format="%.1f", width="small"),
                    "planned_amount": st.column_config.TextColumn("투입금액", width="small"),
                    "executable_amount": st.column_config.TextColumn("집행금액", width="small"),
                    "reserve_amount": st.column_config.TextColumn("현금보류", width="small"),
                    "horizon": st.column_config.TextColumn("기간", width="small"),
                    "unit_price": st.column_config.TextColumn("현재가", width="small"),
                    "estimated_units": st.column_config.NumberColumn("예상수량", format="%d", width="small"),
                    "split_1": st.column_config.TextColumn("1차", width="small"),
                    "split_2": st.column_config.TextColumn("2차", width="small"),
                    "split_3": st.column_config.TextColumn("3차", width="small"),
                    "candidates": st.column_config.TextColumn("추천 후보", width="large"),
                    "execution_rule": st.column_config.TextColumn("실행 원칙", width="large"),
                    "note": st.column_config.TextColumn("운용 메모", width="large"),
                },
            )
            if not kr_detail.empty:
                with st.expander("한국 세부 배분 후보"):
                    _show_table(
                        kr_detail,
                        currency_columns=[
                            "planned_amount",
                            "ref_price",
                            "first_amount",
                            "second_amount",
                            "third_amount",
                            "buy_now_limit",
                            "add_on_pullback",
                            "stop_loss",
                            "reentry_above",
                            "target_1",
                            "target_2",
                        ],
                        default_market="KR",
                        column_config={
                            "bucket": st.column_config.TextColumn("유형", width="small"),
                            "ticker": st.column_config.TextColumn("티커", width="small"),
                            "name": st.column_config.TextColumn("종목명", width="small"),
                            "setup": st.column_config.TextColumn("세팅", width="small"),
                            "score": st.column_config.NumberColumn("점수", format="%d", width="small"),
                            "confidence_view": st.column_config.TextColumn("신뢰도", width="small"),
                            "execution_status": st.column_config.TextColumn("집행상태", width="small"),
                            "score_weight_pct": st.column_config.NumberColumn("후보내비중", format="%.1f", width="small"),
                            "planned_amount": st.column_config.TextColumn("배정금액", width="small"),
                            "ref_price": st.column_config.TextColumn("현재가", width="small"),
                            "estimated_units": st.column_config.NumberColumn("예상수량", format="%d", width="small"),
                            "first_amount": st.column_config.TextColumn("1차금액", width="small"),
                            "second_amount": st.column_config.TextColumn("2차금액", width="small"),
                            "third_amount": st.column_config.TextColumn("3차금액", width="small"),
                            "buy_now_limit": st.column_config.TextColumn("진입가", width="small"),
                            "add_on_pullback": st.column_config.TextColumn("눌림추가", width="small"),
                            "stop_loss": st.column_config.TextColumn("손절/보류", width="small"),
                            "reentry_above": st.column_config.TextColumn("재진입", width="small"),
                            "target_1": st.column_config.TextColumn("1차목표", width="small"),
                            "target_2": st.column_config.TextColumn("2차목표", width="small"),
                            "execution_rule": st.column_config.TextColumn("실행 원칙", width="large"),
                            "allocation_reason": st.column_config.TextColumn("배정근거", width="large"),
                            "quality_note": st.column_config.TextColumn("품질메모", width="medium"),
                            "reason": st.column_config.TextColumn("핵심 사유", width="large"),
                            },
                        )
                with st.expander("한국 직접 선택 플랜"):
                    st.caption("유형별로 직접 고릅니다.")
                    kr_selection_map: dict[str, str] = {}
                    for bucket in kr_plan["bucket"].astype(str).tolist():
                        bucket_detail = kr_detail[kr_detail["bucket"].astype(str) == bucket].copy()
                        if bucket_detail.empty:
                            continue
                        option_labels = {
                            f"{str(row.get('ticker', ''))} | {str(row.get('name', ''))} | "
                            f"{str(row.get('setup', '')) or '기본'} | {_safe_int(row.get('score', 0))}점": str(row.get("ticker", ""))
                            for _, row in bucket_detail.iterrows()
                        }
                        default_label = next(iter(option_labels.keys()))
                        picked_label = st.selectbox(
                            f"{bucket} 직접 선택",
                            options=list(option_labels.keys()),
                            index=0,
                            key=f"budget_pick_kr_{bucket}",
                        )
                        kr_selection_map[bucket] = option_labels.get(picked_label, option_labels[default_label])

                    picked_kr = _build_selected_budget_actions(
                        market="KR",
                        plan_df=kr_plan,
                        detail_df=kr_detail,
                        selection_map=kr_selection_map,
                    )
                    if not picked_kr.empty:
                        _show_table(
                            picked_kr,
                            plain_numeric_columns=["planned_amount", "executable_amount", "reserve_amount"],
                            currency_columns=[
                                "ref_price",
                                "split_1",
                                "split_2",
                                "split_3",
                                "buy_now_limit",
                                "add_on_pullback",
                                "stop_loss",
                                "reentry_above",
                                "target_1",
                                "target_2",
                            ],
                            default_market="KR",
                            column_config={
                                "market": st.column_config.TextColumn("시장", width="small"),
                                "bucket": st.column_config.TextColumn("유형", width="small"),
                                "ticker": st.column_config.TextColumn("티커", width="small"),
                                "name": st.column_config.TextColumn("종목명", width="small"),
                                "setup": st.column_config.TextColumn("세팅", width="small"),
                                "score": st.column_config.NumberColumn("점수", format="%d", width="small"),
                                "confidence_view": st.column_config.TextColumn("신뢰도", width="small"),
                                "execution_status": st.column_config.TextColumn("집행상태", width="small"),
                                "planned_amount": st.column_config.TextColumn("배정금액", width="small"),
                                "executable_amount": st.column_config.TextColumn("집행금액", width="small"),
                                "reserve_amount": st.column_config.TextColumn("현금보류", width="small"),
                                "ref_price": st.column_config.TextColumn("현재가", width="small"),
                                "estimated_units": st.column_config.NumberColumn("예상수량", format="%d", width="small"),
                                "split_1": st.column_config.TextColumn("1차", width="small"),
                                "split_2": st.column_config.TextColumn("2차", width="small"),
                                "split_3": st.column_config.TextColumn("3차", width="small"),
                                "buy_now_limit": st.column_config.TextColumn("진입가", width="small"),
                                "add_on_pullback": st.column_config.TextColumn("눌림추가", width="small"),
                                "stop_loss": st.column_config.TextColumn("손절/보류", width="small"),
                                "reentry_above": st.column_config.TextColumn("재진입", width="small"),
                                "target_1": st.column_config.TextColumn("1차목표", width="small"),
                                "target_2": st.column_config.TextColumn("2차목표", width="small"),
                                "horizon": st.column_config.TextColumn("기간", width="small"),
                                "execution_rule": st.column_config.TextColumn("실행 원칙", width="large"),
                                "allocation_reason": st.column_config.TextColumn("배정근거", width="large"),
                                "quality_note": st.column_config.TextColumn("품질메모", width="medium"),
                                "note": st.column_config.TextColumn("운용 메모", width="large"),
                                "reason": st.column_config.TextColumn("핵심 사유", width="large"),
                            },
                        )


def render_news_event_tab() -> None:
    st.subheader("뉴스 / 이벤트 모니터")
    st.caption("실적, 배당락, 뉴스 흐름을 한 번에 봅니다.")

    run_col, hint_col = st.columns([1, 3])
    with run_col:
        if st.button("뉴스/이벤트 조회", width="stretch"):
            st.session_state["news_event_ready"] = True
    with hint_col:
        st.caption("버튼을 눌러야 조회합니다.")

    if not st.session_state.get("news_event_ready", False):
        st.info("필요할 때 `뉴스/이벤트 조회`.")
        return

    candidates: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    for row in st.session_state.portfolio.to_dict("records"):
        market = str(row.get("market", "")).strip().upper()
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker or (market, ticker) in seen:
            continue
        seen.add((market, ticker))
        candidates.append(
            {
                "market": market,
                "ticker": ticker,
                "name": str(row.get("name", "") or ""),
                "source": "보유 종목",
            }
        )

    tracked = load_manual_tracking()
    if not tracked.empty:
        for row in tracked.to_dict("records"):
            market = str(row.get("market", "")).strip().upper()
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker or (market, ticker) in seen:
                continue
            seen.add((market, ticker))
            candidates.append(
                {
                    "market": market,
                    "ticker": ticker,
                    "name": str(row.get("name", "") or ""),
                    "source": "관심 추적",
                }
            )

    min_score = int(st.session_state.scanner_settings["min_score"])
    top_n = min(6, int(st.session_state.scanner_settings["top_n"]))
    scan_limit = int(st.session_state.scanner_settings["scan_limit"])
    for market in ["US", "KR"]:
        scan = get_today_scan_state(
            market=market,
            watchlist_key=_watchlist_cache_key(market),
            min_score=min_score,
            scan_limit=scan_limit,
        )
        if scan.empty:
            continue
        for row in scan.head(top_n).to_dict("records"):
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker or (market, ticker) in seen:
                continue
            seen.add((market, ticker))
            candidates.append(
                {
                    "market": market,
                    "ticker": ticker,
                    "name": str(row.get("name", "") or ""),
                    "source": "오늘 후보",
                }
            )

    if not candidates:
        st.info("뉴스/이벤트를 확인할 종목이 아직 없습니다.")
        return

    rows: list[dict[str, object]] = []
    for item in candidates:
        ticker = str(item["ticker"])
        market = str(item["market"])
        event_summary = get_stock_event_summary(ticker)
        news_summary = get_stock_news_summary(ticker)
        quote = get_latest_quote(ticker)
        rows.append(
            {
                "market": market,
                "ticker": ticker,
                "name": item["name"],
                "source": item["source"],
                "current_price": pd.to_numeric(quote.get("current_price", None), errors="coerce"),
                "change_pct": pd.to_numeric(quote.get("change_pct", None), errors="coerce"),
                "quote_as_of": str(quote.get("as_of", "")),
                "event_risk": str(event_summary.get("event_risk", "")),
                "earnings_date": str(event_summary.get("earnings_date", "")),
                "ex_dividend_date": str(event_summary.get("ex_dividend_date", "")),
                "event_note": str(event_summary.get("event_note", "")),
                "news_bias": str(news_summary.get("news_bias", "중립")),
                "news_count": int(news_summary.get("news_count", 0) or 0),
                "headline": str(news_summary.get("headline", "")),
                "news_note": str(news_summary.get("news_note", "")),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        st.info("뉴스/이벤트 데이터가 아직 부족합니다.")
        return

    urgent = frame[frame["event_risk"] == "높음"]
    positive = frame[frame["news_bias"] == "긍정"]
    negative = frame[frame["news_bias"] == "부정"]
    a, b, c = st.columns(3)
    a.metric("가까운 핵심 일정", f"{len(urgent)}")
    b.metric("긍정 뉴스 흐름", f"{len(positive)}")
    c.metric("부정 뉴스 흐름", f"{len(negative)}")

    _show_table(
        frame[
            [
                "market",
                "ticker",
                "name",
                "source",
                "current_price",
                "change_pct",
                "event_risk",
                "earnings_date",
                "ex_dividend_date",
                "news_bias",
                "news_count",
                "quote_as_of",
                "event_note",
                "news_note",
            ]
        ].sort_values(by=["event_risk", "news_count", "ticker"], ascending=[True, False, True]),
        datetime_columns=["quote_as_of"],
        currency_columns=["current_price"],
        column_config={
            "market": st.column_config.TextColumn("시장", width="small"),
            "ticker": st.column_config.TextColumn("티커", width="small"),
            "name": st.column_config.TextColumn("종목명", width="small"),
            "source": st.column_config.TextColumn("출처", width="small"),
            "current_price": st.column_config.TextColumn("현재가", width="small"),
            "change_pct": st.column_config.NumberColumn("당일등락", format="%.2f", width="small"),
            "event_risk": st.column_config.TextColumn("이벤트리스크", width="small"),
            "earnings_date": st.column_config.TextColumn("실적예정", width="small"),
            "ex_dividend_date": st.column_config.TextColumn("배당락일", width="small"),
            "news_bias": st.column_config.TextColumn("뉴스흐름", width="small"),
            "news_count": st.column_config.NumberColumn("뉴스건수", format="%d", width="small"),
            "quote_as_of": st.column_config.TextColumn("시세기준", width="small"),
            "event_note": st.column_config.TextColumn("이벤트 메모", width="medium"),
            "news_note": st.column_config.TextColumn("뉴스 해석", width="medium"),
        },
    )

    with st.expander("최근 헤드라인 보기"):
        for _, row in frame.iterrows():
            st.markdown(f"**{row['ticker']} {row['name']}**")
            if str(row.get("headline", "")).strip():
                st.write(f"- 헤드라인: {row['headline']}")
            st.write(f"- 이벤트: {row['event_note']}")
            st.write(f"- 뉴스 해석: {row['news_note']}")


def render_market_scanner() -> None:
    st.subheader("오늘 매수 후보 스캐너")
    st.caption("오늘 진입했을 때 이번 주 또는 근시일 안의 상승 가능성이 높은 최근 차트 후보를 점수화합니다.")
    min_score = int(st.session_state.scanner_settings["min_score"])
    scan_limit = min(30, int(st.session_state.scanner_settings["scan_limit"]))
    st.caption(f"상세 스캐너는 화면 멈춤을 막기 위해 이번 실행에서 시장별 최대 {scan_limit}개만 확인합니다.")
    us_regime = classify_market_regime("US")
    kr_regime = classify_market_regime("KR")

    us_scan = get_today_scan_state(
        market="US",
        watchlist_key=_watchlist_cache_key("US"),
        min_score=min_score,
        scan_limit=scan_limit,
    )
    kr_scan = get_today_scan_state(
        market="KR",
        watchlist_key=_watchlist_cache_key("KR"),
        min_score=min_score,
        scan_limit=scan_limit,
    )

    us_tab, kr_tab = st.tabs(["미국", "한국"])
    with us_tab:
        st.caption(f"장세: {us_regime.regime} / {us_regime.note}")
        if us_scan.empty:
            st.info("조건을 만족한 미국 후보가 없습니다.")
        else:
            us_view = _prepare_recommendation_execution_view(us_scan, include_market=True, default_market="US")
            _show_table(
                us_view,
                datetime_columns=["quote_as_of"],
                currency_columns=["current_price", "entry_price", "stop_loss", "target_1", "target_2", "target_3"],
                default_market="US",
                column_config=_candidate_column_config(),
            )
            _render_manual_tracking_quick_add(us_scan, source_label="오늘추천", key_prefix="today_scan_us", default_market="US")

    with kr_tab:
        st.caption(f"장세: {kr_regime.regime} / {kr_regime.note}")
        if kr_scan.empty:
            st.info("조건을 만족한 한국 후보가 없습니다.")
        else:
            kr_view = _prepare_recommendation_execution_view(kr_scan, include_market=True, default_market="KR")
            _show_table(
                kr_view,
                datetime_columns=["quote_as_of"],
                currency_columns=["current_price", "entry_price", "stop_loss", "target_1", "target_2", "target_3"],
                default_market="KR",
                column_config=_candidate_column_config(),
            )
            _render_manual_tracking_quick_add(kr_scan, source_label="오늘추천", key_prefix="today_scan_kr", default_market="KR")

    combined = pd.concat([us_scan.assign(market="US"), kr_scan.assign(market="KR")], ignore_index=True)
    if not combined.empty:
        remembered_markets: list[str] = []
        for market_name, frame in [("US", us_scan), ("KR", kr_scan)]:
            if _remember_recommendation_snapshot("today_scan", market_name, frame):
                remembered_markets.append(market_name)
        if remembered_markets:
            st.caption(f"학습 메모리 자동 저장: {', '.join(remembered_markets)} 오늘 추천")
        if st.button("오늘 추천 스냅샷 저장", width="stretch"):
            for market_name, frame in [("US", us_scan), ("KR", kr_scan)]:
                if not frame.empty:
                    append_scan_history("today_scan", market_name, frame)
            st.success("오늘 추천 결과를 누적 저장했습니다.")

        st.download_button(
            "추천 결과 CSV 다운로드",
            data=combined.to_csv(index=False).encode("utf-8-sig"),
            file_name="today_candidates.csv",
            mime="text/csv",
            width="stretch",
        )


def render_buy_now_panel() -> None:
    st.subheader("오늘 바로 볼 종목")
    st.caption("오늘 진입 검토가 가능한 후보와 대안을 봅니다.")
    min_score = int(st.session_state.scanner_settings["min_score"])
    top_n = int(st.session_state.scanner_settings["top_n"])
    scan_limit = min(12, int(st.session_state.scanner_settings["scan_limit"]))
    st.caption(f"빠른 추천 모드: 시장별 상위 후보 {scan_limit}개를 먼저 확인합니다.")

    us_scan = get_today_scan_state(
        market="US",
        watchlist_key=_watchlist_cache_key("US"),
        min_score=min_score,
        scan_limit=scan_limit,
    )
    kr_scan = get_today_scan_state(
        market="KR",
        watchlist_key=_watchlist_cache_key("KR"),
        min_score=min_score,
        scan_limit=scan_limit,
    )
    combined = pd.concat([us_scan.assign(market="US"), kr_scan.assign(market="KR")], ignore_index=True)

    if combined.empty:
        st.info("오늘 조건을 만족한 종목이 없습니다.")
        return

    top = combined.sort_values(by=["score", "ticker"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    best_row = top.iloc[0]
    col1, col2, col3 = st.columns(3)
    best_market = str(best_row.get("market", "") or "")
    best_label = f"{best_market} {best_row['ticker']}".strip()
    col1.metric("최상위 후보", best_label)
    col2.metric("최고 판단", str(best_row["score_view"]))
    col3.metric("후보 수", len(combined))
    _render_action_deck(top.assign(bucket="오늘추천"), title="오늘 결정 카드", limit=4)
    _show_table(
        _prepare_recommendation_execution_view(top),
        datetime_columns=["quote_as_of"],
        currency_columns=["current_price", "entry_price", "stop_loss", "target_1", "target_2", "target_3"],
        column_config=_candidate_column_config(),
    )
    _render_manual_tracking_quick_add(top, source_label="오늘바로볼종목", key_prefix="buy_now_top")

    st.markdown("#### 급등 후보 보드")
    momentum_board = combined.sort_values(
        by=["volume_score", "breakout_score", "momentum_score", "score"],
        ascending=[False, False, False, False],
    ).head(top_n)
    _show_table(
        _prepare_recommendation_execution_view(momentum_board),
        datetime_columns=["quote_as_of"],
        currency_columns=["current_price", "entry_price", "stop_loss", "target_1", "target_2", "target_3"],
        column_config=_candidate_column_config(),
    )
    _render_manual_tracking_quick_add(momentum_board, source_label="급등후보보드", key_prefix="momentum_board")

    if not st.session_state.portfolio.empty:
        portfolio_scores: list[int] = []
        for row in st.session_state.portfolio.to_dict("records"):
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            data = get_stock_data(ticker)
            if data.empty:
                continue
            result = analyze_position(
                data=data,
                position=PositionInput(
                    ticker=ticker,
                    quantity=float(row.get("quantity", 0) or 0),
                    avg_price=float(row.get("avg_price", 0) or 0),
                    cash_budget=float(row.get("cash_budget", 0) or 0),
                    target_weight=float(row.get("target_weight", 0) or 0),
                ),
            )
            portfolio_scores.append(result.score)

        if portfolio_scores:
            avg_score = sum(portfolio_scores) / len(portfolio_scores)
            stronger = top[top["score"] > avg_score]
            st.markdown("#### 보유 종목보다 강한 후보")
            if stronger.empty:
                st.write("현재 보유 종목 평균 점수보다 확실히 강한 후보는 많지 않습니다.")
            else:
                _show_table(
                    _prepare_recommendation_execution_view(stronger),
                    datetime_columns=["quote_as_of"],
                    currency_columns=["current_price", "entry_price", "stop_loss", "target_1", "target_2", "target_3"],
                    column_config=_candidate_column_config(),
                )


def render_today_recommendation_page() -> None:
    st.subheader("오늘 추천")
    _render_recommendation_definition("today")
    st.caption("버튼을 눌러 오늘 후보를 계산합니다.")

    if not _render_data_freshness_gate(purpose="오늘 추천", require_intraday=False):
        st.info("최신 일봉 기준이 확인될 때까지 오늘 추천을 표시하지 않습니다.")
        return

    run_col, hint_col = st.columns([1, 3])
    with run_col:
        if st.button("오늘 추천 계산", width="stretch"):
            st.session_state["today_recommendation_ready"] = True
            st.session_state["today_detail_scanner_ready"] = False
            get_today_scan_state.clear()
            st.rerun()
    with hint_col:
        st.caption("전체 스캔은 버튼 실행.")

    if not st.session_state.get("today_recommendation_ready", False):
        st.info("오늘 추천을 보려면 `오늘 추천 계산`을 눌러 주세요.")
        return

    with st.spinner("오늘 바로 볼 종목을 빠르게 계산하는 중입니다. 시장별 12개 이하만 먼저 확인합니다."):
        _safe_render("오늘 바로 볼 종목", render_buy_now_panel)

    st.divider()
    detail_col, detail_hint = st.columns([1, 3])
    with detail_col:
        if st.button("상세 스캐너 실행", width="stretch", key="today_detail_scanner_button"):
            st.session_state["today_detail_scanner_ready"] = True
            st.rerun()
    with detail_hint:
        st.caption("상세 스캐너는 추가 후보를 더 보려는 경우에만 따로 실행합니다.")
    if st.session_state.get("today_detail_scanner_ready", False):
        with st.spinner("상세 스캐너 실행 중입니다. 조건 통과/탈락 결과를 정리합니다."):
            _safe_render("오늘 추천 스캐너", render_market_scanner)


def render_new_recommendation_hub() -> None:
    st.subheader("신규 추천")
    st.caption("추천 방향만 고르면 같은 추천 엔진과 예산 기준으로 종목, 진입가, 손절가, 분할매도를 봅니다.")
    direction = st.radio(
        "추천 방향",
        ["오늘 당장", "단타", "장기 보유", "배당주", "예산으로 추천"],
        horizontal=True,
        label_visibility="collapsed",
        key="recommendation_direction",
    )

    if direction == "오늘 당장":
        _safe_render("오늘 추천", render_today_recommendation_page)
        with st.expander("실시간 급등주 스캐너 열기", expanded=False):
            _safe_render("실시간", render_realtime_tab)
    elif direction == "단타":
        _safe_render("단타", render_short_term_trade_tab)
    elif direction == "장기 보유":
        if _render_data_freshness_gate(purpose="장기 보유 추천", require_intraday=False):
            _safe_render("장기/전략 추천", render_strategy_profiles_tab)
    elif direction == "배당주":
        if _render_data_freshness_gate(purpose="배당주 추천", require_intraday=False):
            _safe_render("배당주", render_dividend_tab)
    elif direction == "예산으로 추천":
        _safe_render("예산 플래너", render_budget_planner)


def render_watchlist_editor() -> None:
    st.subheader("추천 후보군 편집")
    st.caption("스캔 후보 목록을 관리합니다.")

    st.markdown("#### 데이터 품질 점검")
    audit_limit = st.slider("시장별 점검 종목 수", 20, 200, 80, 20, key="data_quality_audit_limit")
    if st.button("현재 기준 데이터 점검", width="stretch", key="run_data_quality_audit"):
        with st.spinner("유니버스/가격 최신성을 점검하는 중입니다."):
            summary, detail = _build_data_quality_summary(int(audit_limit))
        if summary.empty:
            st.warning("점검할 데이터가 없습니다.")
        else:
            _show_table(
                summary,
                column_config={
                    "market": st.column_config.TextColumn("시장", width="small"),
                    "status": st.column_config.TextColumn("상태", width="medium"),
                    "count": st.column_config.NumberColumn("종목수", format="%d", width="small"),
                },
            )
            bad = detail[detail["status"] != "정상"].copy()
            if bad.empty:
                st.success("점검 샘플에서는 오래되거나 거래중단 의심 데이터가 없습니다.")
            else:
                st.error(f"오래됨/거래중단 의심 또는 데이터 없음: {len(bad)}개. 이 종목들은 추천 후보에서 제외됩니다.")
                _show_table(
                    bad.head(40),
                    column_config={
                        "market": st.column_config.TextColumn("시장", width="small"),
                        "ticker": st.column_config.TextColumn("티커", width="small"),
                        "name": st.column_config.TextColumn("종목명", width="medium"),
                        "latest_date": st.column_config.TextColumn("최근가격일", width="small"),
                        "rows": st.column_config.NumberColumn("행수", format="%d", width="small"),
                        "status": st.column_config.TextColumn("상태", width="medium"),
                    },
                )

    us_df = pd.DataFrame(st.session_state.watchlists["US"])
    kr_df = pd.DataFrame(st.session_state.watchlists["KR"])
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 미국 후보군")
        edited_us = st.data_editor(
            us_df,
            num_rows="dynamic",
            width="stretch",
            key="us_watchlist_editor",
            column_config={
                "ticker": st.column_config.TextColumn("티커", required=True),
                "name": st.column_config.TextColumn("종목명"),
            },
        )

    with col2:
        st.markdown("#### 한국 후보군")
        edited_kr = st.data_editor(
            kr_df,
            num_rows="dynamic",
            width="stretch",
            key="kr_watchlist_editor",
            column_config={
                "ticker": st.column_config.TextColumn("티커", required=True),
                "name": st.column_config.TextColumn("종목명"),
            },
        )

    left, center, right = st.columns([1, 1, 1])
    with left:
        if st.button("후보군 저장", width="stretch"):
            st.session_state.watchlists = {
                "US": normalize_watchlist_frame(edited_us),
                "KR": normalize_watchlist_frame(edited_kr),
            }
            save_watchlists(st.session_state.watchlists)
            st.success("추천 후보군을 저장했습니다.")
    with center:
        if st.button("후보군 다시 불러오기", width="stretch"):
            loaded = load_watchlists()
            if loaded is None:
                st.warning("저장된 후보군 파일이 아직 없습니다.")
            else:
                st.session_state.watchlists = loaded
                st.rerun()
    with right:
        if st.button("기본 후보군 복원", width="stretch"):
            st.session_state.watchlists = get_default_watchlists()
            st.rerun()


def render_backtest_tab(ticker: str) -> None:
    st.subheader("백테스트")
    st.caption("현재 점수 로직을 기준으로 간단한 과거 성과를 확인합니다.")

    col1, col2, col3 = st.columns(3)
    with col1:
        initial_cash = st.number_input("초기 자금", min_value=100000, value=10000000, step=100000)
    with col2:
        buy_threshold = st.slider("매수 점수", min_value=55, max_value=90, value=70, step=1)
    with col3:
        sell_threshold = st.slider("매도 점수", min_value=20, max_value=60, value=45, step=1)

    data = get_stock_data(ticker)
    if data.empty:
        st.info("백테스트용 데이터를 불러오지 못했습니다.")
        return

    result, metrics = run_backtest(
        data=data,
        initial_cash=float(initial_cash),
        buy_threshold=int(buy_threshold),
        sell_threshold=int(sell_threshold),
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("최종 자산", f"{metrics.final_value:,.0f}")
    m2.metric("총 수익률", f"{metrics.total_return_pct:.2f}%")
    m3.metric("CAGR", f"{metrics.cagr_pct:.2f}%")
    m4.metric("MDD", f"{metrics.mdd_pct:.2f}%")
    m5.metric("승률", f"{metrics.win_rate_pct:.2f}%")
    st.caption(f"총 거래 횟수: {metrics.trade_count}")

    chart_df = result[["Date", "Close", "equity", "signal_score"]].copy()
    st.line_chart(chart_df.set_index("Date")[["Close", "equity"]], width="stretch")
    st.dataframe(chart_df.tail(30), width="stretch")


def render_realtime_tab() -> None:
    st.subheader("실시간 급등주 스캐너")
    st.caption("분봉 기준 급등/돌파/VWAP 확인.")

    if not _render_data_freshness_gate(purpose="실시간 스캐너", require_intraday=True):
        st.info("최신 분봉 기준이 확인될 때까지 실시간 추천을 표시하지 않습니다.")
        return

    interval = str(st.session_state.realtime_settings["interval"])
    min_score = int(st.session_state.realtime_settings["min_score"])

    left, right = st.columns([1, 1])
    with left:
        if st.button("실시간 스캔 실행/새로고침", width="stretch"):
            st.session_state["realtime_scan_ready"] = True
            st.session_state["realtime_force_refresh"] = True
            st.session_state["realtime_requested_at"] = pd.Timestamp.now(tz="Asia/Seoul").isoformat(timespec="seconds")
            st.session_state["cache_only_mode"] = False
            get_intraday_stock_data.clear()
            get_latest_quote.clear()
            _build_market_data_basis.clear()
            st.rerun()
    with right:
        st.write(f"현재 기준: `{interval}` / 최소점수 `{min_score}`")

    if not st.session_state.get("realtime_scan_ready", False):
        st.info("분봉 스캔은 버튼 실행.")
        return

    learning_adjustments, _, event_adjustments, news_adjustments, _ = get_learning_state()
    _, _, _, _, pattern_stats = get_tracking_state()
    pattern_lookup = _build_pattern_lookup(pattern_stats)
    us_regime = classify_market_regime("US")
    kr_regime = classify_market_regime("KR")
    force_refresh = bool(st.session_state.pop("realtime_force_refresh", False))
    realtime_limit = min(20, int(st.session_state.scanner_settings["scan_limit"]))
    us_universe = st.session_state.watchlists["US"][:realtime_limit]
    kr_universe = st.session_state.watchlists["KR"][:realtime_limit]
    requested_at = str(st.session_state.get("realtime_requested_at", "") or "")
    if requested_at:
        st.caption(f"실시간 요청시각: {_compact_timestamp(requested_at)} · 이번 실행은 시장별 최대 {realtime_limit}개를 새 분봉 기준으로 확인합니다.")

    with st.spinner("미국 실시간 분봉을 새로 조회하는 중입니다."):
        us_scan = scan_intraday_market(
            "US",
            us_universe,
            interval=interval,
            min_score=min_score,
            force_refresh=force_refresh,
            learning_adjustments=learning_adjustments,
            event_adjustments=event_adjustments,
            news_adjustments=news_adjustments,
        )
    with st.spinner("한국 실시간 분봉을 새로 조회하는 중입니다."):
        kr_scan = scan_intraday_market(
            "KR",
            kr_universe,
            interval=interval,
            min_score=min_score,
            force_refresh=force_refresh,
            learning_adjustments=learning_adjustments,
            event_adjustments=event_adjustments,
            news_adjustments=news_adjustments,
        )
    us_scan = _enrich_recommendation_frame(us_scan, scan_type="realtime_scan", market="US", pattern_lookup=pattern_lookup)
    kr_scan = _enrich_recommendation_frame(kr_scan, scan_type="realtime_scan", market="KR", pattern_lookup=pattern_lookup)
    us_scan = _attach_latest_quotes(us_scan, default_market="US", force_refresh=force_refresh)
    kr_scan = _attach_latest_quotes(kr_scan, default_market="KR", force_refresh=force_refresh)

    us_tab, kr_tab = st.tabs(["미국", "한국"])
    with us_tab:
        st.caption(f"장세: {us_regime.regime} / {us_regime.note}")
        if us_scan.empty:
            st.info("실시간 조건을 만족한 미국 후보가 없습니다.")
        else:
            us_view = us_scan[
                [
                    "ticker",
                    "name",
                    "setup",
                    "score_view",
                    "score_band",
                    "prediction_view",
                    "delta_view",
                    "recent_hit_rate_20d",
                    "recent_target_rate_5d",
                    "score",
                    "current_price",
                    "change_pct",
                    "quote_as_of",
                    "event_risk",
                    "earnings_date",
                    "ex_dividend_date",
                    "news_bias",
                    "news_count",
                    "regime",
                    "regime_delta",
                    "context_delta",
                    "learning_delta",
                    "rs_score",
                    "atr_pct",
                    "volume_ratio",
                    "reason",
                ]
            ]
            _show_table(
                us_view,
                datetime_columns=["quote_as_of"],
                currency_columns=["current_price"],
                default_market="US",
                column_config=_candidate_column_config(),
            )
            _render_manual_tracking_quick_add(us_scan, source_label="실시간", key_prefix="realtime_us", default_market="US")

    with kr_tab:
        st.caption(f"장세: {kr_regime.regime} / {kr_regime.note}")
        if kr_scan.empty:
            st.info("실시간 조건을 만족한 한국 후보가 없습니다.")
        else:
            kr_view = kr_scan[
                [
                    "ticker",
                    "name",
                    "setup",
                    "score_view",
                    "score_band",
                    "prediction_view",
                    "delta_view",
                    "recent_hit_rate_20d",
                    "recent_target_rate_5d",
                    "score",
                    "current_price",
                    "change_pct",
                    "quote_as_of",
                    "event_risk",
                    "earnings_date",
                    "ex_dividend_date",
                    "news_bias",
                    "news_count",
                    "regime",
                    "regime_delta",
                    "context_delta",
                    "learning_delta",
                    "rs_score",
                    "atr_pct",
                    "volume_ratio",
                    "reason",
                ]
            ]
            _show_table(
                kr_view,
                datetime_columns=["quote_as_of"],
                currency_columns=["current_price"],
                default_market="KR",
                column_config=_candidate_column_config(),
            )
            _render_manual_tracking_quick_add(kr_scan, source_label="실시간", key_prefix="realtime_kr", default_market="KR")

    combined = pd.concat([us_scan.assign(market="US"), kr_scan.assign(market="KR")], ignore_index=True)
    if combined.empty:
        return
    remembered_markets: list[str] = []
    for market_name, frame in [("US", us_scan), ("KR", kr_scan)]:
        if _remember_recommendation_snapshot("realtime_scan", market_name, frame):
            remembered_markets.append(market_name)
    if remembered_markets:
        st.caption(f"학습 메모리 자동 저장: {', '.join(remembered_markets)} 실시간 추천")
    _render_action_deck(combined.sort_values(by=["score", "ticker"], ascending=[False, True]), title="실시간 결정 카드", limit=4)

    if st.button("실시간 스냅샷 저장", width="stretch"):
        for market_name, frame in [("US", us_scan), ("KR", kr_scan)]:
            if not frame.empty:
                append_scan_history("realtime_scan", market_name, frame)
        st.success("실시간 후보를 누적 저장했습니다.")

    st.markdown("#### 실시간 탑픽")
    top = combined.head(int(st.session_state.scanner_settings["top_n"]))
    top_view = _prepare_recommendation_execution_view(top, include_market=True, include_source=False)
    _show_table(
        top_view,
        datetime_columns=["quote_as_of"],
        currency_columns=["current_price", "entry_price", "stop_loss", "target_1", "target_2", "target_3"],
        column_config=_candidate_column_config(),
    )


def render_auto_candidates_tab() -> None:
    st.subheader("자동 후보군")
    st.caption("월간/주간/일간/내일 후보 압축.")

    top_n = int(st.session_state.scanner_settings["top_n"])
    button_col, note_col = st.columns([2, 3])
    with button_col:
        if st.button("자동 후보군 계산/새로고침", width="stretch"):
            get_auto_candidate_sets_state.clear()
            st.session_state["auto_candidates_ready"] = True
            st.rerun()
    with note_col:
        st.caption("전체 계산은 버튼 실행. 30분 캐시.")

    if not st.session_state.get("auto_candidates_ready", False):
        history = load_recent_scan_history(limit=20)
        st.info("자동 후보군은 버튼으로 갱신합니다. 최근 스냅샷은 아래에서 확인.")
        st.markdown("#### 누적 스냅샷 이력")
        if not history:
            st.info("저장된 스냅샷이 아직 없습니다.")
        else:
            history_rows = []
            for item in history:
                history_rows.append(
                    {
                        "snapshot_id": item.get("snapshot_id", ""),
                        "scan_type": item.get("scan_type", ""),
                        "market": item.get("market", ""),
                        "row_count": item.get("row_count", len(item.get("rows", []))),
                        "saved_at": item.get("saved_at", ""),
                    }
                )
            history_df = pd.DataFrame(history_rows).sort_values(by="saved_at", ascending=False).reset_index(drop=True)
            _show_table(history_df, datetime_columns=["saved_at"], column_config=_history_column_config())
        return

    us_sets = get_auto_candidate_sets_state("US", top_n=top_n)
    kr_sets = get_auto_candidate_sets_state("KR", top_n=top_n)

    if st.button("자동 후보군 스냅샷 저장", width="stretch"):
        for market, bucket in [("US", us_sets), ("KR", kr_sets)]:
            for scan_type, frame in bucket.items():
                if not frame.empty:
                    append_scan_history(scan_type, market, frame)
        st.success("자동 후보군 스냅샷을 누적 저장했습니다.")

    bucket_labels = {
        "monthly": "월간 후보",
        "weekly": "주간 후보",
        "daily": "일간 후보",
        "next_day": "내일 급등 후보",
    }

    for key, label in bucket_labels.items():
        st.markdown(f"#### {label}")
        left, right = st.columns(2)
        with left:
            st.caption("미국")
            if us_sets[key].empty:
                st.info("미국 후보가 없습니다.")
            else:
                us_view = _prepare_recommendation_execution_view(us_sets[key], include_market=True, default_market="US")
                _show_table(
                    us_view,
                    datetime_columns=["quote_as_of"],
                    currency_columns=["current_price", "entry_price", "stop_loss", "target_1", "target_2", "target_3"],
                    default_market="US",
                    column_config=_candidate_column_config(),
                )
        with right:
            st.caption("한국")
            if kr_sets[key].empty:
                st.info("한국 후보가 없습니다.")
            else:
                kr_view = _prepare_recommendation_execution_view(kr_sets[key], include_market=True, default_market="KR")
                _show_table(
                    kr_view,
                    datetime_columns=["quote_as_of"],
                    currency_columns=["current_price", "entry_price", "stop_loss", "target_1", "target_2", "target_3"],
                    default_market="KR",
                    column_config=_candidate_column_config(),
                )

    history = load_recent_scan_history(limit=20)
    st.markdown("#### 누적 스냅샷 이력")
    if not history:
        st.info("저장된 스냅샷이 아직 없습니다.")
    else:
        history_rows = []
        for item in history:
            history_rows.append(
                {
                    "snapshot_id": item.get("snapshot_id", ""),
                    "scan_type": item.get("scan_type", ""),
                    "market": item.get("market", ""),
                    "row_count": item.get("row_count", len(item.get("rows", []))),
                    "saved_at": item.get("saved_at", ""),
                }
            )
        history_df = pd.DataFrame(history_rows).sort_values(by="saved_at", ascending=False).reset_index(drop=True)
        _show_table(history_df, datetime_columns=["saved_at"], column_config=_history_column_config())


def render_tracking_tab() -> None:
    st.subheader("후보 추적 성과")
    st.caption("저장 후보의 수익/목표/손절을 추적합니다.")

    detail, summary, leaderboard, pending, pattern_stats = get_tracking_state()
    _, learning_df, _, _, context_df = get_learning_state()
    if summary.empty:
        st.info("스냅샷이 없습니다. 자동 후보군에서 먼저 저장.")
    else:
        event_stats, news_stats = _build_event_learning_stats(detail)
        mature_detail = detail[detail["status"] == "평가완료"].copy()
        total_saved = len(detail)
        matured_count = len(mature_detail)
        pending_count = len(detail[detail["status"] == "평가대기"])
        avg_20d = mature_detail["ret_20d_pct"].mean() if not mature_detail.empty else 0.0
        target_first_5d = (
            (((mature_detail["path_5d"] == "목표가 선도달") | (mature_detail["path_5d"] == "목표가 도달")).mean() * 100)
            if not mature_detail.empty
            else 0.0
        )
        stop_first_5d = (
            (((mature_detail["path_5d"] == "손절 선도달") | (mature_detail["path_5d"] == "손절 도달")).mean() * 100)
            if not mature_detail.empty
            else 0.0
        )
        feature_log_count = len(load_recent_feature_log(limit=3000))

        a, b, c, d, e, f, g = st.columns(7)
        a.metric("누적 후보 수", f"{total_saved:,}")
        b.metric("평가 완료", f"{matured_count:,}")
        c.metric("평가 대기", f"{pending_count:,}")
        d.metric("평균 20일 수익률", f"{avg_20d:.2f}%")
        e.metric("5일 목표도달", f"{target_first_5d:.1f}%")
        f.metric("5일 손절도달", f"{stop_first_5d:.1f}%")
        g.metric("최근 피처 로그", f"{feature_log_count:,}")

        st.markdown("#### 추적 요약")
        _show_table(summary, column_config=_tracking_summary_column_config())

        if not learning_df.empty:
            st.markdown("#### 현재 학습 보정")
            st.caption("누적 성과 보정값.")
            _show_table(learning_df.head(20))

        if not context_df.empty:
            st.markdown("#### 이벤트/뉴스 학습 보정")
            st.caption("이벤트/뉴스 보정.")
            _show_table(context_df.head(20))

        if not pattern_stats.empty:
            st.markdown("#### 패턴 적중률")
            st.caption("잘 맞은 유형 통계.")
            _show_table(pattern_stats.head(20), column_config=_pattern_stats_column_config())

        if not event_stats.empty:
            st.markdown("#### 이벤트 리스크 성과")
            st.caption("일정 있는 후보 성과.")
            _show_table(event_stats.head(10))

        if not news_stats.empty:
            st.markdown("#### 뉴스 흐름 성과")
            st.caption("뉴스 흐름별 성과.")
            _show_table(news_stats.head(10))

        if not pending.empty:
            st.markdown("#### 평가 대기 현황")
            _show_table(pending, datetime_columns=["latest_saved_at"])

        if not leaderboard.empty:
            st.markdown("#### 반복 등장 강세 종목")
            st.caption("자주 맞은 종목 기억.")
            _show_table(
                leaderboard.head(20),
                datetime_columns=["latest_saved_at"],
                column_config=_leaderboard_column_config(),
            )

        st.markdown("#### 추적 상세")
        _show_table(
            detail.head(100)[
                [
                    "scan_type",
                    "market",
                    "ticker",
                    "name",
                    "score",
                    "current_price",
                    "entry_price",
                    "stop_loss",
                    "target_1",
                    "setup",
                    "action",
                    "event_risk",
                    "news_bias",
                    "ret_1d_pct",
                    "ret_3d_pct",
                    "ret_5d_pct",
                    "ret_20d_pct",
                    "max_5d_pct",
                    "min_5d_pct",
                    "path_5d",
                    "path_20d",
                    "best_forward_pct",
                    "label_5d",
                    "label_20d",
                    "status",
                    "saved_at",
                    "captured_at",
                ]
            ],
            datetime_columns=["saved_at", "captured_at"],
            currency_columns=["current_price", "entry_price", "stop_loss", "target_1"],
            column_config=_tracking_column_config(),
        )

    st.divider()
    st.subheader("장기 복리 후보")
    _render_recommendation_definition("long")
    st.caption("계속 모아갈 만한 장기 추세형 후보입니다.")

    top_n = int(st.session_state.scanner_settings["top_n"])
    us_compounders = build_compounder_candidates("US", top_n=top_n)
    kr_compounders = build_compounder_candidates("KR", top_n=top_n)

    us_tab, kr_tab = st.tabs(["미국 장기 후보", "한국 장기 후보"])
    with us_tab:
        if us_compounders.empty:
            st.info("미국 장기 후보가 없습니다.")
        else:
            us_view = _prepare_recommendation_execution_view(us_compounders, include_market=True, default_market="US")
            _show_table(
                us_view,
                datetime_columns=["quote_as_of"],
                currency_columns=["current_price", "entry_price", "stop_loss", "target_1", "target_2", "target_3"],
                default_market="US",
                column_config=_candidate_column_config(),
            )
    with kr_tab:
        if kr_compounders.empty:
            st.info("한국 장기 후보가 없습니다.")
        else:
            kr_view = _prepare_recommendation_execution_view(kr_compounders, include_market=True, default_market="KR")
            _show_table(
                kr_view,
                datetime_columns=["quote_as_of"],
                currency_columns=["current_price", "entry_price", "stop_loss", "target_1", "target_2", "target_3"],
                default_market="KR",
                column_config=_candidate_column_config(),
            )


def render_manual_tracking_tab() -> None:
    st.subheader("관심 추적")
    st.caption("직접 고른 종목만 따로 추적.")

    tracked = load_manual_tracking()
    detail, _, _, _, _ = get_tracking_state()
    manual_detail = detail[detail["scan_type"] == "manual_track"].copy() if not detail.empty else pd.DataFrame()

    st.markdown("#### 추적 추가")
    load_pool = st.button("추천 후보 불러오기", width="stretch", key="manual_tracking_load_pool")
    if not load_pool:
        pool = pd.DataFrame()
        st.caption("관심 추적 화면은 먼저 빠르게 열고, 후보 검색은 버튼을 눌렀을 때만 실행합니다.")
    else:
        with st.spinner("추천 후보를 불러오는 중입니다. 오늘 추천/실시간/단타 후보를 확인하고 있어요."):
            pool = _build_manual_tracking_pool()

    if load_pool and pool.empty:
        st.info("지금 추가할 수 있는 추천 후보가 없습니다.")
    elif not pool.empty:
        option_map = {
            f"{row['source']} | {row['market']} | {row['ticker']} | {row['name']}": row
            for _, row in pool.iterrows()
        }
        selected_label = st.selectbox("추천 후보에서 고르기", options=list(option_map.keys()))
        memo = st.text_input("메모", placeholder="왜 관심 있는지 간단히 남겨도 됩니다.")
        if st.button("관심 추적에 추가", width="stretch"):
            selected = option_map[selected_label]
            row_dict = selected.to_dict()
            row_dict["memo"] = memo.strip()
            append_manual_tracking(row_dict)
            append_scan_history("manual_track", str(selected["market"]), pd.DataFrame([row_dict]))
            st.success(f"{selected['ticker']}를 관심 추적에 추가했습니다.")
            st.rerun()

    st.markdown("#### 현재 관심 종목")
    if tracked.empty:
        st.info("아직 직접 고른 관심 종목이 없습니다.")
    else:
        tracked_view = tracked.copy()
        if not manual_detail.empty:
            latest_manual = (
                manual_detail.sort_values(by="saved_at", ascending=False)
                .drop_duplicates(subset=["market", "ticker"], keep="first")
                [
                    [
                        "market",
                        "ticker",
                        "ret_3d_pct",
                        "ret_5d_pct",
                        "ret_20d_pct",
                        "path_5d",
                        "path_20d",
                    ]
                ]
            )
            tracked_view = tracked_view.merge(latest_manual, on=["market", "ticker"], how="left")
        tracked_view["score_view"] = tracked_view["score"].apply(
            lambda value: _score_view(_safe_float(value)) if str(value).strip() else ""
        )
        tracked_view["recent_hit_rate_20d"] = 0.0
        tracked_view["recent_target_rate_5d"] = 0.0
        refresh_quotes = st.button("관심 종목 현재가 갱신", width="stretch", key="manual_tracking_quote_refresh")
        if not refresh_quotes:
            st.caption("화면은 빠르게 열고, 현재가/수익률은 필요할 때만 갱신합니다.")
        current_prices: list[float | None] = []
        change_pcts: list[float | None] = []
        current_returns: list[float | None] = []
        alert_statuses: list[str] = []
        quote_as_ofs: list[str] = []
        for _, row in tracked_view.iterrows():
            quote = get_latest_quote(str(row.get("ticker", ""))) if refresh_quotes else {}
            current_price = pd.to_numeric(quote.get("current_price", None), errors="coerce")
            entry_price = pd.to_numeric(row.get("entry_price", None), errors="coerce")
            stop_loss = pd.to_numeric(row.get("stop_loss", None), errors="coerce")
            target_1 = pd.to_numeric(row.get("target_1", None), errors="coerce")

            current_prices.append(None if pd.isna(current_price) else float(current_price))
            change_pcts.append(pd.to_numeric(quote.get("change_pct", None), errors="coerce"))
            if pd.isna(current_price) or pd.isna(entry_price) or float(entry_price) <= 0:
                current_returns.append(None)
            else:
                current_returns.append((float(current_price) - float(entry_price)) / float(entry_price) * 100)
            if refresh_quotes:
                alert_statuses.append(
                    _manual_alert_status(
                        None if pd.isna(current_price) else float(current_price),
                        None if pd.isna(stop_loss) else float(stop_loss),
                        None if pd.isna(target_1) else float(target_1),
                    )
                )
            else:
                alert_statuses.append("갱신 대기")
            quote_as_ofs.append(str(quote.get("as_of", "")))

        tracked_view["current_price"] = current_prices
        tracked_view["change_pct"] = change_pcts
        tracked_view["current_return_pct"] = current_returns
        tracked_view["alert_status"] = alert_statuses
        tracked_view["quote_as_of"] = quote_as_ofs

        active_count = int((tracked_view["alert_status"] == "추적중").sum())
        target_count = int((tracked_view["alert_status"] == "목표도달").sum())
        risk_count = int(tracked_view["alert_status"].isin(["손절근접", "손절구간"]).sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("현재 추적중", f"{active_count}")
        m2.metric("목표 도달", f"{target_count}")
        m3.metric("주의 필요", f"{risk_count}")

        _show_table(
            tracked_view[
                [
                    "market",
                    "ticker",
                    "name",
                    "source",
                    "setup",
                    "score_view",
                    "current_price",
                    "change_pct",
                    "current_return_pct",
                    "entry_price",
                    "stop_loss",
                    "target_1",
                    "ret_3d_pct",
                    "ret_5d_pct",
                    "ret_20d_pct",
                    "path_5d",
                    "path_20d",
                    "alert_status",
                    "quote_as_of",
                    "memo",
                    "created_at",
                ]
            ],
            datetime_columns=["created_at", "quote_as_of"],
            currency_columns=["current_price", "entry_price", "stop_loss", "target_1"],
            column_config=_manual_tracking_column_config(),
        )

        removable = st.selectbox(
            "삭제할 관심 종목",
            options=[""] + [f"{row['tracking_id']} | {row['market']} | {row['ticker']}" for _, row in tracked.iterrows()],
        )
        if removable and st.button("관심 추적에서 제거", width="stretch"):
            tracking_id = removable.split("|")[0].strip()
            remove_manual_tracking(tracking_id)
            st.success("관심 추적 종목을 제거했습니다.")
            st.rerun()


def render_strategy_profiles_tab() -> None:
    st.subheader("전략별 추천")
    _render_recommendation_definition("long")
    st.caption("안정/배당/성장 후보를 장기 보유 관점으로 봅니다.")

    top_n = int(st.session_state.scanner_settings["top_n"])
    us_profiles = get_strategy_profiles_state("US", top_n)
    kr_profiles = get_strategy_profiles_state("KR", top_n)

    sections = [
        ("stable", "안정형 적립"),
        ("dividend", "우량 배당"),
        ("growth", "고위험 성장"),
    ]

    for key, label in sections:
        st.markdown(f"#### {label}")
        left, right = st.columns(2)
        with left:
            st.caption("미국")
            if us_profiles[key].empty:
                st.info("미국 후보가 없습니다.")
            else:
                us_view = _prepare_recommendation_execution_view(us_profiles[key], include_market=True, default_market="US")
                _show_table(
                    us_view,
                    currency_columns=["current_price", "entry_price", "stop_loss", "target_1", "target_2", "target_3"],
                    default_market="US",
                    column_config=_candidate_column_config(),
                )
        with right:
            st.caption("한국")
            if kr_profiles[key].empty:
                st.info("한국 후보가 없습니다.")
            else:
                kr_view = _prepare_recommendation_execution_view(kr_profiles[key], include_market=True, default_market="KR")
                _show_table(
                    kr_view,
                    currency_columns=["current_price", "entry_price", "stop_loss", "target_1", "target_2", "target_3"],
                    default_market="KR",
                    column_config=_candidate_column_config(),
                )

    st.markdown("#### 읽는 기준")
    st.write("- `안정형 적립`: VOO 같은 꾸준한 우상향, 낮은 변동성, 낮은 낙폭 중심")
    st.write("- `우량 배당`: 배당과 추세가 함께 받쳐주는 종목 중심")
    st.write("- `고위험 성장`: 변동성은 높지만 장기 주도주로 커질 가능성이 큰 종목 중심")


def render_dividend_tab() -> None:
    st.subheader("배당주 분석")
    st.caption("배당률, 성장, 배당락, 모아가기 구간.")

    top_n = int(st.session_state.scanner_settings["top_n"])
    us_profiles = get_dividend_profiles_state("US", top_n)
    kr_profiles = get_dividend_profiles_state("KR", top_n)

    us_tab, kr_tab = st.tabs(["미국", "한국"])
    for market_label, market_code, profiles, tab in [
        ("미국", "US", us_profiles, us_tab),
        ("한국", "KR", kr_profiles, kr_tab),
    ]:
        with tab:
            stable_tab, growth_tab = st.tabs(["안정 배당", "배당 성장"])

            with stable_tab:
                if profiles["stable"].empty:
                    st.info(f"{market_label} 안정 배당 후보가 없습니다.")
                else:
                    stable_view = _prepare_recommendation_execution_view(
                        profiles["stable"],
                        include_market=True,
                        default_market=market_code,
                        include_targets=False,
                        extra_columns=[
                            "dividend_yield_pct",
                            "annual_dividend",
                            "dividend_growth_1y_pct",
                            "ex_dividend_date",
                            "accumulate_low",
                            "accumulate_high",
                        ],
                    )
                    _show_table(
                        stable_view,
                        currency_columns=["current_price", "entry_price", "stop_loss", "target_1", "annual_dividend", "accumulate_low", "accumulate_high"],
                        default_market=market_code,
                        column_config={**_candidate_column_config(), **_dividend_column_config()},
                    )

            with growth_tab:
                if profiles["growth"].empty:
                    st.info(f"{market_label} 배당 성장 후보가 없습니다.")
                else:
                    growth_view = _prepare_recommendation_execution_view(
                        profiles["growth"],
                        include_market=True,
                        default_market=market_code,
                        include_targets=False,
                        extra_columns=[
                            "dividend_yield_pct",
                            "annual_dividend",
                            "dividend_growth_1y_pct",
                            "ex_dividend_date",
                            "accumulate_low",
                            "accumulate_high",
                        ],
                    )
                    _show_table(
                        growth_view,
                        currency_columns=["current_price", "entry_price", "stop_loss", "target_1", "annual_dividend", "accumulate_low", "accumulate_high"],
                        default_market=market_code,
                        column_config={**_candidate_column_config(), **_dividend_column_config()},
                    )

            st.markdown("#### 읽는 기준")
            st.write("- `배당수익률`은 최근 1년 배당 기준입니다.")
            st.write("- `배당성장`은 최근 배당이 얼마나 커졌는지 보는 값입니다.")
            st.write("- `모으기하단~상단`: 분할매수 참고 구간.")
            st.write("- `배당락일`: 가까우면 배당락 후 흔들림도 확인.")


def render_short_term_trade_tab() -> None:
    st.subheader("단타 트레이드 플랜")
    _render_recommendation_definition("trade")
    st.caption("단타/고위험 단타의 즉시 진입, 손절, 목표.")

    if not _render_data_freshness_gate(purpose="단타 추천", require_intraday=True):
        st.info("최신 분봉 기준이 확인될 때까지 단타 추천을 표시하지 않습니다.")
        return

    top_n = int(st.session_state.scanner_settings["top_n"])
    interval = str(st.session_state.realtime_settings["interval"])
    min_score = max(60, int(st.session_state.realtime_settings["min_score"]))
    market_sweep_limit = int(st.session_state.scanner_settings["market_sweep_limit"])

    st.markdown("#### 단타 실행 모드 제안")
    _render_scan_advisor(purpose="short_term", key_prefix="short_term")

    run_col, hint_col = st.columns([1, 3])
    with run_col:
        if st.button("단타 후보 계산", width="stretch"):
            st.session_state["short_trade_ready"] = True
    with hint_col:
        st.caption(
            f"현재 기준: `{interval}` / 최소점수 `{min_score}` / 시장 탐색 `{market_sweep_limit}`개. "
            "단타 후보는 분봉 조회가 많아 버튼을 눌렀을 때만 계산합니다."
        )

    if not st.session_state.get("short_trade_ready", False):
        st.info("단타 후보를 보려면 `단타 후보 계산`을 눌러 주세요.")
        return

    learning_adjustments, _, event_adjustments, news_adjustments, _ = get_learning_state()
    _, _, _, _, pattern_stats = get_tracking_state()
    pattern_lookup = _build_pattern_lookup(pattern_stats)
    us_regime = classify_market_regime("US")
    kr_regime = classify_market_regime("KR")
    us_universe = get_market_sweep_universe("US")[: max(1, market_sweep_limit)]
    kr_universe = get_market_sweep_universe("KR")[: max(1, market_sweep_limit)]
    progress_cols = st.columns(4)
    us_trade_progress = progress_cols[0].progress(0.0, text="미국 일반 단타 대기")
    kr_trade_progress = progress_cols[1].progress(0.0, text="한국 일반 단타 대기")
    us_risk_progress = progress_cols[2].progress(0.0, text="미국 고위험 대기")
    kr_risk_progress = progress_cols[3].progress(0.0, text="한국 고위험 대기")

    def _make_progress_callback(label: str, widget):
        def _callback(event: dict[str, object]) -> None:
            total = max(1, int(event.get("total", 1) or 1))
            done = min(total, max(0, int(event.get("done", 0) or 0)))
            ticker = str(event.get("ticker", "") or "")
            stage = str(event.get("stage", "") or "")
            widget.progress(done / total, text=f"{label} {done}/{total} · {stage} · {ticker}")

        return _callback

    us_trades = build_short_term_trade_candidates(
        "US",
        top_n=top_n,
        interval=interval,
        min_score=min_score,
        universe=us_universe,
        progress_callback=_make_progress_callback("US 일반", us_trade_progress),
        learning_adjustments=learning_adjustments,
        event_adjustments=event_adjustments,
        news_adjustments=news_adjustments,
    )
    us_trade_progress.progress(1.0, text="US 일반 단타 완료")
    kr_trades = build_short_term_trade_candidates(
        "KR",
        top_n=top_n,
        interval=interval,
        min_score=min_score,
        universe=kr_universe,
        progress_callback=_make_progress_callback("KR 일반", kr_trade_progress),
        learning_adjustments=learning_adjustments,
        event_adjustments=event_adjustments,
        news_adjustments=news_adjustments,
    )
    kr_trade_progress.progress(1.0, text="KR 일반 단타 완료")
    us_high_risk = build_high_risk_trade_candidates(
        "US",
        top_n=top_n,
        interval=interval,
        min_score=60,
        universe=us_universe,
        progress_callback=_make_progress_callback("US 고위험", us_risk_progress),
        learning_adjustments=learning_adjustments,
        event_adjustments=event_adjustments,
        news_adjustments=news_adjustments,
    )
    us_risk_progress.progress(1.0, text="US 고위험 단타 완료")
    kr_high_risk = build_high_risk_trade_candidates(
        "KR",
        top_n=top_n,
        interval=interval,
        min_score=60,
        universe=kr_universe,
        progress_callback=_make_progress_callback("KR 고위험", kr_risk_progress),
        learning_adjustments=learning_adjustments,
        event_adjustments=event_adjustments,
        news_adjustments=news_adjustments,
    )
    kr_risk_progress.progress(1.0, text="KR 고위험 단타 완료")
    diagnostic_rows = []
    for label, frame in [
        ("US 일반", us_trades),
        ("KR 일반", kr_trades),
        ("US 고위험", us_high_risk),
        ("KR 고위험", kr_high_risk),
    ]:
        stats = frame.attrs.get("diagnostics", {})
        diagnostic_rows.append(
            {
                "scan": label,
                "scanned": int(stats.get("scanned", 0) or 0),
                "daily_pass": int(stats.get("daily_pass", 0) or 0),
                "intraday_pass": int(stats.get("intraday_pass", 0) or 0),
                "selected": int(stats.get("selected", 0) or 0),
                "fallback_selected": int(stats.get("fallback_selected", 0) or 0),
                "errors": int(stats.get("errors", 0) or 0),
            }
        )
    us_trades = _enrich_recommendation_frame(us_trades, scan_type="short_term_trade", market="US", pattern_lookup=pattern_lookup)
    kr_trades = _enrich_recommendation_frame(kr_trades, scan_type="short_term_trade", market="KR", pattern_lookup=pattern_lookup)
    us_high_risk = _enrich_recommendation_frame(us_high_risk, scan_type="high_risk_trade", market="US", pattern_lookup=pattern_lookup)
    kr_high_risk = _enrich_recommendation_frame(kr_high_risk, scan_type="high_risk_trade", market="KR", pattern_lookup=pattern_lookup)
    us_trades = _attach_latest_quotes(us_trades, default_market="US")
    kr_trades = _attach_latest_quotes(kr_trades, default_market="KR")
    us_high_risk = _attach_latest_quotes(us_high_risk, default_market="US")
    kr_high_risk = _attach_latest_quotes(kr_high_risk, default_market="KR")
    with st.expander("스캔 진단", expanded=False):
        st.caption("일봉 선별을 먼저 통과한 종목만 분봉 정밀 분석으로 넘어갑니다.")
        _show_table(
            pd.DataFrame(diagnostic_rows),
            column_config={
                "scan": st.column_config.TextColumn("스캔", width="small"),
                "scanned": st.column_config.NumberColumn("조회", format="%d", width="small"),
                "daily_pass": st.column_config.NumberColumn("일봉통과", format="%d", width="small"),
                "intraday_pass": st.column_config.NumberColumn("분봉통과", format="%d", width="small"),
                "selected": st.column_config.NumberColumn("후보", format="%d", width="small"),
                "fallback_selected": st.column_config.NumberColumn("예비후보", format="%d", width="small"),
                "errors": st.column_config.NumberColumn("오류", format="%d", width="small"),
            },
        )

    combined = pd.concat(
        [
            us_trades.assign(market="US", trade_group="일반단타"),
            kr_trades.assign(market="KR", trade_group="일반단타"),
            us_high_risk.assign(market="US", trade_group="고위험단타"),
            kr_high_risk.assign(market="KR", trade_group="고위험단타"),
        ],
        ignore_index=True,
    )
    if not combined.empty:
        remembered_markets: list[str] = []
        for market_name, frame, scan_type in [
            ("US", us_trades, "short_term_trade"),
            ("KR", kr_trades, "short_term_trade"),
            ("US", us_high_risk, "high_risk_trade"),
            ("KR", kr_high_risk, "high_risk_trade"),
        ]:
            if _remember_recommendation_snapshot(scan_type, market_name, frame):
                remembered_markets.append(f"{market_name} {scan_type}")
        if remembered_markets:
            st.caption(f"학습 메모리 자동 저장: {', '.join(remembered_markets)}")
        _render_action_deck(
            combined.sort_values(by=["score", "ticker"], ascending=[False, True]).assign(bucket=combined.get("trade_group", "")),
            title="단타 결정 카드",
            limit=4,
        )
        if st.button("단타 후보 스냅샷 저장", width="stretch"):
            for market_name, frame, scan_type in [
                ("US", us_trades, "short_term_trade"),
                ("KR", kr_trades, "short_term_trade"),
                ("US", us_high_risk, "high_risk_trade"),
                ("KR", kr_high_risk, "high_risk_trade"),
            ]:
                if not frame.empty:
                    append_scan_history(scan_type, market_name, frame)
            st.success("단타 후보를 누적 저장했습니다.")

    us_tab, kr_tab = st.tabs(["미국", "한국"])
    with us_tab:
        st.caption(f"장세: {us_regime.regime} / {us_regime.note}")
        normal_tab, risky_tab = st.tabs(["일반 단타", "고위험 단타"])
        with normal_tab:
            if us_trades.empty:
                st.info("미국 일반 단타 후보가 없습니다.")
            else:
                us_normal_view = _prepare_trade_execution_view(us_trades)
                _show_table(
                    us_normal_view,
                    datetime_columns=["quote_as_of"],
                    currency_columns=["current_price", "entry_price", "stop_loss", "target_1", "target_2", "target_3"],
                    default_market="US",
                    column_config=_trade_column_config(),
                )
                _render_manual_tracking_quick_add(us_trades, source_label="일반단타", key_prefix="short_trade_us", default_market="US")
        with risky_tab:
            if us_high_risk.empty:
                st.info("미국 고위험 단타 후보가 없습니다.")
            else:
                st.warning("고위험 단타는 변동성이 매우 크니 소액/짧게 보는 전제가 필요합니다.")
                us_risky_view = _prepare_trade_execution_view(us_high_risk, include_risk_level=True)
                _show_table(
                    us_risky_view,
                    datetime_columns=["quote_as_of"],
                    currency_columns=["current_price", "entry_price", "stop_loss", "target_1", "target_2", "target_3"],
                    default_market="US",
                    column_config=_trade_column_config(),
                )
                _render_manual_tracking_quick_add(us_high_risk, source_label="고위험단타", key_prefix="high_risk_us", default_market="US")
    with kr_tab:
        st.caption(f"장세: {kr_regime.regime} / {kr_regime.note}")
        normal_tab, risky_tab = st.tabs(["일반 단타", "고위험 단타"])
        with normal_tab:
            if kr_trades.empty:
                st.info("한국 일반 단타 후보가 없습니다.")
            else:
                kr_normal_view = _prepare_trade_execution_view(kr_trades)
                _show_table(
                    kr_normal_view,
                    datetime_columns=["quote_as_of"],
                    currency_columns=["current_price", "entry_price", "stop_loss", "target_1", "target_2", "target_3"],
                    default_market="KR",
                    column_config=_trade_column_config(),
                )
                _render_manual_tracking_quick_add(kr_trades, source_label="일반단타", key_prefix="short_trade_kr", default_market="KR")
        with risky_tab:
            if kr_high_risk.empty:
                st.info("한국 고위험 단타 후보가 없습니다.")
            else:
                st.warning("고위험 단타는 급등과 급락이 모두 빠르니 손절 기준을 더 엄격하게 봐야 합니다.")
                kr_risky_view = _prepare_trade_execution_view(kr_high_risk, include_risk_level=True)
                _show_table(
                    kr_risky_view,
                    datetime_columns=["quote_as_of"],
                    currency_columns=["current_price", "entry_price", "stop_loss", "target_1", "target_2", "target_3"],
                    default_market="KR",
                    column_config=_trade_column_config(),
                )
                _render_manual_tracking_quick_add(kr_high_risk, source_label="고위험단타", key_prefix="high_risk_kr", default_market="KR")

    if not combined.empty:
        st.markdown("#### 단타 운용 원칙")
        st.write("- `entry_price` 위에서 거래량이 유지될 때만 진입 후보로 봅니다.")
        st.write("- `stop_loss` 이탈 시 손절, `target_1` 도달 시 일부 익절을 기본으로 봅니다.")
        st.write("- `target_2`까지 가면 남은 물량은 추세 보며 정리하는 식으로 씁니다.")


def _safe_render(label: str, render_fn, *args, **kwargs) -> None:
    try:
        render_fn(*args, **kwargs)
    except Exception as exc:
        st.error(f"{label} 화면을 불러오는 중 문제가 생겼습니다. 다른 화면은 계속 사용할 수 있습니다.")
        with st.expander("오류 상세"):
            st.exception(exc)


def render_holding_operation_hub() -> None:
    st.subheader("보유 운용")
    st.caption("내 보유 종목 기준으로 물타기, 불타기, 추가매수, 비중 조절, 예산 배분을 판단합니다.")
    section = st.radio(
        "보유 운용 화면",
        ["보유 입력", "포트폴리오", "물타기/불타기", "보유 기준 예산"],
        horizontal=True,
        label_visibility="collapsed",
        key="holding_operation_section",
    )
    st.caption("보유 종목 판단은 신규 추천/예산 추천과 같은 가격 기준을 참고합니다.")

    if section == "보유 입력":
        _safe_render("보유 종목 입력", render_portfolio_editor)
    elif section == "포트폴리오":
        _safe_render("포트폴리오 분석", render_portfolio_insights)
        _safe_render("포트폴리오 전망", render_portfolio_outlook)
    elif section == "물타기/불타기":
        _safe_render("보유 종목 매매 판단", render_portfolio_analysis)
        st.divider()
        _safe_render("리밸런싱 힌트", render_rebalance_hint)
        st.divider()
        _safe_render("리밸런싱 제안", render_rebalance_suggestions)
    elif section == "보유 기준 예산":
        st.info("보유 종목과 신규 추천을 함께 참고해서 추가 예산 배분을 계산합니다.")
        _safe_render("예산 플래너", render_budget_planner)


def render_portfolio_page() -> None:
    render_holding_operation_hub()


def render_learning_management_hub(ticker: str) -> None:
    st.subheader("학습·관리")
    st.caption("추천 결과가 실제로 어땠는지 추적하고, 후보군/뉴스/백테스트를 관리합니다.")
    section = st.radio(
        "관리 화면",
        ["학습 추적", "관심 종목", "뉴스/이벤트", "백테스트", "후보군 관리", "자동 후보군"],
        horizontal=True,
        label_visibility="collapsed",
        key="learning_management_section",
    )

    if section == "학습 추적":
        _safe_render("추적", render_tracking_tab)
    elif section == "관심 종목":
        _safe_render("관심 추적", render_manual_tracking_tab)
    elif section == "뉴스/이벤트":
        _safe_render("뉴스/이벤트", render_news_event_tab)
    elif section == "백테스트":
        _safe_render("백테스트", render_backtest_tab, ticker)
    elif section == "후보군 관리":
        _safe_render("후보군 관리", render_watchlist_editor)
    elif section == "자동 후보군":
        _safe_render("자동 후보군", render_auto_candidates_tab)


def _build_recent_snapshot_candidates(limit: int = 8) -> pd.DataFrame:
    snapshots = load_recent_scan_history(limit=30)
    rows: list[dict[str, object]] = []
    priority = {
        "realtime_scan": 1,
        "short_term_trade": 2,
        "high_risk_trade": 3,
        "next_day": 4,
        "daily": 5,
        "weekly": 6,
        "monthly": 7,
    }
    for snapshot in snapshots:
        scan_type = str(snapshot.get("scan_type", "") or "")
        market = str(snapshot.get("market", "") or "")
        saved_at = str(snapshot.get("saved_at", "") or "")
        for row in snapshot.get("rows", [])[:5]:
            ticker = str(row.get("ticker", "") or "").strip().upper()
            if not is_tradable_ticker(ticker):
                continue
            rows.append(
                {
                    "market": market,
                    "ticker": ticker,
                    "name": row.get("name", ""),
                    "scan_type": scan_type,
                    "score": pd.to_numeric(row.get("score", 0), errors="coerce"),
                    "current_price": pd.to_numeric(row.get("current_price", row.get("entry_price", None)), errors="coerce"),
                    "entry_price": pd.to_numeric(row.get("entry_price", None), errors="coerce"),
                    "stop_loss": pd.to_numeric(row.get("stop_loss", None), errors="coerce"),
                    "target_1": pd.to_numeric(row.get("target_1", None), errors="coerce"),
                    "reason": row.get("reason", ""),
                    "saved_at": saved_at,
                    "data_basis": "저장 스냅샷",
                    "scan_priority": priority.get(scan_type, 9),
                }
            )

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(by=["scan_priority", "score", "saved_at"], ascending=[True, False, False])
    frame = frame.drop_duplicates(subset=["market", "ticker"], keep="first").head(limit).reset_index(drop=True)
    return frame


def _quote_freshness_label(value: object) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "기준시각 없음"
    now = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)
    age_minutes = max(0, int((now - pd.Timestamp(ts)).total_seconds() // 60))
    if age_minutes <= 20:
        return "방금 갱신"
    if age_minutes <= 180:
        return "오늘 갱신"
    if age_minutes <= 60 * 36:
        return "일봉/전일 기준"
    return "오래된 기준"


def _build_dashboard_checklist(outlook: pd.DataFrame, analysis: pd.DataFrame, recent_candidates: pd.DataFrame) -> list[str]:
    checklist: list[str] = []
    if not outlook.empty:
        if "data_freshness" not in outlook.columns and "quote_as_of" in outlook.columns:
            outlook["data_freshness"] = outlook["quote_as_of"].apply(_quote_freshness_label)
        caution = outlook[outlook["outlook"].isin(["하방 주의"])]
        event_high = outlook[outlook["event_risk"] == "높음"]
        addable = outlook[outlook["action_hint"].isin(["눌림 추가 가능"])]
        if not caution.empty:
            row = caution.iloc[0]
            checklist.append(f"보유 종목: `{row['ticker']}`는 방어 기준부터 확인하세요. {row.get('next_step', '')}")
        if not event_high.empty:
            row = event_high.iloc[0]
            checklist.append(f"뉴스/이벤트: `{row['ticker']}` 일정이 가까워요. {row.get('event_note', '')}")
        if not addable.empty:
            row = addable.sort_values(by=["outlook_score"], ascending=[False]).iloc[0]
            checklist.append(f"추가 후보: `{row['ticker']}`. {row.get('reason_summary', '')}")
    if not analysis.empty:
        high_risk = analysis[analysis["risk_level"] == "높음"]
        if not high_risk.empty:
            row = high_risk.sort_values(by=["weight_pct"], ascending=[False]).iloc[0]
            checklist.append(f"위험 체크: `{row['ticker']}` 비중 {float(row.get('weight_pct', 0) or 0):.1f}%.")
    if not recent_candidates.empty:
        row = recent_candidates.iloc[0]
        basis = str(row.get("data_basis", "저장 스냅샷") or "저장 스냅샷")
        checklist.append(f"강한 후보: `{row['ticker']}` {_safe_int(row.get('score', 0))}점. 기준: {basis}.")
    if not checklist:
        checklist.append("급한 신호 적음. 보유 전망만 확인.")
    return checklist[:5]


def _render_learning_data_stack(db_stats: dict[str, object]) -> None:
    st.markdown("#### 데이터/학습 엔진 상태")
    feature_rows = load_recent_feature_log(limit=3000)
    feature_frame = pd.DataFrame(feature_rows)
    learned_patterns = 0
    if not feature_frame.empty and {"scan_type", "market", "setup"}.issubset(feature_frame.columns):
        learned_patterns = int(feature_frame[["scan_type", "market", "setup"]].drop_duplicates().shape[0])

    try:
        _, tracking_summary, _, _, pattern_stats = get_tracking_state()
    except Exception:
        tracking_summary = pd.DataFrame()
        pattern_stats = pd.DataFrame()

    evaluated_picks = int(pd.to_numeric(tracking_summary.get("picks", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not tracking_summary.empty else 0
    pattern_count = int(len(pattern_stats)) if not pattern_stats.empty else 0

    basis = _build_market_data_basis(include_intraday=True)
    latest_daily = "없음"
    latest_intraday = "없음"
    if not basis.empty:
        daily_basis = basis[basis["kind"] == "일봉"]
        intraday_basis = basis[basis["kind"] == "분봉"]
        if not daily_basis.empty:
            latest_daily = " / ".join(f"{row.market} {row.basis_time}" for row in daily_basis.itertuples())
        if not intraday_basis.empty:
            latest_intraday = " / ".join(f"{row.market} {row.basis_time}" for row in intraday_basis.itertuples())

    cols = st.columns(4)
    cols[0].metric("장기 가격 데이터", "확인")
    cols[0].caption(latest_daily)
    cols[1].metric("실시간 분봉 데이터", "확인")
    cols[1].caption(latest_intraday)
    cols[2].metric("학습 스냅샷", f"{int(db_stats.get('scan_snapshots', 0) or 0):,}")
    cols[3].metric("평가 패턴", f"{pattern_count:,}")

    st.caption(
        "판단 흐름: 장기 일봉/추세/상대강도 -> 실시간 분봉/거래량/VWAP -> 뉴스/이벤트 -> "
        "저장된 추천 성과 학습 순서로 점수와 신뢰도를 보정합니다."
    )
    if int(db_stats.get("scan_snapshots", 0) or 0) == 0:
        st.warning("아직 학습 표본이 부족합니다. 추천 화면에서 후보가 나올 때 자동 저장되고, 시간이 지나면 1일/3일/5일 성과가 학습에 반영됩니다.")
    else:
        st.success(
            f"학습 로그 {int(db_stats.get('feature_entries', 0) or 0):,}개, "
            f"패턴 표본 {learned_patterns:,}개, 평가 후보 {evaluated_picks:,}개를 추천 신뢰도에 반영 중입니다."
        )


def render_daily_dashboard() -> None:
    st.subheader("오늘의 투자 브리핑")
    st.caption("보유, 후보, 추적 상태를 먼저 봅니다.")

    recent_candidates = _build_recent_snapshot_candidates(limit=8)
    db_stats = get_sqlite_cache_stats()
    detail_ready = bool(st.session_state.get("dashboard_detail_ready", False))

    action_col, hint_col = st.columns([1, 3])
    with action_col:
        if st.button("보유 종목 정밀 브리핑 계산", width="stretch", key="dashboard_detail_refresh"):
            st.session_state["dashboard_detail_ready"] = True
            get_portfolio_analysis_state.clear()
            get_portfolio_outlook_state.clear()
            st.rerun()
    with hint_col:
        st.caption("기본 대시보드는 빠르게 열고, 보유 종목 시세/전망 분석은 필요할 때만 계산합니다.")

    collect_col, collect_hint = st.columns([1, 3])
    with collect_col:
        if st.button("오늘 데이터 한방 수집", width="stretch", key="dashboard_collect_all"):
            with st.spinner("오늘/예산/실시간/단타/고위험 단타 후보를 한 번에 수집하고 저장하는 중입니다."):
                st.session_state["dashboard_collect_result"] = _collect_daily_recommendation_snapshots()
            st.success("한방 수집을 완료했습니다.")
    with collect_hint:
        st.caption("매일 한 번 누르면 주요 추천 스냅샷과 학습 로그를 빠른 범위로 쌓습니다. 기본은 시장별 8~16개 후보만 확인합니다.")

    collect_result = st.session_state.get("dashboard_collect_result", pd.DataFrame())
    if isinstance(collect_result, pd.DataFrame) and not collect_result.empty:
        with st.expander("최근 한방 수집 결과", expanded=True):
            _show_table(
                collect_result,
                column_config={
                    "scan_type": st.column_config.TextColumn("수집유형", width="small"),
                    "market": st.column_config.TextColumn("시장", width="small"),
                    "candidates": st.column_config.NumberColumn("후보수", format="%d", width="small"),
                    "saved": st.column_config.TextColumn("저장상태", width="small"),
                },
            )

    st.markdown("#### 전체 시장 수집")
    full_state = _load_full_collection_state()
    full_batch_col, full_run_col, full_reset_col = st.columns([1, 1, 1])
    with full_batch_col:
        full_batch_size = st.number_input("전체 수집 배치 크기", min_value=20, max_value=1000, value=120, step=20)
    with full_run_col:
        if st.button("전체 수집 이어하기", width="stretch", key="dashboard_full_collect"):
            with st.spinner("미국/한국 전체 유니버스를 배치로 수집 중입니다. 오래 걸릴 수 있습니다."):
                st.session_state["dashboard_full_collect_result"] = _collect_full_market_batch(int(full_batch_size))
            st.success("전체 수집 배치를 완료했습니다.")
    with full_reset_col:
        if st.button("전체 수집 처음부터", width="stretch", key="dashboard_full_collect_reset"):
            _reset_full_collection_state()
            st.session_state["dashboard_full_collect_result"] = pd.DataFrame()
            st.success("전체 수집 진행 위치를 초기화했습니다.")
            st.rerun()

    st.caption(
        f"진행 위치: US {int(full_state.get('US', 0) or 0):,} / KR {int(full_state.get('KR', 0) or 0):,}. "
        "전수 수집은 수천 종목이라 한 번에 끝내기보다 배치로 이어받는 방식이 안전합니다."
    )
    full_result = st.session_state.get("dashboard_full_collect_result", pd.DataFrame())
    if isinstance(full_result, pd.DataFrame) and not full_result.empty:
        with st.expander("최근 전체 수집 결과", expanded=True):
            _show_table(
                full_result,
                column_config={
                    "scan_type": st.column_config.TextColumn("수집유형", width="small"),
                    "market": st.column_config.TextColumn("시장", width="small"),
                    "range": st.column_config.TextColumn("범위", width="small"),
                    "candidates": st.column_config.NumberColumn("후보/처리수", format="%d", width="small"),
                    "fresh_prices": st.column_config.NumberColumn("최신가격", format="%d", width="small"),
                    "missing_prices": st.column_config.NumberColumn("누락/오래됨", format="%d", width="small"),
                    "saved": st.column_config.TextColumn("상태", width="small"),
                },
            )

    if detail_ready:
        with st.spinner("보유 종목 정밀 브리핑 계산 중입니다. 시세와 이벤트 데이터를 확인하고 있어요."):
            portfolio_key = _portfolio_cache_key(st.session_state.portfolio)
            analysis, summary, sector_summary, _ = get_portfolio_analysis_state(portfolio_key)
            summary = _portfolio_summary_defaults(summary)
            outlook = get_portfolio_outlook_state(portfolio_key)
            _, _, event_adjustments, news_adjustments, _ = get_learning_state()
            outlook = _enrich_outlook_frame(outlook, event_adjustments=event_adjustments, news_adjustments=news_adjustments)
    else:
        analysis = pd.DataFrame()
        outlook = pd.DataFrame()
        summary = _portfolio_summary_defaults()
        st.info("빠른 대시보드 모드입니다. 보유 종목 수익률/전망은 `보유 종목 정밀 브리핑 계산`을 눌러 확인하세요.")

    if analysis.empty and outlook.empty:
        st.info("보유 종목을 입력하면 브리핑이 더 정확해집니다. 지금은 저장된 후보/가이드 중심으로 보여드립니다.")

    holding_count = len(st.session_state.portfolio) if not st.session_state.portfolio.empty else len(analysis)
    strong_count = int(outlook["outlook"].isin(["상승 기대", "완만한 우상향"]).sum()) if not outlook.empty else 0
    caution_count = int((outlook["outlook"] == "하방 주의").sum()) if not outlook.empty else 0
    event_count = int((outlook["event_risk"] == "높음").sum()) if not outlook.empty else 0
    top_weight = float(summary.get("top_weight", 0) or 0)
    a, b, c, d, e = st.columns(5)
    a.metric("보유종목", f"{holding_count:,}")
    b.metric("우호적 전망", f"{strong_count:,}")
    c.metric("하방 주의", f"{caution_count:,}")
    d.metric("이벤트 임박", f"{event_count:,}")
    e.metric("최대비중", f"{top_weight:.1f}%" if detail_ready else "계산 전")

    st.markdown("#### 시장 탐색 준비도")
    readiness_limit = int(st.session_state.scanner_settings["market_sweep_limit"])
    ready_cols = st.columns(2)
    for col, ready in zip(ready_cols, [_daily_cache_readiness("US", readiness_limit), _daily_cache_readiness("KR", readiness_limit)]):
        with col:
            st.metric(
                f"{ready['market']} 탐색 캐시",
                f"{float(ready['coverage_pct']):.0f}%",
                f"{ready['label']}",
            )
            st.progress(min(1.0, max(0.0, float(ready["coverage_pct"]) / 100)))
            st.caption(
                f"사용 가능 {int(ready['usable']):,}/{int(ready['total']):,}개 · "
                f"신선 {int(ready['fresh']):,} / 만료 {int(ready['stale']):,} / 없음 {int(ready['missing']):,}. "
                f"{ready['action']}"
            )

    _render_learning_data_stack(db_stats)

    st.markdown("#### 오늘 먼저 볼 것")
    for item in _build_dashboard_checklist(outlook, analysis, recent_candidates):
        st.write(f"- {item}")

    _render_decision_log_overview(compact=True)

    col_left, col_right = st.columns([1.15, 1])
    with col_left:
        st.markdown("#### 내 종목 현황 요약")
        if outlook.empty:
            if not st.session_state.portfolio.empty:
                quick_view = st.session_state.portfolio.copy()
                quick_columns = [column for column in ["market", "ticker", "name", "quantity", "avg_price", "cash_budget", "target_weight"] if column in quick_view.columns]
                st.info("정밀 계산 전입니다. 아래는 입력된 보유 종목 요약입니다.")
                _show_table(
                    quick_view[quick_columns].head(12),
                    plain_numeric_columns=["quantity", "target_weight"],
                    currency_columns=["avg_price", "cash_budget"],
                    column_config={
                        "market": st.column_config.TextColumn("시장", width="small"),
                        "ticker": st.column_config.TextColumn("티커", width="small"),
                        "name": st.column_config.TextColumn("종목명", width="small"),
                        "quantity": st.column_config.NumberColumn("수량", format="%.4f", width="small"),
                        "avg_price": st.column_config.TextColumn("평균단가", width="small"),
                        "cash_budget": st.column_config.TextColumn("추가예산", width="small"),
                        "target_weight": st.column_config.NumberColumn("목표비중", format="%.2f", width="small"),
                    },
                )
            else:
                st.info("전망 데이터가 부족합니다.")
        else:
            view = outlook.sort_values(by=["outlook_score", "ticker"], ascending=[False, True]).head(8)
            _show_table(
                view[
                    [
                        "market",
                        "ticker",
                        "name",
                        "outlook",
                        "outlook_label",
                        "action_hint",
                        "next_step",
                        "current_price",
                        "return_pct",
                        "weight_status",
                        "momentum_state",
                        "risk_level",
                        "data_freshness",
                        "reason_summary",
                    ]
                ],
                currency_columns=["current_price"],
                column_config={
                    "market": st.column_config.TextColumn("시장", width="small"),
                    "ticker": st.column_config.TextColumn("티커", width="small"),
                    "name": st.column_config.TextColumn("종목명", width="small"),
                    "outlook": st.column_config.TextColumn("전망", width="small"),
                    "outlook_label": st.column_config.TextColumn("점수해석", width="small"),
                    "action_hint": st.column_config.TextColumn("판단", width="small"),
                    "next_step": st.column_config.TextColumn("다음 행동", width="medium"),
                    "current_price": st.column_config.TextColumn("현재가", width="small"),
                    "return_pct": st.column_config.NumberColumn("수익률", format="%.2f", width="small"),
                    "weight_status": st.column_config.TextColumn("비중", width="small"),
                    "momentum_state": st.column_config.TextColumn("차트", width="small"),
                    "risk_level": st.column_config.TextColumn("위험", width="small"),
                    "data_freshness": st.column_config.TextColumn("데이터", width="small"),
                    "reason_summary": st.column_config.TextColumn("한줄 요약", width="large"),
                },
            )

    with col_right:
        st.markdown("#### 강한 후보 요약")
        scan_col, note_col = st.columns([1, 1.4])
        with scan_col:
            if st.button("오늘 강한 후보 새로 스캔", width="stretch"):
                get_today_scan_state.clear()
                st.session_state["dashboard_scan_ready"] = True
                st.rerun()
        with note_col:
            st.caption("저장본 우선. 버튼 누르면 새 스캔.")

        if st.session_state.get("dashboard_scan_ready", False):
            min_score = int(st.session_state.scanner_settings["min_score"])
            scan_limit = int(st.session_state.scanner_settings["scan_limit"])
            frames = []
            for market in ["US", "KR"]:
                scan = get_today_scan_state(
                    market=market,
                    watchlist_key=_watchlist_cache_key(market),
                    min_score=min_score,
                    scan_limit=scan_limit,
                )
                if not scan.empty:
                    frames.append(scan.assign(market=market, scan_type="today_scan", data_basis="오늘 새 스캔"))
            candidate_view = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if not candidate_view.empty:
                candidate_view = candidate_view.sort_values(by=["score", "ticker"], ascending=[False, True]).head(8)
        else:
            candidate_view = recent_candidates

        if candidate_view.empty:
            st.info("후보 스냅샷이 없습니다. 새 스캔 또는 자동 후보군 저장.")
        else:
            for column in ["entry_price", "stop_loss", "target_1", "confidence_view", "recent_sample_count", "data_basis"]:
                if column not in candidate_view.columns:
                    candidate_view[column] = None
            candidate_view["data_basis"] = candidate_view["data_basis"].fillna("저장 스냅샷")
            _render_action_deck(candidate_view.assign(bucket="강한 후보"), title="강한 후보 카드", limit=3)
            candidate_table = _prepare_recommendation_execution_view(
                candidate_view,
                include_market=True,
                include_source=True,
                include_targets=False,
            )
            _show_table(
                candidate_table,
                currency_columns=["current_price", "entry_price", "stop_loss", "target_1"],
                column_config={
                    **_candidate_column_config(),
                    "scan_type": st.column_config.TextColumn("출처", width="small"),
                    "data_basis": st.column_config.TextColumn("기준", width="small"),
                },
            )
            _render_manual_tracking_quick_add(
                candidate_view,
                source_label="대시보드",
                key_prefix="dashboard_candidates",
            )

    st.markdown("#### 탭별 빠른 길잡이")
    guide_cols = st.columns(4)
    guide_items = [
        ("보유", "전망은 방향, 매매판단은 가격."),
        ("추천", "신뢰도 낮으면 추적 먼저."),
        ("예산", "진입가 넘으면 현금."),
        ("학습", f"스냅샷 {db_stats['scan_snapshots']:,}개 / 로그 {db_stats['feature_entries']:,}개."),
    ]
    for idx, (title, body) in enumerate(guide_items):
        with guide_cols[idx]:
            st.markdown(f"**{title}**")
            st.caption(body)

    st.markdown("#### 오늘의 운영 원칙")
    st.write("- 저장 스냅샷은 참고용. 진입 전 새로 스캔.")
    st.write("- 신규/미검증은 소액 또는 추적 먼저.")
    st.write("- 보유 종목은 매매 판단의 가격 기준 우선.")


def main() -> None:
    init_state()
    _inject_ui_style()
    _, ticker = render_sidebar()

    title_col, mode_col = st.columns([4.8, 1.2], vertical_alignment="center")
    with title_col:
        st.title("Stock Decision Helper")
        st.caption("신규 추천은 방향별로, 보유 종목은 운용 판단으로 나눠 한 화면 흐름으로 정리합니다.")
    with mode_col:
        mode_label = "로컬 DB" if st.session_state.get("cache_only_mode", True) else "온라인"
        st.metric("데이터 모드", mode_label)

    nav_col, refresh_col = st.columns([5.4, 0.9], vertical_alignment="bottom")
    current_page = st.session_state.get("main_page", "대시보드")
    current_page = PAGE_ALIASES.get(str(current_page), str(current_page))
    if current_page not in MAIN_PAGES:
        current_page = "대시보드"
    with nav_col:
        st.caption("화면 선택")
        page = st.pills(
            "화면 선택",
            MAIN_PAGES,
            default=current_page,
            key="main_page",
            label_visibility="collapsed",
            width="stretch",
        )
        if page is None:
            page = current_page
    with refresh_col:
        if st.button("전체 새로고침", width="stretch"):
            _clear_all_cached_data()
            st.rerun()

    if page == "대시보드":
        _safe_render("대시보드", render_daily_dashboard)
    elif page == "신규 추천":
        render_new_recommendation_hub()
    elif page == "보유 운용":
        render_holding_operation_hub()
    elif page == "종목 분석":
        _safe_render("종목 분석", render_selected_analysis, ticker)
    elif page == "학습·관리":
        render_learning_management_hub(ticker)


if __name__ == "__main__":
    main()
