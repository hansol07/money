from __future__ import annotations

import pandas as pd
import streamlit as st

from src.automation.scheduler import BackgroundAnalyzer
from src.backtest.engine import run_backtest
from src.data.fetch import get_intraday_stock_data, get_stock_data
from src.portfolio.analytics import analyze_portfolio, build_rebalance_suggestions
from src.portfolio.models import PositionInput
from src.storage.local_store import (
    append_scan_history,
    append_manual_tracking,
    load_feature_log,
    load_manual_tracking,
    load_portfolio,
    load_scan_history,
    load_watchlists,
    normalize_portfolio_frame,
    remove_manual_tracking,
    save_portfolio,
    save_watchlists,
)
from src.strategy.auto_candidates import build_auto_candidate_sets, build_compounder_candidates
from src.strategy.dividend import build_dividend_profiles
from src.strategy.learning import build_learning_adjustments
from src.strategy.profiles import build_high_risk_trade_candidates, build_short_term_trade_candidates, build_strategy_profiles
from src.strategy.regime import classify_market_regime
from src.strategy.recommendation import analyze_position
from src.strategy.realtime import scan_intraday_market
from src.strategy.scanner import scan_market
from src.strategy.tracker import evaluate_scan_history
from src.strategy.universe import get_default_watchlists, normalize_watchlist_frame
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
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.2rem;
        }
        div[role="radiogroup"] {
            gap: 0.5rem;
            padding: 0.2rem 0 0.6rem 0;
            flex-wrap: wrap;
        }
        div[role="radiogroup"] label {
            border: 1px solid #d6dde8;
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            background: #ffffff;
        }
        div[role="radiogroup"] label:has(input:checked) {
            border-color: #1f4f8a;
            background: #eef5ff;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #e7edf5;
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            background: #fbfdff;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }
        .stCaptionContainer p {
            color: #5a6678;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _compact_timestamp(value: object) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d %H:%M")


def _format_price(value: object, market: str | None = None) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    if market == "KR":
        return f"{int(round(float(numeric), 0)):,}원"
    return f"${float(numeric):,.2f}"


def _format_mixed_currency(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    return f"{float(numeric):,.0f}"


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
            if "market" in frame.columns:
                frame[column] = frame.apply(lambda row: _format_price(row.get(column), str(row.get("market", "")).upper()), axis=1)
            else:
                frame[column] = frame[column].apply(lambda value: _format_price(value, default_market))
    for column in plain_numeric_columns or []:
        if column in frame.columns:
            frame[column] = frame[column].apply(_format_mixed_currency)
    return frame


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
    st.dataframe(
        prepared,
        use_container_width=True,
        hide_index=hide_index,
        column_config=column_config,
    )


def _candidate_column_config() -> dict[str, object]:
    return {
        "market": st.column_config.TextColumn("시장", width="small"),
        "ticker": st.column_config.TextColumn("티커", width="small"),
        "name": st.column_config.TextColumn("종목명", width="small"),
        "setup": st.column_config.TextColumn("세팅", width="small"),
        "action": st.column_config.TextColumn("액션", width="small"),
        "score_view": st.column_config.TextColumn("점수판단", width="small"),
        "score": st.column_config.NumberColumn("점수", format="%d", width="small"),
        "score_band": st.column_config.TextColumn("등급", width="small"),
        "recent_hit_rate_20d": st.column_config.NumberColumn("최근적중률", format="%.1f", width="small"),
        "recent_target_rate_5d": st.column_config.NumberColumn("최근목표도달", format="%.1f", width="small"),
        "current_price": st.column_config.TextColumn("현재가", width="small"),
        "entry_price": st.column_config.TextColumn("진입가", width="small"),
        "stop_loss": st.column_config.TextColumn("손절가", width="small"),
        "target_1": st.column_config.TextColumn("1차목표", width="small"),
        "regime": st.column_config.TextColumn("장세", width="small"),
        "regime_delta": st.column_config.NumberColumn("장세보정", format="%d", width="small"),
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
        "setup": st.column_config.TextColumn("세팅", width="small"),
        "score_view": st.column_config.TextColumn("점수판단", width="small"),
        "score": st.column_config.NumberColumn("점수", format="%d", width="small"),
        "score_band": st.column_config.TextColumn("등급", width="small"),
        "recent_hit_rate_20d": st.column_config.NumberColumn("최근적중률", format="%.1f", width="small"),
        "recent_target_rate_5d": st.column_config.NumberColumn("최근목표도달", format="%.1f", width="small"),
        "entry_price": st.column_config.TextColumn("진입가", width="small"),
        "stop_loss": st.column_config.TextColumn("손절가", width="small"),
        "target_1": st.column_config.TextColumn("1차목표", width="small"),
        "target_2": st.column_config.TextColumn("2차목표", width="small"),
        "regime": st.column_config.TextColumn("장세", width="small"),
        "regime_delta": st.column_config.NumberColumn("장세보정", format="%d", width="small"),
        "learning_delta": st.column_config.NumberColumn("학습보정", format="%d", width="small"),
        "short_return_pct": st.column_config.NumberColumn("단기탄력", format="%.2f", width="small"),
        "volume_ratio": st.column_config.NumberColumn("거래량배수", format="%.2f", width="small"),
        "atr_pct": st.column_config.NumberColumn("ATR%", format="%.2f", width="small"),
        "rs_score": st.column_config.NumberColumn("상대강도", format="%.2f", width="small"),
        "risk_reward_1": st.column_config.NumberColumn("1차손익비", format="%.2f", width="small"),
        "exit_rule": st.column_config.TextColumn("청산기준", width="medium"),
        "reason": st.column_config.TextColumn("핵심 사유", width="large"),
    }


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
        "current_price": st.column_config.TextColumn("현재가", width="small"),
        "entry_price": st.column_config.TextColumn("진입가", width="small"),
        "stop_loss": st.column_config.TextColumn("손절가", width="small"),
        "target_1": st.column_config.TextColumn("목표가", width="small"),
        "ret_3d_pct": st.column_config.NumberColumn("3일", format="%.2f", width="small"),
        "ret_5d_pct": st.column_config.NumberColumn("5일", format="%.2f", width="small"),
        "ret_20d_pct": st.column_config.NumberColumn("20일", format="%.2f", width="small"),
        "path_5d": st.column_config.TextColumn("5일경로", width="small"),
        "path_20d": st.column_config.TextColumn("20일경로", width="small"),
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
        st.session_state.watchlists = load_watchlists() or get_default_watchlists()
    if "scanner_settings" not in st.session_state:
        st.session_state.scanner_settings = {
            "min_score": 65,
            "top_n": 8,
        }
    if "realtime_settings" not in st.session_state:
        st.session_state.realtime_settings = {
            "min_score": 60,
            "interval": "5m",
        }


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

    analyzer = get_background_analyzer()
    status = analyzer.get_status()
    st.sidebar.divider()
    st.sidebar.subheader("자동 분석 엔진")
    st.sidebar.caption("서버가 켜져 있는 동안 자동 후보군을 주기적으로 저장합니다.")
    st.sidebar.write(f"상태: {'실행중' if status['running'] else '중지'}")
    st.sidebar.write(f"주기: {int(status['interval_seconds']) // 60}분")
    if status["last_run_at"]:
        st.sidebar.write(f"최근 실행: {_compact_timestamp(status['last_run_at'])}")
    st.sidebar.write(f"최근 결과: {status['last_result']}")
    if status.get("error_message"):
        st.sidebar.error(str(status["error_message"]))
    if st.sidebar.button("지금 자동 분석 실행", use_container_width=True):
        saved_count = analyzer.run_once()
        st.sidebar.success(f"자동 분석을 실행했고 {saved_count}개 스냅샷을 저장했습니다.")

    return market, ticker


@st.cache_data(ttl=1800, show_spinner=False)
def get_learning_state() -> tuple[dict[tuple[str, str, str], object], pd.DataFrame]:
    return build_learning_adjustments(limit=400, min_samples=3)


@st.cache_data(ttl=1800, show_spinner=False)
def get_tracking_state() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        return evaluate_scan_history(limit=400)
    except Exception:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty


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
    enriched["score_band"] = enriched["score"].apply(lambda value: _score_band(float(value)))
    enriched["score_view"] = enriched["score"].apply(lambda value: _score_view(float(value)))

    recent_hits: list[float] = []
    recent_targets: list[float] = []
    for _, row in enriched.iterrows():
        setup = str(row.get("setup", "") or "미분류")
        stats = pattern_lookup.get((scan_type, market, setup), {})
        recent_hits.append(round(float(stats.get("hit_rate_20d_pct", 0.0)), 1))
        recent_targets.append(round(float(stats.get("target_first_5d_pct", 0.0)), 1))

    enriched["recent_hit_rate_20d"] = recent_hits
    enriched["recent_target_rate_5d"] = recent_targets
    return enriched


def _build_manual_tracking_pool() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    learning_adjustments, _ = get_learning_state()
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
                        scan_market("US", st.session_state.watchlists["US"], min_score=min_score, learning_adjustments=learning_adjustments),
                        scan_type="today_scan",
                        market="US",
                        pattern_lookup=pattern_lookup,
                    ).assign(market="US"),
                    _enrich_recommendation_frame(
                        scan_market("KR", st.session_state.watchlists["KR"], min_score=min_score, learning_adjustments=learning_adjustments),
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
                        scan_intraday_market("US", st.session_state.watchlists["US"], interval=interval, min_score=realtime_min_score, learning_adjustments=learning_adjustments),
                        scan_type="realtime_scan",
                        market="US",
                        pattern_lookup=pattern_lookup,
                    ).assign(market="US"),
                    _enrich_recommendation_frame(
                        scan_intraday_market("KR", st.session_state.watchlists["KR"], interval=interval, min_score=realtime_min_score, learning_adjustments=learning_adjustments),
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
                        build_short_term_trade_candidates("US", top_n=top_n, interval=interval, min_score=max(60, realtime_min_score), learning_adjustments=learning_adjustments),
                        scan_type="short_term_trade",
                        market="US",
                        pattern_lookup=pattern_lookup,
                    ).assign(market="US"),
                    _enrich_recommendation_frame(
                        build_short_term_trade_candidates("KR", top_n=top_n, interval=interval, min_score=max(60, realtime_min_score), learning_adjustments=learning_adjustments),
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
                        build_high_risk_trade_candidates("US", top_n=top_n, interval=interval, min_score=60, learning_adjustments=learning_adjustments),
                        scan_type="high_risk_trade",
                        market="US",
                        pattern_lookup=pattern_lookup,
                    ).assign(market="US"),
                    _enrich_recommendation_frame(
                        build_high_risk_trade_candidates("KR", top_n=top_n, interval=interval, min_score=60, learning_adjustments=learning_adjustments),
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
    if st.button("이 종목 관심 추적 추가", key=f"{key_prefix}_tracking_button", use_container_width=True):
        selected = dict(labels[selected_label])
        selected["market"] = selected.get("market", default_market or "")
        selected["source"] = selected.get("source", source_label)
        selected["memo"] = memo.strip()
        append_manual_tracking(selected)
        append_scan_history("manual_track", str(selected["market"]), pd.DataFrame([selected]))
        st.success(f"{selected['ticker']}를 관심 추적에 추가했습니다.")


def render_portfolio_editor() -> None:
    st.subheader("보유 종목 입력")
    st.caption("평균단가와 보유수량을 넣으면 현재 차트 기준으로 간단한 액션을 계산합니다.")

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
            st.dataframe(normalized, use_container_width=True, hide_index=True)

            left, right = st.columns([1, 1])
            with left:
                if st.button("CSV 가져오기", use_container_width=True):
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
        use_container_width=True,
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
        use_container_width=True,
    )

    left, right = st.columns([1, 1])
    with left:
        if st.button("보유 종목 저장", use_container_width=True):
            save_portfolio(st.session_state.portfolio)
            st.success("보유 종목을 로컬 파일에 저장했습니다.")
    with right:
        if st.button("보유 종목 다시 불러오기", use_container_width=True):
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

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(build_price_chart(data), use_container_width=True)
    with right:
        market = "KR" if ticker.endswith((".KS", ".KQ")) else "US"
        st.metric("현재가", _format_price(latest["Close"], market))
        st.metric("RSI", f"{latest['rsi']:.1f}")
        st.metric("MACD", f"{latest['macd']:.2f}")
        st.metric("액션", recommendation.action)
        st.metric("제안 매수 비중", f"{recommendation.suggested_buy_pct}%")
        st.metric("제안 매도 비중", f"{recommendation.suggested_sell_pct}%")
        st.write("사유")
        for reason in recommendation.reasons:
            st.write(f"- {reason}")

    st.dataframe(data.tail(20), use_container_width=True)


def render_portfolio_analysis() -> None:
    st.subheader("포트폴리오 진단")

    rows: list[dict[str, object]] = []
    for row in st.session_state.portfolio.to_dict("records"):
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue

        data = get_stock_data(ticker)
        if data.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "name": row.get("name", ""),
                    "action": "데이터 없음",
                    "score": None,
                    "current_price": None,
                    "return_pct": None,
                    "reason": "가격 데이터를 불러오지 못했습니다.",
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

        rows.append(
            {
                "ticker": ticker,
                "name": row.get("name", ""),
                "action": result.action,
                "score": result.score,
                "current_price": round(result.current_price, 2),
                "return_pct": round(result.return_pct, 2),
                "buy_pct": result.suggested_buy_pct,
                "sell_pct": result.suggested_sell_pct,
                "reason": " / ".join(result.reasons),
            }
        )

    if not rows:
        st.info("보유 종목을 하나 이상 입력해 주세요.")
        return

    summary = pd.DataFrame(rows)
    st.dataframe(summary, use_container_width=True)


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

    st.dataframe(pd.DataFrame(recommendations), use_container_width=True, hide_index=True)


def render_rebalance_suggestions() -> None:
    st.subheader("리밸런싱 제안")
    st.caption("집중도, 섹터 쏠림, 위험도, 스타일 균형을 기준으로 포트폴리오 재조정 힌트를 제공합니다.")

    analysis, summary, sector_summary, _ = analyze_portfolio(st.session_state.portfolio)
    suggestions = build_rebalance_suggestions(analysis, summary, sector_summary)
    if suggestions.empty:
        st.info("리밸런싱 제안을 만들 데이터가 부족합니다.")
        return

    st.dataframe(suggestions, use_container_width=True, hide_index=True)


def render_portfolio_insights() -> None:
    st.subheader("포트폴리오 분석")
    st.caption("보유 종목을 위험도, 성장성, 배당 성향, 집중도로 나눠서 봅니다.")

    analysis, summary, sector_summary, correlation = analyze_portfolio(st.session_state.portfolio)
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
    st.dataframe(style_summary, use_container_width=True, hide_index=True)

    if not sector_summary.empty:
        st.markdown("#### 섹터 비중")
        st.dataframe(sector_summary, use_container_width=True, hide_index=True)

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
        st.dataframe(correlation, use_container_width=True)

    st.markdown("#### 보유 종목 상세 스타일")
    st.dataframe(
        analysis[
            [
                "market",
                "ticker",
                "name",
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
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_market_scanner() -> None:
    st.subheader("오늘 매수 후보 스캐너")
    st.caption("대표 후보군을 빠르게 훑어서 오늘 상대적으로 강한 종목을 점수화합니다.")
    min_score = int(st.session_state.scanner_settings["min_score"])
    learning_adjustments, _ = get_learning_state()
    _, _, _, _, pattern_stats = get_tracking_state()
    pattern_lookup = _build_pattern_lookup(pattern_stats)
    us_regime = classify_market_regime("US")
    kr_regime = classify_market_regime("KR")

    us_scan = scan_market("US", st.session_state.watchlists["US"], min_score=min_score, learning_adjustments=learning_adjustments)
    kr_scan = scan_market("KR", st.session_state.watchlists["KR"], min_score=min_score, learning_adjustments=learning_adjustments)
    us_scan = _enrich_recommendation_frame(us_scan, scan_type="today_scan", market="US", pattern_lookup=pattern_lookup)
    kr_scan = _enrich_recommendation_frame(kr_scan, scan_type="today_scan", market="KR", pattern_lookup=pattern_lookup)

    us_tab, kr_tab = st.tabs(["미국", "한국"])
    with us_tab:
        st.caption(f"장세: {us_regime.regime} / {us_regime.note}")
        if us_scan.empty:
            st.info("조건을 만족한 미국 후보가 없습니다.")
        else:
            us_view = us_scan[
                [
                    "ticker",
                    "name",
                    "setup",
                    "action",
                    "score_view",
                    "score_band",
                    "recent_hit_rate_20d",
                    "recent_target_rate_5d",
                    "score",
                    "regime",
                    "regime_delta",
                    "learning_delta",
                    "rs_score",
                    "atr_pct",
                    "from_52w_high_pct",
                    "entry_price",
                    "stop_loss",
                    "target_1",
                    "trend_score",
                    "momentum_score",
                    "volume_score",
                    "breakout_score",
                    "reason",
                ]
            ]
            _show_table(
                us_view,
                currency_columns=["entry_price", "stop_loss", "target_1"],
                default_market="US",
                column_config=_candidate_column_config(),
            )
            _render_manual_tracking_quick_add(us_scan, source_label="오늘추천", key_prefix="today_scan_us", default_market="US")

    with kr_tab:
        st.caption(f"장세: {kr_regime.regime} / {kr_regime.note}")
        if kr_scan.empty:
            st.info("조건을 만족한 한국 후보가 없습니다.")
        else:
            kr_view = kr_scan[
                [
                    "ticker",
                    "name",
                    "setup",
                    "action",
                    "score_view",
                    "score_band",
                    "recent_hit_rate_20d",
                    "recent_target_rate_5d",
                    "score",
                    "regime",
                    "regime_delta",
                    "learning_delta",
                    "rs_score",
                    "atr_pct",
                    "from_52w_high_pct",
                    "entry_price",
                    "stop_loss",
                    "target_1",
                    "trend_score",
                    "momentum_score",
                    "volume_score",
                    "breakout_score",
                    "reason",
                ]
            ]
            _show_table(
                kr_view,
                currency_columns=["entry_price", "stop_loss", "target_1"],
                default_market="KR",
                column_config=_candidate_column_config(),
            )
            _render_manual_tracking_quick_add(kr_scan, source_label="오늘추천", key_prefix="today_scan_kr", default_market="KR")

    combined = pd.concat([us_scan.assign(market="US"), kr_scan.assign(market="KR")], ignore_index=True)
    if not combined.empty:
        if st.button("오늘 추천 스냅샷 저장", use_container_width=True):
            for market_name, frame in [("US", us_scan), ("KR", kr_scan)]:
                if not frame.empty:
                    append_scan_history("today_scan", market_name, frame)
            st.success("오늘 추천 결과를 누적 저장했습니다.")

        st.download_button(
            "추천 결과 CSV 다운로드",
            data=combined.to_csv(index=False).encode("utf-8-sig"),
            file_name="today_candidates.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_buy_now_panel() -> None:
    st.subheader("오늘 바로 볼 종목")
    st.caption("점수가 높은 후보를 먼저 보여주고, 보유 종목보다 더 강한 대안을 찾는 용도입니다.")
    min_score = int(st.session_state.scanner_settings["min_score"])
    top_n = int(st.session_state.scanner_settings["top_n"])
    learning_adjustments, _ = get_learning_state()
    _, _, _, _, pattern_stats = get_tracking_state()
    pattern_lookup = _build_pattern_lookup(pattern_stats)

    us_scan = scan_market("US", st.session_state.watchlists["US"], min_score=min_score, learning_adjustments=learning_adjustments)
    kr_scan = scan_market("KR", st.session_state.watchlists["KR"], min_score=min_score, learning_adjustments=learning_adjustments)
    us_scan = _enrich_recommendation_frame(us_scan, scan_type="today_scan", market="US", pattern_lookup=pattern_lookup)
    kr_scan = _enrich_recommendation_frame(kr_scan, scan_type="today_scan", market="KR", pattern_lookup=pattern_lookup)
    combined = pd.concat([us_scan.assign(market="US"), kr_scan.assign(market="KR")], ignore_index=True)

    if combined.empty:
        st.info("오늘 조건을 만족한 종목이 없습니다.")
        return

    top = combined.sort_values(by=["score", "ticker"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    best_row = top.iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("최상위 후보", str(best_row["ticker"]))
    col2.metric("최고 판단", str(best_row["score_view"]))
    col3.metric("후보 수", len(combined))
    _show_table(
        top[
            [
                "market",
                "ticker",
                "name",
                "setup",
                "action",
                "score_view",
                "score_band",
                "recent_hit_rate_20d",
                "recent_target_rate_5d",
                "score",
                "regime",
                "regime_delta",
                "learning_delta",
                "rs_score",
                "atr_pct",
                "from_52w_high_pct",
                "entry_price",
                "stop_loss",
                "target_1",
                "volume_ratio",
                "return_20d",
                "reason",
            ]
        ],
        currency_columns=["entry_price", "stop_loss", "target_1"],
        column_config=_candidate_column_config(),
    )
    _render_manual_tracking_quick_add(top, source_label="오늘바로볼종목", key_prefix="buy_now_top")

    st.markdown("#### 급등 후보 보드")
    momentum_board = combined.sort_values(
        by=["volume_score", "breakout_score", "momentum_score", "score"],
        ascending=[False, False, False, False],
    ).head(top_n)
    _show_table(
        momentum_board[
            [
                "market",
                "ticker",
                "name",
                "setup",
                "score_view",
                "score_band",
                "recent_hit_rate_20d",
                "recent_target_rate_5d",
                "regime",
                "regime_delta",
                "learning_delta",
                "rs_score",
                "atr_pct",
                "entry_price",
                "stop_loss",
                "volume_score",
                "breakout_score",
                "momentum_score",
                "volume_ratio",
                "score",
            ]
        ],
        currency_columns=["entry_price", "stop_loss"],
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
                    stronger[
                        [
                            "market",
                            "ticker",
                            "name",
                            "setup",
                            "score",
                            "regime",
                            "regime_delta",
                            "learning_delta",
                            "entry_price",
                            "stop_loss",
                            "volume_ratio",
                            "return_20d",
                            "reason",
                        ]
                    ],
                    currency_columns=["entry_price", "stop_loss"],
                    column_config=_candidate_column_config(),
                )


def render_watchlist_editor() -> None:
    st.subheader("추천 후보군 편집")
    st.caption("오늘 추천 탭에서 스캔할 미국/한국 후보 종목 목록을 직접 관리할 수 있습니다.")

    us_df = pd.DataFrame(st.session_state.watchlists["US"])
    kr_df = pd.DataFrame(st.session_state.watchlists["KR"])
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 미국 후보군")
        edited_us = st.data_editor(
            us_df,
            num_rows="dynamic",
            use_container_width=True,
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
            use_container_width=True,
            key="kr_watchlist_editor",
            column_config={
                "ticker": st.column_config.TextColumn("티커", required=True),
                "name": st.column_config.TextColumn("종목명"),
            },
        )

    left, center, right = st.columns([1, 1, 1])
    with left:
        if st.button("후보군 저장", use_container_width=True):
            st.session_state.watchlists = {
                "US": normalize_watchlist_frame(edited_us),
                "KR": normalize_watchlist_frame(edited_kr),
            }
            save_watchlists(st.session_state.watchlists)
            st.success("추천 후보군을 저장했습니다.")
    with center:
        if st.button("후보군 다시 불러오기", use_container_width=True):
            loaded = load_watchlists()
            if loaded is None:
                st.warning("저장된 후보군 파일이 아직 없습니다.")
            else:
                st.session_state.watchlists = loaded
                st.rerun()
    with right:
        if st.button("기본 후보군 복원", use_container_width=True):
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
    st.line_chart(chart_df.set_index("Date")[["Close", "equity"]], use_container_width=True)
    st.dataframe(chart_df.tail(30), use_container_width=True)


def render_realtime_tab() -> None:
    st.subheader("실시간 급등주 스캐너")
    st.caption("분봉 데이터를 기준으로 거래량 급증, 장중 돌파, VWAP 상단 여부를 빠르게 스캔합니다.")

    interval = str(st.session_state.realtime_settings["interval"])
    min_score = int(st.session_state.realtime_settings["min_score"])
    learning_adjustments, _ = get_learning_state()
    _, _, _, _, pattern_stats = get_tracking_state()
    pattern_lookup = _build_pattern_lookup(pattern_stats)
    us_regime = classify_market_regime("US")
    kr_regime = classify_market_regime("KR")

    left, right = st.columns([1, 1])
    with left:
        if st.button("실시간 스캐너 새로고침", use_container_width=True):
            get_intraday_stock_data.clear()
            st.rerun()
    with right:
        st.write(f"현재 기준: `{interval}` / 최소점수 `{min_score}`")

    us_scan = scan_intraday_market(
        "US",
        st.session_state.watchlists["US"],
        interval=interval,
        min_score=min_score,
        learning_adjustments=learning_adjustments,
    )
    kr_scan = scan_intraday_market(
        "KR",
        st.session_state.watchlists["KR"],
        interval=interval,
        min_score=min_score,
        learning_adjustments=learning_adjustments,
    )
    us_scan = _enrich_recommendation_frame(us_scan, scan_type="realtime_scan", market="US", pattern_lookup=pattern_lookup)
    kr_scan = _enrich_recommendation_frame(kr_scan, scan_type="realtime_scan", market="KR", pattern_lookup=pattern_lookup)

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
                    "recent_hit_rate_20d",
                    "recent_target_rate_5d",
                    "score",
                    "current_price",
                    "regime",
                    "regime_delta",
                    "learning_delta",
                    "rs_score",
                    "atr_pct",
                    "volume_ratio",
                    "reason",
                ]
            ]
            _show_table(
                us_view,
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
                    "recent_hit_rate_20d",
                    "recent_target_rate_5d",
                    "score",
                    "current_price",
                    "regime",
                    "regime_delta",
                    "learning_delta",
                    "rs_score",
                    "atr_pct",
                    "volume_ratio",
                    "reason",
                ]
            ]
            _show_table(
                kr_view,
                currency_columns=["current_price"],
                default_market="KR",
                column_config=_candidate_column_config(),
            )
            _render_manual_tracking_quick_add(kr_scan, source_label="실시간", key_prefix="realtime_kr", default_market="KR")

    combined = pd.concat([us_scan.assign(market="US"), kr_scan.assign(market="KR")], ignore_index=True)
    if combined.empty:
        return

    if st.button("실시간 스냅샷 저장", use_container_width=True):
        for market_name, frame in [("US", us_scan), ("KR", kr_scan)]:
            if not frame.empty:
                append_scan_history("realtime_scan", market_name, frame)
        st.success("실시간 후보를 누적 저장했습니다.")

    st.markdown("#### 실시간 탑픽")
    top = combined.head(int(st.session_state.scanner_settings["top_n"]))
    _show_table(
        top[
            [
                "market",
                "ticker",
                "name",
                "setup",
                "score",
                "regime",
                "regime_delta",
                "learning_delta",
                "current_price",
                "volume_ratio",
                "short_return_pct",
                "above_vwap",
                "reason",
            ]
        ],
        currency_columns=["current_price"],
        column_config=_candidate_column_config(),
    )


def render_auto_candidates_tab() -> None:
    st.subheader("자동 후보군")
    st.caption("시스템이 시장 전체 유니버스를 기준으로 월간/주간/일간/내일 후보를 자동으로 압축합니다.")

    top_n = int(st.session_state.scanner_settings["top_n"])
    us_sets = build_auto_candidate_sets("US", top_n=top_n)
    kr_sets = build_auto_candidate_sets("KR", top_n=top_n)

    if st.button("자동 후보군 스냅샷 저장", use_container_width=True):
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
                _show_table(us_sets[key], currency_columns=["current_price"], default_market="US", column_config=_candidate_column_config())
        with right:
            st.caption("한국")
            if kr_sets[key].empty:
                st.info("한국 후보가 없습니다.")
            else:
                _show_table(kr_sets[key], currency_columns=["current_price"], default_market="KR", column_config=_candidate_column_config())

    history = load_scan_history()
    st.markdown("#### 누적 스냅샷 이력")
    if not history:
        st.info("저장된 스냅샷이 아직 없습니다.")
    else:
        history_rows = []
        for item in history[-20:]:
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
    st.caption("저장된 후보가 이후 얼마나 갔는지뿐 아니라, 목표가를 먼저 쳤는지 손절이 먼저 났는지도 같이 추적합니다.")

    detail, summary, leaderboard, pending, pattern_stats = evaluate_scan_history()
    _, learning_df = get_learning_state()
    if summary.empty:
        st.info("추적할 스냅샷이 아직 없습니다. 자동 후보군에서 먼저 저장해 주세요.")
    else:
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
        feature_log_count = len(load_feature_log())

        a, b, c, d, e, f, g = st.columns(7)
        a.metric("누적 후보 수", f"{total_saved:,}")
        b.metric("평가 완료", f"{matured_count:,}")
        c.metric("평가 대기", f"{pending_count:,}")
        d.metric("평균 20일 수익률", f"{avg_20d:.2f}%")
        e.metric("5일 목표도달", f"{target_first_5d:.1f}%")
        f.metric("5일 손절도달", f"{stop_first_5d:.1f}%")
        g.metric("피처 로그", f"{feature_log_count:,}")

        st.markdown("#### 추적 요약")
        _show_table(summary, column_config=_tracking_summary_column_config())

        if not learning_df.empty:
            st.markdown("#### 현재 학습 보정")
            st.caption("최근 누적 성과를 반영해 지금 점수에 더해지는 가산점/감점입니다.")
            _show_table(learning_df.head(20))

        if not pattern_stats.empty:
            st.markdown("#### 패턴 적중률")
            st.caption("어떤 유형과 세팅이 실제로 잘 맞았는지 누적 통계로 봅니다.")
            _show_table(pattern_stats.head(20), column_config=_pattern_stats_column_config())

        if not pending.empty:
            st.markdown("#### 평가 대기 현황")
            _show_table(pending, datetime_columns=["latest_saved_at"])

        if not leaderboard.empty:
            st.markdown("#### 반복 등장 강세 종목")
            st.caption("여러 번 후보로 잡히면서 실제 성과도 나왔던 종목을 기억 점수 기준으로 정렬합니다.")
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
    st.caption("팔란티어처럼 장기적으로 크게 올라갈 가능성이 있는 추세형 후보를 따로 봅니다.")

    top_n = int(st.session_state.scanner_settings["top_n"])
    us_compounders = build_compounder_candidates("US", top_n=top_n)
    kr_compounders = build_compounder_candidates("KR", top_n=top_n)

    us_tab, kr_tab = st.tabs(["미국 장기 후보", "한국 장기 후보"])
    with us_tab:
        if us_compounders.empty:
            st.info("미국 장기 후보가 없습니다.")
        else:
            _show_table(us_compounders, currency_columns=["current_price"], default_market="US", column_config=_candidate_column_config())
    with kr_tab:
        if kr_compounders.empty:
            st.info("한국 장기 후보가 없습니다.")
        else:
            _show_table(kr_compounders, currency_columns=["current_price"], default_market="KR", column_config=_candidate_column_config())


def render_manual_tracking_tab() -> None:
    st.subheader("관심 추적")
    st.caption("추천 후보 중에서 네가 직접 고른 종목만 따로 저장해두고, 그 시점부터 이후 흐름을 별도로 봅니다.")

    pool = _build_manual_tracking_pool()
    tracked = load_manual_tracking()
    detail, _, _, _, _ = get_tracking_state()
    manual_detail = detail[detail["scan_type"] == "manual_track"].copy() if not detail.empty else pd.DataFrame()

    st.markdown("#### 추적 추가")
    if pool.empty:
        st.info("지금 추가할 수 있는 추천 후보가 없습니다.")
    else:
        option_map = {
            f"{row['source']} | {row['market']} | {row['ticker']} | {row['name']}": row
            for _, row in pool.iterrows()
        }
        selected_label = st.selectbox("추천 후보에서 고르기", options=list(option_map.keys()))
        memo = st.text_input("메모", placeholder="왜 관심 있는지 간단히 남겨도 됩니다.")
        if st.button("관심 추적에 추가", use_container_width=True):
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
        tracked_view["score_view"] = tracked_view["score"].apply(lambda value: _score_view(float(value)) if str(value) != "" else "")
        tracked_view["recent_hit_rate_20d"] = 0.0
        tracked_view["recent_target_rate_5d"] = 0.0
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
                    "entry_price",
                    "stop_loss",
                    "target_1",
                    "ret_3d_pct",
                    "ret_5d_pct",
                    "ret_20d_pct",
                    "path_5d",
                    "path_20d",
                    "memo",
                    "created_at",
                ]
            ],
            datetime_columns=["created_at"],
            currency_columns=["current_price", "entry_price", "stop_loss", "target_1"],
            column_config=_manual_tracking_column_config(),
        )

        removable = st.selectbox(
            "삭제할 관심 종목",
            options=[""] + [f"{row['tracking_id']} | {row['market']} | {row['ticker']}" for _, row in tracked.iterrows()],
        )
        if removable and st.button("관심 추적에서 제거", use_container_width=True):
            tracking_id = removable.split("|")[0].strip()
            remove_manual_tracking(tracking_id)
            st.success("관심 추적 종목을 제거했습니다.")
            st.rerun()


def render_strategy_profiles_tab() -> None:
    st.subheader("전략별 추천")
    st.caption("안정형 적립, 우량 배당, 고위험 성장으로 성격을 나눠서 장기적으로 모아갈 후보를 봅니다.")

    top_n = int(st.session_state.scanner_settings["top_n"])
    us_profiles = build_strategy_profiles("US", top_n=top_n)
    kr_profiles = build_strategy_profiles("KR", top_n=top_n)

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
                _show_table(us_profiles[key], currency_columns=["current_price"], default_market="US", column_config=_candidate_column_config())
        with right:
            st.caption("한국")
            if kr_profiles[key].empty:
                st.info("한국 후보가 없습니다.")
            else:
                _show_table(kr_profiles[key], currency_columns=["current_price"], default_market="KR", column_config=_candidate_column_config())

    st.markdown("#### 읽는 기준")
    st.write("- `안정형 적립`: VOO 같은 꾸준한 우상향, 낮은 변동성, 낮은 낙폭 중심")
    st.write("- `우량 배당`: 배당과 추세가 함께 받쳐주는 종목 중심")
    st.write("- `고위험 성장`: 변동성은 높지만 장기 주도주로 커질 가능성이 큰 종목 중심")


def render_dividend_tab() -> None:
    st.subheader("배당주 분석")
    st.caption("미국/한국 배당 후보를 따로 보면서 배당수익률, 배당성장, 배당락일, 모아가기 구간을 같이 확인합니다.")

    top_n = int(st.session_state.scanner_settings["top_n"])
    us_profiles = build_dividend_profiles("US", top_n=top_n)
    kr_profiles = build_dividend_profiles("KR", top_n=top_n)

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
                    _show_table(
                        profiles["stable"],
                        currency_columns=["current_price", "annual_dividend", "accumulate_low", "accumulate_high"],
                        default_market=market_code,
                        column_config=_dividend_column_config(),
                    )

            with growth_tab:
                if profiles["growth"].empty:
                    st.info(f"{market_label} 배당 성장 후보가 없습니다.")
                else:
                    _show_table(
                        profiles["growth"],
                        currency_columns=["current_price", "annual_dividend", "accumulate_low", "accumulate_high"],
                        default_market=market_code,
                        column_config=_dividend_column_config(),
                    )

            st.markdown("#### 읽는 기준")
            st.write("- `배당수익률`은 최근 1년 배당 기준입니다.")
            st.write("- `배당성장`은 최근 배당이 얼마나 커졌는지 보는 값입니다.")
            st.write("- `모으기하단~상단`은 지금 가격이 과열인지, 분할로 모아도 되는 구간인지 보려는 참고 범위입니다.")
            st.write("- `배당락일`은 최근 확인된 배당락 기준일이라, 가까운 시기라면 배당락 후 흔들림도 같이 보는 편이 좋습니다.")


def render_short_term_trade_tab() -> None:
    st.subheader("단타 트레이드 플랜")
    st.caption("장중 강도와 일봉 추세를 같이 보고, 일반 단타와 고위험 단타를 나눠서 진입가, 손절가, 목표가를 계산합니다.")

    top_n = int(st.session_state.scanner_settings["top_n"])
    interval = str(st.session_state.realtime_settings["interval"])
    min_score = max(60, int(st.session_state.realtime_settings["min_score"]))
    learning_adjustments, _ = get_learning_state()
    _, _, _, _, pattern_stats = get_tracking_state()
    pattern_lookup = _build_pattern_lookup(pattern_stats)
    us_regime = classify_market_regime("US")
    kr_regime = classify_market_regime("KR")

    us_trades = build_short_term_trade_candidates(
        "US",
        top_n=top_n,
        interval=interval,
        min_score=min_score,
        learning_adjustments=learning_adjustments,
    )
    kr_trades = build_short_term_trade_candidates(
        "KR",
        top_n=top_n,
        interval=interval,
        min_score=min_score,
        learning_adjustments=learning_adjustments,
    )
    us_high_risk = build_high_risk_trade_candidates(
        "US",
        top_n=top_n,
        interval=interval,
        min_score=60,
        learning_adjustments=learning_adjustments,
    )
    kr_high_risk = build_high_risk_trade_candidates(
        "KR",
        top_n=top_n,
        interval=interval,
        min_score=60,
        learning_adjustments=learning_adjustments,
    )
    us_trades = _enrich_recommendation_frame(us_trades, scan_type="short_term_trade", market="US", pattern_lookup=pattern_lookup)
    kr_trades = _enrich_recommendation_frame(kr_trades, scan_type="short_term_trade", market="KR", pattern_lookup=pattern_lookup)
    us_high_risk = _enrich_recommendation_frame(us_high_risk, scan_type="high_risk_trade", market="US", pattern_lookup=pattern_lookup)
    kr_high_risk = _enrich_recommendation_frame(kr_high_risk, scan_type="high_risk_trade", market="KR", pattern_lookup=pattern_lookup)

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
        if st.button("단타 후보 스냅샷 저장", use_container_width=True):
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
                us_normal_view = us_trades[
                    [
                        "ticker",
                        "name",
                        "setup",
                        "score_view",
                        "score_band",
                        "recent_hit_rate_20d",
                        "recent_target_rate_5d",
                        "score",
                        "entry_price",
                        "stop_loss",
                        "target_1",
                        "target_2",
                        "atr_pct",
                        "rs_score",
                        "risk_reward_1",
                        "regime",
                        "learning_delta",
                        "reason",
                    ]
                ]
                _show_table(
                    us_normal_view,
                    currency_columns=["entry_price", "stop_loss", "target_1", "target_2"],
                    default_market="US",
                    column_config=_trade_column_config(),
                )
                _render_manual_tracking_quick_add(us_trades, source_label="일반단타", key_prefix="short_trade_us", default_market="US")
        with risky_tab:
            if us_high_risk.empty:
                st.info("미국 고위험 단타 후보가 없습니다.")
            else:
                st.warning("고위험 단타는 변동성이 매우 크니 소액/짧게 보는 전제가 필요합니다.")
                us_risky_view = us_high_risk[
                    [
                        "ticker",
                        "name",
                        "risk_level",
                        "setup",
                        "score_view",
                        "score_band",
                        "recent_hit_rate_20d",
                        "recent_target_rate_5d",
                        "score",
                        "entry_price",
                        "stop_loss",
                        "target_1",
                        "target_2",
                        "atr_pct",
                        "rs_score",
                        "risk_reward_1",
                        "regime",
                        "learning_delta",
                        "reason",
                    ]
                ]
                _show_table(
                    us_risky_view,
                    currency_columns=["entry_price", "stop_loss", "target_1", "target_2"],
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
                kr_normal_view = kr_trades[
                    [
                        "ticker",
                        "name",
                        "setup",
                        "score_view",
                        "score_band",
                        "recent_hit_rate_20d",
                        "recent_target_rate_5d",
                        "score",
                        "entry_price",
                        "stop_loss",
                        "target_1",
                        "target_2",
                        "atr_pct",
                        "rs_score",
                        "risk_reward_1",
                        "regime",
                        "learning_delta",
                        "reason",
                    ]
                ]
                _show_table(
                    kr_normal_view,
                    currency_columns=["entry_price", "stop_loss", "target_1", "target_2"],
                    default_market="KR",
                    column_config=_trade_column_config(),
                )
                _render_manual_tracking_quick_add(kr_trades, source_label="일반단타", key_prefix="short_trade_kr", default_market="KR")
        with risky_tab:
            if kr_high_risk.empty:
                st.info("한국 고위험 단타 후보가 없습니다.")
            else:
                st.warning("고위험 단타는 급등과 급락이 모두 빠르니 손절 기준을 더 엄격하게 봐야 합니다.")
                kr_risky_view = kr_high_risk[
                    [
                        "ticker",
                        "name",
                        "risk_level",
                        "setup",
                        "score_view",
                        "score_band",
                        "recent_hit_rate_20d",
                        "recent_target_rate_5d",
                        "score",
                        "entry_price",
                        "stop_loss",
                        "target_1",
                        "target_2",
                        "atr_pct",
                        "rs_score",
                        "risk_reward_1",
                        "regime",
                        "learning_delta",
                        "reason",
                    ]
                ]
                _show_table(
                    kr_risky_view,
                    currency_columns=["entry_price", "stop_loss", "target_1", "target_2"],
                    default_market="KR",
                    column_config=_trade_column_config(),
                )
                _render_manual_tracking_quick_add(kr_high_risk, source_label="고위험단타", key_prefix="high_risk_kr", default_market="KR")

    if not combined.empty:
        st.markdown("#### 단타 운용 원칙")
        st.write("- `entry_price` 위에서 거래량이 유지될 때만 진입 후보로 봅니다.")
        st.write("- `stop_loss` 이탈 시 손절, `target_1` 도달 시 일부 익절을 기본으로 봅니다.")
        st.write("- `target_2`까지 가면 남은 물량은 추세 보며 정리하는 식으로 씁니다.")


def main() -> None:
    init_state()
    _inject_ui_style()
    st.title("Stock Decision Helper")
    st.caption("한국/미국 주식을 함께 보며 보유 종목 관리와 간단한 추천 액션을 확인하는 1차 MVP입니다.")

    _, ticker = render_sidebar()
    page = st.radio(
        "화면 선택",
        ["종목 분석", "보유 종목", "오늘 추천", "전략별 추천", "배당주", "단타", "관심 추적", "자동 후보군", "추적", "실시간", "백테스트", "후보군 관리"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if page == "종목 분석":
        render_selected_analysis(ticker)
    elif page == "보유 종목":
        render_portfolio_editor()
        st.divider()
        render_portfolio_insights()
        st.divider()
        render_portfolio_analysis()
        st.divider()
        render_rebalance_hint()
        st.divider()
        render_rebalance_suggestions()
    elif page == "오늘 추천":
        render_buy_now_panel()
        st.divider()
        render_market_scanner()
    elif page == "전략별 추천":
        render_strategy_profiles_tab()
    elif page == "배당주":
        render_dividend_tab()
    elif page == "단타":
        render_short_term_trade_tab()
    elif page == "관심 추적":
        render_manual_tracking_tab()
    elif page == "자동 후보군":
        render_auto_candidates_tab()
    elif page == "추적":
        render_tracking_tab()
    elif page == "실시간":
        render_realtime_tab()
    elif page == "백테스트":
        render_backtest_tab(ticker)
    elif page == "후보군 관리":
        render_watchlist_editor()


if __name__ == "__main__":
    main()
