from __future__ import annotations

import pandas as pd
import streamlit as st


def compact_timestamp(value: object) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d %H:%M")


def format_price(value: object, market: str | None = None) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    if market == "KR":
        return f"{int(round(float(numeric), 0)):,}원"
    return f"${float(numeric):,.2f}"


def format_mixed_currency(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    return f"{float(numeric):,.0f}"


def prepare_table(
    df: pd.DataFrame,
    datetime_columns: list[str] | None = None,
    currency_columns: list[str] | None = None,
    plain_numeric_columns: list[str] | None = None,
    default_market: str | None = None,
) -> pd.DataFrame:
    frame = df.copy()
    for column in datetime_columns or []:
        if column in frame.columns:
            frame[column] = frame[column].apply(compact_timestamp)
    for column in currency_columns or []:
        if column in frame.columns:
            if "market" in frame.columns:
                frame[column] = frame.apply(
                    lambda row: format_price(row.get(column), str(row.get("market", "")).upper()),
                    axis=1,
                )
            else:
                frame[column] = frame[column].apply(lambda value: format_price(value, default_market))
    for column in plain_numeric_columns or []:
        if column in frame.columns:
            frame[column] = frame[column].apply(format_mixed_currency)
    return frame


def show_table(
    df: pd.DataFrame,
    *,
    datetime_columns: list[str] | None = None,
    currency_columns: list[str] | None = None,
    plain_numeric_columns: list[str] | None = None,
    default_market: str | None = None,
    column_config: dict[str, object] | None = None,
    hide_index: bool = True,
) -> None:
    prepared = prepare_table(
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
