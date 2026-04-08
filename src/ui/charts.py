from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def build_price_chart(data: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=data["Date"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
        )
    )
    figure.add_trace(go.Scatter(x=data["Date"], y=data["ma20"], mode="lines", name="MA20"))
    figure.add_trace(go.Scatter(x=data["Date"], y=data["ma60"], mode="lines", name="MA60"))

    figure.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h"),
    )
    return figure
