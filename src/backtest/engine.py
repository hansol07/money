from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class BacktestMetrics:
    final_value: float
    total_return_pct: float
    cagr_pct: float
    mdd_pct: float
    win_rate_pct: float
    trade_count: int


def build_signal_scores(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["signal_score"] = 50

    frame.loc[frame["Close"] > frame["ma20"], "signal_score"] += 10
    frame.loc[frame["ma20"] > frame["ma60"], "signal_score"] += 15
    frame.loc[frame["rsi"].between(45, 68), "signal_score"] += 10
    frame.loc[frame["rsi"] > 72, "signal_score"] -= 10
    frame.loc[frame["macd"] > frame["macd_signal"], "signal_score"] += 10
    frame.loc[frame["volume_ratio"] > 1.3, "signal_score"] += 5
    frame.loc[frame["return_20d"] > 8, "signal_score"] += 5
    frame.loc[frame["return_20d"] < -12, "signal_score"] -= 10
    frame["signal_score"] = frame["signal_score"].clip(0, 100)
    return frame


def run_backtest(
    data: pd.DataFrame,
    initial_cash: float = 10_000_000,
    buy_threshold: int = 70,
    sell_threshold: int = 45,
    fee_rate: float = 0.001,
) -> tuple[pd.DataFrame, BacktestMetrics]:
    frame = build_signal_scores(data)
    cash = initial_cash
    shares = 0.0
    entry_price = 0.0
    trades: list[float] = []
    equity_curve: list[float] = []

    for row in frame.itertuples(index=False):
        price = float(row.Close)
        score = int(row.signal_score)

        if score >= buy_threshold and shares == 0:
            shares = (cash * (1 - fee_rate)) / price
            cash = 0.0
            entry_price = price
        elif score <= sell_threshold and shares > 0:
            cash = shares * price * (1 - fee_rate)
            trades.append(((price - entry_price) / entry_price) * 100)
            shares = 0.0
            entry_price = 0.0

        equity = cash + shares * price
        equity_curve.append(equity)

    if shares > 0:
        last_price = float(frame.iloc[-1]["Close"])
        final_value = shares * last_price
        trades.append(((last_price - entry_price) / entry_price) * 100)
    else:
        final_value = cash

    frame["equity"] = equity_curve
    frame["drawdown_pct"] = (frame["equity"] / frame["equity"].cummax() - 1.0) * 100

    total_return_pct = ((final_value - initial_cash) / initial_cash) * 100
    years = max(len(frame) / 252, 1 / 252)
    cagr_pct = ((final_value / initial_cash) ** (1 / years) - 1) * 100 if final_value > 0 else -100.0
    mdd_pct = float(frame["drawdown_pct"].min())
    trade_count = len(trades)
    win_rate_pct = (sum(1 for item in trades if item > 0) / trade_count * 100) if trade_count else 0.0

    metrics = BacktestMetrics(
        final_value=final_value,
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        mdd_pct=mdd_pct,
        win_rate_pct=win_rate_pct,
        trade_count=trade_count,
    )
    return frame, metrics
