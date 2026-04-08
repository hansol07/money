from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PositionInput:
    ticker: str
    quantity: float
    avg_price: float
    cash_budget: float
    target_weight: float
