from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"


def _clean_symbol(value: object) -> str:
    symbol = str(value or "").strip().upper()
    return symbol.replace(".", "-")


def _write_universe(path: Path, rows: list[dict[str, str]]) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["ticker", "name"])
    frame = frame.fillna("")
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["name"] = frame["name"].astype(str).str.strip()
    frame = frame[frame["ticker"] != ""]
    frame = frame.drop_duplicates(subset=["ticker"], keep="first")
    frame = frame.sort_values(by=["ticker"]).reset_index(drop=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return int(len(frame))


def _load_fdr() -> Any:
    try:
        import FinanceDataReader as fdr  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "FinanceDataReader가 설치되어 있지 않습니다. `pip install -r requirements.txt` 후 다시 실행해 주세요."
        ) from exc
    return fdr


def refresh_us_universe() -> int:
    fdr = _load_fdr()
    rows: list[dict[str, str]] = []
    for exchange in ["NASDAQ", "NYSE", "AMEX"]:
        try:
            frame = fdr.StockListing(exchange)
        except Exception:
            continue
        if frame is None or frame.empty:
            continue
        symbol_col = "Symbol" if "Symbol" in frame.columns else "Code"
        name_col = "Name" if "Name" in frame.columns else symbol_col
        for _, row in frame.iterrows():
            ticker = _clean_symbol(row.get(symbol_col, ""))
            name = str(row.get(name_col, "") or "").strip()
            if ticker:
                rows.append({"ticker": ticker, "name": name})
    return _write_universe(DATA_DIR / "universe_US.csv", rows)


def refresh_kr_universe() -> int:
    fdr = _load_fdr()
    frame = fdr.StockListing("KRX")
    rows: list[dict[str, str]] = []
    if frame is not None and not frame.empty:
        for _, row in frame.iterrows():
            code = str(row.get("Code", "") or "").strip()
            name = str(row.get("Name", "") or "").strip()
            market = str(row.get("Market", "") or "").strip().upper()
            if not code:
                continue
            if market == "KOSPI":
                ticker = f"{code}.KS"
            elif market == "KOSDAQ":
                ticker = f"{code}.KQ"
            else:
                continue
            rows.append({"ticker": ticker, "name": name})
    return _write_universe(DATA_DIR / "universe_KR.csv", rows)


def refresh_universe_files() -> dict[str, int]:
    return {
        "US": refresh_us_universe(),
        "KR": refresh_kr_universe(),
    }


if __name__ == "__main__":
    counts = refresh_universe_files()
    print(f"US={counts['US']:,}")
    print(f"KR={counts['KR']:,}")
