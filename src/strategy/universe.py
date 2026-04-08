from __future__ import annotations

import pandas as pd


US_UNIVERSE: list[dict[str, str]] = [
    {"ticker": "VOO", "name": "Vanguard S&P 500 ETF"},
    {"ticker": "SPY", "name": "SPDR S&P 500 ETF"},
    {"ticker": "QQQ", "name": "Invesco QQQ Trust"},
    {"ticker": "SCHD", "name": "Schwab U.S. Dividend Equity ETF"},
    {"ticker": "BRK-B", "name": "Berkshire Hathaway"},
    {"ticker": "AAPL", "name": "Apple"},
    {"ticker": "MSFT", "name": "Microsoft"},
    {"ticker": "NVDA", "name": "NVIDIA"},
    {"ticker": "AMZN", "name": "Amazon"},
    {"ticker": "META", "name": "Meta"},
    {"ticker": "GOOGL", "name": "Alphabet"},
    {"ticker": "TSLA", "name": "Tesla"},
    {"ticker": "AMD", "name": "AMD"},
    {"ticker": "PLTR", "name": "Palantir"},
    {"ticker": "SMCI", "name": "Super Micro Computer"},
    {"ticker": "AVGO", "name": "Broadcom"},
    {"ticker": "NFLX", "name": "Netflix"},
    {"ticker": "CRM", "name": "Salesforce"},
    {"ticker": "UBER", "name": "Uber"},
    {"ticker": "MU", "name": "Micron"},
    {"ticker": "JPM", "name": "JPMorgan"},
    {"ticker": "LLY", "name": "Eli Lilly"},
    {"ticker": "COST", "name": "Costco"},
    {"ticker": "QCOM", "name": "Qualcomm"},
    {"ticker": "ARM", "name": "ARM"},
    {"ticker": "JNJ", "name": "Johnson & Johnson"},
    {"ticker": "PG", "name": "Procter & Gamble"},
    {"ticker": "KO", "name": "Coca-Cola"},
]

KR_UNIVERSE: list[dict[str, str]] = [
    {"ticker": "005930.KS", "name": "Samsung Electronics"},
    {"ticker": "000660.KS", "name": "SK Hynix"},
    {"ticker": "035420.KS", "name": "NAVER"},
    {"ticker": "005380.KS", "name": "Hyundai Motor"},
    {"ticker": "035720.KQ", "name": "Kakao"},
    {"ticker": "068270.KS", "name": "Celltrion"},
    {"ticker": "207940.KS", "name": "Samsung Biologics"},
    {"ticker": "042700.KS", "name": "Hanmi Semiconductor"},
    {"ticker": "247540.KQ", "name": "EcoPro BM"},
    {"ticker": "066570.KS", "name": "LG Electronics"},
    {"ticker": "005490.KS", "name": "POSCO Holdings"},
    {"ticker": "051910.KS", "name": "LG Chem"},
    {"ticker": "006400.KS", "name": "Samsung SDI"},
    {"ticker": "096770.KQ", "name": "SK Innovation"},
    {"ticker": "034730.KS", "name": "SK"},
    {"ticker": "105560.KS", "name": "KB Financial"},
    {"ticker": "012330.KS", "name": "Hyundai Mobis"},
    {"ticker": "329180.KQ", "name": "HD Hyundai Heavy"},
    {"ticker": "352820.KS", "name": "HYBE"},
    {"ticker": "373220.KS", "name": "LG Energy Solution"},
]

US_HIGH_RISK_UNIVERSE: list[dict[str, str]] = [
    {"ticker": "SOUN", "name": "SoundHound AI"},
    {"ticker": "IONQ", "name": "IonQ"},
    {"ticker": "SOFI", "name": "SoFi"},
    {"ticker": "RIVN", "name": "Rivian"},
    {"ticker": "OPEN", "name": "Opendoor"},
    {"ticker": "UPST", "name": "Upstart"},
    {"ticker": "MARA", "name": "Marathon Digital"},
    {"ticker": "RIOT", "name": "Riot Platforms"},
    {"ticker": "HIMS", "name": "Hims & Hers"},
    {"ticker": "ASTS", "name": "AST SpaceMobile"},
]

KR_HIGH_RISK_UNIVERSE: list[dict[str, str]] = [
    {"ticker": "196170.KQ", "name": "알테오젠"},
    {"ticker": "091990.KQ", "name": "셀트리온헬스케어"},
    {"ticker": "095340.KQ", "name": "ISC"},
    {"ticker": "214150.KQ", "name": "클래시스"},
    {"ticker": "122870.KQ", "name": "와이지엔터테인먼트"},
    {"ticker": "086520.KQ", "name": "에코프로"},
    {"ticker": "277810.KQ", "name": "레인보우로보틱스"},
    {"ticker": "058470.KQ", "name": "리노공업"},
    {"ticker": "253450.KQ", "name": "스튜디오드래곤"},
    {"ticker": "214370.KQ", "name": "케어젠"},
]

US_DIVIDEND_UNIVERSE: list[dict[str, str]] = [
    {"ticker": "VOO", "name": "Vanguard S&P 500 ETF"},
    {"ticker": "SPY", "name": "SPDR S&P 500 ETF"},
    {"ticker": "SCHD", "name": "Schwab U.S. Dividend Equity ETF"},
    {"ticker": "VYM", "name": "Vanguard High Dividend Yield ETF"},
    {"ticker": "DGRO", "name": "iShares Core Dividend Growth ETF"},
    {"ticker": "HDV", "name": "iShares Core High Dividend ETF"},
    {"ticker": "JEPI", "name": "JPMorgan Equity Premium Income ETF"},
    {"ticker": "JNJ", "name": "Johnson & Johnson"},
    {"ticker": "PG", "name": "Procter & Gamble"},
    {"ticker": "KO", "name": "Coca-Cola"},
    {"ticker": "PEP", "name": "PepsiCo"},
    {"ticker": "ABBV", "name": "AbbVie"},
    {"ticker": "XOM", "name": "Exxon Mobil"},
    {"ticker": "CVX", "name": "Chevron"},
    {"ticker": "JPM", "name": "JPMorgan"},
    {"ticker": "O", "name": "Realty Income"},
]

KR_DIVIDEND_UNIVERSE: list[dict[str, str]] = [
    {"ticker": "005930.KS", "name": "Samsung Electronics"},
    {"ticker": "005380.KS", "name": "Hyundai Motor"},
    {"ticker": "005490.KS", "name": "POSCO Holdings"},
    {"ticker": "034730.KS", "name": "SK"},
    {"ticker": "105560.KS", "name": "KB Financial"},
    {"ticker": "055550.KS", "name": "Shinhan Financial"},
    {"ticker": "086790.KS", "name": "Hana Financial"},
    {"ticker": "017670.KS", "name": "SK Telecom"},
    {"ticker": "033780.KS", "name": "KT&G"},
    {"ticker": "088980.KS", "name": "Macquarie Korea Infrastructure"},
]


def get_default_watchlists() -> dict[str, list[dict[str, str]]]:
    return {"US": list(US_UNIVERSE), "KR": list(KR_UNIVERSE)}


def get_universe(market: str) -> list[dict[str, str]]:
    return US_UNIVERSE if market == "US" else KR_UNIVERSE


def get_high_risk_universe(market: str) -> list[dict[str, str]]:
    return US_HIGH_RISK_UNIVERSE if market == "US" else KR_HIGH_RISK_UNIVERSE


def get_dividend_universe(market: str) -> list[dict[str, str]]:
    return US_DIVIDEND_UNIVERSE if market == "US" else KR_DIVIDEND_UNIVERSE


def normalize_watchlist_frame(frame: pd.DataFrame) -> list[dict[str, str]]:
    if frame.empty:
        return []

    cleaned = frame.fillna("").copy()
    if "ticker" not in cleaned.columns:
        cleaned["ticker"] = ""
    if "name" not in cleaned.columns:
        cleaned["name"] = ""

    cleaned["ticker"] = cleaned["ticker"].astype(str).str.strip().str.upper()
    cleaned["name"] = cleaned["name"].astype(str).str.strip()
    cleaned = cleaned[(cleaned["ticker"] != "")]
    cleaned = cleaned.drop_duplicates(subset=["ticker"], keep="first")
    return cleaned[["ticker", "name"]].to_dict("records")
