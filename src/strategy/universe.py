from __future__ import annotations

import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MARKET_SWEEP_FILES = {
    "US": DATA_DIR / "universe_US.csv",
    "KR": DATA_DIR / "universe_KR.csv",
}

BLOCKED_TICKERS = {
    "CFLT",  # IBM acquisition completed 2026-03-17; delisted from Nasdaq.
}


def is_tradable_ticker(ticker: str) -> bool:
    cleaned = str(ticker or "").strip().upper()
    return bool(cleaned) and cleaned not in BLOCKED_TICKERS


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
    {"ticker": "VRT", "name": "Vertiv"},
    {"ticker": "AXON", "name": "Axon Enterprise"},
    {"ticker": "CRDO", "name": "Credo Technology"},
    {"ticker": "TMDX", "name": "TransMedics"},
    {"ticker": "CELH", "name": "Celsius"},
    {"ticker": "ELF", "name": "e.l.f. Beauty"},
    {"ticker": "FIX", "name": "Comfort Systems USA"},
    {"ticker": "STRL", "name": "Sterling Infrastructure"},
    {"ticker": "IOT", "name": "Samsara"},
    {"ticker": "FOUR", "name": "Shift4 Payments"},
    {"ticker": "FN", "name": "Fabrinet"},
    {"ticker": "ONON", "name": "On Holding"},
    {"ticker": "JNJ", "name": "Johnson & Johnson"},
    {"ticker": "PG", "name": "Procter & Gamble"},
    {"ticker": "KO", "name": "Coca-Cola"},
]

KR_UNIVERSE: list[dict[str, str]] = [
    {"ticker": "005930.KS", "name": "Samsung Electronics"},
    {"ticker": "000660.KS", "name": "SK Hynix"},
    {"ticker": "035420.KS", "name": "NAVER"},
    {"ticker": "005380.KS", "name": "Hyundai Motor"},
    {"ticker": "035720.KS", "name": "Kakao"},
    {"ticker": "068270.KS", "name": "Celltrion"},
    {"ticker": "207940.KS", "name": "Samsung Biologics"},
    {"ticker": "042700.KS", "name": "Hanmi Semiconductor"},
    {"ticker": "247540.KQ", "name": "EcoPro BM"},
    {"ticker": "066570.KS", "name": "LG Electronics"},
    {"ticker": "005490.KS", "name": "POSCO Holdings"},
    {"ticker": "051910.KS", "name": "LG Chem"},
    {"ticker": "006400.KS", "name": "Samsung SDI"},
    {"ticker": "096770.KS", "name": "SK Innovation"},
    {"ticker": "034730.KS", "name": "SK"},
    {"ticker": "105560.KS", "name": "KB Financial"},
    {"ticker": "012330.KS", "name": "Hyundai Mobis"},
    {"ticker": "329180.KS", "name": "HD Hyundai Heavy"},
    {"ticker": "352820.KS", "name": "HYBE"},
    {"ticker": "373220.KS", "name": "LG Energy Solution"},
    {"ticker": "039030.KQ", "name": "EO Technics"},
    {"ticker": "240810.KQ", "name": "Wonik IPS"},
    {"ticker": "036930.KQ", "name": "Jusung Engineering"},
    {"ticker": "067310.KQ", "name": "Hana Micron"},
    {"ticker": "222800.KQ", "name": "SimMTech"},
    {"ticker": "403870.KQ", "name": "HPSP"},
    {"ticker": "214450.KQ", "name": "PharmaResearch"},
    {"ticker": "145020.KQ", "name": "Hugel"},
    {"ticker": "237690.KQ", "name": "ST Pharm"},
    {"ticker": "000250.KQ", "name": "Sam Chun Dang Pharm"},
    {"ticker": "348370.KQ", "name": "Enchem"},
    {"ticker": "112040.KQ", "name": "Wemade"},
    {"ticker": "263750.KQ", "name": "Pearl Abyss"},
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
    {"ticker": "RKLB", "name": "Rocket Lab"},
    {"ticker": "JOBY", "name": "Joby Aviation"},
    {"ticker": "ACHR", "name": "Archer Aviation"},
    {"ticker": "LUNR", "name": "Intuitive Machines"},
    {"ticker": "ENVX", "name": "Enovix"},
    {"ticker": "RXRX", "name": "Recursion Pharmaceuticals"},
    {"ticker": "SYM", "name": "Symbotic"},
    {"ticker": "AEHR", "name": "Aehr Test Systems"},
    {"ticker": "BEAM", "name": "Beam Therapeutics"},
    {"ticker": "EOSE", "name": "Eos Energy"},
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
    {"ticker": "039030.KQ", "name": "이오테크닉스"},
    {"ticker": "240810.KQ", "name": "원익IPS"},
    {"ticker": "036930.KQ", "name": "주성엔지니어링"},
    {"ticker": "067310.KQ", "name": "하나마이크론"},
    {"ticker": "222800.KQ", "name": "심텍"},
    {"ticker": "403870.KQ", "name": "HPSP"},
    {"ticker": "348370.KQ", "name": "엔켐"},
    {"ticker": "112040.KQ", "name": "위메이드"},
    {"ticker": "263750.KQ", "name": "펄어비스"},
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


US_DISCOVERY_UNIVERSE: list[dict[str, str]] = [
    {"ticker": "ANET", "name": "Arista Networks"},
    {"ticker": "APP", "name": "AppLovin"},
    {"ticker": "DDOG", "name": "Datadog"},
    {"ticker": "NET", "name": "Cloudflare"},
    {"ticker": "SNOW", "name": "Snowflake"},
    {"ticker": "MDB", "name": "MongoDB"},
    {"ticker": "ZS", "name": "Zscaler"},
    {"ticker": "CRWD", "name": "CrowdStrike"},
    {"ticker": "PANW", "name": "Palo Alto Networks"},
    {"ticker": "FTNT", "name": "Fortinet"},
    {"ticker": "OKTA", "name": "Okta"},
    {"ticker": "ESTC", "name": "Elastic"},
    {"ticker": "BILL", "name": "BILL Holdings"},
    {"ticker": "TOST", "name": "Toast"},
    {"ticker": "AFRM", "name": "Affirm"},
    {"ticker": "HOOD", "name": "Robinhood"},
    {"ticker": "COIN", "name": "Coinbase"},
    {"ticker": "MSTR", "name": "MicroStrategy"},
    {"ticker": "CLSK", "name": "CleanSpark"},
    {"ticker": "HUT", "name": "Hut 8"},
    {"ticker": "IREN", "name": "IREN"},
    {"ticker": "WULF", "name": "TeraWulf"},
    {"ticker": "ALAB", "name": "Astera Labs"},
    {"ticker": "MRVL", "name": "Marvell"},
    {"ticker": "LSCC", "name": "Lattice Semiconductor"},
    {"ticker": "ONTO", "name": "Onto Innovation"},
    {"ticker": "ACLS", "name": "Axcelis Technologies"},
    {"ticker": "AMKR", "name": "Amkor Technology"},
    {"ticker": "FORM", "name": "FormFactor"},
    {"ticker": "CAMT", "name": "Camtek"},
    {"ticker": "NVMI", "name": "Nova"},
    {"ticker": "TER", "name": "Teradyne"},
    {"ticker": "AEIS", "name": "Advanced Energy"},
    {"ticker": "ENPH", "name": "Enphase Energy"},
    {"ticker": "FSLR", "name": "First Solar"},
    {"ticker": "NXT", "name": "Nextracker"},
    {"ticker": "RUN", "name": "Sunrun"},
    {"ticker": "FLNC", "name": "Fluence Energy"},
    {"ticker": "STEM", "name": "Stem"},
    {"ticker": "GEV", "name": "GE Vernova"},
    {"ticker": "CEG", "name": "Constellation Energy"},
    {"ticker": "VST", "name": "Vistra"},
    {"ticker": "NRG", "name": "NRG Energy"},
    {"ticker": "PWR", "name": "Quanta Services"},
    {"ticker": "EME", "name": "EMCOR"},
    {"ticker": "DY", "name": "Dycom"},
    {"ticker": "TOL", "name": "Toll Brothers"},
    {"ticker": "BLDR", "name": "Builders FirstSource"},
    {"ticker": "TREX", "name": "Trex"},
    {"ticker": "WING", "name": "Wingstop"},
    {"ticker": "CAVA", "name": "Cava"},
    {"ticker": "SG", "name": "Sweetgreen"},
    {"ticker": "SHAK", "name": "Shake Shack"},
    {"ticker": "TXRH", "name": "Texas Roadhouse"},
    {"ticker": "BROS", "name": "Dutch Bros"},
    {"ticker": "DECK", "name": "Deckers Outdoor"},
    {"ticker": "CROX", "name": "Crocs"},
    {"ticker": "BOOT", "name": "Boot Barn"},
    {"ticker": "BIRK", "name": "Birkenstock"},
    {"ticker": "DUOL", "name": "Duolingo"},
    {"ticker": "SPOT", "name": "Spotify"},
    {"ticker": "TTD", "name": "The Trade Desk"},
    {"ticker": "PINS", "name": "Pinterest"},
    {"ticker": "RBLX", "name": "Roblox"},
    {"ticker": "SE", "name": "Sea"},
    {"ticker": "MELI", "name": "MercadoLibre"},
    {"ticker": "CPNG", "name": "Coupang"},
    {"ticker": "GLBE", "name": "Global-e"},
    {"ticker": "SHOP", "name": "Shopify"},
    {"ticker": "FVRR", "name": "Fiverr"},
    {"ticker": "UPWK", "name": "Upwork"},
    {"ticker": "LMND", "name": "Lemonade"},
    {"ticker": "ROOT", "name": "Root"},
    {"ticker": "OSCR", "name": "Oscar Health"},
    {"ticker": "TDOC", "name": "Teladoc"},
    {"ticker": "GH", "name": "Guardant Health"},
    {"ticker": "NTRA", "name": "Natera"},
    {"ticker": "TGTX", "name": "TG Therapeutics"},
    {"ticker": "HALO", "name": "Halozyme"},
    {"ticker": "INSM", "name": "Insmed"},
    {"ticker": "SMMT", "name": "Summit Therapeutics"},
    {"ticker": "VKTX", "name": "Viking Therapeutics"},
    {"ticker": "TARS", "name": "Tarsus Pharmaceuticals"},
    {"ticker": "ALNY", "name": "Alnylam"},
    {"ticker": "EXAS", "name": "Exact Sciences"},
    {"ticker": "PODD", "name": "Insulet"},
    {"ticker": "GMED", "name": "Globus Medical"},
    {"ticker": "MMSI", "name": "Merit Medical"},
    {"ticker": "IINN", "name": "Inspira Technologies"},
    {"ticker": "KTOS", "name": "Kratos Defense"},
    {"ticker": "AVAV", "name": "AeroVironment"},
    {"ticker": "HWM", "name": "Howmet Aerospace"},
    {"ticker": "BWXT", "name": "BWX Technologies"},
    {"ticker": "FIX", "name": "Comfort Systems USA"},
    {"ticker": "GFF", "name": "Griffon"},
    {"ticker": "ATKR", "name": "Atkore"},
    {"ticker": "AIT", "name": "Applied Industrial"},
    {"ticker": "WFRD", "name": "Weatherford"},
    {"ticker": "TMDX", "name": "TransMedics"},
]


KR_DISCOVERY_UNIVERSE: list[dict[str, str]] = [
    {"ticker": "064760.KQ", "name": "티씨케이"},
    {"ticker": "089030.KQ", "name": "테크윙"},
    {"ticker": "091700.KQ", "name": "파트론"},
    {"ticker": "098460.KQ", "name": "고영"},
    {"ticker": "101490.KQ", "name": "에스앤에스텍"},
    {"ticker": "108320.KQ", "name": "LX세미콘"},
    {"ticker": "121600.KQ", "name": "나노신소재"},
    {"ticker": "131970.KQ", "name": "두산테스나"},
    {"ticker": "140860.KQ", "name": "파크시스템스"},
    {"ticker": "166090.KQ", "name": "하나머티리얼즈"},
    {"ticker": "171090.KQ", "name": "선익시스템"},
    {"ticker": "178320.KQ", "name": "서진시스템"},
    {"ticker": "183300.KQ", "name": "코미코"},
    {"ticker": "189300.KQ", "name": "인텔리안테크"},
    {"ticker": "195940.KQ", "name": "HK이노엔"},
    {"ticker": "213420.KQ", "name": "덕산네오룩스"},
    {"ticker": "215200.KQ", "name": "메가스터디교육"},
    {"ticker": "230360.KQ", "name": "에코마케팅"},
    {"ticker": "241560.KS", "name": "두산밥캣"},
    {"ticker": "272290.KQ", "name": "이녹스첨단소재"},
    {"ticker": "281740.KQ", "name": "레이크머티리얼즈"},
    {"ticker": "290650.KQ", "name": "엘앤씨바이오"},
    {"ticker": "317330.KQ", "name": "덕산테코피아"},
    {"ticker": "319660.KQ", "name": "피에스케이"},
    {"ticker": "328130.KQ", "name": "루닛"},
    {"ticker": "376300.KQ", "name": "디어유"},
    {"ticker": "383310.KQ", "name": "에코프로에이치엔"},
    {"ticker": "389030.KQ", "name": "지니너스"},
    {"ticker": "394280.KQ", "name": "오픈엣지테크놀로지"},
    {"ticker": "417200.KQ", "name": "LS머트리얼즈"},
    {"ticker": "418470.KQ", "name": "밀리의서재"},
    {"ticker": "425040.KQ", "name": "티이엠씨"},
    {"ticker": "432720.KQ", "name": "퀄리타스반도체"},
    {"ticker": "445090.KQ", "name": "에이직랜드"},
    {"ticker": "450080.KS", "name": "에코프로머티"},
    {"ticker": "451760.KQ", "name": "컨텍"},
    {"ticker": "452280.KQ", "name": "한선엔지니어링"},
    {"ticker": "457190.KS", "name": "이수스페셜티케미컬"},
    {"ticker": "462870.KS", "name": "시프트업"},
    {"ticker": "950220.KQ", "name": "네오이뮨텍"},
    {"ticker": "010620.KS", "name": "HD현대미포"},
    {"ticker": "011200.KS", "name": "HMM"},
    {"ticker": "018260.KS", "name": "삼성에스디에스"},
    {"ticker": "028050.KS", "name": "삼성엔지니어링"},
    {"ticker": "032830.KS", "name": "삼성생명"},
    {"ticker": "047810.KS", "name": "한국항공우주"},
    {"ticker": "064350.KS", "name": "현대로템"},
    {"ticker": "079550.KS", "name": "LIG넥스원"},
    {"ticker": "090430.KS", "name": "아모레퍼시픽"},
    {"ticker": "128940.KS", "name": "한미약품"},
    {"ticker": "137310.KS", "name": "에스디바이오센서"},
    {"ticker": "138040.KS", "name": "메리츠금융지주"},
    {"ticker": "161390.KS", "name": "한국타이어앤테크놀로지"},
    {"ticker": "180640.KS", "name": "한진칼"},
    {"ticker": "267260.KS", "name": "HD현대일렉트릭"},
    {"ticker": "272210.KS", "name": "한화시스템"},
    {"ticker": "298040.KS", "name": "효성중공업"},
    {"ticker": "307950.KS", "name": "현대오토에버"},
    {"ticker": "329180.KS", "name": "HD현대중공업"},
    {"ticker": "402340.KS", "name": "SK스퀘어"},
]


def _dedupe_universe(*groups: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            ticker = str(item.get("ticker", "") or "").strip().upper()
            if not is_tradable_ticker(ticker) or ticker in seen:
                continue
            rows.append({"ticker": ticker, "name": str(item.get("name", "") or "")})
            seen.add(ticker)
    return rows


def _load_universe_file(market: str) -> list[dict[str, str]]:
    path = MARKET_SWEEP_FILES.get(market)
    if path is None or not path.exists():
        return []
    try:
        frame = pd.read_csv(path).fillna("")
    except Exception:
        return []
    if "ticker" not in frame.columns:
        return []
    if "name" not in frame.columns:
        frame["name"] = ""
    return normalize_watchlist_frame(frame[["ticker", "name"]])


def get_default_watchlists() -> dict[str, list[dict[str, str]]]:
    return {"US": get_universe("US"), "KR": get_universe("KR")}


def get_universe(market: str) -> list[dict[str, str]]:
    if market == "US":
        return _dedupe_universe(US_UNIVERSE, US_DISCOVERY_UNIVERSE)
    return _dedupe_universe(KR_UNIVERSE, KR_DISCOVERY_UNIVERSE)


def get_market_sweep_universe(market: str) -> list[dict[str, str]]:
    file_rows = _load_universe_file(market)
    if market == "US":
        return _dedupe_universe(US_UNIVERSE, US_DISCOVERY_UNIVERSE, US_HIGH_RISK_UNIVERSE, file_rows)
    return _dedupe_universe(KR_UNIVERSE, KR_DISCOVERY_UNIVERSE, KR_HIGH_RISK_UNIVERSE, file_rows)


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
    cleaned = cleaned[cleaned["ticker"].apply(is_tradable_ticker)]
    cleaned = cleaned.drop_duplicates(subset=["ticker"], keep="first")
    return cleaned[["ticker", "name"]].to_dict("records")
