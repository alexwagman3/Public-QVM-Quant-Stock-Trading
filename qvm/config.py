"""Configuration: universe, factor weights, market-timing weights, output paths.

All API keys come from environment variables (FMP_API_KEY, DEEPSEEK_API_KEY).
Both keys are optional — the pipeline degrades gracefully (yfinance fallback
for fundamentals; template bear case when DeepSeek is unavailable).
"""

from __future__ import annotations

import os
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(os.environ.get("QVM_OUTPUT_DIR", "output")).resolve()
DB_PATH = str(OUTPUT_DIR / "qvm.db")

# ─── Portfolio sizing ───────────────────────────────────────────────────────

CAPITAL = float(os.environ.get("QVM_CAPITAL", "10000"))
MAX_SECTOR_PCT = 0.40           # no single sector > 40% of deployed capital
FUNDAMENTAL_TOP_N = 100         # depth of fundamental dive after momentum screen
TOP_N_REPORT = 30               # rows shown in the HTML report

# ─── Universe: 500+ liquid U.S. equities (no ETFs) ──────────────────────────

TICKERS: list[str] = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK-B",
    "TSLA", "AVGO", "LLY", "JPM", "V", "XOM", "UNH", "WMT",
    "MA", "PG", "JNJ", "HD", "CVX", "MRK", "COST", "ABBV",
    "ADBE", "BAC", "KO", "PEP", "TMO", "PFE", "MCD", "CSCO",
    "ABT", "DIS", "ACN", "WFC", "CRM", "VZ", "DHR", "TXN",
    "NKE", "PM", "MS", "RTX", "NEE", "LIN", "UPS", "QCOM",
    "UNP", "LOW", "AMGN", "HON", "SBUX", "IBM", "GS", "BA",
    "CAT", "AMD", "INTC", "SPGI", "GILD", "MDLZ", "DE", "LMT",
    "BKNG", "C", "TJX", "BLK", "AXP", "ADP", "T", "ISRG",
    "GE", "AMAT", "CVS", "CB", "VRTX", "MO", "PGR", "CI",
    "SYK", "DUK", "SO", "AON", "EW", "SLB", "REGN", "ETN",
    "F", "ITW", "SHW", "PLTR", "ORCL", "NFLX", "LRCX", "GEV",
    "MSI", "MSCI", "NDAQ", "NTAP", "NEM", "NWSA", "NWS", "NI",
    "NSC", "NTRS", "NOC", "NCLH", "NOV", "NRG", "NUE", "NVR",
    "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "OTIS", "OXY",
    "PANW", "PAYC", "PAYX", "PCAR", "PEG", "PH", "PHM", "PKG",
    "PLD", "PNC", "PNR", "PNW", "POOL", "PPG", "PPL", "PRU",
    "PSA", "PSX", "PTC", "PWR", "PYPL", "QRVO", "RCL", "REG",
    "RF", "RHI", "RJF", "RL", "RMD", "ROK", "ROL", "ROP",
    "ROST", "RSG", "RVTY", "SBAC", "SCHW", "SEDG", "SEE", "SNA",
    "SNPS", "SPG", "SRE", "STE", "STLD", "STT", "STX", "STZ",
    "SWK", "SWKS", "SYF", "SYY", "TAP", "TDG", "TDY", "TECH",
    "TEL", "TER", "TFC", "TFX", "TGT", "TMUS", "TPR", "TRMB",
    "TROW", "TRV", "TSCO", "TSN", "TT", "TTWO", "TXT", "TYL",
    "UAL", "UDR", "UHS", "ULTA", "UNM", "URI", "USB", "VFC",
    "VICI", "VLO", "VMC", "VRSK", "VRSN", "VST", "VTRS", "WAB",
    "WAT", "WBD", "WDC", "WEC", "WELL", "WHR", "WM", "WMB",
    "WRB", "WST", "WY", "WYNN", "XEL", "XYL", "YUM", "ZBH",
    "ZBRA", "ZTS", "MMM", "AOS", "AAP", "AFL", "A", "APD",
    "AKAM", "ALK", "ALB", "ARE", "ALGN", "ALLE", "LNT", "AMCR",
    "AEE", "AAL", "AEP", "AIG", "AMT", "AWK", "AMP", "AME",
    "APH", "ADI", "APA", "APTV", "ADM", "ANET", "AJG", "AIZ",
    "ATO", "AZO", "AVB", "AVY", "BKR", "BAX", "BDX", "BBY",
    "BIO", "BIIB", "CDW", "CE", "CNC", "CNP", "CF", "CHTR",
    "CMG", "CINF", "CTAS", "CFG", "CME", "CMS", "CTSH", "CL",
    "CMCSA", "COP", "ED", "CPRT", "GLW", "CTVA", "CCI", "CSX",
    "CMI", "D", "DAL", "DVN", "DXCM", "FANG", "DLR", "DG",
    "DHI", "DRI", "DTE", "DVA", "DXC", "EMN", "EBAY", "ECL",
    "EIX", "EA", "EMR", "ETR", "EOG", "EFX", "EQIX", "EQR",
    "ESS", "EL", "ETSY", "EVRG", "ES", "EXC", "EXPE", "EXPD",
    "EXR", "FDS", "FAST", "FRT", "FDX", "FITB", "FE", "FISV",
    "FMC", "FTNT", "FTV", "FOX", "FOXA", "BEN", "FCX", "GRMN",
    "IT", "GD", "GIS", "GM", "GPC", "GPN", "GL", "HAL",
    "HCA", "HSIC", "HPE", "HLT", "HOLX", "HRL", "HST", "HWM",
    "HPQ", "HUM", "HBAN", "HII", "IEX", "IDXX", "ILMN", "INCY",
    "IR", "ICE", "IFF", "IP", "INTU", "IVZ", "IPGP", "IQV",
    "IRM", "JBHT", "JKHY", "J", "JCI", "KEY", "KEYS", "KMB",
    "KIM", "KMI", "KLAC", "LHX", "LH", "LW", "LVS", "LEG",
    "LDOS", "LEN", "MCK", "MDT", "MTD", "MGM", "MCHP", "MU",
    "MAA", "MHK", "MNST", "MCO", "COR", "VLTO", "SOLV", "KVUE",
    "GEHC", "CPAY", "EG", "ARM", "SMCI", "APP", "CRWD", "SNOW",
    "DDOG", "NET", "MDB", "OKTA", "ZS", "S", "GEN", "LYV",
    "CARR", "AXON", "HEI", "BWXT", "SAIA", "CHRW", "LSTR", "ARCB",
    "TFII", "GFL", "WCN", "DOV", "GGG", "LECO", "TOL", "CVCO",
    "LPX", "RYN", "CLF", "RS", "ATI", "CMC", "VALE", "SCCO",
    "TECK", "AR", "EQT", "RRC", "CNX", "CRK", "CTRA", "MPC",
    "OVV", "CVE", "SU", "IMO", "PBR", "KBR", "FLR", "DY",
    "PRIM", "APG", "ARMK", "USFD", "PFGC", "CHEF", "UNFI", "ANDE",
    "CALM", "POST", "LANC", "FLO", "CPB", "MKC", "BG", "INGR",
    "NTR", "DPZ", "TXRH", "WEN", "QSR", "DIN", "HELE", "NWL",
    "ELF", "YETI", "BC", "MAT", "HAS", "CHDN", "BYD", "PENN",
    "CZR", "BURL", "DECK", "SKX", "CROX", "LEVI", "PVH", "COLM",
    "DLTR", "FIVE", "OLLI", "BROS", "BBWI", "ORLY", "WDAY", "RBLX",
    "COIN", "HOOD", "SOFI", "AFRM", "SQ", "TOST", "SHOP", "MELI",
    "CPNG", "DASH", "ABNB", "MAR", "TCOM", "U", "AI", "GTLB",
    "ESTC", "CFLT", "BILL", "PATH", "ASAN", "MNDY", "TDOC", "AMWL",
    "ACCD", "PGNY", "WOLF", "MP", "STWD", "BXMT", "NRZ", "UPST",
    "ENVA",
]

# ─── Factor Weights ─────────────────────────────────────────────────────────
# Weights match the academic literature. Total momentum exposure
# (jt_12_1 + 6m) = 0.30, matching Jegadeesh-Titman / Asness common practice.
# Gross profitability gets the heaviest single weight because Novy-Marx (2013)
# showed it is the strongest quality factor.

WEIGHTS: dict[str, float] = {
    "jt_12_1_momentum":    0.20,  # 12-minus-1 month momentum (Jegadeesh-Titman 1993)
    "momentum_6m":         0.10,  # 6-month momentum
    "inv_volatility_252d": 0.20,  # inverse 252-day vol (Blitz & van Vliet 2007)
    "gross_profitability": 0.25,  # gross profit / total assets (Novy-Marx 2013)
    "fcf_yield":           0.10,  # FCF / enterprise value (Fama-French 2006)
    "accruals":            0.10,  # Sloan (1996) accruals — earnings quality
    "inv_asset_growth":    0.05,  # inverse asset growth (Titman-Wei-Xie 2004)
    "piotroski_gate":      6,     # minimum F-Score to pass gate (Piotroski 2000)
}

# ─── Market-Timing Config ───────────────────────────────────────────────────

MARKET_TIMING_CONFIG: dict[str, float] = {
    "weight_trend":      0.30,
    "weight_volatility": 0.25,
    "weight_credit":     0.20,
    "weight_breadth":    0.10,
    "weight_sentiment":  0.10,
    "weight_safe_haven": 0.05,
}

# ─── Factor display config (used by screener.py and report.py) ──────────────

FACTOR_DISPLAY: list[tuple[str, str, str, float]] = [
    ("jt_12_1_momentum",   "Momentum",   "pct",  0.20),
    ("momentum_6m",        "6M Mom",     "pct",  0.10),
    ("inv_volatility_252d", "Volatility", "mult", 0.20),
    ("gross_profitability", "Quality",   "pct",  0.25),
    ("fcf_yield",          "Value",      "pct",  0.10),
    ("accruals",           "Accruals",   "pct",  0.10),
    ("inv_asset_growth",   "Inv Growth", "pct",  0.05),
]


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
