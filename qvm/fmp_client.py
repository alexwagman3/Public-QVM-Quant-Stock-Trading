"""Financial Modeling Prep (FMP) API client.

Thin wrapper around the FMP v3 endpoints. Every method swallows errors
and returns None / empty list so callers can transparently fall back to
yfinance.

Set FMP_API_KEY in the environment to enable. Without the key, every call
short-circuits to None and the caller uses yfinance.
"""

from __future__ import annotations

import os
from typing import Any

import requests

_BASE_V3 = "https://financialmodelingprep.com/api/v3"
_BASE_V4 = "https://financialmodelingprep.com/api/v4"
_TIMEOUT = 15

# Cache identical requests inside one process run — fundamentals get
# fetched multiple times across stages, no reason to round-trip.
_CACHE: dict[str, Any] = {}


def _api_key() -> str | None:
    key = os.environ.get("FMP_API_KEY")
    return key.strip() if key else None


def is_enabled() -> bool:
    """True if FMP_API_KEY is present in the environment."""
    return _api_key() is not None


def _get(path: str, base: str = _BASE_V3, **params: Any) -> Any:
    key = _api_key()
    if not key:
        return None
    params["apikey"] = key
    url = f"{base}/{path}"
    cache_key = f"{url}?{sorted(params.items())}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    try:
        r = requests.get(url, params=params, timeout=_TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json()
        _CACHE[cache_key] = data
        return data
    except Exception:
        return None


# ─── Profile / identity ─────────────────────────────────────────────────────

def get_profile(ticker: str) -> dict | None:
    data = _get(f"profile/{ticker}")
    if isinstance(data, list) and data:
        return data[0]
    return None


# ─── Statements (annual, N years) ───────────────────────────────────────────

def get_income_statements(ticker: str, years: int = 5) -> list[dict]:
    data = _get(f"income-statement/{ticker}", limit=years)
    return data if isinstance(data, list) else []


def get_balance_sheets(ticker: str, years: int = 5) -> list[dict]:
    data = _get(f"balance-sheet-statement/{ticker}", limit=years)
    return data if isinstance(data, list) else []


def get_cash_flow_statements(ticker: str, years: int = 5) -> list[dict]:
    data = _get(f"cash-flow-statement/{ticker}", limit=years)
    return data if isinstance(data, list) else []


def get_key_metrics(ticker: str, years: int = 1) -> list[dict]:
    data = _get(f"key-metrics/{ticker}", limit=years)
    return data if isinstance(data, list) else []


# ─── Bear-case enrichment ───────────────────────────────────────────────────

def get_latest_transcript(ticker: str) -> dict | None:
    data = _get(f"earning_call_transcript/{ticker}", limit=1)
    if isinstance(data, list) and data:
        return data[0]
    return None


def get_analyst_targets(ticker: str) -> dict | None:
    data = _get("price-target-consensus", base=_BASE_V4, symbol=ticker)
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    return None


def get_recent_news(ticker: str, limit: int = 5) -> list[dict]:
    data = _get("stock_news", tickers=ticker, limit=limit)
    return data if isinstance(data, list) else []


def get_insider_trades(ticker: str, limit: int = 10) -> list[dict]:
    data = _get(
        "insider-trading", base=_BASE_V4, symbol=ticker, limit=limit, page=0
    )
    return data if isinstance(data, list) else []


def get_historical_prices(ticker: str, years: int = 2) -> list[dict]:
    """Daily closes. yfinance is preferred for prices."""
    data = _get(f"historical-price-full/{ticker}", serietype="line")
    if isinstance(data, dict):
        return data.get("historical", [])
    return []
