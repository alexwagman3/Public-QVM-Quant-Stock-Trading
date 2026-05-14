"""Unified data-access layer.

Each function returns a normalized dict that callers can use without
knowing the underlying source. FMP is tried first when FMP_API_KEY is
set; on any failure (no key, network error, missing field) the call
falls back to yfinance.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from . import fmp_client as _fmp


def _f(v: Any) -> Optional[float]:
    """Coerce to float, returning None on failure or NaN."""
    if v is None:
        return None
    try:
        f = float(v)
        if pd.isna(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


# ─── FMP path ───────────────────────────────────────────────────────────────

def fundamentals_from_fmp(ticker: str) -> dict | None:
    """Build the normalized fundamentals dict from FMP. Returns None on miss."""
    if not _fmp.is_enabled():
        return None
    income = _fmp.get_income_statements(ticker, years=2)
    balance = _fmp.get_balance_sheets(ticker, years=2)
    cashflow = _fmp.get_cash_flow_statements(ticker, years=2)
    profile = _fmp.get_profile(ticker) or {}
    metrics = _fmp.get_key_metrics(ticker, years=1)

    if not income or not balance:
        return None

    i0 = income[0]
    b0 = balance[0]
    c0 = cashflow[0] if cashflow else {}
    m0 = metrics[0] if metrics else {}

    return {
        "ticker": ticker,
        "fiscal_date": i0.get("date", ""),
        "gross_profit": _f(i0.get("grossProfit")),
        "total_assets": _f(b0.get("totalAssets")),
        "free_cash_flow": _f(c0.get("freeCashFlow")),
        "operating_cash_flow": _f(
            c0.get("operatingCashFlow")
            or c0.get("netCashProvidedByOperatingActivities")
        ),
        "enterprise_value": _f(m0.get("enterpriseValue")),
        "market_cap": _f(m0.get("marketCap") or profile.get("mktCap")),
        "stockholders_equity": _f(b0.get("totalStockholdersEquity")),
        "net_income": _f(i0.get("netIncome")),
        "total_revenue": _f(i0.get("revenue")),
        "total_debt": _f(b0.get("totalDebt")),
        "long_term_debt": _f(b0.get("longTermDebt")),
        "current_assets": _f(b0.get("totalCurrentAssets")),
        "current_liabilities": _f(b0.get("totalCurrentLiabilities")),
        "shares_outstanding": _f(i0.get("weightedAverageShsOut")),
        "sector": profile.get("sector") or "Unknown",
        "industry": profile.get("industry") or "Unknown",
        "name": profile.get("companyName") or ticker,
        "source": "fmp",
    }


def two_year_balance_from_fmp(ticker: str) -> dict | None:
    """Get current + prior year fundamentals for YoY signals."""
    if not _fmp.is_enabled():
        return None
    income = _fmp.get_income_statements(ticker, years=2)
    balance = _fmp.get_balance_sheets(ticker, years=2)
    cashflow = _fmp.get_cash_flow_statements(ticker, years=2)
    if len(income) < 2 or len(balance) < 2:
        return None
    i0, i1 = income[0], income[1]
    b0, b1 = balance[0], balance[1]
    c0 = cashflow[0] if cashflow else {}
    return {
        "curr": {
            "net_income": _f(i0.get("netIncome")),
            "total_assets": _f(b0.get("totalAssets")),
            "long_term_debt": _f(b0.get("longTermDebt")),
            "current_assets": _f(b0.get("totalCurrentAssets")),
            "current_liabilities": _f(b0.get("totalCurrentLiabilities")),
            "shares_outstanding": _f(i0.get("weightedAverageShsOut")),
            "gross_profit": _f(i0.get("grossProfit")),
            "total_revenue": _f(i0.get("revenue")),
            "operating_cash_flow": _f(
                c0.get("operatingCashFlow")
                or c0.get("netCashProvidedByOperatingActivities")
            ),
        },
        "prior": {
            "net_income": _f(i1.get("netIncome")),
            "total_assets": _f(b1.get("totalAssets")),
            "long_term_debt": _f(b1.get("longTermDebt")),
            "current_assets": _f(b1.get("totalCurrentAssets")),
            "current_liabilities": _f(b1.get("totalCurrentLiabilities")),
            "shares_outstanding": _f(i1.get("weightedAverageShsOut")),
            "gross_profit": _f(i1.get("grossProfit")),
            "total_revenue": _f(i1.get("revenue")),
        },
        "source": "fmp",
    }


# ─── Bear-case enrichment ───────────────────────────────────────────────────

def get_bear_case_context(ticker: str, max_chars: int = 4000) -> dict:
    """Return structured context for the LLM bear-case prompt.

    Each field is best-effort. Empty strings/lists indicate the data couldn't
    be fetched.
    """
    out = {
        "transcript_excerpt": "",
        "analyst_consensus": None,
        "insider_summary": "",
        "news_headlines": [],
        "sources": [],
    }

    if _fmp.is_enabled():
        transcript = _fmp.get_latest_transcript(ticker)
        if transcript and transcript.get("content"):
            content = transcript["content"]
            if len(content) > max_chars:
                content = content[-max_chars:]
            quarter = transcript.get("quarter")
            year = transcript.get("year")
            label = f"Q{quarter} {year}" if quarter and year else "latest"
            out["transcript_excerpt"] = (
                f"From the {label} earnings call (most recent Q&A segment):\n{content}"
            )
            out["sources"].append(f"FMP earnings call transcript ({label})")

        targets = _fmp.get_analyst_targets(ticker)
        if targets:
            out["analyst_consensus"] = targets
            out["sources"].append("FMP analyst price target consensus")

        trades = _fmp.get_insider_trades(ticker, limit=10)
        if trades:
            sells = [t for t in trades if (t.get("transactionType") or "").upper().startswith("S")]
            buys = [t for t in trades if (t.get("transactionType") or "").upper().startswith("P")]
            sell_value = sum(_f(t.get("securitiesTransacted")) or 0 for t in sells)
            buy_value = sum(_f(t.get("securitiesTransacted")) or 0 for t in buys)
            if sells or buys:
                out["insider_summary"] = (
                    f"Last 10 insider transactions: {len(sells)} sells "
                    f"({sell_value:,.0f} shares), {len(buys)} buys "
                    f"({buy_value:,.0f} shares)."
                )
                out["sources"].append("FMP insider transactions (Form 4)")

        news = _fmp.get_recent_news(ticker, limit=5)
        for item in news:
            title = item.get("title")
            if title:
                site = item.get("site", "")
                out["news_headlines"].append(
                    f"- {title}" + (f" ({site})" if site else "")
                )
        if out["news_headlines"]:
            out["sources"].append("FMP stock news feed")

    if not out["news_headlines"]:
        try:
            import yfinance as yf
            yf_news = getattr(yf.Ticker(ticker), "news", None) or []
            for item in yf_news[:5]:
                title = item.get("title") or (item.get("content") or {}).get("title")
                if title:
                    out["news_headlines"].append(f"- {title}")
            if out["news_headlines"]:
                out["sources"].append("yfinance news feed (fallback)")
        except Exception:
            pass

    return out
