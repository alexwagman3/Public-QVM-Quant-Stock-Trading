"""Unified data pipeline: download prices, score by momentum/vol, fetch
fundamentals for the top N, persist to SQLite.

Tables written: prices, fundamentals, company_info.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from time import sleep

import numpy as np
import pandas as pd
import yfinance as yf

from . import data_sources, fmp_client
from .config import DB_PATH, FUNDAMENTAL_TOP_N, TICKERS


def _safe_get(df, row_label, col_idx=0):
    """Safely extract a scalar value from a yfinance DataFrame."""
    if df is None or df.empty:
        return None
    if row_label not in df.index:
        return None
    cols = df.columns.tolist()
    if col_idx >= len(cols):
        return None
    val = df.loc[row_label, cols[col_idx]]
    if pd.isna(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _fetch_prices(tickers: list[str]) -> pd.DataFrame:
    prices_all = yf.download(
        tickers=tickers,
        period="2y",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if isinstance(prices_all.columns, pd.MultiIndex):
        closes = prices_all.xs("Close", level=1, axis=1)
    else:
        closes = (
            prices_all[["Close"]].copy()
            if "Close" in prices_all.columns
            else prices_all.iloc[:, 0:1]
        )
        closes.columns = [tickers[0]] if len(tickers) == 1 else closes.columns

    closes = closes.reset_index().melt(
        id_vars="Date", var_name="ticker", value_name="close"
    )
    closes.rename(columns={"Date": "date"}, inplace=True)
    closes.dropna(subset=["close"], inplace=True)
    closes["date"] = pd.to_datetime(closes["date"])
    return closes


def _select_top_n_momentum(closes: pd.DataFrame, n: int) -> list[str]:
    pivot = closes.pivot(index="date", columns="ticker", values="close").sort_index()

    monthly_prices = pivot.resample("ME").last().dropna(how="all")
    if len(monthly_prices) >= 14:
        price_1m_ago = monthly_prices.iloc[-2]
        price_13m_ago = monthly_prices.iloc[-14]
        jt_momentum = (price_1m_ago / price_13m_ago) - 1
    else:
        jt_momentum = pd.Series(0, index=pivot.columns)

    if len(pivot) >= 126:
        mom_6m = (pivot.iloc[-1] / pivot.iloc[-126]) - 1
    else:
        mom_6m = pd.Series(0, index=pivot.columns)

    log_rets = np.log(pivot / pivot.shift(1))
    rolling_vol = log_rets.iloc[-252:].std() * np.sqrt(252)
    inv_vol = 1.0 / rolling_vol.replace(0, np.nan)

    jt_rank = jt_momentum.rank(pct=True, na_option="bottom")
    iv_rank = inv_vol.rank(pct=True, na_option="bottom")
    composite_rank = 0.5 * jt_rank + 0.5 * iv_rank
    _ = mom_6m  # kept for parity with Cell 4

    return composite_rank.dropna().sort_values(ascending=False).head(n).index.tolist()


def _fetch_fundamentals(top_tickers: list[str]) -> tuple[list[dict], list[dict]]:
    fund_records: list[dict] = []
    info_records: list[dict] = []
    fmp_hits = yf_hits = 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for i, ticker in enumerate(top_tickers):
        if i % 10 == 0:
            print(f"      ...{i:3d}/{len(top_tickers)}: {ticker} (fmp={fmp_hits} yf={yf_hits})")

        record = info_rec = None

        fmp_data = data_sources.fundamentals_from_fmp(ticker)
        if fmp_data is not None and fmp_data.get("total_assets"):
            fmp_hits += 1
            record = {
                "ticker": ticker,
                "fiscal_date": fmp_data["fiscal_date"],
                "gross_profit": fmp_data["gross_profit"],
                "total_assets": fmp_data["total_assets"],
                "free_cash_flow": fmp_data["free_cash_flow"],
                "operating_cash_flow": fmp_data["operating_cash_flow"],
                "enterprise_value": fmp_data["enterprise_value"],
                "stockholders_equity": fmp_data["stockholders_equity"],
                "net_income": fmp_data["net_income"],
                "total_revenue": fmp_data["total_revenue"],
                "total_debt": fmp_data["total_debt"],
                "current_assets": fmp_data["current_assets"],
                "current_liabilities": fmp_data["current_liabilities"],
                "shares_outstanding": fmp_data["shares_outstanding"],
                "fcf_yf": fmp_data["free_cash_flow"],
                "data_source": "fmp",
                "fetched_at": now,
            }
            info_rec = {
                "ticker": ticker,
                "name": fmp_data["name"],
                "sector": fmp_data["sector"],
                "industry": fmp_data["industry"],
            }

        if record is None:
            try:
                t = yf.Ticker(ticker)
                info = t.info or {}
                fin = t.financials
                bs = t.balance_sheet
                cf = t.cashflow
                if fin is None or fin.empty or bs is None or bs.empty:
                    continue

                c1_fin = fin.columns[0]
                c1_cf = cf.columns[0] if cf is not None and not cf.empty else None
                fiscal_date = (
                    str(c1_fin.date()) if hasattr(c1_fin, "date") else str(c1_fin)
                )
                ocf_yf = _safe_get(cf, "Operating Cash Flow", 0) if c1_cf else None

                record = {
                    "ticker": ticker,
                    "fiscal_date": fiscal_date,
                    "gross_profit": _safe_get(fin, "Gross Profit", 0),
                    "total_assets": _safe_get(bs, "Total Assets", 0),
                    "free_cash_flow": (
                        _safe_get(cf, "Free Cash Flow", 0)
                        if c1_cf
                        else info.get("freeCashflow")
                    ),
                    "operating_cash_flow": ocf_yf,
                    "enterprise_value": info.get("enterpriseValue"),
                    "stockholders_equity": _safe_get(bs, "Stockholders Equity", 0),
                    "net_income": _safe_get(fin, "Net Income", 0),
                    "total_revenue": _safe_get(fin, "Total Revenue", 0),
                    "total_debt": _safe_get(bs, "Total Debt", 0),
                    "current_assets": _safe_get(bs, "Current Assets", 0),
                    "current_liabilities": _safe_get(bs, "Current Liabilities", 0),
                    "shares_outstanding": info.get("sharesOutstanding"),
                    "fcf_yf": info.get("freeCashflow"),
                    "data_source": "yfinance",
                    "fetched_at": now,
                }
                info_rec = {
                    "ticker": ticker,
                    "name": info.get("longName", ticker),
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                }
                yf_hits += 1
                sleep(0.5)
            except Exception:
                continue

        fund_records.append(record)
        if info_rec is not None:
            info_records.append(info_rec)

    print(f"    Fundamentals fetched: FMP={fmp_hits}, yfinance={yf_hits}")
    return fund_records, info_records


def run(
    db_path: str = DB_PATH,
    tickers: list[str] | None = None,
    top_n: int = FUNDAMENTAL_TOP_N,
) -> None:
    """Run the full data pipeline. Writes prices, fundamentals, and
    company_info tables to SQLite at ``db_path``."""
    tickers = tickers or TICKERS
    print("=" * 60)
    print("PIPELINE: Unified Data Pipeline")
    print("=" * 60)
    if fmp_client.is_enabled():
        print("    FMP_API_KEY detected → using FMP for fundamentals (yfinance fallback).")
    else:
        print("    FMP_API_KEY not set → using yfinance for everything.")

    print(f"\n[1/3] Fetching prices for {len(tickers)} tickers via yfinance...")
    closes = _fetch_prices(tickers)

    conn = sqlite3.connect(db_path)
    closes.to_sql("prices", conn, if_exists="replace", index=False)
    print(f"    Saved {len(closes):,} price rows to table 'prices'")

    print("\n[2/3] Ranking by momentum + inv-volatility to pick top N for deep dive...")
    top_tickers = _select_top_n_momentum(closes, top_n)
    print(f"    Top {len(top_tickers)} tickers selected for fundamental deep-dive")

    print(f"\n[3/3] Fetching fundamentals for top {len(top_tickers)} (FMP → yfinance fallback)...")
    fund_records, info_records = _fetch_fundamentals(top_tickers)

    if info_records:
        pd.DataFrame(info_records).to_sql("company_info", conn, if_exists="replace", index=False)
        print(f"    Saved {len(info_records)} rows to 'company_info'")
    if fund_records:
        pd.DataFrame(fund_records).to_sql("fundamentals", conn, if_exists="replace", index=False)
        print(f"    Saved {len(fund_records)} rows to 'fundamentals'")
    else:
        print("    WARNING: No fundamental records fetched!")

    conn.close()
    print(f"\nPIPELINE COMPLETE: {len(fund_records)} fundamentals loaded.")
