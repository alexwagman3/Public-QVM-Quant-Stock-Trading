"""Factor computation: Piotroski F-Score (Cell 3) + QVM scoring (Cell 4).

Implements the 7 academic factors:
  1. Jegadeesh-Titman 12-1 momentum         (Jegadeesh-Titman 1993)
  2. 6-month momentum                       (secondary trend factor)
  3. Inverse 252-day volatility             (Blitz & van Vliet 2007)
  4. Gross profitability (GP / TA)          (Novy-Marx 2013)
  5. Free cash flow yield (FCF / EV)        (Fama-French 2006)
  6. Sloan accruals                         (Sloan 1996)
  7. Inverse asset growth                   (Titman-Wei-Xie 2004)

Plus the Piotroski F-Score gate (Piotroski 2000).
"""

from __future__ import annotations

import sqlite3
from time import sleep

import numpy as np
import pandas as pd
import yfinance as yf

from . import data_sources
from .config import DB_PATH, WEIGHTS

_BATCH_SIZE = 5
_BATCH_DELAY = 0.5


def _safe_get(df, row_label, col_idx=0):
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


def _safe_shares(info_dict):
    if not info_dict:
        return None
    for key in (
        "sharesOutstanding",
        "shares outstanding",
        "impliedSharesOutstanding",
    ):
        val = info_dict.get(key)
        if val is not None and not pd.isna(val):
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


# ─── Piotroski F-Score (9 signals) ──────────────────────────────────────────


def compute_piotroski(db_path: str = DB_PATH) -> pd.DataFrame:
    """Compute the 9-point Piotroski F-Score for every ticker in
    `fundamentals`. Writes results to the `f_scores` table.

    Reference: Piotroski, Joseph D. (2000), "Value Investing: The Use of
    Historical Financial Statement Information to Separate Winners from
    Losers." Journal of Accounting Research.
    """
    print("=" * 60)
    print("FACTORS: Piotroski F-Score (9 binary signals)")
    print("=" * 60)

    conn = sqlite3.connect(db_path)
    fundamentals = pd.read_sql("SELECT * FROM fundamentals", conn)
    if fundamentals.empty:
        raise ValueError("fundamentals table is empty. Run pipeline first.")

    tickers = fundamentals["ticker"].unique().tolist()
    print(f"   Tickers to score: {len(tickers)}")

    results = []
    for idx, ticker in enumerate(tickers):
        if idx % 20 == 0:
            print(f"   ...{idx}/{len(tickers)} tickers processed")

        row = fundamentals[fundamentals["ticker"] == ticker]
        if row.empty:
            continue
        row = row.iloc[0]

        def _flt(v):
            if v is None:
                return None
            try:
                f = float(v)
                if pd.isna(f):
                    return None
                return f
            except (TypeError, ValueError):
                return None

        net_income = _flt(row.get("net_income"))
        total_assets = _flt(row.get("total_assets"))
        free_cash_flow = _flt(row.get("free_cash_flow"))
        gross_profit = _flt(row.get("gross_profit"))
        total_revenue = _flt(row.get("total_revenue"))

        ocf = long_term_debt = current_assets = current_liab = shares = None
        prior_net_income = prior_total_assets = None
        prior_total_debt = prior_current_assets = prior_current_liab = None
        prior_shares = prior_gross_profit = prior_total_revenue = None
        has_hist = False

        fmp_two_year = data_sources.two_year_balance_from_fmp(ticker)
        if fmp_two_year is not None:
            has_hist = True
            cur = fmp_two_year["curr"]
            pri = fmp_two_year["prior"]
            ocf = cur["operating_cash_flow"]
            long_term_debt = cur["long_term_debt"]
            current_assets = cur["current_assets"]
            current_liab = cur["current_liabilities"]
            shares = cur["shares_outstanding"]
            prior_net_income = pri["net_income"]
            prior_total_assets = pri["total_assets"]
            prior_total_debt = pri["long_term_debt"]
            prior_current_assets = pri["current_assets"]
            prior_current_liab = pri["current_liabilities"]
            prior_shares = pri["shares_outstanding"]
            prior_gross_profit = pri["gross_profit"]
            prior_total_revenue = pri["total_revenue"]

        if not has_hist:
            try:
                t = yf.Ticker(ticker)
                info = t.info or {}
                bs = t.balance_sheet
                fin = t.financials
                cf = t.cashflow

                ocf = _safe_get(cf, "Operating Cash Flow", 0)
                long_term_debt = _safe_get(bs, "Long Term Debt", 0) or _safe_get(
                    bs, "Long-Term Debt", 0
                )
                current_assets = _safe_get(bs, "Current Assets", 0) or _safe_get(
                    bs, "Total Current Assets", 0
                )
                current_liab = _safe_get(bs, "Current Liabilities", 0) or _safe_get(
                    bs, "Total Current Liabilities", 0
                )
                shares = _safe_shares(info) or _safe_get(bs, "Common Stock", 0)

                if bs is not None and len(bs.columns) >= 2:
                    prior_net_income = _safe_get(fin, "Net Income", 1)
                    prior_total_assets = _safe_get(bs, "Total Assets", 1)
                    prior_total_debt = _safe_get(bs, "Long Term Debt", 1) or _safe_get(
                        bs, "Long-Term Debt", 1
                    )
                    prior_current_assets = _safe_get(bs, "Current Assets", 1) or _safe_get(
                        bs, "Total Current Assets", 1
                    )
                    prior_current_liab = _safe_get(bs, "Current Liabilities", 1) or _safe_get(
                        bs, "Total Current Liabilities", 1
                    )
                    prior_shares = _safe_get(bs, "Common Stock", 1)
                    prior_gross_profit = _safe_get(fin, "Gross Profit", 1)
                    prior_total_revenue = _safe_get(fin, "Total Revenue", 1)
            except Exception:
                pass

        # Signal 1: ROA > 0
        if net_income is not None and total_assets:
            s_roa = 1.0 if (net_income / total_assets) > 0 else 0.0
        else:
            s_roa = np.nan

        # Signal 2: OCF > 0
        if ocf is not None and total_assets:
            s_ocf = 1.0 if (ocf / total_assets) > 0 else 0.0
        elif free_cash_flow is not None and total_assets:
            s_ocf = 1.0 if (free_cash_flow / total_assets) > 0 else 0.0
        else:
            s_ocf = np.nan

        # Signal 3: dROA > 0
        if (
            net_income is not None
            and total_assets
            and prior_net_income is not None
            and prior_total_assets
        ):
            s_droa = 1.0 if (net_income / total_assets) > (prior_net_income / prior_total_assets) else 0.0
        elif net_income is not None and prior_net_income is not None:
            s_droa = 1.0 if net_income > prior_net_income else 0.0
        else:
            s_droa = np.nan

        # Signal 4: Accruals (OCF > NI) — Sloan (1996) uses OCF, not FCF.
        # FCF = OCF − CapEx, so falling back to FCF biases the signal against
        # capital-intensive firms. Only fall back when OCF is truly missing.
        if ocf is not None and net_income is not None:
            s_accruals = 1.0 if ocf > net_income else 0.0
        elif free_cash_flow is not None and net_income is not None:
            s_accruals = 1.0 if free_cash_flow > net_income else 0.0
        else:
            s_accruals = np.nan

        # Signal 5: dLeverage < 0 (deleveraging)
        if (
            long_term_debt is not None
            and total_assets
            and prior_total_debt is not None
            and prior_total_assets
        ):
            s_dlev = 1.0 if (long_term_debt / total_assets) < (prior_total_debt / prior_total_assets) else 0.0
        elif long_term_debt is not None and prior_total_debt is not None:
            s_dlev = 1.0 if long_term_debt < prior_total_debt else 0.0
        else:
            s_dlev = np.nan

        # Signal 6: dCurrent Ratio > 0
        if (
            current_assets is not None
            and current_liab
            and prior_current_assets is not None
            and prior_current_liab
        ):
            s_dcurr = 1.0 if (current_assets / current_liab) > (prior_current_assets / prior_current_liab) else 0.0
        elif current_assets is not None and current_liab:
            s_dcurr = 0.5
        else:
            s_dcurr = np.nan

        # Signal 7: dShares <= 0 (no dilution)
        if shares is not None and prior_shares:
            s_dshares = 1.0 if shares <= prior_shares else 0.0
        elif shares is not None:
            s_dshares = 0.5
        else:
            s_dshares = np.nan

        # Signal 8: dGross Margin > 0 — single-year case is neutral (0.5),
        # not unfavorable (0).
        if (
            gross_profit is not None
            and total_revenue
            and prior_gross_profit is not None
            and prior_total_revenue
        ):
            s_dmargin = 1.0 if (gross_profit / total_revenue) > (prior_gross_profit / prior_total_revenue) else 0.0
        elif gross_profit is not None and total_revenue:
            s_dmargin = 0.5
        else:
            s_dmargin = np.nan

        # Signal 9: dAsset Turnover > 0
        if (
            total_revenue is not None
            and total_assets
            and prior_total_revenue is not None
            and prior_total_assets
        ):
            s_dturn = 1.0 if (total_revenue / total_assets) > (prior_total_revenue / prior_total_assets) else 0.0
        elif total_revenue is not None and prior_total_revenue:
            s_dturn = 1.0 if ((total_revenue - prior_total_revenue) / abs(prior_total_revenue)) > 0 else 0.0
        else:
            s_dturn = np.nan

        signals = [
            s_roa, s_ocf, s_droa, s_accruals, s_dlev,
            s_dcurr, s_dshares, s_dmargin, s_dturn,
        ]
        available = [s for s in signals if not pd.isna(s)]
        if len(available) >= 7:
            quality = "complete"
        elif len(available) >= 4:
            quality = "partial"
        else:
            quality = "estimated"

        f_score = sum(0 if pd.isna(s) else s for s in signals)
        results.append({
            "ticker": ticker,
            "f_score": round(f_score, 1),
            "roa_positive":         1 if s_roa == 1.0 else 0 if s_roa == 0.0 else None,
            "ocf_positive":         1 if s_ocf == 1.0 else 0 if s_ocf == 0.0 else None,
            "droa_positive":        1 if s_droa == 1.0 else 0 if s_droa == 0.0 else None,
            "accruals_good":        1 if s_accruals == 1.0 else 0 if s_accruals == 0.0 else None,
            "dleverage_positive":   1 if s_dlev == 1.0 else 0 if s_dlev == 0.0 else None,
            "dcurrent_ratio_positive": 1 if s_dcurr == 1.0 else 0 if s_dcurr == 0.0 else None,
            "dshares_non_dilutive": 1 if s_dshares == 1.0 else 0 if s_dshares == 0.0 else None,
            "dmargin_positive":     1 if s_dmargin == 1.0 else 0 if s_dmargin == 0.0 else None,
            "dturnover_positive":   1 if s_dturn == 1.0 else 0 if s_dturn == 0.0 else None,
            "data_quality": quality,
            "piotroski_pass": "PASS" if f_score >= 6 else "FAIL",
            "signals_available": len(available),
            "signals_missing": 9 - len(available),
        })

        if (idx + 1) % _BATCH_SIZE == 0:
            sleep(_BATCH_DELAY)

    f_score_df = pd.DataFrame(results)
    conn.execute("DROP TABLE IF EXISTS f_scores")
    f_score_df.to_sql("f_scores", conn, if_exists="replace", index=False)
    conn.close()

    n_pass = int((f_score_df["f_score"] >= 6).sum())
    print(f"\n   Saved {len(f_score_df)} F-Score records.")
    print(f"   PASS (F-Score >= 6): {n_pass} / {len(f_score_df)}")
    return f_score_df


# ─── QVM scoring engine (z-score normalize, composite, F-Score gate) ────────


def compute_qvm_signals(db_path: str = DB_PATH, weights: dict | None = None) -> pd.DataFrame:
    """Compute the 7 QVM factors, z-score normalize, build composite score,
    apply Piotroski gate, and return the ranked signals DataFrame.
    """
    print("=" * 60)
    print("FACTORS: QVM Scoring Engine")
    print("=" * 60)
    weights = weights or WEIGHTS

    fw = {f"z_{k}": v for k, v in weights.items() if k != "piotroski_gate"}

    conn = sqlite3.connect(db_path)
    prices_raw = pd.read_sql("SELECT * FROM prices", conn, parse_dates=["date"])
    if prices_raw.empty:
        raise ValueError("prices table is empty — run the data pipeline first.")
    price_matrix = (
        prices_raw.pivot(index="date", columns="ticker", values="close").sort_index()
    )

    fundamentals_raw = pd.read_sql("SELECT * FROM fundamentals", conn)
    if fundamentals_raw.empty:
        raise ValueError("fundamentals table is empty.")
    fundamentals = (
        fundamentals_raw.sort_values(["ticker", "fiscal_date"], ascending=[True, False])
        .drop_duplicates(subset="ticker", keep="first")
    )
    f_scores_df = pd.read_sql("SELECT ticker, f_score FROM f_scores", conn)
    company_info = pd.read_sql("SELECT ticker, name, sector FROM company_info", conn)
    conn.close()

    signals = fundamentals[["ticker"]].copy()
    signals = signals.merge(company_info[["ticker", "name", "sector"]], on="ticker", how="left")
    signals["name"] = signals["name"].fillna(signals["ticker"])
    signals["sector"] = signals["sector"].fillna("Unknown")

    # ── Momentum factors
    monthly_prices = price_matrix.resample("ME").last().dropna(how="all")
    if len(monthly_prices) < 14:
        raise ValueError(
            f"Need >= 14 months of price data. Got {len(monthly_prices)} months."
        )
    jt_mom = (monthly_prices.iloc[-2] / monthly_prices.iloc[-14] - 1).rename("jt_12_1_momentum")

    if len(price_matrix) >= 126:
        m6 = (price_matrix.iloc[-1] / price_matrix.iloc[-126] - 1).rename("momentum_6m")
    else:
        m6 = pd.Series(np.nan, index=price_matrix.columns).rename("momentum_6m")

    # ── Volatility
    log_returns = np.log(price_matrix / price_matrix.shift(1))
    rolling_std = log_returns.rolling(window=252, min_periods=180).std()
    ann_vol = rolling_std * np.sqrt(252)
    inv_vol = (1.0 / ann_vol).replace([np.inf, -np.inf], np.nan)
    finite_max = inv_vol.stack().dropna().quantile(0.99)
    inv_vol = inv_vol.clip(upper=finite_max)
    inv_vol_latest = inv_vol.iloc[-1].rename("inv_volatility_252d")

    # ── Quality + Value
    fund_idx = fundamentals.set_index("ticker")
    gp = fund_idx["gross_profit"] / fund_idx["total_assets"]
    fcf_yld = fund_idx["free_cash_flow"] / fund_idx["enterprise_value"]

    # Accruals — Sloan (1996) defines accruals using OCF. Prefer OCF; fall
    # back to FCF only when OCF is missing.
    if "operating_cash_flow" in fund_idx.columns:
        cash_proxy = fund_idx["operating_cash_flow"].fillna(fund_idx["free_cash_flow"])
    else:
        cash_proxy = fund_idx["free_cash_flow"]
    acc = -((fund_idx["net_income"] - cash_proxy) / fund_idx["total_assets"])

    # ── Inv asset growth (re-fetch 2-year history)
    iag_rows = []
    for i, ticker in enumerate(signals["ticker"].tolist()):
        if i % 20 == 0 and i > 0:
            print(f"      ...{i}/{len(signals)} balance sheets fetched")
        growth = np.nan
        fmp_two = data_sources.two_year_balance_from_fmp(ticker)
        if fmp_two is not None:
            ca = fmp_two["curr"]["total_assets"]
            pa = fmp_two["prior"]["total_assets"]
            if ca and pa:
                growth = -(ca / pa - 1)
        if pd.isna(growth):
            try:
                t = yf.Ticker(ticker)
                bs = t.balance_sheet
                if bs is not None and not bs.empty and len(bs.columns) >= 2:
                    ca = _safe_get(bs, "Total Assets", 0)
                    pa = _safe_get(bs, "Total Assets", 1)
                    if ca is not None and pa:
                        growth = -(ca / pa - 1)
                sleep(0.3)
            except Exception:
                pass
        iag_rows.append({"ticker": ticker, "inv_asset_growth": growth})
    inv_asset_df = pd.DataFrame(iag_rows)

    # ── Assemble
    signals = signals.set_index("ticker")
    signals["jt_12_1_momentum"] = jt_mom
    signals["momentum_6m"] = m6
    signals["inv_volatility_252d"] = inv_vol_latest
    signals["gross_profitability"] = gp
    signals["fcf_yield"] = fcf_yld
    signals["accruals"] = acc
    signals = signals.reset_index()
    signals = signals.merge(inv_asset_df, on="ticker", how="left")
    signals = signals.set_index("ticker")
    signals["latest_price"] = price_matrix.iloc[-1]
    signals = signals.reset_index()

    # ── Sector-aware median imputation
    impute_cols = [
        "gross_profitability", "fcf_yield", "accruals", "inv_asset_growth",
        "jt_12_1_momentum", "momentum_6m", "inv_volatility_252d",
    ]
    for col in impute_cols:
        if signals[col].isna().any():
            sector_medians = signals.groupby("sector")[col].transform("median")
            signals[col] = signals[col].fillna(sector_medians)
            signals[col] = signals[col].fillna(signals[col].median())

    # ── Z-score normalization (winsorize 1/99, clip [-3, 3])
    raw_factor_map = {
        "z_jt_12_1_momentum":   "jt_12_1_momentum",
        "z_momentum_6m":         "momentum_6m",
        "z_inv_volatility_252d": "inv_volatility_252d",
        "z_gross_profitability": "gross_profitability",
        "z_fcf_yield":           "fcf_yield",
        "z_accruals":            "accruals",
        "z_inv_asset_growth":    "inv_asset_growth",
    }
    for z_col, raw_col in raw_factor_map.items():
        if z_col not in fw:
            continue
        raw = signals[raw_col]
        lo, hi = raw.quantile(0.01), raw.quantile(0.99)
        winsorized = raw.clip(lower=lo, upper=hi)
        mean, std = winsorized.mean(), winsorized.std()
        if std == 0 or pd.isna(std):
            signals[z_col] = 0.0
        else:
            signals[z_col] = ((winsorized - mean) / std).clip(-3, 3)

    # ── Composite score
    signals["composite_score"] = sum(
        signals[z_col] * w for z_col, w in fw.items() if z_col in signals.columns
    )

    # ── F-Score gate
    f_scores_df = f_scores_df.set_index("ticker")["f_score"]
    signals = signals.set_index("ticker")
    signals["f_score"] = f_scores_df.reindex(signals.index).fillna(5)
    signals = signals.reset_index()
    gate_threshold = weights.get("piotroski_gate", 6)
    signals = signals[signals["f_score"] >= gate_threshold].copy()
    signals = signals.sort_values("composite_score", ascending=False).reset_index(drop=True)
    signals["rank"] = range(1, len(signals) + 1)

    output_cols = [
        "ticker", "name", "sector", "rank", "composite_score", "latest_price",
        "z_jt_12_1_momentum", "z_momentum_6m", "z_inv_volatility_252d",
        "z_gross_profitability", "z_fcf_yield", "z_accruals", "z_inv_asset_growth",
        "jt_12_1_momentum", "momentum_6m", "inv_volatility_252d",
        "gross_profitability", "fcf_yield", "accruals", "inv_asset_growth",
        "f_score",
    ]
    for col in output_cols:
        if col not in signals.columns:
            signals[col] = np.nan
    signals = signals[output_cols].copy()

    print(f"\n   {len(signals)} stocks passed F-Score gate (>= {gate_threshold}).")
    print(f"   Composite score: mean={signals['composite_score'].mean():.4f}, "
          f"std={signals['composite_score'].std():.4f}")
    return signals
