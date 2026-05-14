"""Market-timing v2: 6-category composite regime score (0-100).

Fetches all data via yfinance (free). Each indicator wrapped in try/except;
failure → neutral score 50. Returns a single dict consumed by the screener
and the HTML report.

Categories and weights:
  TREND        30%   SPY vs 200/50-DMA, 12m momentum, MA200 slope
  VOLATILITY   25%   VIX level, VIX/VIX3M term, realized 20d vs 60d
  CREDIT       20%   10Y-3M yield spread, HYG/LQD ratio
  BREADTH      10%   RSP/SPY (equal-weight vs cap-weight)
  SENTIMENT    10%   VIX 5-day change
  SAFE HAVEN    5%   GLD/SPY trend
"""

from __future__ import annotations

import numpy as np
import yfinance as yf


def _fetch_history(ticker: str, period: str = "2y", interval: str = "1d"):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _fetch_current_price(ticker: str):
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if hist is not None and not hist.empty:
            return float(hist["Close"].iloc[-1])
        return None
    except Exception:
        return None


def _interp_score(val, x1, y1, x2, y2, x3=None, y3=None, clamp=(0, 100)):
    try:
        points = sorted([(x1, y1), (x2, y2)] + ([(x3, y3)] if x3 is not None else []))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if val <= xs[0]:
            s = ys[0]
        elif val >= xs[-1]:
            s = ys[-1]
        else:
            s = 50
            for i in range(len(xs) - 1):
                if xs[i] <= val <= xs[i + 1]:
                    t = (val - xs[i]) / (xs[i + 1] - xs[i]) if xs[i + 1] != xs[i] else 0
                    s = ys[i] + t * (ys[i + 1] - ys[i])
                    break
        return float(np.clip(s, clamp[0], clamp[1]))
    except Exception:
        return 50.0


# ─── TREND (30%) ────────────────────────────────────────────────────────────


def _trend():
    scores, details = {}, {}
    spy_df = _fetch_history("SPY", period="2y")
    if spy_df is None or len(spy_df) < 252:
        return 50.0, {"spy_price": None, "ma200": None, "ma50": None}

    close = spy_df["Close"]
    cur = float(close.iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    details.update(spy_price=cur, ma200=ma200, ma50=ma50)

    try:
        spy_vs_200 = (cur - ma200) / ma200 * 100
        scores["spy_vs_200dma"] = _interp_score(spy_vs_200, -10, 0, 0, 50, 5, 100)
        details["spy_vs_200dma"] = spy_vs_200
    except Exception:
        scores["spy_vs_200dma"] = 50; details["spy_vs_200dma"] = None

    try:
        spy_vs_50 = (cur - ma50) / ma50 * 100
        scores["spy_vs_50dma"] = _interp_score(spy_vs_50, -5, 0, 0, 50, 2, 100)
        details["spy_vs_50dma"] = spy_vs_50
    except Exception:
        scores["spy_vs_50dma"] = 50; details["spy_vs_50dma"] = None

    try:
        price_252d_ago = float(close.iloc[-252])
        mom_12m = (cur - price_252d_ago) / price_252d_ago * 100
        scores["momentum_12m"] = _interp_score(mom_12m, -20, 0, 0, 50, 20, 100)
        details["momentum_12m"] = mom_12m
    except Exception:
        scores["momentum_12m"] = 50; details["momentum_12m"] = None

    try:
        ma200_series = close.rolling(200).mean()
        ma200_now = float(ma200_series.iloc[-1])
        ma200_21d_ago = float(ma200_series.iloc[-21])
        slope_pct = (ma200_now - ma200_21d_ago) / ma200_21d_ago * 100 if ma200_21d_ago else 0
        if slope_pct > 0.5:
            scores["ma200_slope"] = 75
        elif slope_pct < -0.5:
            scores["ma200_slope"] = 25
        else:
            scores["ma200_slope"] = 50
        details["ma200_slope"] = slope_pct
    except Exception:
        scores["ma200_slope"] = 50; details["ma200_slope"] = None

    trend_score = (
        scores.get("spy_vs_200dma", 50) * 0.40
        + scores.get("spy_vs_50dma", 50) * 0.30
        + scores.get("momentum_12m", 50) * 0.20
        + scores.get("ma200_slope", 50) * 0.10
    )
    return trend_score, details


# ─── VOLATILITY (25%) ───────────────────────────────────────────────────────


def _volatility():
    scores, details = {}, {}
    try:
        vix_df = _fetch_history("^VIX", period="1mo")
        vix = float(vix_df["Close"].iloc[-1]) if vix_df is not None and not vix_df.empty else None
        details["vix"] = vix
        if vix is None:
            raise ValueError
        if vix < 12: scores["vix_level"] = 100
        elif vix < 17: scores["vix_level"] = 85
        elif vix < 22: scores["vix_level"] = 65
        elif vix < 28: scores["vix_level"] = 40
        elif vix < 35: scores["vix_level"] = 20
        else: scores["vix_level"] = 0
    except Exception:
        scores["vix_level"] = 50; details["vix"] = None

    try:
        vix3m_df = _fetch_history("^VIX3M", period="1mo")
        vix_cur = details.get("vix")
        if vix3m_df is not None and not vix3m_df.empty and vix_cur:
            vix3m = float(vix3m_df["Close"].iloc[-1])
            ratio = vix3m / vix_cur
            scores["vix_term"] = _interp_score(ratio, 0.95, 20, 1.0, 50, 1.05, 85)
            details["vix3m_vix_ratio"] = ratio
        else:
            raise ValueError
    except Exception:
        scores["vix_term"] = 50; details["vix3m_vix_ratio"] = None

    try:
        spy_df = _fetch_history("SPY", period="1y")
        if spy_df is not None and len(spy_df) > 60:
            rets = spy_df["Close"].pct_change().dropna()
            vol20 = rets.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            vol60 = rets.rolling(60).std().iloc[-1] * np.sqrt(252) * 100
            if vol20 < vol60 * 0.95: scores["realized_vol"] = 80
            elif vol20 > vol60 * 1.05: scores["realized_vol"] = 30
            else: scores["realized_vol"] = 55
            details["vol20"] = vol20; details["vol60"] = vol60
        else:
            raise ValueError
    except Exception:
        scores["realized_vol"] = 50; details["vol20"] = None; details["vol60"] = None

    vol_score = (
        scores.get("vix_level", 50) * 0.50
        + scores.get("vix_term", 50) * 0.30
        + scores.get("realized_vol", 50) * 0.20
    )
    return vol_score, details


# ─── CREDIT/MACRO (20%) ─────────────────────────────────────────────────────


def _credit_macro():
    scores, details = {}, {}
    try:
        tnx = _fetch_current_price("^TNX")
        irx = _fetch_current_price("^IRX")
        if tnx is None or irx is None:
            raise ValueError
        spread = tnx - irx
        if spread >= 1.0: scores["yield_spread"] = 100
        elif spread >= 0.25: scores["yield_spread"] = 75
        elif spread >= 0.0: scores["yield_spread"] = 50
        elif spread >= -0.25: scores["yield_spread"] = 25
        else: scores["yield_spread"] = 0
        details["yield_spread"] = spread
    except Exception:
        scores["yield_spread"] = 50; details["yield_spread"] = None

    try:
        hyg = _fetch_current_price("HYG")
        lqd = _fetch_current_price("LQD")
        if hyg is None or lqd is None or lqd <= 0:
            raise ValueError
        ratio = hyg / lqd
        if ratio >= 0.92: scores["hyg_lqd_ratio"] = 100
        elif ratio >= 0.88: scores["hyg_lqd_ratio"] = 70
        elif ratio >= 0.84: scores["hyg_lqd_ratio"] = 40
        else: scores["hyg_lqd_ratio"] = 10
        details["hyg_lqd_ratio"] = ratio
    except Exception:
        scores["hyg_lqd_ratio"] = 50; details["hyg_lqd_ratio"] = None

    credit_score = (
        scores.get("yield_spread", 50) * 0.60
        + scores.get("hyg_lqd_ratio", 50) * 0.40
    )
    return credit_score, details


# ─── BREADTH (10%) ──────────────────────────────────────────────────────────


def _breadth():
    scores, details = {}, {}
    try:
        rsp = _fetch_history("RSP", period="6mo")
        spy = _fetch_history("SPY", period="6mo")
        if rsp is None or spy is None or len(rsp) <= 20 or len(spy) <= 20:
            raise ValueError
        ratio = rsp["Close"] / spy["Close"]
        ma20 = ratio.rolling(20).mean()
        recent, past = float(ma20.iloc[-1]), float(ma20.iloc[-6])
        if recent > past * 1.002: scores["rsp_spy_ratio"] = 75
        elif recent < past * 0.998: scores["rsp_spy_ratio"] = 25
        else: scores["rsp_spy_ratio"] = 50
        details["rsp_spy_ratio"] = float(ratio.iloc[-1])
        details["rsp_spy_trend"] = (recent / past - 1) * 100 if past else 0
    except Exception:
        scores["rsp_spy_ratio"] = 50
        details["rsp_spy_ratio"] = None
        details["rsp_spy_trend"] = None
    return scores.get("rsp_spy_ratio", 50), details


# ─── SENTIMENT (10%) ────────────────────────────────────────────────────────


def _sentiment():
    scores, details = {}, {}
    try:
        vix_df = _fetch_history("^VIX", period="1mo")
        if vix_df is None or len(vix_df) < 6:
            raise ValueError
        vix_now = float(vix_df["Close"].iloc[-1])
        vix_5d_ago = float(vix_df["Close"].iloc[-6])
        change_pct = (vix_now - vix_5d_ago) / vix_5d_ago * 100 if vix_5d_ago else 0
        if change_pct > 30: scores["vix_5d_change"] = 75
        elif change_pct > 15: scores["vix_5d_change"] = 60
        elif change_pct < -20: scores["vix_5d_change"] = 55
        else: scores["vix_5d_change"] = 50
        details["vix_5d_change_pct"] = change_pct
    except Exception:
        scores["vix_5d_change"] = 50; details["vix_5d_change_pct"] = None
    return scores.get("vix_5d_change", 50), details


# ─── SAFE HAVEN (5%) ────────────────────────────────────────────────────────


def _safe_haven():
    scores, details = {}, {}
    try:
        gld = _fetch_history("GLD", period="6mo")
        spy = _fetch_history("SPY", period="6mo")
        if gld is None or spy is None or len(gld) <= 20 or len(spy) <= 20:
            raise ValueError
        ratio = gld["Close"] / spy["Close"]
        ma20 = ratio.rolling(20).mean()
        recent, past = float(ma20.iloc[-1]), float(ma20.iloc[-6])
        if recent > past * 1.002: scores["gld_spy_ratio"] = 30
        elif recent < past * 0.998: scores["gld_spy_ratio"] = 70
        else: scores["gld_spy_ratio"] = 50
        details["gld_spy_ratio"] = float(ratio.iloc[-1])
        details["gld_spy_trend"] = (recent / past - 1) * 100 if past else 0
    except Exception:
        scores["gld_spy_ratio"] = 50
        details["gld_spy_ratio"] = None
        details["gld_spy_trend"] = None
    return scores.get("gld_spy_ratio", 50), details


# ─── Regime mapping ─────────────────────────────────────────────────────────


def _map_regime(score: float) -> tuple[str, float, int]:
    if score >= 80:  return "STRONG_BULL", 1.00, 10
    if score >= 65:  return "BULLISH",     0.85, 8
    if score >= 50:  return "NEUTRAL",     0.70, 6
    if score >= 35:  return "CAUTIOUS",    0.50, 4
    if score >= 20:  return "BEARISH",     0.25, 2
    return "CRISIS", 0.00, 0


def compute() -> dict:
    """Run all 6 sub-scores and return the market_timing dict."""
    print("=" * 60)
    print("MARKET TIMING: 6-category composite regime")
    print("=" * 60)

    print("  [1/6] TREND..."); trend, td = _trend()
    print("  [2/6] VOLATILITY..."); vol, vd = _volatility()
    print("  [3/6] CREDIT/MACRO..."); credit, cd = _credit_macro()
    print("  [4/6] BREADTH..."); breadth, bd = _breadth()
    print("  [5/6] SENTIMENT..."); sentiment, sd = _sentiment()
    print("  [6/6] SAFE HAVEN..."); safe, hd = _safe_haven()

    composite = (
        trend * 0.30
        + vol * 0.25
        + credit * 0.20
        + breadth * 0.10
        + sentiment * 0.10
        + safe * 0.05
    )
    composite = max(0, min(100, composite))
    regime, equity_pct, n_positions = _map_regime(composite)

    print(f"\n  COMPOSITE: {composite:.0f}/100 → {regime}")
    print(f"  Equity: {equity_pct*100:.0f}% | Positions: {n_positions}")

    return {
        "regime": regime,
        "score": round(composite),
        "composite": round(composite),
        "message": regime,
        "advice": f"Deploy {int(equity_pct * 100)}% equity ({n_positions} positions)",
        "equity_pct": equity_pct,
        "n_positions": n_positions,
        "cash_pct": 1.0 - equity_pct,
        "trend_score": round(trend),
        "vol_score": round(vol),
        "credit_score": round(credit),
        "breadth_score": round(breadth),
        "sentiment_score": round(sentiment),
        "safe_haven_score": round(safe),
        "spy_price": td.get("spy_price"),
        "ma200": td.get("ma200"),
        "ma50": td.get("ma50"),
        "spy_vs_200dma": td.get("spy_vs_200dma") or 0,
        "spy_vs_50dma": td.get("spy_vs_50dma"),
        "vix": vd.get("vix") or 0,
        "yield_spread": cd.get("yield_spread"),
        "hyg_lqd_ratio": cd.get("hyg_lqd_ratio"),
    }
