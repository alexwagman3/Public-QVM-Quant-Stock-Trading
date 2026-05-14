"""Build the top-N portfolio: sector-capped selection, fractional-share
allocation, factor breakdown per stock, and AI-generated bear case.

Bear-case context (FMP transcripts, analyst targets, insider trades, news)
is passed to DeepSeek when DEEPSEEK_API_KEY is set; otherwise a deterministic
template is used.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests

from . import data_sources
from .config import CAPITAL, FACTOR_DISPLAY, MAX_SECTOR_PCT, TOP_N_REPORT

_DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
_DEEPSEEK_MODEL = "deepseek-chat"


def _format_raw(factor_name: str, raw_val) -> str:
    if pd.isna(raw_val):
        return "N/A"
    if factor_name == "inv_volatility_252d":
        return f"{raw_val:.1f}x"
    return f"{raw_val:+.1f}%"


def _format_z(z) -> str:
    if pd.isna(z):
        return "N/A"
    return f"{z:+.2f}"


def _z_color(z) -> str:
    if pd.isna(z):
        return "gray"
    if z > 0.5:
        return "green"
    if z < -0.5:
        return "red"
    return "amber"


def _template_bear_case(name: str, ticker: str, sector: str, row) -> tuple[str, str, str]:
    """Deterministic template bear case (fallback when DeepSeek unavailable)."""
    biz = f"{name} ({ticker}) operates in the {sector} sector."
    if "gross_profitability" in row and not pd.isna(row["gross_profitability"]):
        biz += f" Gross profitability of {row['gross_profitability']:.1f}%."
    biz += " Reflects core business economics and competitive positioning."

    focus_parts = []
    for fcol, flabel, _, _ in FACTOR_DISPLAY:
        z = row.get(f"z_{fcol}")
        if z is not None and not pd.isna(z):
            direction = "strong" if z > 0.5 else "weak" if z < -0.5 else "neutral"
            focus_parts.append(f"{flabel}: {direction} ({z:+.2f})")
    focus = f"Composite score {row['composite_score']:.1f}. " + ", ".join(focus_parts) + "."

    warnings = []
    for fcol, flabel, _, _ in FACTOR_DISPLAY:
        z = row.get(f"z_{fcol}")
        if z is not None and not pd.isna(z) and z < -0.5:
            warnings.append(f"{flabel} concerning at {z:+.2f} z-score")
    if not warnings:
        warnings.append("No major red flags from factor data alone.")
    warning = "; ".join(warnings) + "."

    return biz, focus, warning


def _llm_bear_case(name, ticker, sector, row) -> tuple[str | None, str | None, str | None]:
    """Use DeepSeek to write a structured skeptic report.

    Context priority (richest first):
      1. FMP earnings-call transcript Q&A
      2. FMP analyst consensus + price target
      3. FMP insider trading summary (Form 4)
      4. FMP/yfinance recent news headlines
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None, None, None

    factor_lines = []
    for fcol, flabel, _, _ in FACTOR_DISPLAY:
        z = row.get(f"z_{fcol}")
        if z is not None and not pd.isna(z):
            factor_lines.append(f"  {flabel}: z-score {z:+.2f}")
    factor_block = "\n".join(factor_lines) or "  (no factor data)"

    try:
        ctx = data_sources.get_bear_case_context(ticker, max_chars=3500)
    except Exception:
        ctx = {
            "transcript_excerpt": "",
            "analyst_consensus": None,
            "insider_summary": "",
            "news_headlines": [],
            "sources": [],
        }

    blocks: list[str] = []
    if ctx.get("transcript_excerpt"):
        blocks.append(ctx["transcript_excerpt"])
    if ctx.get("analyst_consensus"):
        ac = ctx["analyst_consensus"]
        target = ac.get("targetConsensus") or ac.get("targetMean")
        high, low = ac.get("targetHigh"), ac.get("targetLow")
        if any(v is not None for v in (target, high, low)):
            blocks.append(
                f"Analyst price target consensus: ${target} (high ${high}, low ${low})."
            )
    if ctx.get("insider_summary"):
        blocks.append(f"Insider activity: {ctx['insider_summary']}")
    if ctx.get("news_headlines"):
        blocks.append("Recent headlines:\n" + "\n".join(ctx["news_headlines"]))
    enriched = "\n\n".join(blocks) if blocks else "(no external context available)"
    sources_line = "; ".join(ctx.get("sources") or []) or "factor data only"

    prompt = (
        f"You are a skeptical equity analyst writing a brief report for a retail investor.\n"
        f"Stock: {name} ({ticker})\n"
        f"Sector: {sector}\n"
        f"Composite quant score: {row.get('composite_score', 0):.1f}/10\n"
        f"Piotroski F-Score (financial-strength gate, 0-9): {int(row.get('f_score', 0))}\n\n"
        f"Factor z-scores (positive = better):\n{factor_block}\n\n"
        f"External context ({sources_line}):\n{enriched}\n\n"
        "Write three sections in plain English. Each section must be 1-2 sentences. "
        "No jargon, no generic hedging. Be specific. When the transcript or analyst data "
        "reveals a real risk, cite it explicitly.\n\n"
        "Return JSON with exactly these keys (no preamble, no code fences):\n"
        '  "business": What the company actually does and how it makes money.\n'
        '  "focus":    Which 1-2 quant factors are driving the high rank, concretely.\n'
        '  "warning":  A specific skeptical bear case grounded in the external context.'
    )

    try:
        resp = requests.post(
            _DEEPSEEK_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": _DEEPSEEK_MODEL,
                "max_tokens": 600,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        text = (resp.json()["choices"][0]["message"]["content"] or "").strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:].lstrip()
            text = text.rstrip("`").strip()
        data = json.loads(text)
        return (
            (data.get("business") or "").strip(),
            (data.get("focus") or "").strip(),
            (data.get("warning") or "").strip(),
        )
    except Exception as e:
        print(f"  ! LLM call failed for {ticker}: {e}")
        return None, None, None


def _bar_width(z) -> int:
    if pd.isna(z):
        return 0
    return min(5, max(0, int((z + 2.5) / 0.6)))


def select_top_n(
    signals: pd.DataFrame,
    n_positions: int,
    max_sector_pct: float = MAX_SECTOR_PCT,
) -> pd.DataFrame:
    """Sector-capped greedy selection from a ranked signals DataFrame.

    Real funds enforce sector concentration limits. Without this, all top
    picks could come from a single sector in a hot regime. Walk the ranked
    list and skip any stock that would push its sector over max_sector_pct.
    """
    desired = min(n_positions, len(signals))
    sector_counts: dict[str, int] = {}
    selected: list[int] = []
    sector_cap = max(1, int(desired * max_sector_pct + 0.5))

    for idx, row in signals.iterrows():
        if len(selected) >= desired:
            break
        sec = row.get("sector", "Unknown") or "Unknown"
        if sector_counts.get(sec, 0) >= sector_cap:
            continue
        selected.append(idx)
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    if len(selected) < desired:
        for idx in signals.index:
            if idx not in selected:
                selected.append(idx)
                if len(selected) >= desired:
                    break

    return signals.loc[selected].copy()


def build_allocation(
    signals: pd.DataFrame,
    market_timing: dict,
    capital: float = CAPITAL,
) -> dict[str, Any]:
    """Compute fractional-share allocation given the regime-implied position
    count and equity percentage."""
    equity_pct = market_timing.get("equity_pct", 0.85)
    n_positions = market_timing.get("n_positions", 8)
    if n_positions <= 0 or len(signals) == 0:
        return {
            "capital": capital,
            "equity_pct": equity_pct,
            "n_positions": 0,
            "regime": market_timing.get("regime", "NEUTRAL"),
            "positions": [],
            "total_invested": 0.0,
            "cash_reserve": capital,
        }

    selected = select_top_n(signals, n_positions)
    position_value = (capital * equity_pct) / len(selected)
    positions = []
    for _, row in selected.iterrows():
        price = float(row["latest_price"]) if not pd.isna(row["latest_price"]) else 0.0
        shares = position_value / price if price > 0 else 0.0
        positions.append({
            "ticker": row["ticker"],
            "name": row.get("name", row["ticker"]),
            "sector": row.get("sector", "Unknown"),
            "latest_price": price,
            "shares": round(shares, 4),
            "position_value": round(position_value, 2),
            "weight_pct": round(position_value / capital * 100, 2),
        })
    total_invested = sum(p["position_value"] for p in positions)
    return {
        "capital": capital,
        "equity_pct": equity_pct,
        "n_positions": len(positions),
        "regime": market_timing.get("regime", "NEUTRAL"),
        "regime_score": market_timing.get("score", 50),
        "positions": positions,
        "total_invested": round(total_invested, 2),
        "cash_reserve": round(capital - total_invested, 2),
    }


def build_top_n_list(signals: pd.DataFrame, n: int = TOP_N_REPORT) -> list[dict]:
    """Build the report-ready list (factor breakdowns + AI bear case)."""
    print("=" * 60)
    print(f"SCREENER: Building top {n} list with factor breakdowns + bear case")
    print("=" * 60)

    top_df = signals.head(n).copy()
    out: list[dict] = []

    for _, row in top_df.iterrows():
        factors = []
        for fcol, flabel, ffmt, fwght in FACTOR_DISPLAY:
            raw_val = row.get(fcol)
            z_val = row.get(f"z_{fcol}")
            factors.append({
                "key": fcol,
                "label": flabel,
                "format": ffmt,
                "weight": fwght,
                "raw_value": None if pd.isna(raw_val) else float(raw_val),
                "z_score": None if pd.isna(z_val) else float(z_val),
                "formatted_raw": _format_raw(fcol, raw_val),
                "formatted_z": _format_z(z_val),
                "color": _z_color(z_val),
                "bar_width": _bar_width(z_val),
            })

        name = row.get("name", row.get("ticker", "?"))
        ticker = row["ticker"]
        sector = row.get("sector", "Unknown")

        biz, focus, warning = _llm_bear_case(name, ticker, sector, row)
        if not biz or not focus or not warning:
            biz, focus, warning = _template_bear_case(name, ticker, sector, row)
        else:
            print(f"  ✓ AI analysis for {ticker}")

        out.append({
            "rank": int(row.get("rank", 0)),
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "composite_score": float(row.get("composite_score", 0)),
            "latest_price": float(row.get("latest_price", 0)),
            "f_score": float(row.get("f_score", 0)),
            "factors": factors,
            "biz_text": biz,
            "focus_text": focus,
            "warning_text": warning,
        })

    print(f"  Built {len(out)} entries with {len(FACTOR_DISPLAY)} factor breakdowns each.")
    return out
