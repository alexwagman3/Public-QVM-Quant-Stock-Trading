"""Render the top-N list as a self-contained HTML report. Also write
signals.json and allocation.json to the output directory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ensure_output_dir


# ─── Regime-aware color scheme ──────────────────────────────────────────────


def _regime_colors(regime: str) -> dict[str, str]:
    r = (regime or "NEUTRAL").upper()
    if r in ("STRONG_BULL", "BULLISH"):
        return {
            "header_bg": "linear-gradient(135deg, #1a5c3a 0%, #2d8a5e 50%, #3cb371 100%)",
            "header_border": "#2d8a5e",
            "accent": "#3cb371",
            "row_hover": "rgba(61,179,113,0.08)",
        }
    if r in ("BEARISH", "CRISIS"):
        return {
            "header_bg": "linear-gradient(135deg, #8b2020 0%, #c0392b 50%, #e74c3c 100%)",
            "header_border": "#c0392b",
            "accent": "#e74c3c",
            "row_hover": "rgba(231,76,60,0.08)",
        }
    return {
        "header_bg": "linear-gradient(135deg, #7a5c1a 0%, #c9921a 50%, #e6a817 100%)",
        "header_border": "#c9921a",
        "accent": "#e6a817",
        "row_hover": "rgba(230,168,23,0.08)",
    }


# ─── HTML helpers ───────────────────────────────────────────────────────────


def _factor_bar(factor: dict) -> str:
    color = factor["color"]
    bar_w = factor["bar_width"]
    if color == "green":
        bar_color, text_color = "#2d8a5e", "#1a5c3a"
    elif color == "red":
        bar_color, text_color = "#c0392b", "#8b2020"
    else:
        bar_color, text_color = "#c9921a", "#7a5c1a"
    filled = "&#9608;" * bar_w
    empty = "&#9617;" * (5 - bar_w)
    return (
        '<div style="font-size:10px;line-height:1.6;display:flex;align-items:center;gap:4px;white-space:nowrap;">'
        f'<span style="color:#555;width:48px;text-align:right;font-weight:600;">{factor["label"]}</span>'
        f'<span style="color:{bar_color};letter-spacing:-1px;">{filled}{empty}</span>'
        f'<span style="color:{text_color};font-family:monospace;font-size:9px;min-width:36px;">{factor["formatted_z"]}</span>'
        '</div>'
    )


def _factor_grid(factors: list[dict]) -> str:
    mid = (len(factors) + 1) // 2
    left = "".join(_factor_bar(f) for f in factors[:mid])
    right = "".join(_factor_bar(f) for f in factors[mid:])
    return (
        '<div style="display:flex;gap:8px;">'
        f'<div style="flex:1;">{left}</div>'
        f'<div style="flex:1;">{right}</div>'
        '</div>'
    )


def _ai_analysis(entry: dict) -> str:
    biz = entry["biz_text"]
    warn = entry["warning_text"]
    biz_t = biz[:140] + ("..." if len(biz) > 140 else "")
    warn_t = warn[:140] + ("..." if len(warn) > 140 else "")
    return (
        '<div style="display:flex;flex-direction:column;gap:4px;">'
        '<div style="border-left:3px solid #2d8a5e;padding:4px 6px;background:rgba(45,138,94,0.06);'
        'border-radius:0 4px 4px 0;font-size:10px;line-height:1.5;color:#1a3a2a;">'
        '<strong style="color:#1a5c3a;font-size:9px;text-transform:uppercase;letter-spacing:0.5px;">Business</strong><br/>'
        f'{biz_t}</div>'
        '<div style="border-left:3px solid #c0392b;padding:4px 6px;background:rgba(192,57,43,0.06);'
        'border-radius:0 4px 4px 0;font-size:10px;line-height:1.5;color:#3a1a1a;">'
        '<strong style="color:#8b2020;font-size:9px;text-transform:uppercase;letter-spacing:0.5px;">Caution</strong><br/>'
        f'{warn_t}</div></div>'
    )


def _composite_badge(score: float) -> str:
    if score >= 7.0:
        bg, color = "#2d8a5e", "#fff"
    elif score >= 4.0:
        bg, color = "#c9921a", "#fff"
    else:
        bg, color = "#c0392b", "#fff"
    return (
        f'<span style="display:inline-block;background:{bg};color:{color};padding:3px 10px;'
        f'border-radius:12px;font-weight:700;font-size:13px;">{score:.1f}</span>'
    )


def _rank_badge(rank: int) -> str:
    if rank <= 3:
        bg, color = "#f0c040", "#5a4a00"
    elif rank <= 10:
        bg, color = "#e0e0e0", "#555"
    else:
        bg, color = "#f5f5f5", "#888"
    return (
        f'<span style="display:inline-flex;align-items:center;justify-content:center;width:28px;'
        f'height:28px;border-radius:50%;background:{bg};color:{color};font-weight:700;font-size:12px;">'
        f'{rank}</span>'
    )


def _detail_row(entry: dict, colors: dict, total_count: int) -> str:
    factor_rows = []
    for f in entry["factors"]:
        if f["color"] == "green":
            bar_color = "#2d8a5e"
        elif f["color"] == "red":
            bar_color = "#c0392b"
        else:
            bar_color = "#c9921a"
        bar_w = f["bar_width"]
        filled = "&#9608;" * bar_w
        empty = "&#9617;" * (5 - bar_w)
        wtpct = f"{f['weight'] * 100:.0f}%"
        factor_rows.append(
            '<tr>'
            f'<td style="padding:4px 8px;font-size:11px;color:#444;font-weight:600;">{f["label"]}</td>'
            f'<td style="padding:4px 8px;font-size:11px;color:#666;font-family:monospace;text-align:right;">{f["formatted_raw"]}</td>'
            f'<td style="padding:4px 8px;font-size:11px;color:#666;font-family:monospace;text-align:right;">{f["formatted_z"]}</td>'
            f'<td style="padding:4px 8px;font-size:11px;color:#888;text-align:right;">{wtpct}</td>'
            f'<td style="padding:4px 8px;font-size:11px;color:{bar_color};letter-spacing:-1px;">{filled}{empty}</td>'
            '</tr>'
        )
    factor_html = "\n".join(factor_rows)
    composite = entry["composite_score"]
    rank = entry["rank"]
    pctile = (1 - rank / total_count) * 100 if total_count else 0

    return (
        '<td colspan="6" style="padding:0;border:none;">'
        '<div class="detail-panel" style="display:none;background:#fafbfc;padding:16px 20px;border-top:1px solid #e8e8e8;">'
        '<div style="display:flex;gap:16px;">'
        '<div style="flex:1.4;min-width:0;">'
        '<div style="font-size:11px;font-weight:700;color:#2c3e50;text-transform:uppercase;letter-spacing:0.8px;'
        f'margin-bottom:10px;border-bottom:2px solid {colors["accent"]};padding-bottom:4px;display:inline-block;">'
        'AI Fundamental Analysis</div>'
        '<div style="border-left:3px solid #2d8a5e;padding:10px 14px;background:rgba(45,138,94,0.04);'
        'border-radius:0 6px 6px 0;margin-bottom:8px;">'
        '<div style="font-size:9px;font-weight:700;color:#1a5c3a;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">The Business</div>'
        f'<div style="font-size:11px;line-height:1.7;color:#2c3e50;">{entry["biz_text"]}</div></div>'
        '<div style="border-left:3px solid #4a86c8;padding:10px 14px;background:rgba(74,134,200,0.04);'
        'border-radius:0 6px 6px 0;margin-bottom:8px;">'
        '<div style="font-size:9px;font-weight:700;color:#2a5a8a;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Focus</div>'
        f'<div style="font-size:11px;line-height:1.7;color:#2c3e50;">{entry["focus_text"]}</div></div>'
        '<div style="border-left:3px solid #c0392b;padding:10px 14px;background:rgba(192,57,43,0.04);'
        'border-radius:0 6px 6px 0;">'
        '<div style="font-size:9px;font-weight:700;color:#8b2020;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Caution</div>'
        f'<div style="font-size:11px;line-height:1.7;color:#5a2a2a;">{entry["warning_text"]}</div></div>'
        '</div>'
        '<div style="flex:1.0;min-width:0;">'
        '<div style="font-size:11px;font-weight:700;color:#2c3e50;text-transform:uppercase;letter-spacing:0.8px;'
        f'margin-bottom:10px;border-bottom:2px solid {colors["accent"]};padding-bottom:4px;display:inline-block;">'
        'Factor Breakdown</div>'
        '<table style="width:100%;border-collapse:collapse;font-size:11px;">'
        '<thead><tr style="border-bottom:1px solid #ddd;">'
        '<th style="padding:4px 8px;text-align:left;font-size:9px;color:#888;text-transform:uppercase;letter-spacing:0.5px;">Factor</th>'
        '<th style="padding:4px 8px;text-align:right;font-size:9px;color:#888;text-transform:uppercase;letter-spacing:0.5px;">Raw</th>'
        '<th style="padding:4px 8px;text-align:right;font-size:9px;color:#888;text-transform:uppercase;letter-spacing:0.5px;">Z</th>'
        '<th style="padding:4px 8px;text-align:right;font-size:9px;color:#888;text-transform:uppercase;letter-spacing:0.5px;">Wt</th>'
        '<th style="padding:4px 8px;font-size:9px;color:#888;text-transform:uppercase;letter-spacing:0.5px;">Bar</th>'
        '</tr></thead><tbody>' + factor_html + '</tbody></table>'
        '<div style="margin-top:10px;padding:8px 12px;background:linear-gradient(135deg,rgba(44,62,80,0.06),rgba(44,62,80,0.02));'
        'border-radius:6px;border:1px solid rgba(44,62,80,0.1);">'
        '<div style="font-size:12px;font-weight:700;color:#2c3e50;text-align:center;">'
        f'COMPOSITE: {composite:.1f} <span style="font-weight:400;color:#888;font-size:10px;">'
        f'(Rank #{rank}, top {pctile:.0f}%)</span>'
        '</div></div></div></div></div></td>'
    )


def render_html(top_n_list: list[dict], market_timing: dict) -> tuple[str, str]:
    """Return (body_fragment, full_html). The fragment is suitable for
    embedding inside an existing page (Streamlit, Astro, etc.).
    """
    regime = market_timing.get("regime", "NEUTRAL")
    regime_score = market_timing.get("score", 50)
    colors = _regime_colors(regime)
    today = datetime.now().strftime("%B %d, %Y")
    total = len(top_n_list)

    rows = []
    for entry in top_n_list:
        ticker = entry["ticker"]
        row_id = f"row-{ticker}"
        rows.append(
            f'<tr class="stock-row" data-target="{row_id}" '
            f'style="cursor:pointer;transition:background 0.15s;" '
            f"onmouseenter=\"this.style.background='{colors['row_hover']}'\" "
            f"onmouseleave=\"this.style.background=''\">"
            f'<td style="padding:10px 8px;text-align:center;vertical-align:middle;">{_rank_badge(entry["rank"])}</td>'
            '<td style="padding:10px 8px;vertical-align:middle;">'
            f'<div style="font-weight:700;font-size:13px;color:#1a1a2e;">{ticker}</div>'
            f'<div style="font-size:10px;color:#888;">${entry["latest_price"]:.2f}</div></td>'
            '<td style="padding:10px 8px;vertical-align:middle;">'
            f'<div style="font-weight:600;font-size:12px;color:#2c3e50;">{entry["name"]}</div>'
            f'<div style="font-size:10px;color:#888;">{entry["sector"]}</div></td>'
            f'<td style="padding:10px 8px;text-align:center;vertical-align:middle;">{_composite_badge(entry["composite_score"])}</td>'
            f'<td style="padding:10px 8px;vertical-align:middle;min-width:180px;">{_factor_grid(entry["factors"])}</td>'
            f'<td style="padding:10px 8px;vertical-align:middle;min-width:200px;">{_ai_analysis(entry)}</td>'
            '</tr>'
            f'<tr class="detail-row" id="{row_id}" style="background:#fff;">'
            f'{_detail_row(entry, colors, total)}'
            '</tr>'
        )

    body_fragment = (
        '<style>'
        '.qvm-report * { box-sizing: border-box; }'
        '.qvm-report { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;'
        ' max-width: 1200px; margin: 0 auto; background: #ffffff; border-radius: 12px;'
        ' overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,0.08); }'
        f'.qvm-report-header {{ background: {colors["header_bg"]}; padding: 20px 28px; color: white;'
        f' border-bottom: 3px solid {colors["header_border"]}; }}'
        '.qvm-report-header h1 { margin: 0; font-size: 22px; font-weight: 700; letter-spacing: -0.3px;'
        ' text-shadow: 0 1px 2px rgba(0,0,0,0.15); }'
        '.qvm-report-header .subtitle { margin-top: 6px; font-size: 12px; opacity: 0.92; }'
        '.qvm-report-header .regime-badge { display: inline-block; background: rgba(255,255,255,0.25);'
        ' padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-left: 8px;'
        ' text-transform: uppercase; letter-spacing: 0.5px; }'
        '.qvm-table-wrap { overflow-x: auto; }'
        '.qvm-table { width: 100%; border-collapse: collapse; font-size: 12px; }'
        '.qvm-table thead th { background: linear-gradient(180deg, #f8f9fa 0%, #eef0f2 100%);'
        ' padding: 10px 8px; text-align: left; font-size: 10px; font-weight: 700; color: #556;'
        ' text-transform: uppercase; letter-spacing: 0.6px; border-bottom: 2px solid #ddd;'
        ' white-space: nowrap; position: sticky; top: 0; z-index: 2; }'
        '.qvm-table thead th:nth-child(1) { text-align: center; width: 48px; }'
        '.qvm-table thead th:nth-child(4) { text-align: center; width: 72px; }'
        '.qvm-table tbody tr.stock-row td { border-bottom: 1px solid #f0f0f0; }'
        '.detail-panel { animation: fadeIn 0.2s ease; }'
        '@keyframes fadeIn { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }'
        '</style>'
        '<div class="qvm-report">'
        '<div class="qvm-report-header">'
        '<h1>QVM Quant Stock Screener</h1>'
        '<div class="subtitle">Market Regime: '
        f'<span class="regime-badge">{regime}</span>'
        f'&nbsp;|&nbsp; Score: {regime_score}/100&nbsp;|&nbsp; {today}'
        f'&nbsp;|&nbsp; Top {total} Ranked Stocks</div>'
        '</div>'
        '<div class="qvm-table-wrap">'
        '<table class="qvm-table">'
        '<thead><tr>'
        '<th>#</th><th>Ticker</th><th>Company + Sector</th>'
        '<th>Composite</th><th>Factor Overview</th><th>AI Analysis</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table></div></div>'
        '<script>'
        '(function() { var rows = document.querySelectorAll(".stock-row");'
        ' rows.forEach(function(row) { row.addEventListener("click", function() {'
        ' var targetId = this.getAttribute("data-target");'
        ' var panel = document.getElementById(targetId).querySelector(".detail-panel");'
        ' var isOpen = panel.style.display === "block";'
        ' document.querySelectorAll(".detail-panel").forEach(function(p) { p.style.display = "none"; });'
        ' panel.style.display = isOpen ? "none" : "block"; }); }); })();'
        '</script>'
    )

    full_html = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<title>QVM Quant Stock Screener — {today}</title>'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '</head><body>' + body_fragment + '</body></html>'
    )
    return body_fragment, full_html


def _to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars to JSON-safe Python types."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def write(
    top_n_list: list[dict],
    market_timing: dict,
    allocation: dict,
    signals: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Write report.html, signals.json, allocation.json (and signals.csv if a
    full ranked DataFrame is supplied). Returns a dict of written paths.
    """
    output_dir = output_dir or ensure_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    body_fragment, full_html = render_html(top_n_list, market_timing)
    paths["report"] = output_dir / "report.html"
    paths["report"].write_text(full_html, encoding="utf-8")
    paths["report_fragment"] = output_dir / "report-fragment.html"
    paths["report_fragment"].write_text(body_fragment, encoding="utf-8")

    signals_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "regime": market_timing,
        "stocks": [
            {k: _to_jsonable(v) for k, v in entry.items()} for entry in top_n_list
        ],
    }
    paths["signals_json"] = output_dir / "signals.json"
    paths["signals_json"].write_text(
        json.dumps(signals_payload, indent=2, default=str), encoding="utf-8"
    )

    paths["allocation_json"] = output_dir / "allocation.json"
    paths["allocation_json"].write_text(
        json.dumps(allocation, indent=2, default=str), encoding="utf-8"
    )

    if signals is not None:
        paths["signals_csv"] = output_dir / "signals.csv"
        signals.to_csv(paths["signals_csv"], index=False)

    return paths
