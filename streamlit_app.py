"""Streamlit front-end for the QVM Quant Stock Screener.

Deploy free on Streamlit Community Cloud:
  1. Push this repo to GitHub.
  2. https://share.streamlit.io → "New app" → point at streamlit_app.py.
  3. In the app's "Secrets" panel, add:
        FMP_API_KEY="..."
        DEEPSEEK_API_KEY="..."
  4. Click "Run".

Locally:
  pip install -r requirements.txt streamlit
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Promote st.secrets → env vars so the same code paths work locally and on
# Streamlit Cloud without modification.
for key in ("FMP_API_KEY", "DEEPSEEK_API_KEY", "QVM_CAPITAL"):
    if key in st.secrets and not os.environ.get(key):
        os.environ[key] = str(st.secrets[key])

from qvm import factors, market_timing, pipeline, report, screener  # noqa: E402
from qvm.config import DB_PATH, ensure_output_dir  # noqa: E402

st.set_page_config(
    page_title="QVM Quant Stock Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("QVM Quant Stock Screener")
st.caption(
    "Seven academic factors + Piotroski gate + 6-category market regime. "
    "Top 30 ranked stocks, with AI-generated bear case per name."
)

with st.sidebar:
    st.header("Run pipeline")
    st.write("FMP key set:", "✅" if os.environ.get("FMP_API_KEY") else "❌ (yfinance fallback)")
    st.write("DeepSeek key set:", "✅" if os.environ.get("DEEPSEEK_API_KEY") else "❌ (template bear case)")
    run_btn = st.button("Run full pipeline", type="primary")
    st.markdown(
        "**Heads up:** a full run downloads prices for the universe, computes "
        "fundamentals for the top 100, and (optionally) calls DeepSeek 30× for "
        "bear-case narratives. It typically takes a few minutes."
    )


@st.cache_data(show_spinner=False)
def _load_cached_outputs(output_dir: str) -> dict:
    out = {}
    p = Path(output_dir)
    if (p / "report-fragment.html").exists():
        out["fragment"] = (p / "report-fragment.html").read_text(encoding="utf-8")
    if (p / "signals.json").exists():
        out["signals"] = (p / "signals.json").read_text(encoding="utf-8")
    if (p / "allocation.json").exists():
        out["allocation"] = (p / "allocation.json").read_text(encoding="utf-8")
    return out


def _run_pipeline() -> dict:
    output_dir = ensure_output_dir()
    with st.status("Running pipeline...", expanded=True) as status:
        st.write("Fetching prices + fundamentals...")
        pipeline.run(db_path=DB_PATH)
        st.write("Computing Piotroski F-Score...")
        factors.compute_piotroski(db_path=DB_PATH)
        st.write("Computing QVM signals...")
        signals = factors.compute_qvm_signals(db_path=DB_PATH)
        st.write("Computing market regime...")
        regime = market_timing.compute()
        st.write("Building top 30 + bear case...")
        top_n_list = screener.build_top_n_list(signals)
        allocation = screener.build_allocation(signals, regime)
        report.write(top_n_list, regime, allocation, signals=signals, output_dir=output_dir)
        status.update(label="Pipeline complete", state="complete")
    _load_cached_outputs.clear()
    return _load_cached_outputs(str(output_dir))


if run_btn:
    outputs = _run_pipeline()
else:
    outputs = _load_cached_outputs(str(ensure_output_dir()))

if "fragment" in outputs:
    st.components.v1.html(outputs["fragment"], height=1400, scrolling=True)
    with st.expander("Allocation JSON"):
        st.code(outputs.get("allocation", "{}"), language="json")
    with st.expander("Signals JSON (first 1000 chars)"):
        st.code(outputs.get("signals", "")[:1000] + " ...", language="json")
    st.caption(f"Loaded from ./output/ — last refreshed at app start ({datetime.now():%Y-%m-%d %H:%M:%S}).")
else:
    st.info("Click **Run full pipeline** in the sidebar to generate a report.")
