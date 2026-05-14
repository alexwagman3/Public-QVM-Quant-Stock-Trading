# QVM Quant Stock Screener

A reproducible value–momentum–quality (QVM) screener for ~500 U.S. equities.
The same factor stack that billion-dollar quant funds run every evening, packaged
as a single Python project with three ways to run it: CLI, Streamlit web app,
or Google Colab notebook.

Each run produces:

1. **Market regime score** (0–100) — a six-category composite that tells you
   whether the environment favors 100% equity exposure, 50%, or sitting in cash.
2. **Top 30 ranked stocks** — seven peer-reviewed factors, z-score normalized,
   gated by the 9-point Piotroski F-Score.
3. **AI bear case** (optional) — DeepSeek reads recent earnings transcripts,
   analyst targets, insider activity, and news to write a specific skeptical
   report per stock.
4. **Regime-aware allocation** — fractional-share portfolio sized to the
   regime's implied equity percentage, sector-capped at 40%.

> **Not investment advice.** Research / educational tool. Past performance is
> not indicative of future results.

---

## The seven factors

| # | Factor | Definition | Citation |
| - | ------ | ---------- | -------- |
| 1 | Jegadeesh–Titman 12-1 momentum | `(price_1m_ago / price_13m_ago) − 1` | Jegadeesh & Titman, *"Returns to Buying Winners and Selling Losers,"* **Journal of Finance**, 1993 |
| 2 | 6-month momentum | `(price_today / price_126d_ago) − 1` | Standard secondary trend factor |
| 3 | Inverse 252-day volatility | `1 / annualized_stdev_of_log_returns` | Blitz & van Vliet, *"The Volatility Effect,"* **JPM**, 2007 |
| 4 | Gross profitability | `gross_profit / total_assets` | Novy-Marx, *"The Other Side of Value: The Gross Profitability Premium,"* **JFE**, 2013 |
| 5 | Free cash flow yield | `free_cash_flow / enterprise_value` | Fama & French, *"Profitability, Investment and Average Returns,"* **JFE**, 2006 |
| 6 | Sloan accruals | `−(net_income − operating_cash_flow) / total_assets` | Sloan, *"Do Stock Prices Fully Reflect Information in Accruals and Cash Flows…?"* **The Accounting Review**, 1996 |
| 7 | Inverse asset growth | `−(total_assets_t / total_assets_{t-1} − 1)` | Titman, Wei & Xie, *"Capital Investments and Stock Returns,"* **JFQA**, 2004 |

Each raw factor is winsorized at the 1st/99th percentile, z-score normalized,
clipped to [-3, 3], and combined into a single composite using the weights in
`qvm/config.py`. Stocks below the **Piotroski F-Score gate** (default ≥ 6) are
filtered out before ranking.

### Piotroski F-Score gate

Nine binary signals from Piotroski (2000): ROA > 0, OCF > 0, ΔROA > 0,
accruals (OCF > NI), ΔLeverage < 0, ΔCurrent ratio > 0, ΔShares ≤ 0,
ΔGross margin > 0, ΔAsset turnover > 0. The default gate of 6/9 follows
Piotroski's original "high-quality" threshold.

---

## Market regime score

A 0–100 composite from six independent categories, weighted as follows:

| Category | Weight | Inputs |
| -------- | ------ | ------ |
| **Trend** | 30% | SPY vs 200-DMA, SPY vs 50-DMA, 12-month momentum, 200-DMA slope |
| **Volatility** | 25% | VIX level, VIX/VIX3M term structure, 20-day vs 60-day realized vol |
| **Credit / macro** | 20% | 10Y–3M Treasury yield spread, HYG/LQD ratio |
| **Breadth** | 10% | RSP/SPY (equal-weight vs cap-weight) ratio trend |
| **Sentiment** | 10% | VIX 5-day change |
| **Safe haven** | 5% | GLD/SPY ratio trend |

The score maps to a regime and an implied equity percentage:

| Score | Regime | Equity % | # Positions |
| ----: | ------ | -------: | ----------: |
| ≥ 80  | STRONG_BULL | 100% | 10 |
| ≥ 65  | BULLISH     |  85% |  8 |
| ≥ 50  | NEUTRAL     |  70% |  6 |
| ≥ 35  | CAUTIOUS    |  50% |  4 |
| ≥ 20  | BEARISH     |  25% |  2 |
| < 20  | CRISIS      |   0% |  0 |

---

## Setup

```bash
git clone <this repo>
cd public-qvm-quant-stock-trading
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env with your keys (both optional)
```

### Environment variables

| Variable | Required | What it does |
| -------- | -------- | ------------ |
| `FMP_API_KEY` | optional | [Financial Modeling Prep](https://site.financialmodelingprep.com) — fundamentals, earnings call transcripts, analyst targets, insider trades. Falls back to yfinance if missing. |
| `DEEPSEEK_API_KEY` | optional | [DeepSeek](https://platform.deepseek.com) — generates per-stock bear cases. Falls back to a deterministic template if missing. |
| `QVM_OUTPUT_DIR` | optional | Where to write outputs. Default `./output`. |
| `QVM_CAPITAL` | optional | Portfolio size used for allocation math. Default `10000`. |

---

## How to run it

### 1. CLI

```bash
python main.py
```

Writes to `./output/`:

```
output/
├── report.html             # self-contained HTML — open in any browser
├── report-fragment.html    # embeddable body fragment (no <html>/<body>)
├── signals.json            # top 30 stocks + factor breakdowns + bear case
├── allocation.json         # regime-aware portfolio allocation
└── signals.csv             # full ranked signals DataFrame
```

### 2. Streamlit (free deploy)

```bash
pip install streamlit
streamlit run streamlit_app.py
```

To deploy on [Streamlit Community Cloud](https://share.streamlit.io):

1. Push this repo to GitHub.
2. New app → point at `streamlit_app.py`.
3. Add `FMP_API_KEY` and `DEEPSEEK_API_KEY` in the app's "Secrets" panel.
4. Click **Run**.

### 3. Google Colab

Open [`notebooks/qvm_colab.ipynb`](notebooks/qvm_colab.ipynb) in Colab. Add
your keys to Colab Secrets (key icon in the sidebar), then **Run All**.

---

## Sample output

A run during a `BULLISH` regime (score 72) produces a `signals.json` whose
top entries look like:

```json
{
  "regime": {"regime": "BULLISH", "score": 72, "equity_pct": 0.85, "n_positions": 8},
  "stocks": [
    {
      "rank": 1, "ticker": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology",
      "composite_score": 1.84, "latest_price": 891.23, "f_score": 8.0,
      "factors": [
        {"label": "Momentum",   "raw_value": 0.94, "z_score":  2.31, "color": "green"},
        {"label": "6M Mom",     "raw_value": 0.42, "z_score":  1.78, "color": "green"},
        {"label": "Volatility", "raw_value": 1.85, "z_score":  0.62, "color": "green"},
        {"label": "Quality",    "raw_value": 0.41, "z_score":  2.95, "color": "green"},
        {"label": "Value",      "raw_value": 0.014,"z_score": -0.81, "color": "red"},
        {"label": "Accruals",   "raw_value": 0.05, "z_score":  0.34, "color": "amber"},
        {"label": "Inv Growth", "raw_value": -0.32,"z_score": -1.14, "color": "red"}
      ],
      "biz_text":   "Designs GPUs and accelerated-compute platforms that power data-center AI training and inference.",
      "focus_text": "Quality (z=+2.95) and momentum (z=+2.31) are doing the heavy lifting — Novy-Marx gross profitability is in the 99th percentile.",
      "warning_text": "Management warned on the Q4 call about hyperscaler digestion of inventory and rising lead times for Blackwell..."
    }
  ]
}
```

The HTML report (`output/report.html`) renders the same data as an
interactive table with expandable detail panels and regime-aware coloring.

---

## Project layout

```
qvm/
├── __init__.py
├── config.py        # universe, weights, paths
├── fmp_client.py    # Financial Modeling Prep API wrapper
├── data_sources.py  # unified data layer (FMP → yfinance fallback)
├── pipeline.py      # download prices, score, fetch fundamentals → SQLite
├── factors.py       # Piotroski + 7 QVM factors + z-score / composite
├── market_timing.py # 6-category regime score
├── screener.py      # top-30 selection, sector cap, fractional shares, AI bear case
└── report.py        # HTML render + JSON outputs

main.py              # CLI entrypoint
streamlit_app.py     # Streamlit front-end
notebooks/
└── qvm_colab.ipynb  # Google Colab notebook
```

---

## License

MIT. See [LICENSE](LICENSE).
