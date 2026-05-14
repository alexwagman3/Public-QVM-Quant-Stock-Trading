"""QVM Quant Stock Screener — entrypoint.

Runs the full pipeline end-to-end and writes outputs to ./output/:
  - report.html            self-contained HTML report
  - report-fragment.html   embeddable body fragment (for Streamlit/Astro)
  - signals.json           top-N stocks with factor breakdowns + bear case
  - allocation.json        regime-aware portfolio allocation
  - signals.csv            full ranked signals DataFrame

Environment variables (both optional):
  FMP_API_KEY        Financial Modeling Prep key (improves fundamentals coverage)
  DEEPSEEK_API_KEY   DeepSeek API key (enables AI-generated bear case)
  QVM_OUTPUT_DIR     output directory (default: ./output)
  QVM_CAPITAL        portfolio size used for allocation math (default: 10000)

Usage:
  pip install -r requirements.txt
  python main.py
"""

from __future__ import annotations

import sys

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

from qvm import factors, market_timing, pipeline, report, screener
from qvm.config import DB_PATH, ensure_output_dir


def main() -> int:
    output_dir = ensure_output_dir()
    print(f"Output directory: {output_dir}")

    pipeline.run(db_path=DB_PATH)
    factors.compute_piotroski(db_path=DB_PATH)
    signals = factors.compute_qvm_signals(db_path=DB_PATH)

    regime = market_timing.compute()
    top_n_list = screener.build_top_n_list(signals)
    allocation = screener.build_allocation(signals, regime)

    paths = report.write(
        top_n_list=top_n_list,
        market_timing=regime,
        allocation=allocation,
        signals=signals,
        output_dir=output_dir,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    for label, path in paths.items():
        print(f"  {label:<18} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
