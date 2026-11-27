from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from riskmanage.commodity import (  # noqa: E402
    build_commodity_spot_pnl,
    save_commodity_spot_case,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a standalone historical risk diagnostic for hypothetical WTI/Brent "
            "linear spot proxies using the full EIA spot history."
        )
    )
    parser.add_argument(
        "--input",
        default="data/processed/eia_petroleum_spot.csv",
        help="Normalized EIA spot CSV produced by download_eia_petroleum_spot.py.",
    )
    parser.add_argument("--wti-quantity", type=float, default=150_000.0)
    parser.add_argument("--brent-quantity", type=float, default=-100_000.0)
    parser.add_argument(
        "--pnl-output",
        default="reports/generated/commodity_spot_pnl.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/generated/commodity_spot_risk.json",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(
            f"{input_path} does not exist; first run "
            "python scripts/download_eia_petroleum_spot.py --start 2007-01-01"
        )

    eia_frame = pd.read_csv(input_path)
    try:
        pnl, summary = build_commodity_spot_pnl(
            eia_frame,
            wti_quantity=args.wti_quantity,
            brent_quantity=args.brent_quantity,
        )
        save_commodity_spot_case(
            pnl,
            summary,
            pnl_path=args.pnl_output,
            summary_path=args.summary_output,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")

    print(
        json.dumps(
            {
                "observations": summary["pnl"]["observations"],
                "start": summary["pnl"]["first_pnl_date"],
                "end": summary["pnl"]["last_pnl_date"],
                "wti_quantity_barrels": summary["portfolio"]["wti_quantity_barrels"],
                "brent_quantity_barrels": summary["portfolio"]["brent_quantity_barrels"],
                "historical_risk": summary["historical_risk"],
                "stress_window": summary["stress_window"],
                "max_absolute_hpl_rtpl_residual": summary["pnl"][
                    "max_absolute_hpl_rtpl_residual"
                ],
                "pnl_csv": args.pnl_output,
                "risk_summary": args.summary_output,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
