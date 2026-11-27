from __future__ import annotations

import argparse
import json
from pathlib import Path

from riskmanage.calibration import HestonOptionQuote, calibrate_heston
from riskmanage.heston import HestonParameters, heston_prices_fourier_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce a synthetic Heston quote-surface calibration."
    )
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument(
        "--objective", choices=("mid", "bid_ask"), default="mid"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/heston_calibration.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.starts < 1 or args.max_nfev < 1:
        raise SystemExit("starts and max-nfev must be positive")

    market = HestonParameters(
        spot=100.0,
        variance0=0.04,
        mean_reversion=1.5,
        long_run_variance=0.04,
        vol_of_variance=0.5,
        correlation=-0.7,
        risk_free_rate=0.03,
        dividend_yield=0.01,
    )
    strikes = (85.0, 92.5, 100.0, 107.5, 115.0)
    maturities = (0.25, 0.5, 1.0)
    quotes = []
    for maturity in maturities:
        option_types = ["put", "put", "call", "call", "call"]
        batch = heston_prices_fourier_batch(
            strikes, maturity, market, option_types
        )
        for strike, option_type, mid in zip(
            strikes, option_types, batch["prices"], strict=True
        ):
            spread = max(0.02, 0.015 * float(mid))
            quotes.append(
                HestonOptionQuote(
                    strike=float(strike),
                    maturity=float(maturity),
                    option_type=option_type,
                    mid_price=float(mid),
                    bid=max(0.0, float(mid) - spread / 2.0),
                    ask=float(mid) + spread / 2.0,
                )
            )

    initial = HestonParameters(
        spot=market.spot,
        variance0=0.05,
        mean_reversion=1.2,
        long_run_variance=0.05,
        vol_of_variance=0.6,
        correlation=-0.6,
        risk_free_rate=market.risk_free_rate,
        dividend_yield=market.dividend_yield,
    )
    result = calibrate_heston(
        quotes,
        market,
        initial,
        objective=args.objective,
        n_starts=args.starts,
        max_nfev=args.max_nfev,
        seed=args.seed,
    )
    report = {
        "title": "Heston synthetic option-surface calibration",
        "classification": "synthetic_quote_panel_not_market_calibration",
        "seed": args.seed,
        "quote_count": len(quotes),
        "maturities_years": list(maturities),
        "strikes": list(strikes),
        "market_assumptions": {
            "spot": market.spot,
            "risk_free_rate": market.risk_free_rate,
            "dividend_yield": market.dividend_yield,
        },
        "true_parameters_used_to_generate_quotes": {
            "variance0": market.variance0,
            "mean_reversion": market.mean_reversion,
            "long_run_variance": market.long_run_variance,
            "vol_of_variance": market.vol_of_variance,
            "correlation": market.correlation,
        },
        "calibration": result,
        "interpretation_boundary": (
            "This deterministic synthetic recovery checks implementation behavior only. "
            "It is not evidence of live-market fit or stable parameter identification. "
            "The Jacobian SVD is local and is not a parameter confidence interval."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Calibrated {len(quotes)} synthetic quotes; "
        f"price RMSE={result['price_rmse']:.8f}; "
        f"Jacobian rank={result['jacobian_rank']}/5"
    )
    print(f"Wrote calibration report to {args.output}")


if __name__ == "__main__":
    main()
