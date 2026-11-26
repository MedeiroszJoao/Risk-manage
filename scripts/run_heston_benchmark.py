from __future__ import annotations

import argparse
import json
from pathlib import Path

from riskmanage.heston import (
    HestonParameters,
    heston_monte_carlo_price,
    heston_price_fourier,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Heston Fourier prices with QE-M Monte Carlo estimates."
    )
    parser.add_argument("--paths", type=int, default=20_000)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--pilot-paths", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/heston_benchmark.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.paths < 4 or args.steps < 1 or args.pilot_paths < 4:
        raise SystemExit("paths, steps and pilot-paths must be positive (minimum 4 paths)")
    if args.paths % 2 or args.pilot_paths % 2:
        raise SystemExit("paths and pilot-paths must be even when antithetic sampling is enabled")

    params = HestonParameters(
        spot=100.0,
        variance0=0.04,
        mean_reversion=1.5,
        long_run_variance=0.04,
        vol_of_variance=0.5,
        correlation=-0.7,
        risk_free_rate=0.03,
        dividend_yield=0.01,
    )
    cases = []
    for maturity in (0.5, 1.0):
        for moneyness in (0.90, 1.00, 1.10):
            strike = params.spot * moneyness
            fourier = heston_price_fourier(strike, maturity, params, "call")
            monte_carlo = heston_monte_carlo_price(
                strike,
                maturity,
                params,
                n_paths=args.paths,
                n_steps=args.steps,
                seed=args.seed + len(cases),
                pilot_paths=args.pilot_paths,
            )
            standard_error = monte_carlo["standard_error"]
            cases.append(
                {
                    "maturity_years": maturity,
                    "strike": strike,
                    "moneyness": moneyness,
                    "fourier_price": fourier["price"],
                    "fourier_quadrature_error_estimate": fourier[
                        "quadrature_error_estimate"
                    ],
                    "qe_monte_carlo_price": monte_carlo["price"],
                    "monte_carlo_standard_error": standard_error,
                    "monte_carlo_confidence_interval": monte_carlo[
                        "confidence_interval"
                    ],
                    "price_difference_mc_minus_fourier": (
                        monte_carlo["price"] - fourier["price"]
                    ),
                    "difference_in_standard_errors": (
                        (monte_carlo["price"] - fourier["price"]) / standard_error
                        if standard_error > 0.0
                        else None
                    ),
                    "variance_reduction_factor": monte_carlo[
                        "variance_reduction_factor"
                    ],
                    "stock_martingale_relative_error": monte_carlo[
                        "stock_martingale_relative_error"
                    ],
                    "n_paths": monte_carlo["n_paths"],
                    "n_steps": monte_carlo["n_steps"],
                    "effective_samples": monte_carlo["effective_samples"],
                }
            )

    report = {
        "title": "Heston Fourier vs QE-M Monte Carlo benchmark",
        "classification": "synthetic_parameters_not_market_calibration",
        "seed": args.seed,
        "parameters": {
            "spot": params.spot,
            "variance0": params.variance0,
            "mean_reversion": params.mean_reversion,
            "long_run_variance": params.long_run_variance,
            "vol_of_variance": params.vol_of_variance,
            "correlation": params.correlation,
            "risk_free_rate": params.risk_free_rate,
            "dividend_yield": params.dividend_yield,
            "feller_ratio": params.feller_ratio,
            "feller_condition_satisfied": params.feller_condition_satisfied,
        },
        "simulation": {
            "paths_per_case": args.paths,
            "time_steps": args.steps,
            "pilot_paths": args.pilot_paths,
            "antithetic": True,
            "control_variate": "discounted stock, beta estimated on independent pilot",
        },
        "uncertainty_boundary": (
            "Monte Carlo confidence intervals are conditional on QE time discretization; "
            "they exclude discretization bias, parameter uncertainty and model risk. "
            "Fourier quadrature error is reported separately."
        ),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} benchmark cases to {args.output}")


if __name__ == "__main__":
    main()
