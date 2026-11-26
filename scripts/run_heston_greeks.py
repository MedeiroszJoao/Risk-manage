from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from riskmanage.greeks import heston_monte_carlo_greeks
from riskmanage.heston import HestonParameters, heston_price_fourier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Heston QE-M Monte Carlo Greeks with Fourier bumps."
    )
    parser.add_argument("--paths", type=int, default=50_000)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/heston_greeks.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.paths < 4 or args.steps < 1 or args.paths % 2:
        raise SystemExit("paths must be even and >= 4; steps must be positive")

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
    strike = 100.0
    maturity = 1.0
    spot_bump_relative = 0.005
    spot_bump = params.spot * spot_bump_relative
    sigma0 = params.variance0**0.5
    volatility_bump = 0.005

    base = heston_price_fourier(strike, maturity, params)["price"]
    spot_up = heston_price_fourier(
        strike, maturity, replace(params, spot=params.spot + spot_bump)
    )["price"]
    spot_down = heston_price_fourier(
        strike, maturity, replace(params, spot=params.spot - spot_bump)
    )["price"]
    vol_up = heston_price_fourier(
        strike,
        maturity,
        replace(params, variance0=(sigma0 + volatility_bump) ** 2),
    )["price"]
    vol_down = heston_price_fourier(
        strike,
        maturity,
        replace(params, variance0=(sigma0 - volatility_bump) ** 2),
    )["price"]
    mixed_prices = {}
    for spot_sign, spot_value in (("up", params.spot + spot_bump), ("down", params.spot - spot_bump)):
        for vol_sign, vol_value in (
            ("up", (sigma0 + volatility_bump) ** 2),
            ("down", (sigma0 - volatility_bump) ** 2),
        ):
            mixed_prices[f"{spot_sign}_{vol_sign}"] = heston_price_fourier(
                strike,
                maturity,
                replace(params, spot=spot_value, variance0=vol_value),
            )["price"]

    fourier_reference = {
        "price": base,
        "delta_central_difference": (spot_up - spot_down) / (2.0 * spot_bump),
        "gamma_central_difference": (
            spot_up - 2.0 * base + spot_down
        ) / (spot_bump**2),
        "vega_initial_volatility_central_difference": (
            vol_up - vol_down
        ) / (2.0 * volatility_bump),
        "vanna_spot_initial_volatility_central_difference": (
            mixed_prices["up_up"]
            - mixed_prices["up_down"]
            - mixed_prices["down_up"]
            + mixed_prices["down_down"]
        ) / (4.0 * spot_bump * volatility_bump),
        "volga_initial_volatility_central_difference": (
            vol_up - 2.0 * base + vol_down
        ) / (volatility_bump**2),
        "spot_bump": spot_bump,
        "initial_volatility_bump": volatility_bump,
    }
    monte_carlo = heston_monte_carlo_greeks(
        strike,
        maturity,
        params,
        n_paths=args.paths,
        n_steps=args.steps,
        seed=args.seed,
        spot_bump_relative=spot_bump_relative,
        initial_volatility_bump=volatility_bump,
    )

    comparisons = {
        "delta_pathwise": {
            "estimate": monte_carlo["delta_pathwise"]["estimate"],
            "standard_error": monte_carlo["delta_pathwise"]["standard_error"],
            "fourier_reference": fourier_reference["delta_central_difference"],
        },
        "delta_common_random_numbers": {
            "estimate": monte_carlo["delta_common_random_numbers"]["estimate"],
            "standard_error": monte_carlo["delta_common_random_numbers"][
                "standard_error"
            ],
            "fourier_reference": fourier_reference["delta_central_difference"],
        },
        "gamma_common_random_numbers": {
            "estimate": monte_carlo["gamma_common_random_numbers"]["estimate"],
            "standard_error": monte_carlo["gamma_common_random_numbers"][
                "standard_error"
            ],
            "fourier_reference": fourier_reference["gamma_central_difference"],
        },
        "vega_initial_volatility_common_random_numbers": {
            "estimate": monte_carlo[
                "vega_initial_volatility_common_random_numbers"
            ]["estimate"],
            "standard_error": monte_carlo[
                "vega_initial_volatility_common_random_numbers"
            ]["standard_error"],
            "fourier_reference": fourier_reference[
                "vega_initial_volatility_central_difference"
            ],
        },
        "vanna_spot_initial_volatility_common_random_numbers": {
            "estimate": monte_carlo[
                "vanna_spot_initial_volatility_common_random_numbers"
            ]["estimate"],
            "standard_error": monte_carlo[
                "vanna_spot_initial_volatility_common_random_numbers"
            ]["standard_error"],
            "fourier_reference": fourier_reference[
                "vanna_spot_initial_volatility_central_difference"
            ],
        },
        "volga_initial_volatility_common_random_numbers": {
            "estimate": monte_carlo[
                "volga_initial_volatility_common_random_numbers"
            ]["estimate"],
            "standard_error": monte_carlo[
                "volga_initial_volatility_common_random_numbers"
            ]["standard_error"],
            "fourier_reference": fourier_reference[
                "volga_initial_volatility_central_difference"
            ],
        },
    }
    for item in comparisons.values():
        item["difference_mc_minus_fourier"] = (
            item["estimate"] - item["fourier_reference"]
        )
        item["difference_in_mc_standard_errors"] = (
            item["difference_mc_minus_fourier"] / item["standard_error"]
            if item["standard_error"] > 0.0
            else None
        )

    report = {
        "title": "Heston QE-M Monte Carlo Greeks benchmark",
        "classification": "synthetic_parameters_not_market_greeks",
        "seed": args.seed,
        "contract": {
            "spot": params.spot,
            "strike": strike,
            "maturity_years": maturity,
            "option_type": "call",
        },
        "parameters": {
            "variance0": params.variance0,
            "mean_reversion": params.mean_reversion,
            "long_run_variance": params.long_run_variance,
            "vol_of_variance": params.vol_of_variance,
            "correlation": params.correlation,
            "risk_free_rate": params.risk_free_rate,
            "dividend_yield": params.dividend_yield,
            "feller_ratio": params.feller_ratio,
        },
        "fourier_finite_difference_reference": fourier_reference,
        "monte_carlo": monte_carlo,
        "comparisons": comparisons,
        "interpretation_boundary": (
            "Fourier finite differences are numerical reference values, not closed-form Heston Greeks. "
            "Monte Carlo confidence intervals are conditional on the QE-M grid and bumps. "
            "They exclude time-discretization and bump truncation bias, parameter uncertainty and model risk."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "Heston MC Greeks: "
        f"Delta(pathwise)={monte_carlo['delta_pathwise']['estimate']:.8f}; "
        f"Gamma(CRN)={monte_carlo['gamma_common_random_numbers']['estimate']:.8f}; "
        f"Vega(initial vol)={monte_carlo['vega_initial_volatility_common_random_numbers']['estimate']:.8f}"
    )
    print(f"Wrote Greek comparison report to {args.output}")


if __name__ == "__main__":
    main()
