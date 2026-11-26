from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from riskmanage.calibration import HestonOptionQuote, calibrate_heston
from riskmanage.heston import HestonParameters
from riskmanage.uncertainty import (
    bootstrap_heston_calibration,
    profile_heston_parameter,
    propagate_heston_parameter_uncertainty,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantify Heston calibration uncertainty with bootstrap and profile likelihood."
    )
    parser.add_argument("--replicates", type=int, default=40)
    parser.add_argument("--profile-points", type=int, default=7)
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--bootstrap-max-nfev", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/heston_parameter_uncertainty.json"),
    )
    return parser


def quote_panel(
    parameters: HestonParameters, seed: int
) -> list[HestonOptionQuote]:
    rng = np.random.default_rng(seed)
    quotes = []
    for maturity in (0.25, 0.5, 1.0):
        strikes = (85.0, 92.5, 100.0, 107.5, 115.0)
        option_types = ("put", "put", "call", "call", "call")
        from riskmanage.heston import heston_prices_fourier_batch

        model_prices = heston_prices_fourier_batch(
            strikes, maturity, parameters, option_types
        )["prices"]
        for strike, option_type, model_price in zip(
            strikes, option_types, model_prices, strict=True
        ):
            spread = max(0.02, 0.015 * float(model_price))
            noise_sd = max(0.5 * spread, 0.01) / np.sqrt(3.0)
            mid = max(1e-6, float(model_price) + rng.normal(0.0, noise_sd))
            quotes.append(
                HestonOptionQuote(
                    strike=float(strike),
                    maturity=float(maturity),
                    option_type=option_type,
                    mid_price=mid,
                    bid=max(0.0, mid - spread / 2.0),
                    ask=mid + spread / 2.0,
                )
            )
    return quotes


def main() -> None:
    args = build_parser().parse_args()
    if args.replicates < 2 or args.profile_points < 3:
        raise SystemExit("replicates must be >= 2 and profile-points >= 3")
    if args.starts < 1 or args.max_nfev < 1 or args.bootstrap_max_nfev < 1:
        raise SystemExit("optimizer starts and evaluation limits must be positive")

    true_parameters = HestonParameters(
        spot=100.0,
        variance0=0.04,
        mean_reversion=1.5,
        long_run_variance=0.04,
        vol_of_variance=0.5,
        correlation=-0.7,
        risk_free_rate=0.03,
        dividend_yield=0.01,
    )
    quotes = quote_panel(true_parameters, args.seed)
    initial = replace(
        true_parameters,
        variance0=0.05,
        mean_reversion=1.2,
        long_run_variance=0.05,
        vol_of_variance=0.6,
        correlation=-0.6,
    )
    calibration = calibrate_heston(
        quotes,
        true_parameters,
        initial,
        objective="mid",
        loss="linear",
        n_starts=args.starts,
        max_nfev=args.max_nfev,
        seed=args.seed,
    )
    if not calibration["success"]:
        raise RuntimeError("base Heston quote calibration did not converge")
    fitted = replace(
        true_parameters,
        variance0=calibration["parameters"]["variance0"],
        mean_reversion=calibration["parameters"]["mean_reversion"],
        long_run_variance=calibration["parameters"]["long_run_variance"],
        vol_of_variance=calibration["parameters"]["vol_of_variance"],
        correlation=calibration["parameters"]["correlation"],
    )

    bootstrap = bootstrap_heston_calibration(
        quotes,
        true_parameters,
        fitted,
        n_bootstrap=args.replicates,
        seed=args.seed + 1000,
        n_starts=1,
        max_nfev=args.bootstrap_max_nfev,
    )
    rho_grid = np.linspace(
        max(-0.98, fitted.correlation - 0.22),
        min(0.98, fitted.correlation + 0.22),
        args.profile_points,
    )
    volvol_grid = np.linspace(
        max(0.02, fitted.vol_of_variance * 0.55),
        min(4.95, fitted.vol_of_variance * 1.45),
        args.profile_points,
    )
    profiles = {
        "correlation": profile_heston_parameter(
            quotes,
            true_parameters,
            fitted,
            "correlation",
            rho_grid,
            max_nfev=args.max_nfev,
        ),
        "vol_of_variance": profile_heston_parameter(
            quotes,
            true_parameters,
            fitted,
            "vol_of_variance",
            volvol_grid,
            max_nfev=args.max_nfev,
        ),
    }
    contracts = [
        {"strike": strike, "maturity": maturity, "option_type": option_type}
        for maturity in (0.25, 1.0)
        for strike, option_type in ((90.0, "put"), (100.0, "call"), (110.0, "call"))
    ]
    propagation = propagate_heston_parameter_uncertainty(
        bootstrap["bootstrap_parameter_samples"],
        true_parameters,
        contracts,
    )
    report = {
        "title": "Heston parameter uncertainty and risk propagation",
        "classification": "synthetic_quote_panel_parametric_uncertainty_experiment",
        "seed": args.seed,
        "quote_count": len(quotes),
        "quote_panel": [
            {
                "strike": quote.strike,
                "maturity": quote.maturity,
                "option_type": quote.option_type,
                "mid": quote.mid_price,
                "bid": quote.bid,
                "ask": quote.ask,
            }
            for quote in quotes
        ],
        "true_parameters_used_to_generate_synthetic_quotes": {
            "variance0": true_parameters.variance0,
            "mean_reversion": true_parameters.mean_reversion,
            "long_run_variance": true_parameters.long_run_variance,
            "vol_of_variance": true_parameters.vol_of_variance,
            "correlation": true_parameters.correlation,
        },
        "base_calibration": calibration,
        "fitted_parameters": {
            "variance0": fitted.variance0,
            "mean_reversion": fitted.mean_reversion,
            "long_run_variance": fitted.long_run_variance,
            "vol_of_variance": fitted.vol_of_variance,
            "correlation": fitted.correlation,
            "feller_ratio": fitted.feller_ratio,
        },
        "parametric_bootstrap": bootstrap,
        "profile_likelihood": profiles,
        "price_and_delta_uncertainty_propagation": propagation,
        "interpretation_boundary": (
            "This experiment uses synthetic Heston quotes and a Gaussian quote-noise model "
            "whose standard deviation is derived from bid-ask spread. Bootstrap and profile "
            "intervals are conditional on the Heston model, fixed quote design and flat curves. "
            "They do not quantify model risk, real-market quote dependence, stale/crossed quotes, "
            "curve uncertainty, physical-measure risk or parameter uncertainty outside the chosen "
            "noise model. Profile intervals are grid approximations using an asymptotic chi-square "
            "likelihood-ratio cutoff."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Base calibration RMSE={calibration['price_rmse']:.6g}; "
        f"bootstrap successes={bootstrap['successful_replicates']}/{args.replicates}"
    )
    print(
        "Correlation profile interval:",
        profiles["correlation"]["profile_interval_refined"],
    )
    print(
        "Vol-of-vol profile interval:",
        profiles["vol_of_variance"]["profile_interval_refined"],
    )
    print(f"Wrote parameter uncertainty report to {args.output}")


if __name__ == "__main__":
    main()
