from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from riskmanage.calibration import HestonOptionQuote, calibrate_heston
from riskmanage.hedging_uncertainty import (
    propagate_heston_uncertainty_to_hedging,
)
from riskmanage.heston import HestonParameters, heston_prices_fourier_batch
from riskmanage.uncertainty import bootstrap_heston_calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Propagate Heston quote-calibration uncertainty into discrete hedge PnL, "
            "VaR and Expected Shortfall using common physical scenarios."
        )
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=30)
    parser.add_argument("--paths", type=int, default=1_000)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cost-bps", type=float, default=2.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/heston_hedge_parameter_uncertainty.json"),
    )
    return parser


def make_quote_panel(params: HestonParameters, seed: int):
    rng = np.random.default_rng(seed)
    quotes = []
    strikes = (85.0, 92.5, 100.0, 107.5, 115.0)
    option_types = ("put", "put", "call", "call", "call")
    for maturity in (0.25, 0.5, 1.0):
        prices = heston_prices_fourier_batch(
            strikes, maturity, params, option_types
        )["prices"]
        for strike, option_type, fair_value in zip(
            strikes, option_types, prices, strict=True
        ):
            spread = max(0.02, 0.015 * float(fair_value))
            noise_sd = max(0.5 * spread, 0.01) / np.sqrt(3.0)
            mid = max(1e-6, float(fair_value) + rng.normal(0.0, noise_sd))
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
    if args.bootstrap_replicates < 2:
        raise SystemExit("bootstrap-replicates must be at least 2")
    if args.paths < 40 or args.steps < 1:
        raise SystemExit("paths must be >= 40 and steps must be positive")
    if args.cost_bps < 0.0 or not 0.5 < args.confidence < 1.0:
        raise SystemExit("cost-bps must be non-negative and confidence in (0.5, 1)")

    q_true = HestonParameters(
        spot=100.0,
        variance0=0.04,
        mean_reversion=1.5,
        long_run_variance=0.04,
        vol_of_variance=0.5,
        correlation=-0.7,
        risk_free_rate=0.03,
        dividend_yield=0.01,
    )
    p_scenario = HestonParameters(
        spot=100.0,
        variance0=0.04,
        mean_reversion=1.25,
        long_run_variance=0.05,
        vol_of_variance=0.55,
        correlation=-0.65,
        risk_free_rate=0.03,
        dividend_yield=0.01,
    )
    quotes = make_quote_panel(q_true, args.seed)
    initial = HestonParameters(
        spot=q_true.spot,
        variance0=0.05,
        mean_reversion=1.2,
        long_run_variance=0.05,
        vol_of_variance=0.6,
        correlation=-0.6,
        risk_free_rate=q_true.risk_free_rate,
        dividend_yield=q_true.dividend_yield,
    )
    calibration = calibrate_heston(
        quotes,
        q_true,
        initial,
        objective="mid",
        loss="linear",
        n_starts=2,
        max_nfev=100,
        seed=args.seed,
    )
    if not calibration["success"]:
        raise RuntimeError("base synthetic Heston calibration did not converge")
    fitted = HestonParameters(
        spot=q_true.spot,
        variance0=calibration["parameters"]["variance0"],
        mean_reversion=calibration["parameters"]["mean_reversion"],
        long_run_variance=calibration["parameters"]["long_run_variance"],
        vol_of_variance=calibration["parameters"]["vol_of_variance"],
        correlation=calibration["parameters"]["correlation"],
        risk_free_rate=q_true.risk_free_rate,
        dividend_yield=q_true.dividend_yield,
    )
    bootstrap = bootstrap_heston_calibration(
        quotes,
        q_true,
        fitted,
        n_bootstrap=args.bootstrap_replicates,
        seed=args.seed + 5_000,
        n_starts=1,
        max_nfev=80,
    )
    if bootstrap["successful_replicates"] < 2:
        raise RuntimeError("fewer than two bootstrap parameter draws succeeded")

    hedge_risk = propagate_heston_uncertainty_to_hedging(
        bootstrap["bootstrap_parameter_samples"],
        fitted,
        p_scenario,
        strike=100.0,
        maturity=1.0,
        physical_total_return=0.06,
        n_paths=args.paths,
        n_steps=args.steps,
        option_type="call",
        transaction_cost_bps=args.cost_bps,
        confidence_level=args.confidence,
        seed=args.seed + 10_000,
    )
    report = {
        "title": "Heston calibration uncertainty propagated into delta-hedge tail risk",
        "classification": (
            "synthetic_quote_parametric_uncertainty_and_common_scenario_hedge_experiment"
        ),
        "seed": args.seed,
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
        "risk_neutral_generating_parameters_Q": {
            "variance0": q_true.variance0,
            "mean_reversion": q_true.mean_reversion,
            "long_run_variance": q_true.long_run_variance,
            "vol_of_variance": q_true.vol_of_variance,
            "correlation": q_true.correlation,
        },
        "risk_neutral_fitted_parameters_Q": calibration["parameters"],
        "physical_scenario_parameters_P": {
            "total_return_drift": 0.06,
            "variance0": p_scenario.variance0,
            "mean_reversion": p_scenario.mean_reversion,
            "long_run_variance": p_scenario.long_run_variance,
            "vol_of_variance": p_scenario.vol_of_variance,
            "correlation": p_scenario.correlation,
        },
        "base_calibration": calibration,
        "parametric_bootstrap": bootstrap,
        "hedge_parameter_uncertainty": hedge_risk,
        "interpretation_boundary": (
            "All hedge strategies share one simulated physical/scenario path set. Variation across "
            "draw-level hedge VaR/ES is therefore conditional on this fixed P scenario sample and "
            "is driven by risk-neutral quote-calibration parameter draws. The Gaussian quote-noise "
            "model is hypothetical; the physical scenario parameters are hypothetical. Within-draw "
            "VaR/ES use finite path samples. This is not a market backtest, forecast, model-risk "
            "interval or regulatory capital calculation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    heston_es = hedge_risk["strategies"]["heston_state_delta"][
        "parameter_conditional_risk_distribution"
    ]["net_es"]
    bs_es = hedge_risk["strategies"]["frozen_bs_delta"][
        "parameter_conditional_risk_distribution"
    ]["net_es"]
    print(
        f"Bootstrap fits={bootstrap['successful_replicates']}/"
        f"{bootstrap['requested_replicates']}; "
        f"Heston ES mean={heston_es['mean']:.6f} "
        f"(parameter SD={heston_es['standard_deviation']:.6f}); "
        f"frozen-BS ES mean={bs_es['mean']:.6f} "
        f"(parameter SD={bs_es['standard_deviation']:.6f})"
    )
    print(f"Wrote hedge parameter-uncertainty report to {args.output}")


if __name__ == "__main__":
    main()
