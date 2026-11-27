from __future__ import annotations

import argparse
import json
from pathlib import Path

from riskmanage.hedging import heston_delta_hedge_experiment
from riskmanage.heston import HestonParameters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Heston-model and frozen-Black-Scholes discrete delta hedges."
    )
    parser.add_argument("--paths", type=int, default=2_000)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--cost-bps", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/heston_delta_hedging.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.paths < 40 or args.steps < 1:
        raise SystemExit("paths must be >= 40 and steps must be positive")
    if args.cost_bps < 0.0:
        raise SystemExit("cost-bps must be non-negative")

    pricing = HestonParameters(
        spot=100.0,
        variance0=0.04,
        mean_reversion=1.5,
        long_run_variance=0.04,
        vol_of_variance=0.5,
        correlation=-0.7,
        risk_free_rate=0.03,
        dividend_yield=0.01,
    )
    physical = HestonParameters(
        spot=100.0,
        variance0=0.04,
        mean_reversion=1.25,
        long_run_variance=0.05,
        vol_of_variance=0.55,
        correlation=-0.65,
        risk_free_rate=0.03,
        dividend_yield=0.01,
    )
    frequencies = sorted(
        {
            max(4, args.steps // 2),
            args.steps,
            min(48, args.steps * 2),
        }
    )
    experiments = []
    for index, steps in enumerate(frequencies):
        result = heston_delta_hedge_experiment(
            pricing,
            physical,
            strike=100.0,
            maturity=1.0,
            physical_total_return=0.06,
            n_paths=args.paths,
            n_steps=steps,
            option_type="call",
            transaction_cost_bps=args.cost_bps,
            confidence_level=0.975,
            seed=args.seed + index,
        )
        experiments.append(result)

    report = {
        "title": "Heston dynamic delta hedge and hedging-loss tail risk",
        "classification": "synthetic_physical_scenarios_not_market_backtest",
        "pricing_parameters_Q": {
            "spot": pricing.spot,
            "variance0": pricing.variance0,
            "mean_reversion": pricing.mean_reversion,
            "long_run_variance": pricing.long_run_variance,
            "vol_of_variance": pricing.vol_of_variance,
            "correlation": pricing.correlation,
            "risk_free_rate": pricing.risk_free_rate,
            "dividend_yield": pricing.dividend_yield,
        },
        "physical_scenario_parameters_P": {
            "total_return_drift": 0.06,
            "variance_mean_reversion": physical.mean_reversion,
            "variance_long_run_level": physical.long_run_variance,
            "vol_of_variance": physical.vol_of_variance,
            "correlation": physical.correlation,
            "interpretation": (
                "Hypothetical scenario parameters used to generate hedging PnL; "
                "not an empirical forecast or calibrated physical-measure model."
            ),
        },
        "contract": {
            "type": "call",
            "spot": 100.0,
            "strike": 100.0,
            "maturity_years": 1.0,
        },
        "transaction_cost_bps": args.cost_bps,
        "seed": args.seed,
        "frequency_experiments": experiments,
        "interpretation_boundary": (
            "Initial option is valued under Q. Paths are simulated under explicit P/scenario "
            "drift and variance inputs. Heston delta is a vector Fourier derivative at each "
            "simulated state; the comparator freezes Black-Scholes volatility at sqrt(theta_Q). "
            "VaR and ES describe finite synthetic scenario samples, not forecasts or regulatory "
            "capital. Discrete hedge frequency, trapezoidal dividends, QE discretization, "
            "parameter uncertainty and model risk remain."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for experiment in experiments:
        heston = experiment["strategies"]["heston_fourier_delta"]
        black_scholes = experiment["strategies"]["frozen_black_scholes_delta"]
        print(
            f"steps={experiment['simulation']['n_steps']}: "
            f"Heston ES97.5={heston['net_loss_var_es']['expected_shortfall']:.5f}; "
            f"frozen-BS ES97.5={black_scholes['net_loss_var_es']['expected_shortfall']:.5f}"
        )
    print(f"Wrote delta-hedging report to {args.output}")


if __name__ == "__main__":
    main()
