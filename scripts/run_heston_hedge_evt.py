from __future__ import annotations

import argparse
import json
from pathlib import Path

from riskmanage.hedging import heston_delta_hedge_experiment
from riskmanage.heston import HestonParameters
from riskmanage.tail_risk import fit_pot_gpd_tail_risk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare empirical hedge VaR/ES with Peaks-over-Threshold GPD "
            "tail extrapolation across thresholds."
        )
    )
    parser.add_argument("--paths", type=int, default=10_000)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--cost-bps", type=float, default=2.0)
    parser.add_argument("--confidence-levels", type=float, nargs="+", default=[0.975, 0.99])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.90, 0.925, 0.95])
    parser.add_argument("--min-exceedances", type=int, default=50)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--full-sample-bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/heston_hedge_evt.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.paths < 100 or args.steps < 1:
        raise SystemExit("paths must be >= 100 and steps must be positive")
    if args.cost_bps < 0.0 or args.bootstrap < 0 or args.min_exceedances < 5:
        raise SystemExit("cost-bps/bootstrap/min-exceedances settings are invalid")
    if any(not 0.5 < value < 1.0 for value in args.confidence_levels):
        raise SystemExit("confidence levels must lie in (0.5, 1)")
    if any(not 0.5 < value < 1.0 for value in args.thresholds):
        raise SystemExit("threshold quantiles must lie in (0.5, 1)")
    if any(alpha <= max(args.thresholds) for alpha in args.confidence_levels):
        raise SystemExit("each confidence level must exceed the maximum POT threshold")

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
    hedge = heston_delta_hedge_experiment(
        pricing,
        physical,
        strike=100.0,
        maturity=1.0,
        physical_total_return=0.06,
        n_paths=args.paths,
        n_steps=args.steps,
        transaction_cost_bps=args.cost_bps,
        confidence_level=min(args.confidence_levels),
        seed=args.seed,
        include_path_samples=True,
    )
    path_samples = hedge.pop("path_samples")
    evt_results = {}
    for strategy_index, (strategy, samples) in enumerate(path_samples.items()):
        evt_results[strategy] = {}
        for measure_index, measure in enumerate(("net_losses", "gross_losses")):
            evt_results[strategy][measure] = fit_pot_gpd_tail_risk(
                samples[measure],
                confidence_levels=args.confidence_levels,
                threshold_quantiles=args.thresholds,
                min_exceedances=args.min_exceedances,
                n_bootstrap=args.bootstrap,
                n_full_sample_bootstrap=args.full_sample_bootstrap,
                seed=args.seed + 10_000 * strategy_index + 100 * measure_index,
            )

    report = {
        "title": "Heston hedge loss tail risk: empirical versus POT-GPD",
        "classification": "synthetic_physical_scenario_evt_tail_model_experiment",
        "seed": args.seed,
        "scenario_parameters_Q": {
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
            "variance0": physical.variance0,
            "mean_reversion": physical.mean_reversion,
            "long_run_variance": physical.long_run_variance,
            "vol_of_variance": physical.vol_of_variance,
            "correlation": physical.correlation,
        },
        "hedge_experiment": hedge,
        "evt_configuration": {
            "confidence_levels": args.confidence_levels,
            "threshold_quantiles": args.thresholds,
            "minimum_exceedances": args.min_exceedances,
            "conditional_exceedance_bootstrap_replicates": args.bootstrap,
            "full_sample_iid_bootstrap_replicates": args.full_sample_bootstrap,
            "bootstrap_method": (
                "non-parametric resampling of fitted GPD exceedances, conditional on "
                "empirical threshold and exceedance count"
            ),
        },
        "tail_risk_models": evt_results,
        "interpretation_boundary": (
            "The POT-GPD fits extrapolate beyond empirical thresholds. The threshold grid is "
            "reported rather than selecting a single optimum. GPD bootstrap intervals condition "
            "on threshold and exceedance count; they exclude threshold-selection uncertainty, "
            "serial dependence, structural breaks and model risk. Compare shape/scale and VaR/ES "
            "across thresholds; do not report a precise ES when shape >= 1 or too few exceedances "
            "are available. The physical dynamics are hypothetical scenario parameters, not forecasts."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for strategy in evt_results:
        print(f"{strategy}:")
        for measure in evt_results[strategy]:
            fits = evt_results[strategy][measure]["threshold_fits"]
            compact = [
                {
                    "threshold_q": fit["threshold_quantile"],
                    "n_exc": fit["exceedance_count"],
                    "shape": fit["fit"]["shape"] if fit["fit"] else None,
                    "scale": fit["fit"]["scale"] if fit["fit"] else None,
                }
                for fit in fits
            ]
            print(f"  {measure}: {compact}")
    print(f"Wrote POT-GPD hedge tail-risk report to {args.output}")


if __name__ == "__main__":
    main()
