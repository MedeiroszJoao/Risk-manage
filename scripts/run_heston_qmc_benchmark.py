from __future__ import annotations

import argparse
import json
from pathlib import Path

from riskmanage.heston import HestonParameters, heston_monte_carlo_price, heston_price_fourier
from riskmanage.qmc import heston_randomized_qmc_price


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare randomized Sobol/Brownian-bridge Heston QMC with pseudo-MC."
    )
    parser.add_argument("--sobol-power", type=int, default=10)
    parser.add_argument("--scrambles", type=int, default=8)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/heston_qmc_benchmark.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sobol_power < 4 or args.scrambles < 4 or args.steps < 1:
        raise SystemExit("sobol-power >= 4, scrambles >= 4 and steps >= 1 are required")

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
    qmc_bridge = heston_randomized_qmc_price(
        strike, maturity, params,
        sobol_power=args.sobol_power,
        n_scrambles=args.scrambles,
        n_steps=args.steps,
        seed=args.seed,
        control_variate=True,
        brownian_bridge=True,
    )
    qmc_standard_order = heston_randomized_qmc_price(
        strike, maturity, params,
        sobol_power=args.sobol_power,
        n_scrambles=args.scrambles,
        n_steps=args.steps,
        seed=args.seed,
        control_variate=True,
        brownian_bridge=False,
    )
    points_per_scramble = 2**args.sobol_power
    total_paths = points_per_scramble * args.scrambles
    # Match both the valuation paths and the independent control-variate pilot
    # path budget; this is a path-count comparison, not equal wall-clock cost.
    pilot_paths = points_per_scramble
    pseudo_mc = heston_monte_carlo_price(
        strike,
        maturity,
        params,
        n_paths=total_paths,
        n_steps=args.steps,
        seed=args.seed,
        pilot_paths=pilot_paths,
    )
    if qmc_bridge["total_valuation_points"] != pseudo_mc["n_paths"]:
        raise RuntimeError("QMC and pseudo-MC valuation path budgets are not equal")
    if qmc_bridge["pilot_points"] != pilot_paths:
        raise RuntimeError("QMC and pseudo-MC independent pilot budgets are not equal")
    if qmc_standard_order["total_valuation_points"] != pseudo_mc["n_paths"]:
        raise RuntimeError("Standard-order QMC and pseudo-MC path budgets are not equal")

    fourier = heston_price_fourier(strike, maturity, params, "call")
    reference = fourier["price"]
    report = {
        "title": "Heston randomized Sobol QMC vs pseudo-Monte Carlo",
        "classification": "synthetic_parameters_numerical_efficiency_experiment",
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
        },
        "fourier_reference": {
            "price": reference,
            "quadrature_error_estimate": fourier["quadrature_error_estimate"],
        },
        "randomized_qmc_brownian_bridge": qmc_bridge,
        "randomized_qmc_standard_order": qmc_standard_order,
        "pseudo_monte_carlo": {
            "price": pseudo_mc["price"],
            "standard_error": pseudo_mc["standard_error"],
            "confidence_interval": pseudo_mc["confidence_interval"],
            "n_paths": pseudo_mc["n_paths"],
            "effective_samples": pseudo_mc["effective_samples"],
            "variance_reduction_factor": pseudo_mc["variance_reduction_factor"],
            "stock_martingale_relative_error": pseudo_mc[
                "stock_martingale_relative_error"
            ],
        },
        "comparison": {
            "qmc_brownian_bridge_abs_error_vs_fourier": abs(
                qmc_bridge["price"] - reference
            ),
            "qmc_standard_order_abs_error_vs_fourier": abs(
                qmc_standard_order["price"] - reference
            ),
            "pseudo_mc_abs_error_vs_fourier": abs(pseudo_mc["price"] - reference),
            "qmc_brownian_bridge_error_in_standard_errors": (
                abs(qmc_bridge["price"] - reference)
                / qmc_bridge["standard_error_across_scrambles"]
                if qmc_bridge["standard_error_across_scrambles"] > 0.0
                else None
            ),
            "qmc_standard_order_error_in_standard_errors": (
                abs(qmc_standard_order["price"] - reference)
                / qmc_standard_order["standard_error_across_scrambles"]
                if qmc_standard_order["standard_error_across_scrambles"] > 0.0
                else None
            ),
            "qmc_bridge_to_pseudo_mc_se_ratio": (
                qmc_bridge["standard_error_across_scrambles"]
                / pseudo_mc["standard_error"]
                if pseudo_mc["standard_error"] > 0.0 else None
            ),
            "qmc_standard_order_to_pseudo_mc_se_ratio": (
                qmc_standard_order["standard_error_across_scrambles"]
                / pseudo_mc["standard_error"]
                if pseudo_mc["standard_error"] > 0.0 else None
            ),
            "equal_valuation_path_budget": (
                qmc_bridge["total_valuation_points"] == pseudo_mc["n_paths"]
            ),
            "qmc_total_path_equivalents_including_pilot": (
                qmc_bridge["total_valuation_points"] + qmc_bridge["pilot_points"]
            ),
            "pseudo_mc_total_path_equivalents_including_pilot": (
                pseudo_mc["n_paths"] + pilot_paths
            ),
            "equal_total_path_equivalent_budget": (
                qmc_bridge["total_valuation_points"] + qmc_bridge["pilot_points"]
                == pseudo_mc["n_paths"] + pilot_paths
            ),
        },
        "interpretation_boundary": (
            "QMC standard error is estimated across independent randomized Sobol scrambles; "
            "pseudo-MC standard error is based on antithetic pair observations. Both are "
            "conditional on the QE time grid and exclude time-discretization bias, parameter "
            "uncertainty and model risk. The control-variate beta is estimated on an independent "
            "Sobol pilot scramble. Equal path budgets do not imply equal computational cost."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Fourier={reference:.8f}; QMC-BB={qmc_bridge['price']:.8f} "
        f"(SE={qmc_bridge['standard_error_across_scrambles']:.6f}); "
        f"QMC-standard={qmc_standard_order['price']:.8f} "
        f"(SE={qmc_standard_order['standard_error_across_scrambles']:.6f}); "
        f"pseudo-MC={pseudo_mc['price']:.8f} (SE={pseudo_mc['standard_error']:.6f})"
    )
    print(f"Wrote randomized QMC benchmark to {args.output}")


if __name__ == "__main__":
    main()
