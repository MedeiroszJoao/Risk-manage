from __future__ import annotations

from typing import Sequence

import numpy as np

from .hedging_uncertainty import propagate_heston_uncertainty_to_hedging
from .heston import HestonParameters


def two_way_random_effects_decomposition(values) -> dict:
    """Method-of-moments variance decomposition for a balanced crossed B x R design.

    Rows represent calibration-parameter bootstrap draws; columns represent
    independent physical/scenario seeds. The residual component combines
    parameter-by-scenario interaction and within-cell hedge Monte Carlo error.
    Negative method-of-moments estimates are retained as raw diagnostics and
    truncated at zero for non-negative variance-component reporting.
    """
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("values must be a finite B x R matrix with B,R >= 2")
    if not np.isfinite(matrix).all():
        raise ValueError("variance-decomposition matrix must be finite")

    n_parameters, n_scenarios = matrix.shape
    grand_mean = float(np.mean(matrix))
    parameter_means = np.mean(matrix, axis=1)
    scenario_means = np.mean(matrix, axis=0)
    residuals = (
        matrix
        - parameter_means[:, None]
        - scenario_means[None, :]
        + grand_mean
    )

    ss_parameter = float(
        n_scenarios * np.sum((parameter_means - grand_mean) ** 2)
    )
    ss_scenario = float(
        n_parameters * np.sum((scenario_means - grand_mean) ** 2)
    )
    ss_residual = float(np.sum(residuals**2))
    df_parameter = n_parameters - 1
    df_scenario = n_scenarios - 1
    df_residual = (n_parameters - 1) * (n_scenarios - 1)
    ms_parameter = ss_parameter / df_parameter
    ms_scenario = ss_scenario / df_scenario
    ms_residual = ss_residual / df_residual

    raw_parameter = (ms_parameter - ms_residual) / n_scenarios
    raw_scenario = (ms_scenario - ms_residual) / n_parameters
    raw_interaction = ms_residual
    components = {
        "calibration_parameter": max(float(raw_parameter), 0.0),
        "scenario_sampling": max(float(raw_scenario), 0.0),
        "interaction_and_within_cell_mc": max(float(raw_interaction), 0.0),
    }
    estimated_total = float(sum(components.values()))
    observed_total = float(np.var(matrix, ddof=1))
    proportions = (
        {
            name: float(component / estimated_total)
            for name, component in components.items()
        }
        if estimated_total > 0.0
        else None
    )

    return {
        "design": "balanced_two_way_random_effects_method_of_moments",
        "n_calibration_draws": int(n_parameters),
        "n_scenario_seeds": int(n_scenarios),
        "grand_mean": grand_mean,
        "parameter_draw_means": [float(value) for value in parameter_means],
        "scenario_seed_means": [float(value) for value in scenario_means],
        "sum_of_squares": {
            "calibration_parameter": ss_parameter,
            "scenario_sampling": ss_scenario,
            "interaction_residual": ss_residual,
        },
        "degrees_of_freedom": {
            "calibration_parameter": int(df_parameter),
            "scenario_sampling": int(df_scenario),
            "interaction_residual": int(df_residual),
        },
        "mean_squares": {
            "calibration_parameter": float(ms_parameter),
            "scenario_sampling": float(ms_scenario),
            "interaction_residual": float(ms_residual),
        },
        "raw_variance_component_estimates": {
            "calibration_parameter": float(raw_parameter),
            "scenario_sampling": float(raw_scenario),
            "interaction_and_within_cell_mc": float(raw_interaction),
        },
        "nonnegative_variance_components": components,
        "estimated_total_variance": estimated_total,
        "observed_matrix_variance": observed_total,
        "variance_component_proportions": proportions,
        "interpretation_boundary": (
            "Method-of-moments two-way random-effects decomposition for a balanced crossed "
            "design. The residual component combines parameter-by-scenario interaction and "
            "within-cell hedge Monte Carlo error. Small B/R, non-Gaussian tail estimators, "
            "clipped negative components and scenario heterogeneity can make estimates noisy. "
            "This is not a confidence interval or a regulatory capital decomposition."
        ),
    }


def decompose_heston_hedge_risk_uncertainty(
    parameter_samples: Sequence[dict],
    pricing_market: HestonParameters,
    physical_scenario: HestonParameters,
    strike: float,
    maturity: float,
    scenario_seeds: Sequence[int],
    *,
    physical_total_return: float = 0.06,
    n_paths: int = 1_000,
    n_steps: int = 8,
    option_type: str = "call",
    transaction_cost_bps: float = 2.0,
    confidence_level: float = 0.95,
    psi_c: float = 1.5,
) -> dict:
    """Cross calibration draws with independent physical/scenario seeds.

    For each scenario seed, FE-009 reuses identical Gaussian innovations across
    calibration draws and aligns the initial variance state to each draw's v0.
    Different scenario seeds create independent physical/scenario innovation
    sets. A balanced B x R matrix of per-draw net VaR/ES is assembled for the
    Heston and frozen-BS strategies, plus their paired ES/VaR differences.
    Two-way random-effects method-of-moments decomposition separates parameter,
    scenario, and residual/interaction components.
    """
    parameter_samples = tuple(parameter_samples)
    scenario_seeds = tuple(int(seed) for seed in scenario_seeds)
    if len(parameter_samples) < 2:
        raise ValueError("at least two calibration parameter draws are required")
    if len(scenario_seeds) < 2:
        raise ValueError("at least two independent scenario seeds are required")
    if any(seed < 0 for seed in scenario_seeds):
        raise ValueError("scenario seeds must be non-negative")
    if len(set(scenario_seeds)) != len(scenario_seeds):
        raise ValueError("scenario seeds must be unique")
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive")
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive")

    b_count = len(parameter_samples)
    r_count = len(scenario_seeds)
    matrices = {
        "heston_net_es": np.empty((b_count, r_count), dtype=float),
        "frozen_bs_net_es": np.empty((b_count, r_count), dtype=float),
        "heston_net_var": np.empty((b_count, r_count), dtype=float),
        "frozen_bs_net_var": np.empty((b_count, r_count), dtype=float),
        "heston_minus_bs_net_es": np.empty((b_count, r_count), dtype=float),
        "heston_minus_bs_net_var": np.empty((b_count, r_count), dtype=float),
    }
    scenario_summaries = []
    for scenario_index, scenario_seed in enumerate(scenario_seeds):
        result = propagate_heston_uncertainty_to_hedging(
            parameter_samples,
            pricing_market,
            physical_scenario,
            strike,
            maturity,
            physical_total_return=physical_total_return,
            n_paths=n_paths,
            n_steps=n_steps,
            option_type=option_type,
            transaction_cost_bps=transaction_cost_bps,
            confidence_level=confidence_level,
            seed=scenario_seed,
            psi_c=psi_c,
        )
        heston_es = np.empty(b_count)
        bs_es = np.empty(b_count)
        heston_var = np.empty(b_count)
        bs_var = np.empty(b_count)
        for draw_index, row in enumerate(result["draw_level_results"]):
            heston_risk = row["heston_state_delta_strategy"]["net_loss_var_es"]
            bs_risk = row["frozen_bs_delta_strategy"]["net_loss_var_es"]
            heston_es[draw_index] = heston_risk["expected_shortfall"]
            bs_es[draw_index] = bs_risk["expected_shortfall"]
            heston_var[draw_index] = heston_risk["var"]
            bs_var[draw_index] = bs_risk["var"]

        matrices["heston_net_es"][:, scenario_index] = heston_es
        matrices["frozen_bs_net_es"][:, scenario_index] = bs_es
        matrices["heston_net_var"][:, scenario_index] = heston_var
        matrices["frozen_bs_net_var"][:, scenario_index] = bs_var
        matrices["heston_minus_bs_net_es"][:, scenario_index] = heston_es - bs_es
        matrices["heston_minus_bs_net_var"][:, scenario_index] = heston_var - bs_var
        scenario_summaries.append(
            {
                "scenario_index": int(scenario_index),
                "seed": int(scenario_seed),
                "same_innovations_across_calibration_draws": True,
                "calibration_draw_count": b_count,
                "n_paths": int(n_paths),
                "n_steps": int(n_steps),
                "tail_observations_per_draw": float(
                    n_paths * (1.0 - confidence_level)
                ),
            }
        )

    decompositions = {
        name: two_way_random_effects_decomposition(matrix)
        for name, matrix in matrices.items()
    }
    return {
        "classification": "synthetic_crossed_calibration_and_scenario_uncertainty_experiment",
        "option": {
            "type": option_type,
            "strike": float(strike),
            "maturity_years": float(maturity),
            "spot": float(pricing_market.spot),
        },
        "design": {
            "calibration_bootstrap_draws": b_count,
            "independent_physical_scenario_seeds": r_count,
            "same_gaussian_innovations_across_calibration_draws_within_seed": True,
            "initial_variance_state_follows_each_calibration_draw": True,
            "other_physical_scenario_parameters_fixed": True,
            "n_paths_per_cell": int(n_paths),
            "hedge_steps": int(n_steps),
            "confidence_level": float(confidence_level),
            "tail_observations_per_cell": float(
                n_paths * (1.0 - confidence_level)
            ),
            "transaction_cost_bps": float(transaction_cost_bps),
        },
        "scenario_replicates": scenario_summaries,
        "metric_matrices": {
            name: matrix.tolist() for name, matrix in matrices.items()
        },
        "variance_decompositions": decompositions,
        "interpretation_boundary": (
            "The crossed two-way method-of-moments estimates are conditional on the Gaussian "
            "quote-noise model, Heston model, physical scenario dynamics, discrete hedge rules "
            "and finite per-cell path count. The residual component combines parameter-by-scenario "
            "interaction and within-cell Monte Carlo error. Small numbers of calibration draws and "
            "scenario seeds produce noisy variance components. No physical parameter uncertainty, "
            "model uncertainty, market impact, live data calibration or regulatory capital claim "
            "is included."
        ),
    }
