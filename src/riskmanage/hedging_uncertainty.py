from __future__ import annotations

from dataclasses import replace
import math
from typing import Sequence

import numpy as np

from .hedging import _black_scholes_delta, _execute_hedge, _loss_var_es
from .heston import (
    HestonParameters,
    heston_option_price_delta_fourier_batch,
    heston_price_fourier,
    heston_simulate_paths,
)
from .uncertainty import _distribution_summary


_PARAMETER_NAMES = (
    "variance0",
    "mean_reversion",
    "long_run_variance",
    "vol_of_variance",
    "correlation",
)


def propagate_heston_uncertainty_to_hedging(
    parameter_samples: Sequence[dict],
    pricing_market: HestonParameters,
    physical_scenario: HestonParameters,
    strike: float,
    maturity: float,
    *,
    physical_total_return: float = 0.06,
    n_paths: int = 1_000,
    n_steps: int = 8,
    option_type: str = "call",
    transaction_cost_bps: float = 2.0,
    confidence_level: float = 0.95,
    seed: int = 20261126,
    psi_c: float = 1.5,
) -> dict:
    """Propagate Q-parameter bootstrap draws into discrete hedge PnL and VaR/ES.

    The same standard-normal innovations are reused for every risk-neutral
    parameter draw. The initial variance v0 is aligned across Q and P for each
    draw because it is the common initial state; all other physical scenario
    dynamics remain fixed. The resulting path states differ through that initial
    variance, but not through independently regenerated random shocks. The
    frozen-BS comparator uses sqrt(theta_Q) separately for each Q draw.

    The empirical VaR/ES distribution across bootstrap draws is conditional on
    common Gaussian innovations, fixed non-initial physical dynamics, and
    draw-specific initial variance states. It captures parameter-induced
    hedge-risk dispersion, not independent scenario-resampling uncertainty
    or model risk.
    """
    parameter_samples = tuple(parameter_samples)
    if len(parameter_samples) < 2:
        raise ValueError("at least two bootstrap parameter samples are required")
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive")
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if not np.isfinite(physical_total_return):
        raise ValueError("physical_total_return must be finite")
    if n_paths < 40 or n_steps < 1:
        raise ValueError("n_paths must be at least 40 and n_steps positive")
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be finite and non-negative")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0.5, 1)")
    if not np.isfinite(psi_c) or psi_c <= 0.0:
        raise ValueError("psi_c must be finite and positive")
    if not np.isclose(pricing_market.spot, physical_scenario.spot):
        raise ValueError("Q and P/scenario spot must agree")
    if not np.isclose(pricing_market.dividend_yield, physical_scenario.dividend_yield):
        raise ValueError("Q and P/scenario dividend yield must agree")
    if not np.isclose(pricing_market.risk_free_rate, physical_scenario.risk_free_rate):
        raise ValueError("Q and P/scenario funding rate must agree")

    # Reuse the exact same standard-normal innovations for every bootstrap
    # draw. The initial variance state follows the Q draw because v0 is the
    # same instantaneous state under Q and P; all other P dynamics stay fixed.
    # This keeps the initial premium, t=0 Delta and P path state coherent.
    dt = maturity / n_steps
    rate = pricing_market.risk_free_rate
    dividend_yield = pricing_market.dividend_yield
    cost_rate = transaction_cost_bps * 1e-4
    rows = []

    for draw_index, sample in enumerate(parameter_samples):
        try:
            q_params = replace(
                pricing_market,
                variance0=float(sample["variance0"]),
                mean_reversion=float(sample["mean_reversion"]),
                long_run_variance=float(sample["long_run_variance"]),
                vol_of_variance=float(sample["vol_of_variance"]),
                correlation=float(sample["correlation"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid bootstrap Heston parameter draw {draw_index}: {exc}") from exc

        if q_params.vol_of_variance <= 0.0:
            raise ValueError("bootstrap vol_of_variance samples must be positive")

        physical_draw = replace(
            physical_scenario, variance0=q_params.variance0
        )
        paths = heston_simulate_paths(
            physical_draw,
            maturity,
            n_steps,
            n_paths,
            seed=seed,
            psi_c=psi_c,
            asset_total_return=physical_total_return,
        )
        spots = paths["spots"]
        variances = paths["variances"]
        terminal_spot = spots[:, -1]
        if option_type == "call":
            terminal_payoff = np.maximum(terminal_spot - strike, 0.0)
        else:
            terminal_payoff = np.maximum(strike - terminal_spot, 0.0)

        initial_premium = heston_price_fourier(
            strike, maturity, q_params, option_type
        )["price"]
        heston_deltas = np.empty((n_paths, n_steps), dtype=float)
        bs_deltas = np.empty_like(heston_deltas)
        frozen_volatility = math.sqrt(q_params.long_run_variance)

        for step in range(n_steps):
            remaining = maturity - step * dt
            state_greeks = heston_option_price_delta_fourier_batch(
                spots[:, step],
                variances[:, step],
                strike,
                remaining,
                q_params,
                option_type,
            )
            heston_deltas[:, step] = state_greeks["deltas"]
            bs_deltas[:, step] = _black_scholes_delta(
                spots[:, step],
                strike,
                remaining,
                frozen_volatility,
                rate,
                dividend_yield,
                option_type,
            )

        heston_result = _execute_hedge(
            spots,
            heston_deltas,
            terminal_payoff,
            initial_premium,
            rate,
            dividend_yield,
            dt,
            cost_rate,
        )
        bs_result = _execute_hedge(
            spots,
            bs_deltas,
            terminal_payoff,
            initial_premium,
            rate,
            dividend_yield,
            dt,
            cost_rate,
        )

        def summarize(result: dict) -> dict:
            net_pnl = result["net_pnl"]
            gross_pnl = result["gross_pnl"]
            net_risk = _loss_var_es(-net_pnl, confidence_level)
            gross_risk = _loss_var_es(-gross_pnl, confidence_level)
            return {
                "net_pnl_mean": float(np.mean(net_pnl)),
                "net_pnl_standard_deviation": float(np.std(net_pnl, ddof=1)),
                "gross_pnl_mean": float(np.mean(gross_pnl)),
                "gross_pnl_standard_deviation": float(np.std(gross_pnl, ddof=1)),
                "probability_of_net_loss": float(np.mean(net_pnl < 0.0)),
                "net_loss_var_es": net_risk,
                "gross_loss_var_es": gross_risk,
                "average_transaction_cost": float(np.mean(result["transaction_costs"])),
                "average_transaction_cost_future_value": float(
                    np.mean(result["transaction_costs_future_value"])
                ),
                "average_turnover": float(np.mean(result["turnover"])),
                "net_pnl": net_pnl,
            }

        heston_summary = summarize(heston_result)
        bs_summary = summarize(bs_result)
        paired_difference = heston_result["net_pnl"] - bs_result["net_pnl"]
        paired_metrics = {
            "mean_heston_minus_bs_net_pnl": float(np.mean(paired_difference)),
            "standard_deviation_paired_net_pnl_difference": float(
                np.std(paired_difference, ddof=1)
            ),
            "probability_heston_net_pnl_exceeds_bs": float(
                np.mean(paired_difference > 0.0)
            ),
        }
        heston_summary.pop("net_pnl")
        bs_summary.pop("net_pnl")
        rows.append(
            {
                "bootstrap_draw": int(draw_index),
                "physical_scenario_initial_variance": float(physical_draw.variance0),
                "risk_neutral_parameters": {
                    name: float(getattr(q_params, name))
                    for name in _PARAMETER_NAMES
                },
                "feller_ratio": float(q_params.feller_ratio),
                "initial_heston_option_premium": float(initial_premium),
                "frozen_bs_volatility": float(frozen_volatility),
                "heston_state_delta_strategy": heston_summary,
                "frozen_bs_delta_strategy": bs_summary,
                "paired_comparison": paired_metrics,
            }
        )

    def risk_distribution(strategy_key: str) -> dict:
        metrics = {
            "net_var": [],
            "net_es": [],
            "gross_var": [],
            "gross_es": [],
            "net_pnl_mean": [],
            "net_pnl_standard_deviation": [],
            "average_turnover": [],
            "average_transaction_cost": [],
        }
        for row in rows:
            summary = row[strategy_key]
            metrics["net_var"].append(summary["net_loss_var_es"]["var"])
            metrics["net_es"].append(
                summary["net_loss_var_es"]["expected_shortfall"]
            )
            metrics["gross_var"].append(summary["gross_loss_var_es"]["var"])
            metrics["gross_es"].append(
                summary["gross_loss_var_es"]["expected_shortfall"]
            )
            metrics["net_pnl_mean"].append(summary["net_pnl_mean"])
            metrics["net_pnl_standard_deviation"].append(
                summary["net_pnl_standard_deviation"]
            )
            metrics["average_turnover"].append(summary["average_turnover"])
            metrics["average_transaction_cost"].append(
                summary["average_transaction_cost"]
            )
        return {
            name: _distribution_summary(np.asarray(values, dtype=float))
            for name, values in metrics.items()
        }

    heston_es = np.array(
        [
            row["heston_state_delta_strategy"]["net_loss_var_es"][
                "expected_shortfall"
            ]
            for row in rows
        ],
        dtype=float,
    )
    bs_es = np.array(
        [
            row["frozen_bs_delta_strategy"]["net_loss_var_es"][
                "expected_shortfall"
            ]
            for row in rows
        ],
        dtype=float,
    )
    heston_var = np.array(
        [
            row["heston_state_delta_strategy"]["net_loss_var_es"]["var"]
            for row in rows
        ],
        dtype=float,
    )
    bs_var = np.array(
        [
            row["frozen_bs_delta_strategy"]["net_loss_var_es"]["var"]
            for row in rows
        ],
        dtype=float,
    )
    return {
        "classification": "synthetic_common_scenario_parameter_uncertainty_experiment",
        "option": {
            "type": option_type,
            "strike": float(strike),
            "maturity_years": float(maturity),
            "spot": float(pricing_market.spot),
        },
        "scenario": {
            "physical_total_return_drift": float(physical_total_return),
            "physical_mean_reversion": float(physical_scenario.mean_reversion),
            "physical_long_run_variance": float(
                physical_scenario.long_run_variance
            ),
            "physical_vol_of_variance": float(physical_scenario.vol_of_variance),
            "physical_correlation": float(physical_scenario.correlation),
            "seed": int(seed),
            "n_paths": int(n_paths),
            "n_steps": int(n_steps),
            "same_standard_normal_innovations_for_all_parameter_draws": True,
            "identical_path_states_for_all_parameter_draws": False,
            "initial_variance_state": (
                "Each Q bootstrap draw's v0 is also used as the P/scenario initial "
                "variance state; all other P parameters and Gaussian innovations are fixed."
            ),
        },
        "market_conventions": {
            "risk_free_rate": float(rate),
            "dividend_yield": float(dividend_yield),
            "transaction_cost_bps": float(transaction_cost_bps),
            "confidence_level": float(confidence_level),
        },
        "bootstrap_parameter_draw_count": int(len(rows)),
        "strategies": {
            "heston_state_delta": {
                "parameter_conditional_risk_distribution": risk_distribution(
                    "heston_state_delta_strategy"
                )
            },
            "frozen_bs_delta": {
                "parameter_conditional_risk_distribution": risk_distribution(
                    "frozen_bs_delta_strategy"
                )
            },
        },
        "paired_parameter_draw_comparison": {
            "probability_heston_es_below_bs_es": float(np.mean(heston_es < bs_es)),
            "probability_heston_var_below_bs_var": float(np.mean(heston_var < bs_var)),
            "heston_minus_bs_es_distribution": _distribution_summary(
                heston_es - bs_es
            ),
            "heston_minus_bs_var_distribution": _distribution_summary(
                heston_var - bs_var
            ),
        },
        "draw_level_results": rows,
        "uncertainty_boundary": (
            "Across-draw distributions capture risk-neutral calibration parameter uncertainty "
            "under common standard-normal innovations and fixed non-initial physical scenario "
            "dynamics. The initial P/scenario variance state follows each bootstrap draw's v0 so "
            "the initial state is consistent under Q and P. Path states therefore differ through "
            "v0, but shocks are not independently regenerated. The results exclude physical "
            "parameter uncertainty beyond this initial-state propagation, independent scenario "
            "resampling uncertainty, quote-model misspecification, market microstructure, curve "
            "uncertainty and model risk. Within-draw VaR/ES use a finite synthetic path sample; "
            "tail precision depends on n_paths*(1-confidence_level). This is not a regulatory "
            "capital measure or predictive confidence interval."
        ),
    }
