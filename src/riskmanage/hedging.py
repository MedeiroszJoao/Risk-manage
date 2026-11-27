from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr
from scipy.stats import t as student_t

from .heston import (
    HestonParameters,
    heston_option_price_delta_fourier_batch,
    heston_price_fourier,
    heston_simulate_paths,
)


def _paired_pathwise_statistics(differences: np.ndarray) -> dict:
    """Summarize matched strategy differences across independent simulation paths."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError(
            "paired path differences must be a finite one-dimensional sample "
            "with at least two pairs"
        )

    n_pairs = int(values.size)
    mean_difference = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    standard_error = float(standard_deviation / math.sqrt(n_pairs))
    critical_value = float(student_t.ppf(0.975, df=n_pairs - 1))
    half_width = critical_value * standard_error

    wins = int(np.count_nonzero(values > 0.0))
    losses = int(np.count_nonzero(values < 0.0))
    ties = int(n_pairs - wins - losses)
    win_fraction = float(wins / n_pairs)

    # Wilson score interval for the pathwise outperformance fraction.
    z_value = 1.959963984540054
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / n_pairs
    centre = (win_fraction + z_squared / (2.0 * n_pairs)) / denominator
    radius = (
        z_value
        * math.sqrt(
            win_fraction * (1.0 - win_fraction) / n_pairs
            + z_squared / (4.0 * n_pairs * n_pairs)
        )
        / denominator
    )

    return {
        "paired_path_count": n_pairs,
        "mean_difference": mean_difference,
        "standard_deviation_difference": standard_deviation,
        "standard_error_of_mean_difference": standard_error,
        "mean_difference_95_percent_t_interval": [
            float(mean_difference - half_width),
            float(mean_difference + half_width),
        ],
        "mean_difference_interval_method": (
            "paired_student_t_interval_across_independent_simulated_paths"
        ),
        "pathwise_wins": wins,
        "pathwise_losses": losses,
        "pathwise_ties": ties,
        "pathwise_outperformance_fraction": win_fraction,
        "pathwise_outperformance_95_percent_wilson_interval": [
            float(max(0.0, centre - radius)),
            float(min(1.0, centre + radius)),
        ],
        "paired_standardized_mean_difference": (
            float(mean_difference / standard_deviation)
            if standard_deviation > 0.0
            else None
        ),
        "sampling_assumption": (
            "Each pair is one independently simulated physical path evaluated under "
            "both hedge policies. Intervals are conditional on the supplied scenario "
            "parameters, pricing model and QE-M discretization; they do not cover "
            "parameter, model, market-data or discretization uncertainty."
        ),
    }


def _black_scholes_delta(
    spots: np.ndarray,
    strike: float,
    maturity: float,
    volatility: float,
    rate: float,
    dividend_yield: float,
    option_type: str,
) -> np.ndarray:
    if maturity <= 0.0:
        if option_type == "call":
            return (spots > strike).astype(float)
        return -(spots < strike).astype(float)
    if volatility <= 0.0:
        forward_intrinsic = spots * math.exp((rate - dividend_yield) * maturity) - strike
        call = math.exp(-dividend_yield * maturity) * (
            (forward_intrinsic > 0.0).astype(float)
        )
        return call if option_type == "call" else call - math.exp(-dividend_yield * maturity)
    total_volatility = volatility * math.sqrt(maturity)
    d1 = (
        np.log(spots / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * maturity
    ) / total_volatility
    call_delta = math.exp(-dividend_yield * maturity) * ndtr(d1)
    if option_type == "call":
        return call_delta
    return call_delta - math.exp(-dividend_yield * maturity)


def _loss_var_es(losses: np.ndarray, confidence_level: float) -> dict:
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or not np.isfinite(losses).all():
        raise ValueError("losses must be a finite one-dimensional array")
    tail_mass = losses.size * (1.0 - confidence_level)
    if tail_mass < 1.0:
        raise ValueError("sample is too small for the requested confidence level")
    var = float(np.quantile(losses, confidence_level, method="inverted_cdf"))
    strict_tail = losses > var
    strict_count = int(np.sum(strict_tail))
    es = (
        float(np.sum(losses[strict_tail])) + (tail_mass - strict_count) * var
    ) / tail_mass
    return {
        "confidence_level": float(confidence_level),
        "var": var,
        "expected_shortfall": float(es),
        "tail_mass_observations": float(tail_mass),
        "strict_tail_observations": strict_count,
        "var_boundary_observations": int(np.sum(losses == var)),
        "method": "inverted-CDF empirical VaR with fractional-boundary ES",
    }


def _execute_hedge(
    spots: np.ndarray,
    deltas: np.ndarray,
    payoff: np.ndarray,
    initial_option_premium: float,
    rate: float,
    dividend_yield: float,
    dt: float,
    transaction_cost_rate: float,
) -> dict:
    n_paths, n_steps = deltas.shape
    initial_spot = spots[:, 0]
    cash = (
        initial_option_premium
        - deltas[:, 0] * initial_spot
    )
    initial_cost = transaction_cost_rate * initial_spot * np.abs(deltas[:, 0])
    cash -= initial_cost
    transaction_costs = initial_cost.copy()
    transaction_costs_future_value = initial_cost.copy()
    turnover = initial_spot * np.abs(deltas[:, 0])
    previous_delta = deltas[:, 0].copy()
    discount_growth = math.exp(rate * dt)

    for step in range(1, n_steps):
        old_spot = spots[:, step - 1]
        new_spot = spots[:, step]
        cash *= discount_growth
        transaction_costs_future_value *= discount_growth
        cash += dividend_yield * previous_delta * 0.5 * (old_spot + new_spot) * dt

        new_delta = deltas[:, step]
        trade = new_delta - previous_delta
        trade_notional = new_spot * np.abs(trade)
        cost = transaction_cost_rate * trade_notional
        cash -= new_spot * trade + cost
        transaction_costs += cost
        transaction_costs_future_value += cost
        turnover += trade_notional
        previous_delta = new_delta

    old_spot = spots[:, -2]
    terminal_spot = spots[:, -1]
    cash *= discount_growth
    transaction_costs_future_value *= discount_growth
    cash += dividend_yield * previous_delta * 0.5 * (
        old_spot + terminal_spot
    ) * dt
    net_pnl = cash + previous_delta * terminal_spot - payoff
    # Gross PnL adds back each cost at maturity value, including foregone funding.
    gross_pnl = net_pnl + transaction_costs_future_value
    return {
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "transaction_costs": transaction_costs,
        "transaction_costs_future_value": transaction_costs_future_value,
        "turnover": turnover,
    }


def heston_delta_hedge_experiment(
    pricing_params: HestonParameters,
    physical_params: HestonParameters,
    strike: float,
    maturity: float,
    *,
    physical_total_return: float = 0.06,
    n_paths: int = 20_000,
    n_steps: int = 24,
    option_type: str = "call",
    transaction_cost_bps: float = 0.0,
    frozen_black_scholes_volatility: float | None = None,
    confidence_level: float = 0.975,
    seed: int = 20261125,
    psi_c: float = 1.5,
    include_path_samples: bool = False,
) -> dict:
    """Compare Heston-model delta hedging with a frozen-vol Black-Scholes hedge.

    The option is sold at its risk-neutral Heston Fourier price. Underlying paths
    are simulated with a separately supplied physical/scenario drift and variance
    dynamics. Both hedges use identical paths, rebalance on the same grid and pay
    the same proportional transaction costs. The output includes self-financing
    hedging P&L, turnover, costs, empirical VaR and fractional-tail ES.

    Dividends are accrued using trapezoidal integration over each hedge interval.
    Discrete rebalancing, this dividend approximation, QE time discretization,
    physical parameter uncertainty and model risk are not removed by the reported
    Monte Carlo sample statistics. The physical parameters are scenario inputs,
    not estimated forecasts.
    """
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
    if not isinstance(include_path_samples, bool):
        raise ValueError("include_path_samples must be boolean")
    if not np.isclose(pricing_params.spot, physical_params.spot):
        raise ValueError("pricing and physical spot must agree")
    if not np.isclose(pricing_params.variance0, physical_params.variance0):
        raise ValueError("pricing and physical initial variance must agree")
    if not np.isclose(
        pricing_params.dividend_yield, physical_params.dividend_yield
    ):
        raise ValueError("pricing and physical dividend yields must agree")

    rate = pricing_params.risk_free_rate
    dividend_yield = pricing_params.dividend_yield
    if frozen_black_scholes_volatility is None:
        frozen_volatility = math.sqrt(pricing_params.long_run_variance)
    else:
        frozen_volatility = float(frozen_black_scholes_volatility)
    if not np.isfinite(frozen_volatility) or frozen_volatility <= 0.0:
        raise ValueError("frozen_black_scholes_volatility must be finite and positive")

    paths = heston_simulate_paths(
        physical_params,
        maturity,
        n_steps,
        n_paths,
        seed=seed,
        psi_c=psi_c,
        asset_total_return=physical_total_return,
    )
    spots = paths["spots"]
    variances = paths["variances"]
    dt = maturity / n_steps

    initial_price = heston_price_fourier(
        strike, maturity, pricing_params, option_type
    )["price"]
    heston_deltas = np.empty((n_paths, n_steps), dtype=float)
    bs_deltas = np.empty_like(heston_deltas)
    for step in range(n_steps):
        remaining = maturity - step * dt
        state_greeks = heston_option_price_delta_fourier_batch(
            spots[:, step],
            variances[:, step],
            strike,
            remaining,
            pricing_params,
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

    terminal_spot = spots[:, -1]
    if option_type == "call":
        terminal_payoff = np.maximum(terminal_spot - strike, 0.0)
    else:
        terminal_payoff = np.maximum(strike - terminal_spot, 0.0)

    cost_rate = transaction_cost_bps * 1e-4
    heston_result = _execute_hedge(
        spots, heston_deltas, terminal_payoff, initial_price, rate,
        dividend_yield, dt, cost_rate
    )
    bs_result = _execute_hedge(
        spots, bs_deltas, terminal_payoff, initial_price, rate,
        dividend_yield, dt, cost_rate
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
            "gross_pnl": gross_pnl,
        }

    heston_summary = summarize(heston_result)
    bs_summary = summarize(bs_result)
    pnl_difference = heston_result["net_pnl"] - bs_result["net_pnl"]
    paired_statistics = _paired_pathwise_statistics(pnl_difference)
    # Remove path vectors from the public summary; the JSON runner writes aggregate metrics.
    for summary in (heston_summary, bs_summary):
        summary.pop("net_pnl")
        summary.pop("gross_pnl")

    output = {
        "classification": "synthetic_physical_scenario_not_market_backtest",
        "option": {
            "type": option_type,
            "strike": float(strike),
            "maturity_years": float(maturity),
            "initial_heston_fair_value": float(initial_price),
            "initial_spot": float(pricing_params.spot),
        },
        "market_conventions": {
            "risk_free_rate_flat": float(rate),
            "dividend_yield_flat": float(dividend_yield),
            "transaction_cost_bps_per_turnover": float(transaction_cost_bps),
            "dividend_accrual": "trapezoidal per hedge interval",
        },
        "physical_scenario": {
            "total_return_drift": float(physical_total_return),
            "mean_reversion": float(physical_params.mean_reversion),
            "long_run_variance": float(physical_params.long_run_variance),
            "vol_of_variance": float(physical_params.vol_of_variance),
            "correlation": float(physical_params.correlation),
            "initial_variance": float(physical_params.variance0),
            "seed": int(seed),
        },
        "simulation": {
            "scheme": "QE-M",
            "n_paths": int(n_paths),
            "n_steps": int(n_steps),
            "hedge_frequency_per_year": float(n_steps / maturity),
            "confidence_level": float(confidence_level),
            "statistical_sampling_unit": (
                "independent simulated physical path; both hedge strategies are evaluated "
                "on the same path index"
            ),
        },
        "strategies": {
            "heston_fourier_delta": heston_summary,
            "frozen_black_scholes_delta": bs_summary,
        },
        "paired_comparison": {
            "mean_pnl_difference_heston_minus_bs": paired_statistics["mean_difference"],
            "standard_deviation_pnl_difference": paired_statistics[
                "standard_deviation_difference"
            ],
            "probability_heston_pnl_exceeds_bs_pnl": paired_statistics[
                "pathwise_outperformance_fraction"
            ],
            **paired_statistics,
            "pairing_unit": (
                "one independently simulated physical path evaluated under both strategies"
            ),
        },
        "interpretation_boundary": (
            "Both strategies use the same underlying scenario paths and initial Heston option premium. "
            "The Heston hedge uses the risk-neutral Heston delta evaluated at each simulated state; "
            "the comparator uses a frozen Black-Scholes volatility. Physical drift/variance parameters "
            "are hypothetical scenario inputs, not forecasts. Empirical VaR/ES describe this finite "
            "scenario sample only; they are not regulatory capital or market-calibrated risk measures."
        ),
    }
    if include_path_samples:
        output["path_samples"] = {
            "heston_fourier_delta": {
                "net_losses": (-heston_result["net_pnl"]).tolist(),
                "gross_losses": (-heston_result["gross_pnl"]).tolist(),
            },
            "frozen_black_scholes_delta": {
                "net_losses": (-bs_result["net_pnl"]).tolist(),
                "gross_losses": (-bs_result["gross_pnl"]).tolist(),
            },
        }
    return output
