from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
from scipy.special import ndtr

from .heston import (
    HestonParameters,
    heston_option_price_delta_fourier_batch,
    heston_price_fourier,
    heston_simulate_paths,
)


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
    turnover = initial_spot * np.abs(deltas[:, 0])
    previous_delta = deltas[:, 0].copy()
    discount_growth = math.exp(rate * dt)

    for step in range(1, n_steps):
        old_spot = spots[:, step - 1]
        new_spot = spots[:, step]
        cash *= discount_growth
        cash += dividend_yield * previous_delta * 0.5 * (old_spot + new_spot) * dt

        new_delta = deltas[:, step]
        trade = new_delta - previous_delta
        trade_notional = new_spot * np.abs(trade)
        cost = transaction_cost_rate * trade_notional
        cash -= new_spot * trade + cost
        transaction_costs += cost
        turnover += trade_notional
        previous_delta = new_delta

    old_spot = spots[:, -2]
    terminal_spot = spots[:, -1]
    cash *= discount_growth
    cash += dividend_yield * previous_delta * 0.5 * (
        old_spot + terminal_spot
    ) * dt
    gross_pnl = cash + previous_delta * terminal_spot - payoff
    net_pnl = gross_pnl
    # Transaction costs have already been debited from the self-financing cash account.
    return {
        "gross_pnl": gross_pnl + transaction_costs,
        "net_pnl": net_pnl,
        "transaction_costs": transaction_costs,
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
            "average_turnover": float(np.mean(result["turnover"])),
            "transaction_cost_total_mean": float(np.mean(result["transaction_costs"])),
            "net_pnl": net_pnl,
            "gross_pnl": gross_pnl,
        }

    heston_summary = summarize(heston_result)
    bs_summary = summarize(bs_result)
    pnl_difference = heston_result["net_pnl"] - bs_result["net_pnl"]
    # Remove path vectors from the public summary; the JSON runner writes aggregate metrics.
    for summary in (heston_summary, bs_summary):
        summary.pop("net_pnl")
        summary.pop("gross_pnl")

    return {
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
        },
        "strategies": {
            "heston_fourier_delta": heston_summary,
            "frozen_black_scholes_delta": bs_summary,
        },
        "paired_comparison": {
            "mean_pnl_difference_heston_minus_bs": float(np.mean(pnl_difference)),
            "standard_deviation_pnl_difference": float(np.std(pnl_difference, ddof=1)),
            "probability_heston_pnl_exceeds_bs_pnl": float(
                np.mean(pnl_difference > 0.0)
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
