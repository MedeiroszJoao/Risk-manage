from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc, t as student_t

from .heston import (
    HestonParameters,
    _qe_conditional_log_mgf,
    _qe_variance_step,
)


def _brownian_bridge_increments(
    bridge_normals: np.ndarray, maturity: float
) -> np.ndarray:
    """Map independent normals to standardized Brownian increments.

    The first coordinate generates the terminal Brownian value. Remaining
    coordinates recursively fill conditional midpoints, longest intervals
    first. Differencing the resulting Brownian path yields independent
    N(0,1) increments while assigning low-discrepancy coordinates to coarse
    path features first.
    """
    normals = np.asarray(bridge_normals, dtype=float)
    if normals.ndim != 2 or normals.shape[1] < 1:
        raise ValueError("bridge_normals must be a 2D array with at least one dimension")
    if not np.isfinite(normals).all():
        raise ValueError("bridge_normals must be finite")
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive")

    n_paths, n_steps = normals.shape
    dt = maturity / n_steps
    brownian = np.zeros((n_paths, n_steps + 1), dtype=float)
    brownian[:, -1] = math.sqrt(maturity) * normals[:, 0]

    intervals = [(0, n_steps)]
    coordinate = 1
    while intervals:
        next_intervals = []
        for left, right in intervals:
            if right - left <= 1:
                continue
            middle = (left + right) // 2
            left_width = middle - left
            right_width = right - middle
            total_width = right - left
            weight = left_width / total_width
            conditional_mean = (
                (1.0 - weight) * brownian[:, left]
                + weight * brownian[:, right]
            )
            conditional_variance = (
                left_width * right_width / total_width * dt
            )
            brownian[:, middle] = (
                conditional_mean
                + math.sqrt(conditional_variance) * normals[:, coordinate]
            )
            coordinate += 1
            next_intervals.append((left, middle))
            next_intervals.append((middle, right))
        intervals = next_intervals

    if coordinate != n_steps:
        raise ArithmeticError("Brownian bridge did not consume the expected dimensions")
    increments = np.diff(brownian, axis=1) / math.sqrt(dt)
    return increments


def _sobol_heston_payoffs(
    params: HestonParameters,
    strike: float,
    maturity: float,
    n_steps: int,
    power: int,
    seed: int,
    option_type: str,
    psi_c: float,
    brownian_bridge: bool,
):
    dimension = 2 * n_steps
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    uniforms = sampler.random_base2(power)
    normals = ndtri(np.clip(uniforms, 1e-14, 1.0 - 1e-14))
    n_paths = normals.shape[0]
    variance_normals = normals[:, :n_steps]
    stock_normals = normals[:, n_steps:]
    if brownian_bridge:
        variance_normals = _brownian_bridge_increments(
            variance_normals, maturity
        )
        stock_normals = _brownian_bridge_increments(stock_normals, maturity)

    dt = maturity / n_steps
    log_spot = np.full(n_paths, math.log(params.spot), dtype=float)
    variance = np.full(n_paths, params.variance0, dtype=float)
    discount = math.exp(-params.risk_free_rate * maturity)

    if params.vol_of_variance > 0.0:
        kappa = params.mean_reversion
        xi = params.vol_of_variance
        rho = params.correlation
        gamma1 = gamma2 = 0.5
        k1 = gamma1 * dt * (kappa * rho / xi - 0.5) - rho / xi
        k2 = gamma2 * dt * (kappa * rho / xi - 0.5) + rho / xi
        k3 = gamma1 * dt * (1.0 - rho * rho)
        k4 = gamma2 * dt * (1.0 - rho * rho)
        mgf_exponent = k2 + 0.5 * k4

    for step in range(n_steps):
        z_variance = variance_normals[:, step]
        z_stock = stock_normals[:, step]
        if params.vol_of_variance > 0.0:
            next_variance, conditional_mean, psi = _qe_variance_step(
                variance, z_variance, params, dt, psi_c
            )
            log_mgf = _qe_conditional_log_mgf(
                mgf_exponent, conditional_mean, psi, psi_c
            )
            correction = -(k1 + 0.5 * k3) * variance - log_mgf
            conditional_variance = np.maximum(
                k3 * variance + k4 * next_variance, 0.0
            )
            log_spot += (
                (params.risk_free_rate - params.dividend_yield) * dt
                + correction
                + k1 * variance
                + k2 * next_variance
                + np.sqrt(conditional_variance) * z_stock
            )
        else:
            decay = math.exp(-params.mean_reversion * dt)
            integrated_variance = (
                params.long_run_variance * dt
                + (variance - params.long_run_variance)
                * (1.0 - decay)
                / params.mean_reversion
            )
            next_variance = (
                params.long_run_variance
                + (variance - params.long_run_variance) * decay
            )
            log_spot += (
                (params.risk_free_rate - params.dividend_yield) * dt
                - 0.5 * integrated_variance
                + np.sqrt(np.maximum(integrated_variance, 0.0)) * z_stock
            )
        variance = next_variance

    terminal = np.exp(log_spot)
    if not np.isfinite(terminal).all():
        raise ArithmeticError("Sobol Heston simulation produced non-finite terminal spots")
    if option_type == "call":
        payoff = discount * np.maximum(terminal - strike, 0.0)
    else:
        payoff = discount * np.maximum(strike - terminal, 0.0)
    discounted_stock = discount * terminal
    return payoff, discounted_stock


def heston_randomized_qmc_price(
    strike: float,
    maturity: float,
    params: HestonParameters,
    *,
    option_type: str = "call",
    sobol_power: int = 10,
    n_scrambles: int = 8,
    n_steps: int = 64,
    seed: int = 20261125,
    control_variate: bool = True,
    brownian_bridge: bool = True,
    psi_c: float = 1.5,
    confidence_level: float = 0.95,
) -> dict:
    """Price a European Heston option with randomized Sobol QMC and QE-M.

    Independent scrambled Sobol replicates provide an error estimate from the
    variation of replicate means. Brownian bridge maps early Sobol dimensions
    to coarse Brownian path features. A discounted-stock control-variate beta
    is estimated on a separate randomized Sobol pilot scramble, independent
    from all valuation scrambles.

    The t interval measures randomization error conditional on the QE time grid.
    It excludes weak time-discretization bias, parameter uncertainty and model
    risk. The pilot beta is held fixed across valuation scrambles.
    """
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive")
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if sobol_power < 4 or n_scrambles < 4 or n_steps < 1:
        raise ValueError("sobol_power >= 4, n_scrambles >= 4 and n_steps >= 1 are required")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not np.isfinite(psi_c) or psi_c <= 0.0:
        raise ValueError("psi_c must be finite and positive")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0.5, 1)")

    n_points = 2**sobol_power
    control_mean = params.spot * math.exp(-params.dividend_yield * maturity)
    discount = math.exp(-params.risk_free_rate * maturity)
    beta = 0.0
    pilot_size = 0
    if control_variate:
        pilot_payoff, pilot_stock = _sobol_heston_payoffs(
            params,
            strike,
            maturity,
            n_steps,
            sobol_power,
            seed + 1_000_003,
            option_type,
            psi_c,
            brownian_bridge,
        )
        stock_variance = float(np.var(pilot_stock, ddof=1))
        if stock_variance <= np.finfo(float).tiny:
            raise ArithmeticError("QMC control-variate pilot has zero stock-price variance")
        beta = float(
            np.cov(pilot_payoff, pilot_stock, ddof=1)[0, 1] / stock_variance
        )
        pilot_size = int(pilot_payoff.size)

    raw_means = []
    adjusted_means = []
    discounted_stock_means = []
    for replicate in range(n_scrambles):
        payoff, discounted_stock = _sobol_heston_payoffs(
            params,
            strike,
            maturity,
            n_steps,
            sobol_power,
            seed + replicate + 1,
            option_type,
            psi_c,
            brownian_bridge,
        )
        raw_means.append(float(np.mean(payoff)))
        adjusted = payoff - beta * (discounted_stock - control_mean)
        adjusted_means.append(float(np.mean(adjusted)))
        discounted_stock_means.append(float(np.mean(discounted_stock)))

    raw_means = np.asarray(raw_means, dtype=float)
    adjusted_means = np.asarray(adjusted_means, dtype=float)
    raw_standard_error = float(
        np.std(raw_means, ddof=1) / math.sqrt(n_scrambles)
    )
    standard_error = float(
        np.std(adjusted_means, ddof=1) / math.sqrt(n_scrambles)
    )
    critical = float(
        student_t.ppf(0.5 + confidence_level / 2.0, n_scrambles - 1)
    )
    price = float(np.mean(adjusted_means))
    interval = (
        price - critical * standard_error,
        price + critical * standard_error,
    )
    discounted_stock_mean = float(np.mean(discounted_stock_means))
    return {
        "price": price,
        "unadjusted_price": float(np.mean(raw_means)),
        "standard_error_across_scrambles": standard_error,
        "unadjusted_standard_error_across_scrambles": raw_standard_error,
        "confidence_interval": (float(interval[0]), float(interval[1])),
        "confidence_level": float(confidence_level),
        "n_scrambles": int(n_scrambles),
        "points_per_scramble": int(n_points),
        "total_valuation_points": int(n_points * n_scrambles),
        "pilot_points": pilot_size,
        "sobol_power": int(sobol_power),
        "n_steps": int(n_steps),
        "scheme": "randomized Sobol QMC + Brownian bridge + Heston QE-M",
        "brownian_bridge": bool(brownian_bridge),
        "control_variate": bool(control_variate),
        "control_beta_from_independent_pilot": float(beta),
        "variance_reduction_factor_across_scrambles": (
            float((raw_standard_error / standard_error) ** 2)
            if standard_error > 0.0
            else None
        ),
        "discounted_terminal_stock_mean": discounted_stock_mean,
        "discounted_terminal_stock_target": float(control_mean),
        "stock_martingale_relative_error": float(
            discounted_stock_mean / control_mean - 1.0
        ),
        "raw_scramble_means": [float(value) for value in raw_means],
        "adjusted_scramble_means": [float(value) for value in adjusted_means],
        "uncertainty_boundary": (
            "The t interval is based on independent randomized Sobol scramble means and is "
            "conditional on QE time discretization. It excludes weak discretization bias, "
            "parameter uncertainty, model risk and uncertainty in the independent pilot beta."
        ),
    }
