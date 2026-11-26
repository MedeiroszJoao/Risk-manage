from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
from scipy.stats import t as student_t

from .heston import HestonParameters, _heston_terminal_spots


def _pair_average(values: np.ndarray, antithetic: bool) -> np.ndarray:
    if not antithetic:
        return values
    half = values.size // 2
    return 0.5 * (values[:half] + values[half:])


def _summarize(values: np.ndarray, confidence_level: float) -> dict:
    estimate = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(values.size))
    critical = float(
        student_t.ppf(0.5 + confidence_level / 2.0, values.size - 1)
    )
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "confidence_interval": (
            estimate - critical * standard_error,
            estimate + critical * standard_error,
        ),
        "confidence_level": confidence_level,
        "independent_observations": int(values.size),
    }


def heston_monte_carlo_greeks(
    strike: float,
    maturity: float,
    params: HestonParameters,
    *,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 128,
    seed: int = 20261125,
    antithetic: bool = True,
    spot_bump_relative: float = 0.005,
    initial_volatility_bump: float = 0.005,
    psi_c: float = 1.5,
    confidence_level: float = 0.95,
) -> dict:
    """Estimate European Heston price, Delta, Gamma and initial-volatility Vega.

    Delta is estimated by a pathwise payoff derivative and independently by a
    central common-random-number (CRN) bump. Gamma and Vega use CRN central
    differences. Vega means dPrice/d(sqrt(v0)), not Black-Scholes implied-vol
    vega. Antithetic pairs are averaged before standard errors are computed.
    Confidence intervals are conditional on the QE-M time grid and finite-
    difference bumps; they exclude time-discretization and bump bias, parameter
    uncertainty and model risk.
    """
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive")
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if n_paths < 4 or n_steps < 1:
        raise ValueError("n_paths must be at least 4 and n_steps at least 1")
    if antithetic and n_paths % 2:
        raise ValueError("n_paths must be even when antithetic variates are enabled")
    if params.vol_of_variance <= 0.0:
        raise ValueError("QE-M Greeks require positive volatility of variance")
    if (
        not np.isfinite(spot_bump_relative)
        or not 0.0 < spot_bump_relative < 0.1
    ):
        raise ValueError("spot_bump_relative must lie in (0, 0.1)")
    if (
        not np.isfinite(initial_volatility_bump)
        or initial_volatility_bump <= 0.0
    ):
        raise ValueError("initial_volatility_bump must be finite and positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")

    discount = math.exp(-params.risk_free_rate * maturity)
    terminal = _heston_terminal_spots(
        params,
        maturity,
        n_steps,
        n_paths,
        np.random.default_rng(seed),
        antithetic,
        psi_c,
    )

    def payoff(terminal_spot: np.ndarray) -> np.ndarray:
        if option_type == "call":
            return discount * np.maximum(terminal_spot - strike, 0.0)
        return discount * np.maximum(strike - terminal_spot, 0.0)

    base_payoff = payoff(terminal)
    price_observations = _pair_average(base_payoff, antithetic)
    pathwise_indicator = (
        terminal > strike if option_type == "call" else terminal < strike
    )
    pathwise_sign = 1.0 if option_type == "call" else -1.0
    pathwise_delta = (
        pathwise_sign * discount * pathwise_indicator * terminal / params.spot
    )
    pathwise_delta_observations = _pair_average(pathwise_delta, antithetic)

    spot_bump = params.spot * spot_bump_relative
    spot_up_payoff = payoff(
        terminal * ((params.spot + spot_bump) / params.spot)
    )
    spot_down_payoff = payoff(
        terminal * ((params.spot - spot_bump) / params.spot)
    )
    delta_crn = (spot_up_payoff - spot_down_payoff) / (2.0 * spot_bump)
    gamma_crn = (
        spot_up_payoff - 2.0 * base_payoff + spot_down_payoff
    ) / (spot_bump * spot_bump)

    initial_volatility = math.sqrt(params.variance0)
    if initial_volatility <= initial_volatility_bump:
        raise ValueError(
            "initial_volatility_bump must be smaller than sqrt(variance0)"
        )
    vol_up_params = replace(
        params, variance0=(initial_volatility + initial_volatility_bump) ** 2
    )
    vol_down_params = replace(
        params, variance0=(initial_volatility - initial_volatility_bump) ** 2
    )
    vol_up_terminal = _heston_terminal_spots(
        vol_up_params,
        maturity,
        n_steps,
        n_paths,
        np.random.default_rng(seed),
        antithetic,
        psi_c,
    )
    vol_down_terminal = _heston_terminal_spots(
        vol_down_params,
        maturity,
        n_steps,
        n_paths,
        np.random.default_rng(seed),
        antithetic,
        psi_c,
    )
    vol_up_payoff = payoff(vol_up_terminal)
    vol_down_payoff = payoff(vol_down_terminal)
    vega_crn = (vol_up_payoff - vol_down_payoff) / (
        2.0 * initial_volatility_bump
    )
    volga_crn = (
        vol_up_payoff - 2.0 * base_payoff + vol_down_payoff
    ) / (initial_volatility_bump * initial_volatility_bump)
    vanna_crn = (
        payoff(vol_up_terminal * ((params.spot + spot_bump) / params.spot))
        - payoff(vol_up_terminal * ((params.spot - spot_bump) / params.spot))
        - payoff(vol_down_terminal * ((params.spot + spot_bump) / params.spot))
        + payoff(vol_down_terminal * ((params.spot - spot_bump) / params.spot))
    ) / (4.0 * spot_bump * initial_volatility_bump)

    return {
        "price": _summarize(price_observations, confidence_level),
        "delta_pathwise": _summarize(
            pathwise_delta_observations, confidence_level
        ),
        "delta_common_random_numbers": _summarize(
            _pair_average(delta_crn, antithetic), confidence_level
        ),
        "gamma_common_random_numbers": _summarize(
            _pair_average(gamma_crn, antithetic), confidence_level
        ),
        "vega_initial_volatility_common_random_numbers": _summarize(
            _pair_average(vega_crn, antithetic), confidence_level
        ),
        "vanna_spot_initial_volatility_common_random_numbers": _summarize(
            _pair_average(vanna_crn, antithetic), confidence_level
        ),
        "volga_initial_volatility_common_random_numbers": _summarize(
            _pair_average(volga_crn, antithetic), confidence_level
        ),
        "greek_conventions": {
            "delta": "dPrice/dSpot; pathwise and CRN central difference",
            "gamma": "d2Price/dSpot2; CRN central difference",
            "vega": (
                "dPrice/dsqrt(v0); CRN central difference, "
                "not implied-volatility vega"
            ),
            "vanna": (
                "d2Price/(dSpot d sqrt(v0)); CRN mixed central difference"
            ),
            "volga": "d2Price/dsqrt(v0)2; CRN central difference",
        },
        "simulation": {
            "scheme": "Heston QE-M",
            "n_paths": int(n_paths),
            "n_steps": int(n_steps),
            "antithetic": bool(antithetic),
            "common_random_numbers": True,
            "seed": int(seed),
            "spot_bump": float(spot_bump),
            "initial_volatility_bump": float(initial_volatility_bump),
            "effective_samples": int(price_observations.size),
            "feller_ratio": float(params.feller_ratio),
        },
        "uncertainty_boundary": (
            "Standard errors and confidence intervals quantify sampling error conditional "
            "on the QE-M time grid and finite-difference bumps. They exclude QE time-"
            "discretization bias, finite-difference truncation bias, parameter uncertainty "
            "and model risk."
        ),
    }
