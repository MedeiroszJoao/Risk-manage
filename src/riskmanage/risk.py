from __future__ import annotations

import math

import numpy as np
import pandas as pd

LIQUIDITY_HORIZONS = (10, 20, 40, 60, 120)
EMPIRICAL_QUANTILE_METHOD = "inverted_cdf"


def _finite_vector(values, name="values"):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a finite 1D vector with at least two observations")
    return x


def historical_var_es(losses, confidence=0.975):
    """Historical VaR and ES using the empirical inverse-CDF convention.

    VaR is the left-continuous empirical quantile (NumPy's inverted_cdf
    convention). ES is the integrated empirical quantile over [confidence, 1].
    When the tail probability cuts through a point mass at VaR, only the
    required fractional mass at that boundary is included.
    """
    x = _finite_vector(losses, "losses")
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be between 0.5 and 1")

    var = float(np.quantile(x, confidence, method=EMPIRICAL_QUANTILE_METHOD))
    strict_tail = x[x > var]
    boundary_ties = int(np.count_nonzero(x == var))

    tail_mass = float(len(x) * (1.0 - confidence))
    nearest_integer = round(tail_mass)
    numerical_tolerance = (
        16.0 * np.finfo(float).eps * max(float(len(x)), tail_mass, 1.0)
    )
    if abs(tail_mass - nearest_integer) <= numerical_tolerance:
        tail_mass = float(nearest_integer)

    boundary_mass = tail_mass - len(strict_tail)
    if (
        boundary_mass < -numerical_tolerance
        or boundary_mass > boundary_ties + numerical_tolerance
    ):
        raise ArithmeticError("empirical ES boundary mass is inconsistent with the quantile")

    boundary_mass = min(max(boundary_mass, 0.0), float(boundary_ties))
    strict_tail_sum = math.fsum(float(value) for value in strict_tail)
    expected_shortfall = (strict_tail_sum + boundary_mass * var) / tail_mass

    return {
        "var": var,
        "expected_shortfall": float(expected_shortfall),
        # Inclusive count is retained for backwards compatibility and shows
        # how many observations lie at or beyond the reported VaR.
        "tail_observations": int(len(strict_tail) + boundary_ties),
        "strict_tail_observations": int(len(strict_tail)),
        "var_tie_observations": boundary_ties,
        "fractional_var_boundary_mass": float(boundary_mass),
        "tail_mass_observations": float(tail_mass),
        "observations": int(len(x)),
        "confidence": float(confidence),
        "quantile_method": EMPIRICAL_QUANTILE_METHOD,
        "expected_tail_observations": float(tail_mass),
    }


def rolling_historical_var(losses, window=250, confidence=0.99):
    x = np.asarray(losses, dtype=float)
    if x.ndim != 1 or window < 20 or len(x) <= window:
        raise ValueError("need a 1D series longer than the forecast window")
    if not np.isfinite(x).all():
        raise ValueError("loss series contains non-finite values")
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be between 0.5 and 1")

    out = np.full(len(x), np.nan)
    for i in range(window, len(x)):
        out[i] = np.quantile(
            x[i - window : i],
            confidence,
            method=EMPIRICAL_QUANTILE_METHOD,
        )
    return pd.Series(out)


def _ten_day_losses(daily_losses, horizon=10):
    x = _finite_vector(daily_losses, "daily losses")
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, (bool, np.bool_)):
        raise ValueError("horizon must be an integer")
    if horizon < 1 or len(x) < horizon:
        raise ValueError("insufficient observations for requested horizon")
    return np.convolve(x, np.ones(horizon), mode="valid")


def liquidity_adjusted_es(
    factor_losses: dict[str, np.ndarray],
    liquidity_horizons: dict[str, int],
    confidence=0.975,
    base_horizon=10,
):
    if not factor_losses:
        raise ValueError("factor loss mapping cannot be empty")
    if base_horizon != 10 or isinstance(base_horizon, (bool, np.bool_)):
        raise ValueError("the Basel-style liquidity adjustment uses a fixed 10-day base horizon")
    if set(factor_losses) != set(liquidity_horizons):
        raise ValueError("each factor must have one liquidity horizon")
    invalid = {
        key: value
        for key, value in liquidity_horizons.items()
        if value not in LIQUIDITY_HORIZONS
    }
    if invalid:
        raise ValueError(f"invalid liquidity horizon mapping: {invalid}")
    arrays = {key: _finite_vector(value, key) for key, value in factor_losses.items()}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("factor loss vectors must be aligned")

    all_daily = np.sum(np.vstack(list(arrays.values())), axis=0)
    base_es = historical_var_es(
        _ten_day_losses(all_daily, base_horizon), confidence
    )["expected_shortfall"]

    components = []
    for previous, liquidity_horizon in zip(
        LIQUIDITY_HORIZONS[:-1], LIQUIDITY_HORIZONS[1:]
    ):
        names = [
            name
            for name, horizon in liquidity_horizons.items()
            if horizon >= liquidity_horizon
        ]
        if not names:
            component_es = 0.0
        else:
            subtotal = np.sum(np.vstack([arrays[name] for name in names]), axis=0)
            component_es = historical_var_es(
                _ten_day_losses(subtotal, base_horizon), confidence
            )["expected_shortfall"]

        weight = np.sqrt((liquidity_horizon - previous) / base_horizon)
        components.append(
            {
                "liquidity_horizon": liquidity_horizon,
                "factors": names,
                "partial_es": float(component_es),
                "scaling": float(weight),
                "scaled_component": float(component_es * weight),
            }
        )

    total = float(
        np.sqrt(base_es**2 + sum(item["scaled_component"] ** 2 for item in components))
    )
    return {
        "base_es_10d": float(base_es),
        "liquidity_adjusted_es": total,
        "confidence": confidence,
        "base_horizon": base_horizon,
        "components": components,
        "empirical_quantile_method": EMPIRICAL_QUANTILE_METHOD,
        "expected_shortfall_convention": "integrated_empirical_quantile_fractional_var_mass",
    }


def select_stress_window(
    factor_losses: dict[str, np.ndarray],
    window_days: int = 250,
    confidence: float = 0.975,
):
    """Select the contiguous window with the largest empirical 10-observation ES.

    Inputs are aligned daily loss series (positive values are losses). For each
    candidate window, the function computes overlapping 10-observation portfolio
    losses and their empirical expected shortfall at the requested confidence.
    The window with the highest ES is selected; ties resolve to the earliest.
    This is a descriptive historical selector, not the supervisory FRTB
    stress-period selection procedure.
    """
    if not factor_losses:
        raise ValueError("factor loss mapping cannot be empty")
    if not isinstance(window_days, (int, np.integer)) or window_days < 1:
        raise ValueError("window_days must be a positive integer")
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be between 0.5 and 1")

    arrays = {
        str(name): _finite_vector(values, f"{name} losses")
        for name, values in factor_losses.items()
    }
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("factor loss vectors must be aligned")
    observations = lengths.pop()
    if window_days > observations:
        raise ValueError("window_days cannot exceed the number of observations")

    if window_days < 11:
        raise ValueError("window_days must be at least 11 to estimate 10-observation ES")

    portfolio_losses = np.sum(np.vstack(list(arrays.values())), axis=0)
    candidate_count = observations - window_days + 1
    candidate_es = np.empty(candidate_count, dtype=float)
    candidate_cumulative = np.empty(candidate_count, dtype=float)
    for candidate_start in range(candidate_count):
        candidate = portfolio_losses[candidate_start : candidate_start + window_days]
        ten_observation_losses = _ten_day_losses(candidate, horizon=10)
        candidate_es[candidate_start] = historical_var_es(
            ten_observation_losses, confidence
        )["expected_shortfall"]
        candidate_cumulative[candidate_start] = float(np.sum(candidate))

    # np.argmax deterministically resolves equal-ES windows to the earliest start.
    start_index = int(np.argmax(candidate_es))
    end_index = start_index + int(window_days) - 1
    factor_contributions = {
        name: float(values[start_index : end_index + 1].sum())
        for name, values in arrays.items()
    }
    return {
        "start_index": start_index,
        "end_index": end_index,
        "window_days": int(window_days),
        "confidence": float(confidence),
        "stress_es_10d": float(candidate_es[start_index]),
        "portfolio_cumulative_loss": float(candidate_cumulative[start_index]),
        "factor_cumulative_losses": factor_contributions,
        "selection_method": "maximum_historical_10_observation_expected_shortfall",
        "tie_break": "earliest_window",
    }
