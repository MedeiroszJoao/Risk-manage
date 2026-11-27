from __future__ import annotations

import math

import numpy as np
import pandas as pd

LIQUIDITY_HORIZONS = (10, 20, 40, 60, 120)
EMPIRICAL_QUANTILE_METHOD = "inverted_cdf"


def _finite_vector(values, name="values"):
    objects = np.asarray(values, dtype=object)
    if any(isinstance(value, (bool, np.bool_)) for value in objects.flat):
        raise ValueError(
            f"{name} must be a 1D numeric vector; boolean observations are not allowed"
        )
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError(
            f"{name} must be a 1D numeric vector; boolean, string and object "
            "coercion is not allowed"
        )
    x = raw.astype(float, copy=False)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite and contain at least two observations")
    return x


def _confidence_level(confidence):
    if isinstance(confidence, (bool, np.bool_)) or not isinstance(
        confidence, (int, float, np.integer, np.floating)
    ):
        raise ValueError("confidence must be a numeric value in (0.5, 1)")
    value = float(confidence)
    if not np.isfinite(value) or not 0.5 < value < 1.0:
        raise ValueError("confidence must be finite and in (0.5, 1)")
    return value


def _strict_integer(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def historical_var_es(losses, confidence=0.975):
    """Historical VaR and ES using the empirical inverse-CDF convention.

    VaR is the left-continuous empirical quantile (NumPy's inverted_cdf
    convention). ES is the integrated empirical quantile over [confidence, 1].
    When the tail probability cuts through a point mass at VaR, only the
    required fractional mass at that boundary is included.
    """
    x = _finite_vector(losses, "losses")
    confidence = _confidence_level(confidence)

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
    """One-step-ahead VaR using only the preceding window of trading observations."""
    x = _finite_vector(losses, "loss series")
    window = _strict_integer(window, "window", minimum=20)
    confidence = _confidence_level(confidence)
    if len(x) <= window:
        raise ValueError("need a 1D series longer than the forecast window")

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
    horizon = _strict_integer(horizon, "horizon")
    if len(x) < horizon:
        raise ValueError("insufficient observations for requested horizon")
    return np.convolve(x, np.ones(horizon), mode="valid")


def liquidity_adjusted_es(
    factor_losses: dict[str, np.ndarray],
    liquidity_horizons: dict[str, int],
    confidence=0.975,
    base_horizon=10,
    joint_horizon_losses: dict[int, np.ndarray] | None = None,
    joint_horizon_metadata: dict | None = None,
):
    if not factor_losses:
        raise ValueError("factor loss mapping cannot be empty")
    base_horizon = _strict_integer(base_horizon, "base_horizon")
    if base_horizon != 10:
        raise ValueError("the Basel-style liquidity adjustment uses a fixed 10-day base horizon")
    if set(factor_losses) != set(liquidity_horizons):
        raise ValueError("each factor must have one liquidity horizon")
    normalized_horizons = {
        key: _strict_integer(value, f"liquidity_horizon[{key}]")
        for key, value in liquidity_horizons.items()
    }
    invalid = {
        key: value
        for key, value in normalized_horizons.items()
        if value not in LIQUIDITY_HORIZONS
    }
    if invalid:
        raise ValueError(f"invalid liquidity horizon mapping: {invalid}")
    confidence = _confidence_level(confidence)
    arrays = {key: _finite_vector(value, key) for key, value in factor_losses.items()}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("factor loss vectors must be aligned")

    expected_length = len(next(iter(arrays.values())))
    joint_arrays = None
    joint_input_horizon_days = None
    horizon_metadata = {}
    if joint_horizon_losses is not None:
        if not isinstance(joint_horizon_losses, dict) or set(joint_horizon_losses) != set(LIQUIDITY_HORIZONS):
            raise ValueError(f"joint_horizon_losses must contain exactly {LIQUIDITY_HORIZONS}")
        if joint_horizon_metadata is not None and not isinstance(joint_horizon_metadata, dict):
            raise ValueError("joint_horizon_metadata must be a dictionary")
        if joint_horizon_metadata is not None:
            horizon_metadata = joint_horizon_metadata.get(
                "liquidity_horizon_metadata", joint_horizon_metadata
            )
            if not isinstance(horizon_metadata, dict):
                raise ValueError("liquidity_horizon_metadata must be a dictionary")

        joint_input_horizon_days = horizon_metadata.get(
            "observation_horizon_days", 1
        )
        if (
            isinstance(joint_input_horizon_days, (bool, np.bool_))
            or not isinstance(joint_input_horizon_days, (int, np.integer))
            or int(joint_input_horizon_days) not in (1, base_horizon)
        ):
            raise ValueError(
                "joint observation_horizon_days must be 1 or the 10-day base horizon"
            )
        joint_input_horizon_days = int(joint_input_horizon_days)
        expected_joint_length = (
            expected_length
            if joint_input_horizon_days == 1
            else expected_length - base_horizon + 1
        )
        if expected_joint_length <= 0:
            raise ValueError("insufficient daily factor history for the 10-day base horizon")

        joint_arrays = {
            horizon: _finite_vector(
                joint_horizon_losses[horizon], f"joint_horizon_losses[{horizon}]"
            )
            for horizon in LIQUIDITY_HORIZONS
        }
        if any(len(values) != expected_joint_length for values in joint_arrays.values()):
            raise ValueError(
                "joint horizon loss vectors must match the declared observation horizon"
            )
        method = horizon_metadata.get(
            "method", "position_level_joint_full_revaluation"
        )
        base_daily = joint_arrays[base_horizon]
        base_losses = (
            base_daily
            if joint_input_horizon_days == base_horizon
            else _ten_day_losses(base_daily, base_horizon)
        )
    else:
        method = "additive_factor_contribution_approximation"
        base_daily = np.sum(np.vstack(list(arrays.values())), axis=0)
        base_losses = _ten_day_losses(base_daily, base_horizon)
        if joint_horizon_metadata is not None:
            raise ValueError("joint_horizon_metadata cannot be supplied without joint_horizon_losses")

    base_es = historical_var_es(base_losses, confidence)["expected_shortfall"]

    factor_subset_by_horizon = horizon_metadata.get("factor_subset_by_horizon", {})
    positions_by_horizon = horizon_metadata.get("positions_by_horizon", {})
    components = []
    for previous, liquidity_horizon in zip(
        LIQUIDITY_HORIZONS[:-1], LIQUIDITY_HORIZONS[1:]
    ):
        names = [
            name
            for name, horizon in normalized_horizons.items()
            if horizon >= liquidity_horizon
        ]
        if joint_arrays is not None:
            subtotal = joint_arrays[liquidity_horizon]
            component_losses = (
                subtotal
                if joint_input_horizon_days == base_horizon
                else _ten_day_losses(subtotal, base_horizon)
            )
            component_es = historical_var_es(
                component_losses, confidence
            )["expected_shortfall"]
            component_factors = factor_subset_by_horizon.get(str(liquidity_horizon), names)
            included_positions = positions_by_horizon.get(str(liquidity_horizon), [])
        elif not names:
            component_es = 0.0
            component_factors = []
            included_positions = []
        else:
            subtotal = np.sum(np.vstack([arrays[name] for name in names]), axis=0)
            component_es = historical_var_es(
                _ten_day_losses(subtotal, base_horizon), confidence
            )["expected_shortfall"]
            component_factors = names
            included_positions = []

        weight = np.sqrt((liquidity_horizon - previous) / base_horizon)
        components.append(
            {
                "liquidity_horizon": liquidity_horizon,
                "factors": component_factors,
                "positions": included_positions,
                "partial_es": float(component_es),
                "scaling": float(weight),
                "scaled_component": float(component_es * weight),
                "partial_es_method": method,
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
        "calculation_method": method,
        "theta_included": (
            bool(horizon_metadata.get("theta_included", False))
            if joint_arrays is not None else None
        ),
        "joint_horizon_input_observation_days": joint_input_horizon_days,
        "joint_horizon_metadata": horizon_metadata if joint_arrays is not None else None,
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
    window_days = _strict_integer(window_days, "window_days")
    confidence = _confidence_level(confidence)

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
